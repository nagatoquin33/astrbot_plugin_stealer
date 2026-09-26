import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import (
    EventMessageType,
    PermissionType,
    PlatformAdapterType,
)
from astrbot.api.message_components import Image as MessageImage
from astrbot.api.star import Context, Star

from .core.commands.command_handler import CommandHandler
from .core.commands.image_mgmt_command import ImageManagementCommand
from .core.commands.index_rebuild_command import IndexRebuildCommand
from .core.commands.target_filter_command import TargetFilterCommand
from .core.config.config import PluginConfig
from .core.db.database_service import DatabaseService
from .core.search.meme_selector import MemeSelector
from .core.events.event_handler import EventHandler
from .core.events.meme_sender_engine import MemeSenderEngine
from .core.db.index_manager import IndexManager
from .core.processing.natural_emotion_analyzer import NaturalEmotionAnalyzer
from .core.processing.image_processor_service import ImageProcessorService
from .core.processing.image_render_service import ImageRenderService
from .core.maintenance.service import MaintenanceService
from .core.sources.source_service import SourceService
from .task_scheduler import TaskScheduler
from .plugin_api import PluginAPI
from .core.tools.meme_search_tool import MemeSearchToolWorkflow
from .core.tools.meme_steal_tool import MemeStealToolWorkflow

try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None


class Main(MemeSearchToolWorkflow, MemeStealToolWorkflow, Star):
    """表情包偷取与发送插件。

    功能：
    - 监听消息中的图片并自动保存到插件数据目录
    - 使用当前会话的多模态模型进行情绪分类与标签生成
    - 建立分类索引，支持自动与手动在合适时机发送表情包
    """

    # 常量定义
    BACKEND_TAG = "emoji_stealer"
    SEARCH_MEME_TOOL_NAME = "search_meme"
    SEND_MEME_TOOL_NAME = "send_meme"

    # 时间间隔常量（单位：秒）
    RAW_CLEANUP_INTERVAL_SECONDS = 30 * 60  # 30分钟
    CAPACITY_CONTROL_INTERVAL_SECONDS = 60 * 60  # 60分钟

    # 超时和处理常量
    IMAGE_PROCESSING_TIMEOUT_SECONDS = 120  # 图片处理超时时间（GIF动图处理需要更长时间）
    MAX_SEARCH_RESULTS = 5  # 搜索表情包最大返回数量（避免 FC 输出过长）
    AUTO_EMOJI_COOLDOWN_SECONDS = 20  # 同一会话自动发表情的最短间隔

    # 从外部文件加载的提示词（已迁移到ImageProcessorService）

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)

        # 初始化插件配置
        self.plugin_config = PluginConfig(config, context)

        self.base_dir: Path = self.plugin_config.data_dir
        self.raw_dir: Path = self.plugin_config.raw_dir
        self.categories_dir: Path = self.plugin_config.categories_dir
        self.cache_dir: Path = self.plugin_config.cache_dir

        # 配置统一通过 self.plugin_config 读取（pydantic 模型）。
        # v2.7.5+ 删除了 _sync_all_config() 实例属性镜像。

        # 初始化核心服务类
        self.db_service = DatabaseService(self.cache_dir / "emoji.db")
        self.source_service = SourceService(self)
        self.command_handler = CommandHandler(self)
        self.image_commands = ImageManagementCommand(self)
        self.index_commands = IndexRebuildCommand(self)
        self.target_commands = TargetFilterCommand(self)
        self.web_server = None
        self.plugin_api = PluginAPI(self)
        self.plugin_api.register(context)

        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.image_render_service = ImageRenderService(self)
        self.meme_selector = MemeSelector(self)
        self.task_scheduler = TaskScheduler()

        # 初始化自然语言情绪分析器（新增）
        self.emotion_analyzer = NaturalEmotionAnalyzer(self)

        self.index_manager = IndexManager(self)
        self._emoji_sender_engine = MemeSenderEngine(self)
        self.maintenance = MaintenanceService(self)

        # 运行时属性
        self._terminated: bool = False  # 终止标志位，防止重复清理
        # 强制捕获窗口已迁移到 EventHandler

    def __getattr__(self, name: str):
        """将未定义的属性访问自动代理到 plugin_config。

        v2.7.5+ 配置统一通过 plugin_config (Pydantic 模型) 管理，
        但 core/ 中部分代码仍直接访问 plugin_instance.steal_meme 等。
        通过 __getattr__ 自动代理，无需逐个修改调用方。
        """
        if name == "plugin_config":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'plugin_config'")
        cfg = self.__dict__.get("plugin_config")
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        # 区分：plugin_config 未初始化 vs. 属性完全不存在
        if cfg is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' "
                f"(plugin_config 尚未初始化)"
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}' "
            f"(plugin_config 中也不存在该属性)"
        )

    def _load_vision_provider_id(self) -> str:
        """加载视觉模型提供商ID。"""
        provider_id = getattr(self.plugin_config, "vision_provider_id", "")
        return str(provider_id).strip() if provider_id else ""

    def _apply_prompts(self, prompts: dict) -> None:
        """应用提示词配置。"""
        for key, value in prompts.items():
            setattr(self, key, value)
        final_prompts = self.plugin_config.get_prompts(prompts)
        self.image_processor_service.update_config(
            emoji_classification_prompt=final_prompts.get("emoji_classification_prompt"),
            emoji_classification_with_filter_prompt=final_prompts.get(
                "emoji_classification_with_filter_prompt"
            ),
        )

    def _auto_merge_existing_categories(self) -> None:
        """自动合并已存在的分类目录到配置中。

        注意：基于用户当前已加载的 categories（来自 categories.json）而非
        DEFAULT_CATEGORIES 作为合并基线。这样用户主动删除的预定义类别不会
        被重新加回，仅自动发现磁盘上用户未配置的自定义类别。
        """
        current = list(getattr(self, "categories", None) or [])
        # 兼容：若 categories 尚未加载，回退到已存储配置或默认列表
        if not current:
            current = list(getattr(self.plugin_config, "categories", None) or [])
        if not current:
            current = list(getattr(self.plugin_config, "DEFAULT_CATEGORIES", []) or [])
        current_set = set(current)
        protected = set(getattr(self.plugin_config, "DEFAULT_CATEGORIES", []) or [])
        discovered: set[str] = set()
        try:
            if self.categories_dir.exists():
                for child in self.categories_dir.iterdir():
                    if not child.is_dir():
                        continue
                    key = child.name.strip()
                    if not key or key == "unknown":
                        continue
                    try:
                        if any(p.is_file() for p in child.iterdir()):
                            discovered.add(key)
                    except OSError:
                        discovered.add(key)
        except Exception as e:
            logger.warning(f"[Config] 扫描分类目录时出错: {e}")
        try:
            index = (
                self.db_service.get_index_cache_readonly()
                if self.db_service.count_total() > 0
                else {}
            )
            for meta in index.values():
                if not isinstance(meta, dict):
                    continue
                cat = str(meta.get("category", "")).strip()
                if not cat or cat == "unknown":
                    continue
                discovered.add(cat)
        except Exception as e:
            logger.warning(f"[Config] 从索引合并分类时出错: {e}")
        to_add = sorted(
            cat
            for cat in (discovered - current_set)
            # 仅自动发现「自定义」类别；用户已删除的预定义类别即使磁盘上
            # 仍有残留文件也不会被重新加回（避免重启后复活已被删除的预定义分类）。
            if cat not in protected
        )
        if not to_add:
            return
        merged_categories = current + to_add
        self.update_config({"categories": merged_categories})
        self.plugin_config.ensure_category_dirs(to_add)

    def _validate_config(self) -> bool:
        """验证配置参数的有效性。"""
        cfg = self.plugin_config
        errors = []
        fixed = []
        fixed_values = {}
        if not isinstance(cfg.max_reg_num, int) or cfg.max_reg_num <= 0:
            errors.append("最大表情数量必须大于0的整数")
            fixed.append("最大表情数量已重置为100")
            fixed_values["max_reg_num"] = 100
        if not isinstance(cfg.meme_chance, (int, float)) or not (0 <= cfg.meme_chance <= 1):
            errors.append("表情发送概率必须在0-1之间")
            fixed.append("表情发送概率已重置为0.4")
            fixed_values["meme_chance"] = 0.4
        if cfg.steal_mode not in ("probability", "cooldown"):
            errors.append(f"偷图模式 '{cfg.steal_mode}' 无效，必须为 probability 或 cooldown")
            fixed.append("偷图模式已重置为 probability")
            fixed_values["steal_mode"] = "probability"
        if not isinstance(cfg.steal_chance, (int, float)) or not (0 <= cfg.steal_chance <= 1):
            errors.append("偷图概率必须在0-1之间")
            fixed.append("偷图概率已重置为0.6")
            fixed_values["steal_chance"] = 0.6
        if not isinstance(cfg.steal_pool_capacity, int) or cfg.steal_pool_capacity < 10:
            errors.append("待审核池容量必须是不小于10的整数")
            fixed.append("待审核池容量已重置为200")
            fixed_values["steal_pool_capacity"] = 200
        source_limits = {
            "external_source_max_items": (1, 20_000, 2000),
            "external_source_max_image_bytes": (1024, 1024 * 1024 * 1024, 32 * 1024 * 1024),
            "external_source_max_archive_bytes": (
                1024,
                8 * 1024 * 1024 * 1024,
                1024 * 1024 * 1024,
            ),
            "external_source_max_uncompressed_bytes": (
                1024,
                16 * 1024 * 1024 * 1024,
                4 * 1024 * 1024 * 1024,
            ),
            "external_source_max_pixels": (1, 200_000_000, 40_000_000),
        }
        for name, (minimum, maximum, default) in source_limits.items():
            value = getattr(cfg, name, default)
            if not isinstance(value, int) or not minimum <= value <= maximum:
                errors.append(f"外部源限制 {name} 超出安全范围")
                fixed_values[name] = default
        if any(name in fixed_values for name in source_limits):
            fixed.append("外部源资源限制已恢复为安全默认值")
        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
        if fixed:
            logger.info(f"配置已自动修复: {'; '.join(fixed)}")
            try:
                self.update_config(fixed_values)
            except Exception as e:
                logger.error(f"持久化配置修复失败: {e}")
        return True

    def _get_event_handler(
        self,
        *,
        log_message: str | None = None,
        log_level: str = "warning",
    ):
        """获取可用的 EventHandler 实例，集中记录缺失日志。"""
        event_handler = getattr(self, "event_handler", None)
        if event_handler is None and log_message:
            if log_level == "debug":
                logger.debug(log_message)
            elif log_level == "error":
                logger.error(log_message)
            else:
                logger.warning(log_message)
        return event_handler

    def _safe_create_task(self, coro, *, name: str = "") -> asyncio.Task:
        """创建 fire-and-forget task，并复用 TaskScheduler 的异常日志。"""
        return TaskScheduler.create_detached_task(coro, name=name)

    async def _migrate_legacy_category_storage(self) -> None:
        """迁移旧中文分类目录、SQLite 分类字段和待审核分类。"""
        mapping = self.plugin_config.get_legacy_category_key_map()
        if not mapping:
            return

        db = self.db_service
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        index = db.get_index_cache_readonly() if db else {}

        for old_key, new_key in mapping.items():
            if old_key == new_key:
                continue
            old_dir = self.categories_dir / old_key
            new_dir = self.plugin_config.ensure_category_dir(new_key)

            for stored_path, meta in list(index.items()):
                if not isinstance(meta, dict):
                    continue
                path_obj = Path(stored_path)
                in_old_dir = path_obj.parent.name.casefold() == old_key.casefold()
                if str(meta.get("category", "") or "") != old_key and not in_old_dir:
                    continue

                target_path = path_obj
                moved_file = False
                if in_old_dir:
                    target_path = new_dir / path_obj.name
                    if target_path.exists() and target_path.resolve() != path_obj.resolve():
                        stem, suffix = target_path.stem, target_path.suffix
                        counter = 1
                        while target_path.exists():
                            target_path = new_dir / f"{stem}_legacy{counter}{suffix}"
                            counter += 1
                    if path_obj.is_file() and target_path != path_obj:
                        try:
                            await asyncio.to_thread(shutil.move, str(path_obj), str(target_path))
                            moved_file = True
                        except OSError as exc:
                            logger.warning(f"旧分类图片迁移失败 {path_obj}: {exc}")
                            target_path = path_obj

                updates = {"category": new_key}
                if db:
                    if str(target_path) != str(path_obj) and moved_file:
                        if not await db.move_path(str(path_obj), str(target_path), new_key, updates):
                            await asyncio.to_thread(shutil.move, str(target_path), str(path_obj))
                            await db.update_path(str(path_obj), updates)
                    else:
                        await db.update_path(str(path_obj), updates)

            if old_dir.is_dir():
                for child in list(old_dir.iterdir()):
                    if not child.is_file() or child.suffix.lower() not in image_exts:
                        continue
                    target = new_dir / child.name
                    if target.exists():
                        stem, suffix = target.stem, target.suffix
                        counter = 1
                        while target.exists():
                            target = new_dir / f"{stem}_legacy{counter}{suffix}"
                            counter += 1
                    try:
                        await asyncio.to_thread(shutil.move, str(child), str(target))
                    except OSError as exc:
                        logger.warning(f"孤立旧分类图片迁移失败 {child}: {exc}")

            if db:
                pending_rows, _, _ = db.get_pending_paginated(page=1, page_size=100000)
                for row in pending_rows:
                    if str(row.get("category", "") or "") == old_key:
                        await db.update_pending(int(row["id"]), {"category": new_key})

            try:
                if old_dir.is_dir() and not any(old_dir.iterdir()):
                    old_dir.rmdir()
            except OSError as exc:
                logger.debug(f"旧分类目录清理跳过 {old_dir}: {exc}")

        logger.info(
            "旧分类兼容迁移完成: "
            + ", ".join(f"{old}->{new}" for old, new in mapping.items())
        )

    def _precheck_image_file(self, file_path: str) -> tuple[bool, str]:
        """轻量校验图片，避免明显无效文件进入 VLM 流水线。"""
        path = Path(file_path)
        if not path.exists():
            return False, f"图片文件不存在: {file_path}"
        if not path.is_file():
            return False, f"路径不是文件: {file_path}"
        if path.suffix.lower() not in PluginAPI.ALLOWED_IMAGE_EXTS:
            return False, f"不支持的图片类型: {path.suffix or '无扩展名'}"
        try:
            size = path.stat().st_size
        except OSError as e:
            return False, f"无法读取图片文件: {e}"
        if size <= 0:
            return False, "图片文件为空"
        if size > 25 * 1024 * 1024:
            return False, "图片文件过大，超过 25MB"
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            return False, f"图片格式校验失败: {e}"
        return True, ""

    def get_event_target(self, event: AstrMessageEvent) -> tuple[str, str]:
        if self.plugin_config is None:
            return "", ""
        try:
            return self.plugin_config.get_event_target(event)
        except Exception:
            return "", ""

    def _is_action_enabled_for_event(self, action: str, event: AstrMessageEvent) -> bool:
        """检查指定操作是否在当前事件中启用。"""
        if self.plugin_config is None:
            return True
        try:
            return bool(self.plugin_config.is_action_allowed(action, event))
        except Exception:
            return True

    def is_send_enabled_for_event(self, event: AstrMessageEvent) -> bool:
        return self._is_action_enabled_for_event("send", event)

    def is_steal_enabled_for_event(self, event: AstrMessageEvent) -> bool:
        return self._is_action_enabled_for_event("steal", event)

    def begin_force_capture(self, event: AstrMessageEvent, seconds: int) -> None:
        """委托给 EventHandler。"""
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法进入强制接收模式"
        )
        if event_handler is None:
            return
        event_handler.begin_force_capture(event, seconds)

    def get_force_capture_entry(self, event: AstrMessageEvent) -> dict[str, object] | None:
        """委托给 EventHandler。"""
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法获取强制接收状态",
            log_level="debug",
        )
        if event_handler is None:
            return None
        return event_handler.get_force_capture_entry(event)

    def consume_force_capture(self, event: AstrMessageEvent) -> None:
        """委托给 EventHandler。"""
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法消费强制接收状态",
            log_level="debug",
        )
        if event_handler is None:
            return
        event_handler.consume_force_capture(event)

    def _apply_plugin_config_updates(self, config_dict: dict) -> bool:
        """持久化已知配置项并同步运行时权重。"""
        if not self.plugin_config.update_config(config_dict):
            return False
        self._sync_similarity_weights()
        return True

    def _sync_similarity_weights(self) -> None:
        """把文字距离融合权重同步到 text_similarity 模块（魔法数字 → 配置项）。"""
        try:
            from .core.search import text_similarity

            cfg = self.plugin_config
            preset = cfg.SIM_WEIGHT_PRESETS.get(
                getattr(cfg, "sim_weight_preset", "balanced"), cfg.SIM_WEIGHT_PRESETS["balanced"]
            )
            text_similarity.configure_similarity(
                weights={
                    "ngram": preset["ngram"],
                    "cosine": preset["cosine"],
                    "substring": preset["substring"],
                    "char": preset["char"],
                    "edit": preset["edit"],
                },
                negation_penalty=preset["negation"],
            )
        except Exception as e:
            logger.warning(f"[Config] 同步相似度权重失败: {e}")

    def _sync_image_processor_from_runtime(self) -> None:
        cfg = self.plugin_config
        final_prompts = cfg.get_prompts(
            {
                "EMOJI_CLASSIFICATION_PROMPT": getattr(self, "EMOJI_CLASSIFICATION_PROMPT", None),
                "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT": getattr(
                    self, "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
                ),
            }
        )
        self.image_processor_service.update_config(
            categories=list(cfg.categories or []) or list(cfg.DEFAULT_CATEGORIES),
            content_filtration=cfg.content_filtration,
            vision_provider_id=self._load_vision_provider_id(),
            emoji_classification_prompt=final_prompts.get("emoji_classification_prompt"),
            emoji_classification_with_filter_prompt=final_prompts.get(
                "emoji_classification_with_filter_prompt"
            ),
        )

    def update_config(self, config_dict: dict) -> bool:
        """从配置字典更新插件配置。"""
        if not config_dict:
            return False
        try:
            if self.plugin_config is None or not self._apply_plugin_config_updates(config_dict):
                return False
            self._sync_image_processor_from_runtime()
            try:
                cats = list(self.plugin_config.categories or []) or list(
                    self.plugin_config.DEFAULT_CATEGORIES
                )
                self.plugin_config.ensure_category_dirs(cats)
            except Exception as e:
                logger.warning(f"[Config] 创建分类目录失败: {e}")
            logger.debug("[Config] 配置已保存，后续处理将使用新配置")
            return True
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            return False

    @filter.command_group("meme")
    def meme(self):
        """表情包管理指令"""
        pass

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("on")
    async def meme_on(self, event: AstrMessageEvent):
        """开启表情包偷取功能，自动收集群聊中的表情包。"""
        async for result in self.command_handler.meme_on(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("off")
    async def meme_off(self, event: AstrMessageEvent):
        """关闭表情包偷取功能，停止收集新表情包。"""
        async for result in self.command_handler.meme_off(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("auto_on")
    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送表情包，聊天时根据情绪自动发送。"""
        async for result in self.command_handler.auto_on(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("auto_off")
    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送表情包。"""
        async for result in self.command_handler.auto_off(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("group")
    async def group_filter(
        self,
        event: AstrMessageEvent,
        scope: str = "",
        list_name: str = "",
        action: str = "",
        target: str = "",
        target_id: str = "",
    ):
        """管理发送与收集目标名单。用法: /meme group <send|steal> <wl|bl> <add|del|clear|show> [目标]"""
        async for result in self.target_commands.group_filter(
            event, scope, list_name, action, target, target_id
        ):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("偷")
    async def capture(self, event: AstrMessageEvent):
        """进入强制接收模式，30秒内发送的图片将直接入库。"""
        async for result in self.command_handler.capture(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("natural_analysis")
    async def toggle_natural_analysis(self, event: AstrMessageEvent, action: str = ""):
        """切换情绪识别模式。用法: /meme natural_analysis <on|off>"""
        async for result in self.command_handler.toggle_natural_analysis(event, action):
            yield result

    @meme.command("emotion_stats")
    async def emotion_analysis_stats(self, event: AstrMessageEvent):
        """查看情绪分析统计信息和当前模式。"""
        async for result in self.command_handler.emotion_analysis_stats(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("clear_emotion_cache")
    async def clear_emotion_cache(self, event: AstrMessageEvent):
        """清空情绪分析缓存，释放内存。"""
        async for result in self.command_handler.clear_emotion_cache(event):
            yield result

    @meme.command("status")
    async def status(self, event: AstrMessageEvent):
        """查看插件运行状态和表情包统计信息。"""
        async for result in self.command_handler.status(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("tag_stats")
    async def tag_stats(self, event: AstrMessageEvent, limit: str = ""):
        """标签/场景统计：高频标签、低频噪声标签、零标签条目（打标质量体检）。用法: /meme tag_stats [N]"""
        async for result in self.command_handler.tag_stats(event, limit):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("clean", priority=-100)
    async def clean(self, event: AstrMessageEvent, mode: str = ""):
        """清理原始图片缓存（不影响已分类的表情包）。"""
        async for result in self.command_handler.clean(event, mode):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("capacity")
    async def enforce_capacity(self, event: AstrMessageEvent):
        """立即执行容量控制，清理超出上限的旧表情包。"""
        async for result in self.command_handler.enforce_capacity(event):
            yield result

    @meme.command("list")
    async def list_images(
        self,
        event: AstrMessageEvent,
        category: str = "",
        limit: str = "10",
        page: str = "1",
    ):
        """列出已收集的表情包。用法: /meme list [分类] [每页数量] [页码]"""
        async for result in self.image_commands.list_images(event, category, limit, page):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("delete")
    async def delete_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定表情包。用法: /meme delete <序号|文件名>"""
        async for result in self.image_commands.delete_image(event, identifier):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("blacklist")
    async def blacklist_image(self, event: AstrMessageEvent, identifier: str = ""):
        """拉黑指定表情包。用法: /meme blacklist <序号|文件名>"""
        async for result in self.image_commands.blacklist_image(event, identifier):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("scope")
    async def set_image_scope(
        self, event: AstrMessageEvent, identifier: str = "", scope_mode: str = ""
    ):
        """设置表情包作用域。用法: /meme scope <序号|文件名> <public|local>"""
        async for result in self.image_commands.set_image_scope(event, identifier, scope_mode):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("rebuild_index")
    async def rebuild_index(self, event: AstrMessageEvent):
        """重建表情包索引，用于修复索引异常或版本迁移。"""
        async for result in self.index_commands.rebuild_index(event):
            yield result

    @filter.llm_tool(name=SEARCH_MEME_TOOL_NAME)
    async def search_meme(self, event: AstrMessageEvent, query: str):
        """从插件索引中搜索表情包候选；不要直接浏览表情库目录或发送本地文件。

        Args:
            query(string): 2-8 个检索关键词。优先写图上文字、角色名、画面或语气，例如“猫猫 震惊 怎么会这样”。

        使用建议：
        - 这是从表情库选图的唯一入口；不要使用文件、终端或通用消息工具绕过它
        - 若候选不合适，换图上原文、角色名或更具体的画面词再次调用本工具

        返回值：
        返回候选表情包列表，每个包含：
        - 编号：用于调用 send_meme
        - 分类：浏览分区，仅供参考
        - 角色 / 图上文字 / 描述：选图时优先看这些

        根据候选的图上文字、角色和描述选择最贴合的一张，然后调用 send_meme；不要猜测或传递文件路径。
        """
        async for result in self._search_meme_impl(event, query):
            yield result

    @filter.llm_tool(name=SEND_MEME_TOOL_NAME)
    async def send_meme(self, event: AstrMessageEvent, emoji_id: int):
        """发送 search_meme 返回的候选表情包；不要猜测或直接传递文件路径。

        选择原则：优先发送能代表你"当前心情词"的候选项。

        Args:
            emoji_id(number): 表情包编号（从 search_meme 返回的候选列表中选择）

        """
        async for result in self._send_meme_impl(event, emoji_id):
            yield result

    @filter.llm_tool(name="steal_meme")
    async def steal_sticker(self, event: AstrMessageEvent, image_ref: str):
        """偷取图片入库。VLM 视觉模型会自动分析图片，打上分类、标签、描述和场景。

        使用时机：
        - 用户说"偷一下"/"收了这张图"时直接调用本工具。
        - 你看到当前消息里有适合作为表情包的图片时，也可以调用本工具补充素材库。

        注意：
        - image_ref 必须从当前消息中已有的图片 URL 或文件路径中选择，必填。
        - 不需要自己打标，工具会交给 VLM 自动完成分类、标签、描述和场景分析。
        - 工具返回的 VLM 分析结果可用于向用户说明偷到了什么。
        - 当插件的表情包偷取总开关关闭，或当前会话被偷取黑白名单禁用时，本工具会拒绝入库。

        Args:
            image_ref(string): 图片 URL 或文件路径，从当前消息已有的 Image URL 中选择。
        """
        async for result in self._steal_sticker_impl(event, image_ref):
            yield result

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 每条新消息到达时重置回合状态，防止上一轮的标记影响当前对话
        if getattr(self, "auto_meme_cancel_on_new_message", True):
            self._emoji_sender_engine.cancel_pending_auto_emoji(event)
        self._emoji_sender_engine.reset_turn_state(event)
        event_handler = self._get_event_handler(
            log_message="[Stealer] event_handler 未初始化，跳过消息处理",
            log_level="debug",
        )
        if event_handler is None:
            return
        try:
            await event_handler.on_message(event)
        except Exception as e:
            logger.error(f"[Stealer] 处理消息时发生错误: {e}", exc_info=True)

    @filter.on_llm_tool_respond()
    async def _track_external_image_delivery(
        self,
        event: AstrMessageEvent,
        tool,
        tool_args: dict | None,
        tool_result,
    ) -> None:
        """记录 LLM 通过 AstrBot 通用消息工具发送的图片，避免同轮再被动发表情。"""
        if str(getattr(tool, "name", "") or "") != "send_message_to_user":
            return
        result_contents = getattr(tool_result, "content", None)
        if not isinstance(result_contents, list) or not any(
            "Message sent to session" in str(getattr(item, "text", "") or "")
            for item in result_contents
        ):
            return
        messages = tool_args.get("messages") if isinstance(tool_args, dict) else None
        if not isinstance(messages, list):
            return
        sent_image = any(
            isinstance(item, dict)
            and str(item.get("type", "") or "").strip().lower() == "image"
            for item in messages
        )
        if not sent_image:
            return
        if getattr(event, "_has_send_oper", False):
            self._emoji_sender_engine.emoji_turn_state(event).mark_active_sent()
        selector = getattr(self, "meme_selector", None)
        if selector is None:
            return
        for item in messages:
            if not isinstance(item, dict) or str(item.get("type", "")).lower() != "image":
                continue
            image_path = str(item.get("path", "") or "").strip()
            if not image_path:
                continue
            try:
                await selector.record_emoji_usage(
                    image_path, trigger="generic_tool"
                )
            except Exception as exc:
                logger.warning(f"[Stealer] 通用消息工具的图片计数失败: {exc}")
        logger.debug("[Stealer] LLM 已通过通用消息工具发送图片，跳过本轮被动表情")

    async def _schedule_passive_emoji_response(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> bool:
        """Apply the passive-emoji gates and schedule at most one task per turn."""

        normalized_text = str(text or "").strip()
        if not normalized_text:
            return False
        turn_state = self._emoji_sender_engine.emoji_turn_state(event)
        if turn_state.is_active_sent():
            return False

        turn_allowed = await self._emoji_sender_engine._resolve_with_log(event)
        if not turn_allowed or self._emoji_sender_engine.should_skip_auto_emoji_by_gate(normalized_text):
            return False
        if not self._emoji_sender_engine.claim_auto_emoji_turn(event):
            return False

        user_message = ""
        try:
            user_message = event.get_message_str() or ""
        except Exception:
            pass
        task = self._safe_create_task(
            self._emoji_sender_engine.async_analyze_and_send_emoji(
                event,
                normalized_text,
                [],
                user_message=user_message,
            ),
            name="emoji_analyze_passive",
        )
        self._emoji_sender_engine.schedule_auto_emoji_task(event, task)
        return True

    @staticmethod
    def _response_chain_has_image(response: Any) -> bool:
        chain = getattr(response, "result_chain", None)
        components = getattr(chain, "chain", chain if isinstance(chain, list) else [])
        return any(isinstance(component, MessageImage) for component in components or [])

    @filter.on_llm_response()
    async def _prepare_emoji_after_llm(self, event: AstrMessageEvent, response: Any):
        """Capture final LLM text before the streaming result is decorated.

        AstrBot skips ``on_decorating_result`` while a result is still a
        streaming result.  The LLM response hook runs after the agent has its
        final response and gives passive emoji scheduling a stable entry point.
        """

        role = str(getattr(response, "role", "assistant") or "assistant").lower()
        if role not in {"", "assistant"}:
            return False
        if getattr(response, "tools_call_args", None):
            return False
        if self._response_chain_has_image(response):
            return False
        text = str(getattr(response, "completion_text", "") or "").strip()
        if not text:
            return False
        return await self._schedule_passive_emoji_response(event, text)

    @filter.on_decorating_result(priority=100)
    async def _prepare_emoji_response(self, event: AstrMessageEvent):
        """LLM 回复完成后异步发送表情包（不阻塞回复）。"""
        result = event.get_result()
        if result is None:
            return False
        if not result.is_llm_result():
            return False
        if any(isinstance(comp, MessageImage) for comp in getattr(result, "chain", [])):
            return False
        text = result.get_plain_text() or ""
        return await self._schedule_passive_emoji_response(event, text)

    async def initialize(self):
        """初始化插件运行时资源。

        加载情绪映射和提示词等运行时需要的资源。
        __init__ 仅做属性赋值，IO/目录/密码等操作统一在此执行。
        """
        await super().initialize()
        try:
            self._validate_config()
            if (
                self._get_event_handler(
                    log_message="[Stealer] event_handler 未初始化，插件无法启动",
                    log_level="error",
                )
                is None
            ):
                raise RuntimeError("event_handler 未初始化")
            self.plugin_config.ensure_base_dirs()
            await self._migrate_legacy_category_storage()
            self.plugin_config.ensure_category_dirs(
                list(self.plugin_config.categories or []) or list(
                    self.plugin_config.DEFAULT_CATEGORIES
                )
            )
            await self.source_service.initialize()
            await self.image_processor_service._auto_migrate_categories()
            self._auto_merge_existing_categories()
            try:
                plugin_dir = Path(__file__).parent
                prompts_path = plugin_dir / "prompts.json"
                if prompts_path.exists():
                    if aiofiles:
                        async with aiofiles.open(prompts_path, encoding="utf-8-sig") as f:
                            content = await f.read()
                        content = content.lstrip("\ufeff")
                        prompts = json.loads(content)
                    else:
                        with open(prompts_path, encoding="utf-8-sig") as f:
                            prompts = json.loads(f.read().lstrip("\ufeff"))
                    self._apply_prompts(prompts)
            except Exception as e:
                logger.error(f"初始化提示词失败: {e}")
            await self.index_manager.load_index()
            await self.index_manager.migrate_blacklist()
            await self.maintenance.run_startup_cleanup()
            self._sync_image_processor_from_runtime()
            self._sync_similarity_weights()  # 启动时把持久化的文字距离权重同步到 text_similarity 模块
            self.maintenance.start_periodic_tasks()
            await self.event_handler.start_background_workers()

            # 初始化嵌入向量服务 + 回填旧数据（仅在开启嵌入检索时）
            if self.plugin_config.enable_embedding_search:
                try:
                    smart_service = getattr(self.meme_selector, "_smart_select_service", None)
                    if smart_service and smart_service._embedding_service:
                        await smart_service._embedding_service.initialize()
                        # 同步回填旧数据（分批处理，每批 20 条）
                        backfilled = await smart_service._embedding_service.backfill_existing(batch_size=20)
                        if backfilled > 0:
                            logger.info(f"[Embedding] 旧数据回填完成: {backfilled} 条新向量")
                except Exception as e:
                    logger.warning(f"[Embedding] 初始化失败: {e}")

            logger.info("[Stealer] 插件初始化完成")
        except Exception as e:
            logger.error(f"初始化插件失败: {e}")
            raise

    async def terminate(self):
        """插件销毁生命周期钩子。"""
        if self._terminated:
            return
        self._terminated = True
        try:
            await self.task_scheduler.cancel_task("raw_cleanup_loop")
            await self.task_scheduler.cancel_task("capacity_control_loop")
        except Exception:
            pass
        if self.source_service:
            try:
                await self.source_service.close()
            except Exception:
                pass
        if self.task_scheduler:
            try:
                await self.task_scheduler.cleanup()
            except Exception:
                pass
        # 关闭嵌入向量服务
        try:
            smart_service = getattr(self.meme_selector, "_smart_select_service", None)
            if smart_service and smart_service._embedding_service:
                await smart_service._embedding_service.close()
        except Exception:
            pass

        if self.image_processor_service:
            try:
                self.image_processor_service.cleanup()
                self.image_render_service.cleanup()
            except Exception:
                pass
        if self.command_handler:
            try:
                self.command_handler.cleanup()
            except Exception:
                pass
        if self.event_handler:
            try:
                await self.event_handler.stop_background_workers()
            except Exception:
                pass
            try:
                await self.event_handler.cleanup_async()
            except Exception:
                pass
            try:
                self.event_handler.cleanup()
            except Exception:
                pass
        await super().terminate()
        logger.info("[Stealer] 插件资源清理完成")
