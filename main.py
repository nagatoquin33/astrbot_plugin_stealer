import asyncio
import json
import os
import re
import time
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

from .cache_service import CacheService
from .core.commands.command_handler import CommandHandler
from .core.config.config import PluginConfig
from .core.db.database_service import DatabaseService
from .core.search.meme_selector import MemeSelector
from .core.events.event_handler import EventHandler
from .core.events.meme_sender_engine import MemeSenderEngine
from .core.db.index_manager import IndexManager
from .core.processing.natural_emotion_analyzer import SmartEmotionMatcher
from .core.processing.image_processor_service import ImageProcessorService
from .core.maintenance.service import MaintenanceService
from .core.util.safe_io import safe_remove_file
from .task_scheduler import TaskScheduler
from .plugin_api import PluginAPI
from .core.search.meme_smart_select_service import _unwrap_event

try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None


class Main(Star):
    """表情包偷取与发送插件。

    功能：
    - 监听消息中的图片并自动保存到插件数据目录
    - 使用当前会话的多模态模型进行情绪分类与标签生成
    - 建立分类索引，支持自动与手动在合适时机发送表情包
    """

    # 常量定义
    BACKEND_TAG = "emoji_stealer"

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

        # 情绪选择标记（用于识别注入的内容）
        self._persona_marker = "<!-- STEALER_PLUGIN_EMOTION_MARKER_v3 -->"  # 更新版本号

        # 初始化插件配置
        self.plugin_config = PluginConfig(config, context)

        self.base_dir: Path = self.plugin_config.data_dir
        self.raw_dir: Path = self.plugin_config.raw_dir
        self.categories_dir: Path = self.plugin_config.categories_dir
        self.cache_dir: Path = self.plugin_config.cache_dir

        # 配置统一通过 self.plugin_config 读取（pydantic 模型）。
        # v2.7.5+ 删除了 _sync_all_config() 实例属性镜像。

        # 初始化核心服务类
        self.cache_service = CacheService(self.cache_dir)
        self.db_service = DatabaseService(self.cache_dir / "emoji.db")
        self.command_handler = CommandHandler(self)
        self.web_server = None
        self.plugin_api = PluginAPI(self)
        self.plugin_api.register(context)

        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.meme_selector = MemeSelector(self)
        self.task_scheduler = TaskScheduler()

        # 初始化自然语言情绪分析器（新增）
        self.smart_emotion_matcher = SmartEmotionMatcher(self)

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

    def _ensure_default_prompts_in_config(self, prompts: dict) -> None:
        """如果配置中的提示词字段为空，将 prompts.json 内容写入配置作为默认显示值。"""
        updates = {}
        current_prompt = getattr(self.plugin_config, "custom_meme_classification_prompt", "")
        if not current_prompt or not current_prompt.strip():
            default_prompt = prompts.get("EMOJI_CLASSIFICATION_PROMPT", "")
            if default_prompt:
                updates["custom_meme_classification_prompt"] = default_prompt
        current_filter_prompt = getattr(
            self.plugin_config, "custom_meme_classification_with_filter_prompt", ""
        )
        if not current_filter_prompt or not current_filter_prompt.strip():
            default_filter_prompt = prompts.get("EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", "")
            if default_filter_prompt:
                updates["custom_meme_classification_with_filter_prompt"] = default_filter_prompt
        if updates:
            self.update_config(updates)
            logger.info(f"已将默认提示词写入配置: {list(updates.keys())}")

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

    def _apply_plugin_config_updates(self, config_dict: dict) -> None:
        """将更新字典写回 PluginConfig。"""
        for k, v in config_dict.items():
            setattr(self.plugin_config, k, v)

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

    def update_config(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return
        try:
            if self.plugin_config:
                self._apply_plugin_config_updates(config_dict)
                self._sync_image_processor_from_runtime()
                try:
                    cats = list(self.plugin_config.categories or []) or list(
                        self.plugin_config.DEFAULT_CATEGORIES
                    )
                    self.plugin_config.ensure_category_dirs(cats)
                except Exception as e:
                    logger.warning(f"[Config] 创建分类目录失败: {e}")
                logger.debug("[Config] 配置已更新，下次 LLM 请求将使用新分类")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")

    # ===== 门面委托：MemeSenderEngine =====
    _emoji_turn_state = lambda self, event: self._emoji_sender_engine.emoji_turn_state(event)  # noqa: E731
    _send_explicit_emojis = (  # noqa: E731
        lambda self, event, paths, text: self._emoji_sender_engine.send_explicit_emojis(
            event, paths, text
        )
    )
    _get_auto_emoji_session_key = (  # noqa: E731
        lambda self, event: self._emoji_sender_engine.get_auto_emoji_session_key(event)
    )
    _should_skip_auto_emoji_by_gate = (  # noqa: E731
        lambda self, text: self._emoji_sender_engine.should_skip_auto_emoji_by_gate(text)
    )
    _is_auto_emoji_cooldown_ready = (  # noqa: E731
        lambda self, event: self._emoji_sender_engine.is_auto_emoji_cooldown_ready(event)
    )
    _normalize_auto_meme_chance = (  # noqa: E731
        lambda self: self._emoji_sender_engine.normalize_auto_meme_chance()
    )
    _resolve_auto_emoji_turn_permission = (  # noqa: E731
        lambda self, event: self._emoji_sender_engine._resolve_with_log(event)
    )
    _claim_auto_emoji_turn = lambda self, event: self._emoji_sender_engine.claim_auto_emoji_turn(  # noqa: E731
        event
    )
    _prune_auto_emoji_cooldowns = (  # noqa: E731
        lambda self, now: self._emoji_sender_engine.prune_auto_emoji_cooldowns(now)
    )
    _mark_auto_emoji_sent = lambda self, event: self._emoji_sender_engine.mark_auto_emoji_sent(  # noqa: E731
        event
    )
    _cancel_pending_auto_emoji = (  # noqa: E731
        lambda self, event, reason="new_message": self._emoji_sender_engine.cancel_pending_auto_emoji(
            event, reason
        )
    )
    _schedule_auto_emoji_task = (  # noqa: E731
        lambda self, event, task: self._emoji_sender_engine.schedule_auto_emoji_task(event, task)
    )
    _try_send_emoji = lambda self, event, emotions, text: self._emoji_sender_engine.try_send_emoji(  # noqa: E731
        event, emotions, text
    )
    _get_meme_send_delay = lambda self: self._emoji_sender_engine.get_meme_send_delay()  # noqa: E731
    _async_analyze_and_send_emoji = (  # noqa: E731
        lambda self,
        event,
        text,
        emotions,
        **kw: self._emoji_sender_engine.async_analyze_and_send_emoji(event, text, emotions, **kw)
    )
    _validate_result = lambda self, result: self._emoji_sender_engine.validate_result(result)  # noqa: E731
    _update_result_with_cleaned_text_safe = (  # noqa: E731
        lambda self,
        event,
        result,
        text: self._emoji_sender_engine.update_result_with_cleaned_text_safe(event, result, text)
    )

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
        """管理群聊黑白名单。用法: /meme group <wl|bl> <add|del|clear|show> [群号]"""
        async for result in self.command_handler.group_filter(
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
        """列出已收集的表情包。用法: /meme list [分类] [数量]"""
        async for result in self.command_handler.list_images(event, category, limit, page):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("delete")
    async def delete_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定表情包。用法: /meme delete <序号|文件名>"""
        async for result in self.command_handler.delete_image(event, identifier):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("blacklist")
    async def blacklist_image(self, event: AstrMessageEvent, identifier: str = ""):
        """拉黑指定表情包。用法: /meme blacklist <序号|文件名>"""
        async for result in self.command_handler.blacklist_image(event, identifier):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("scope")
    async def set_image_scope(
        self, event: AstrMessageEvent, identifier: str = "", scope_mode: str = ""
    ):
        """设置表情包作用域。用法: /meme scope <序号|文件名> <public|local>"""
        async for result in self.command_handler.set_image_scope(event, identifier, scope_mode):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @meme.command("rebuild_index")
    async def rebuild_index(self, event: AstrMessageEvent):
        """重建表情包索引，用于修复索引异常或版本迁移。"""
        async for result in self.command_handler.rebuild_index(event):
            yield result

    async def _search_emoji_candidates(
        self,
        event: AstrMessageEvent,
        query: str,
        *,
        limit: int = 5,
        idx: dict | None = None,
    ):
        """委托给 MemeSelector.smart_search。"""
        if idx is None:
            idx = (
                self.db_service.get_index_cache_readonly()
                if self.db_service.count_total() > 0
                else {}
            )

        return await self.meme_selector.smart_search(query, limit=limit, idx=idx, event=event)

    def _find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的多个分类，委托给 MemeSelector。"""
        return self.meme_selector.find_similar_categories(query, top_n)

    @filter.llm_tool(name="search_meme")
    async def search_emoji(self, event: AstrMessageEvent, query: str):
        """搜索表情包候选，并优先按你当前心情词进行匹配。

        Args:
            query(string): 你当前心情的代表词（也支持描述词、场景词）

        使用建议：
        - 先判断你此刻最能代表自己的心情词（例如：开心、无语、尴尬、感谢）
        - 再用该心情词调用本工具搜索候选
        - 若无结果，可换同义词再搜索（如"无语"->"dumb/尴尬"）

        返回值：
        返回候选表情包列表，每个包含：
        - 编号：用于调用 send_emoji_by_id
        - 分类：表情包的情绪分类
        - 描述：表情包的详细描述（这是你选择时的重要参考）

        请先锁定"当前心情词"，再仔细阅读候选描述，选择最能代表你当前心情与语气的一张。
        """
        event = _unwrap_event(event)
        query = str(query or "").strip()
        logger.info(f"[Tool] LLM 搜索表情包: {query}")

        turn_state = self._emoji_turn_state(event)

        try:
            if not query:
                yield "搜索失败：缺少 query 参数。请传入你当前心情词，例如：开心、无语、尴尬、感谢。"
                return

            if not self.is_send_enabled_for_event(event):
                yield "搜索失败：当前群聊已禁用表情包功能"
                return

            if self.db_service.count_total() > 0:
                idx = self.db_service.get_index_cache_readonly()
            else:
                logger.debug("索引未加载，正在加载...")
                await self.index_manager.load_index()
                idx = self.db_service.get_index_cache_readonly()

            # smart_search 已内置关键词映射和模糊匹配（阈值0.4）
            results = await self._search_emoji_candidates(
                event, query, limit=self.MAX_SEARCH_RESULTS, idx=idx
            )

            if not results:
                similar = self._find_similar_categories(query, top_n=3)
                suggestion = f"未找到与'{query}'匹配的表情包。"
                if similar:
                    suggestion += "\n\n您是否想找以下分类？\n- " + "\n- ".join(similar)
                cats = self.plugin_config.get_categories()
                suggestion += "\n\n可用分类：" + ", ".join(cats[:10])
                if len(cats) > 10:
                    suggestion += f" 等共{len(cats)}个分类"
                logger.warning(f"[Tool] 未找到匹配: {query}, 推荐: {similar}")
                yield suggestion
                return

            candidates = []
            result_lines = [f"找到 {len(results)} 个匹配的表情包：\n"]

            for i, (path, desc, emotion, tags) in enumerate(results):
                if os.path.exists(path):
                    meta = idx.get(path, {}) if isinstance(idx, dict) else {}
                    raw_scenes = meta.get("scenes", None) if isinstance(meta, dict) else None
                    if not raw_scenes:
                        raw_scenes = meta.get("scene", None) if isinstance(meta, dict) else None

                    scenes_items = PluginAPI._split_scenes(raw_scenes)
                    scenes_str = ", ".join(scenes_items)
                    source = str(meta.get("source", "") or "") if isinstance(meta, dict) else ""
                    scope_mode = str(meta.get("scope_mode", "public") or "public") if isinstance(meta, dict) else "public"
                    origin_target = str(meta.get("origin_target", "") or "") if isinstance(meta, dict) else ""
                    use_count = int(meta.get("use_count", 0) or 0) if isinstance(meta, dict) else 0

                    candidate_id = f"emoji_{i + 1}"
                    candidates.append(
                        {
                            "id": candidate_id,
                            "path": path,
                            "desc": desc,
                            "emotion": emotion,
                            "tags": tags,
                            "scenes": scenes_str,
                            "source": source,
                            "scope_mode": scope_mode,
                            "origin_target": origin_target,
                            "use_count": use_count,
                        }
                    )
                    result_lines.append(f"\n[{i + 1}] 分类：{emotion}")
                    if tags:
                        result_lines.append(f"    标签：{tags}")
                    if scenes_str:
                        result_lines.append(f"    场景：{scenes_str}")
                    else:
                        result_lines.append("    场景：无")
                    result_lines.append(f"    作用域：{scope_mode}")
                    if use_count:
                        result_lines.append(f"    使用次数：{use_count}")
                    if source == "qq_store":
                        result_lines.append("    来源：QQ商城")
                    result_lines.append(f"    描述：{desc}")

            if not candidates:
                yield "搜索失败：找到的表情包文件均已丢失"
                return

            turn_state.set_candidates(candidates)
            result_lines.append(
                "\n\n请先确定你当前最能代表自己的心情词，再根据候选描述选择最合适的表情包，最后调用 send_emoji_by_id(编号) 发送。"
            )

            result_text = "\n".join(result_lines)
            logger.info(f"[Tool] 搜索完成，返回 {len(candidates)} 个候选")
            yield result_text

        except Exception as e:
            logger.error(f"[Tool] 搜索表情包失败: {e}", exc_info=True)
            yield f"搜索出错：{e}"

    @filter.llm_tool(name="send_meme")
    async def send_emoji_by_id(self, event: AstrMessageEvent, emoji_id: int):
        """发送你选择的表情包。必须先调用 search_emoji 获取候选列表。

        选择原则：优先发送能代表你"当前心情词"的候选项。

        Args:
            emoji_id(number): 表情包编号（从 search_emoji 返回的列表中选择）

        """
        event = _unwrap_event(event)
        logger.info(f"[Tool] LLM 选择发送表情包编号: {emoji_id}")
        turn_state = self._emoji_turn_state(event)

        try:
            if not self.is_send_enabled_for_event(event):
                yield "发送失败：reason=send_disabled。当前会话已禁用表情包发送功能，请不要继续调用发送工具。"
                return

            if emoji_id is None:
                yield "发送失败：reason=missing_id。缺少 emoji_id 参数。请先调用 search_emoji，再传入候选编号。"
                return

            try:
                emoji_id = int(emoji_id)
            except Exception:
                yield f"发送失败：reason=invalid_id。编号 {emoji_id} 无法解析为整数，请输入有效的数字编号。"
                return

            candidates = turn_state.get_candidates()
            if not candidates:
                yield "发送失败：reason=candidate_expired。没有可用候选列表。请先调用 search_emoji 重新搜索。"
                return

            if emoji_id < 1 or emoji_id > len(candidates):
                yield f"发送失败：reason=invalid_id。编号 {emoji_id} 无效。可选编号范围：1-{len(candidates)}，请重新选择。"
                return

            selected = candidates[emoji_id - 1]
            path = selected["path"]
            desc = selected["desc"]
            emotion = selected["emotion"]

            if not os.path.exists(path):
                yield f"发送失败：reason=file_missing。表情包文件已丢失。\n你选择的是：编号 {emoji_id}，分类 {emotion}，描述 {desc}\n请重新搜索并选择其他表情包。"
                return

            if not self.meme_selector.is_path_allowed_for_event(path, event):
                yield "发送失败：reason=scope_denied。该表情包被限制为仅来源会话可发送，请选择 public 表情或重新搜索。"
                return

            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            send_mode = await self.meme_selector.send_emoji_message(event, path)
            if not send_mode:
                yield "发送失败：reason=send_failed。表情包编码或平台发送失败，请重新搜索或选择其他候选。"
                return
            sent_as_sticker = send_mode == "telegram_sticker"

            await self.meme_selector.record_emoji_usage(path, trigger="llm_tool")
            await self._mark_auto_emoji_sent(event)
            turn_state.mark_active_sent()

            mode_desc = "Telegram贴纸" if sent_as_sticker else "图片"
            success_msg = f"发送成功（{mode_desc}）。\n\n你发送的表情包：\n- 编号：{emoji_id}\n- 分类：{emotion}\n- 描述：{desc}"
            logger.info(f"[Tool] {success_msg}")
            yield success_msg
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield f"发送出错：{e}"
            return

    @filter.llm_tool(name="steal_meme")
    async def steal_sticker(
        self,
        event: AstrMessageEvent,
        image_ref: str,
    ):
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
        event = _unwrap_event(event)
        try:
            if not self.plugin_config.steal_meme:
                yield "偷取失败：表情包偷取功能未开启，请先在插件配置中启用"
                return

            if not self.is_steal_enabled_for_event(event):
                yield "偷取失败：当前群聊已禁用偷取功能"
                return

            event_handler = self._get_event_handler(log_message="event_handler 未初始化，无法下载图片")
            if event_handler is None:
                yield "偷取失败：内部服务未初始化"
                return

            image_ref, source = await self._resolve_steal_image_ref(
                event, image_ref, event_handler
            )
            if not image_ref:
                yield "偷取失败：缺少 image_ref 参数，请提供当前消息中的图片 URL"
                return

            logger.info(f"[Tool] LLM 请求偷取: ref={image_ref[:80]}")

            # 下载图片
            if image_ref.startswith("http://") or image_ref.startswith("https://"):
                temp_path, _is_gif = await event_handler._download_to_temp(image_ref, log_download=True)
                if not temp_path or not os.path.exists(temp_path):
                    yield f"偷取失败：无法下载图片 {image_ref[:100]}"
                    return
                is_temp = True
            elif image_ref.startswith("file:///"):
                local_path = image_ref[8:]
                if len(local_path) > 2 and local_path[0] == "/" and local_path[2] == ":":
                    local_path = local_path[1:]
                temp_path = os.path.abspath(local_path)
                is_temp = False
            else:
                temp_path = os.path.abspath(image_ref)
                is_temp = False

            if not os.path.exists(temp_path):
                hint = ""
                # 当 LLM 传来的是非 URL 形式（相对路径/裸文件名）且仍无法定位时，
                # 提示它从消息中已有的 Image URL 选择（issue #88）。
                ref_value = str(image_ref or "").strip()
                if ref_value and not (
                    ref_value.startswith("http://")
                    or ref_value.startswith("https://")
                    or ref_value.startswith("file:")
                ):
                    hint = "（请确认 image_ref 是当前消息中的图片 URL 或本地绝对路径）"
                yield f"偷取失败：图片文件不存在: {temp_path}{hint}"
                return

            precheck_ok, precheck_reason = self._precheck_image_file(temp_path)
            if not precheck_ok:
                if is_temp:
                    await safe_remove_file(temp_path)
                yield f"偷取失败：{precheck_reason}"
                return

            # 记下入库存前已有的路径，之后 diff 找出 VLM 分析结果
            idx_before = await self.index_manager.load_index()
            before_paths = set(idx_before.keys()) if idx_before else set()

            # 统一走 VLM 流水线
            logger.info(f"[Tool] VLM 分析入库: {temp_path}")
            extra_meta = self._build_steal_tool_extra_meta(
                event, image_ref, source=source
            )
            success, merged_idx = await self._process_image(
                event, temp_path, is_temp=is_temp, extra_meta=extra_meta
            )

            if not success:
                fail_open_hint = (
                    "。已启用审核失败开放策略，但明确审核不通过或重复图片不会入库"
                    if getattr(self, "content_filtration_fail_open", False)
                    else ""
                )
                yield f"偷取失败：VLM 分析未通过（可能已存在、内容不合适或无法识别为表情包）{fail_open_hint}"
                return

            if merged_idx:
                await self.index_manager.save_index(merged_idx)
                new_paths = set(merged_idx.keys()) - before_paths
                if new_paths:
                    new_entry = next((merged_idx[p] for p in new_paths if isinstance(merged_idx.get(p), dict)), None)
                    if new_entry and isinstance(new_entry, dict):
                        cat = new_entry.get("category", "?")
                        tag_list = new_entry.get("tags", [])
                        tags_str = ", ".join(tag_list) if isinstance(tag_list, list) else str(tag_list)
                        desc_text = new_entry.get("desc", "")
                        scene_list = new_entry.get("scenes", [])
                        scenes_str = ", ".join(scene_list) if isinstance(scene_list, list) else str(scene_list)
                        yield (
                            f"偷取成功！VLM 分析结果：\n"
                            f"- 分类：{cat}\n"
                            f"- 标签：{tags_str or '无'}\n"
                            f"- 描述：{desc_text or '无'}\n"
                            f"- 场景：{scenes_str or '无'}"
                        )
                        return
                yield "偷取成功！已通过 VLM 自动分析并入库"
            else:
                yield "偷取成功但索引更新失败"

        except Exception as e:
            logger.error(f"[Tool] 偷取表情包失败: {e}", exc_info=True)
            yield f"偷取出错：{e}"
            return

    async def _resolve_steal_image_ref(
        self,
        event: AstrMessageEvent,
        image_ref: str,
        event_handler: Any,
    ) -> tuple[str, str]:
        """Resolve an explicit or current-message image reference for steal_sticker.

        修复 issue #88：LLM 偶尔会传相对路径（如 ``./image.png``）或仅文件名。
        之前的实现直接把 ``explicit_ref`` 原样返回，下游 ``os.path.abspath``
        会拼到 CWD（如 ``/AstrBot/image.png``），触发"图片文件不存在"。

        现在对非 URL 形式的 ``image_ref``，优先在当前消息的 ``Image`` 组件中
        按 basename 匹配，再回退到组件自带的 url/file/path/convert_to_file_path。
        """
        explicit_ref = str(image_ref or "").strip()

        # 1. URL 形式（http/https/file 协议）直接信任 LLM 传入
        if explicit_ref and (
            explicit_ref.startswith("http://")
            or explicit_ref.startswith("https://")
            or explicit_ref.startswith("file:")
        ):
            return explicit_ref, "llm_tool"

        # 2. 尝试把显式 ref 解析为消息内某张图片的真实位置
        resolved_explicit = ""
        if explicit_ref:
            resolved_explicit = await self._resolve_image_ref_against_event(
                event, explicit_ref
            )
            if resolved_explicit:
                return resolved_explicit, "llm_tool"

        # 3. 未提供 ref 或 ref 解析失败：取当前消息中第一张可用图片
        try:
            for comp in event.get_messages():
                if not isinstance(comp, MessageImage):
                    continue
                for attr in ("url", "file", "path"):
                    value = str(getattr(comp, attr, "") or "").strip()
                    if value:
                        return value, "llm_tool"
                if hasattr(comp, "convert_to_file_path"):
                    path = await comp.convert_to_file_path()
                    path = str(path or "").strip()
                    if path:
                        return path, "llm_tool"
        except Exception:
            pass

        # 4. 兜底：QQ 商城表情 URL
        try:
            store_urls = event_handler._extract_store_emoji_urls(event)
        except Exception:
            store_urls = []
        if store_urls:
            return str(store_urls[0] or "").strip(), "qq_store"

        # 5. 都没有的话才把显式 ref 原样回传（让下游报错时提示更准确）
        return explicit_ref, "llm_tool"

    async def _resolve_image_ref_against_event(
        self,
        event: AstrMessageEvent,
        image_ref: str,
    ) -> str:
        """在当前消息的 Image 组件中按 basename / 绝对路径 / 已有本地路径匹配。

        返回：
            - 命中组件时：组件真实的 url/file/path，或 ``convert_to_file_path()`` 结果；
            - 未命中或异常时：空串。
        """
        ref_norm = image_ref.replace("\\", "/").strip()
        ref_basename = Path(ref_norm).name if ref_norm else ""

        try:
            comps = list(event.get_messages())
        except Exception:
            return ""

        for comp in comps:
            if not isinstance(comp, MessageImage):
                continue

            # a. basename 命中组件的 url/file/path
            if ref_basename:
                for attr in ("url", "file", "path"):
                    value = str(getattr(comp, attr, "") or "").strip()
                    if not value:
                        continue
                    value_norm = value.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
                    value_basename = value_norm.rsplit("/", 1)[-1]
                    if value_basename and value_basename == ref_basename:
                        return value

            # b. ref 是已存在的绝对路径且等于组件本地路径
            if os.path.isabs(image_ref):
                for attr in ("file", "path"):
                    value = str(getattr(comp, attr, "") or "").strip()
                    if value and os.path.normcase(os.path.normpath(value)) == os.path.normcase(
                        os.path.normpath(image_ref)
                    ):
                        return value

            # c. 调用组件自身方法把图片落到本地，返回真实可读路径
            if hasattr(comp, "convert_to_file_path"):
                try:
                    path = await comp.convert_to_file_path()
                    path = str(path or "").strip()
                except Exception:
                    path = ""
                if not path:
                    continue
                path_norm = path.replace("\\", "/")
                path_basename = path_norm.rsplit("/", 1)[-1].split("?")[0]
                if ref_basename and path_basename == ref_basename:
                    return path
                if os.path.isabs(image_ref) and os.path.normcase(
                    os.path.normpath(path)
                ) == os.path.normcase(os.path.normpath(image_ref)):
                    return path

        return ""

    def _build_steal_tool_extra_meta(
        self,
        event: AstrMessageEvent,
        image_ref: str,
        *,
        source: str = "llm_tool",
    ) -> dict[str, Any] | None:
        extra_meta: dict[str, Any] = {}
        try:
            scope, target_id = self.get_event_target(event)
        except Exception:
            scope, target_id = "", ""
        if scope and target_id:
            extra_meta["origin_target"] = f"{scope}:{target_id}"

        if image_ref.startswith("http://") or image_ref.startswith("https://"):
            extra_meta["origin_url"] = image_ref
        if source:
            extra_meta["source"] = source
        return extra_meta or None

    async def _process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        is_platform_emoji: bool = False,
        extra_meta: dict[str, Any] | None = None,
        to_pending: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片的方法，包括过滤、分类、存储和索引更新。"""
        try:
            success, updated_idx = await asyncio.wait_for(
                self.image_processor_service.process_image(
                    event=event,
                    file_path=file_path,
                    is_temp=is_temp,
                    idx=idx,
                    categories=self.plugin_config.get_categories(),
                    content_filtration=self.plugin_config.content_filtration,
                    is_platform_emoji=is_platform_emoji,
                    extra_meta=extra_meta,
                    to_pending=to_pending,
                ),
                timeout=self.IMAGE_PROCESSING_TIMEOUT_SECONDS,
            )
            if idx is None and updated_idx is not None and not to_pending:
                full_idx = await self.index_manager.load_index()
                full_idx.update(updated_idx)
                return success, full_idx
            return success, updated_idx
        except asyncio.TimeoutError:
            logger.warning(f"图片处理超时: {file_path}")
            if is_temp:
                await safe_remove_file(file_path)
            return False, idx if idx is not None else {}
        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            if is_temp:
                await safe_remove_file(file_path)
            return False, idx if idx is not None else {}

    async def _extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            return await self.meme_selector.extract_emotions_from_text(event, text)
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 每条新消息到达时重置回合状态，防止上一轮的标记影响当前对话
        if getattr(self, "auto_meme_cancel_on_new_message", True):
            self._cancel_pending_auto_emoji(event)
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

    @filter.on_llm_request()
    async def _inject_emotion_instruction(self, event: AstrMessageEvent, req):
        """在 LLM 请求时动态注入被动标签模式的情绪选择指令。

        使用 extra_user_content_parts 追加指令，避免修改 system_prompt
        破坏 LLM 提供商的提示词缓存。
        """
        try:
            if not self.plugin_config.auto_send_meme:
                return

            turn_state = self._emoji_turn_state(event)
            if turn_state.is_active_sent():
                return

            if turn_state.is_auto_claimed():
                return

            if not await self._resolve_auto_emoji_turn_permission(event):
                return

            if self.plugin_config.enable_natural_emotion_analysis:
                return

            cats = self.plugin_config.get_categories()
            if not cats:
                return

            categories_str = ", ".join(cats)

            emotion_instruction = f"""
{self._persona_marker}
# 角色指令：情绪表达
你需要根据对话的上下文和你当前的回复态度，从以下列表中选择一个最匹配的情绪：
[{categories_str}]

# 输出格式严格要求
1. 必须在回复的**最开头**，使用双浮点号 '&&' 包裹情绪标签。
2. 格式示例：
   &&happy&& 哈哈，这个太有意思了！
   &&sad&& 唉，怎么会这样...
3. 只能使用列表中的情绪词，严禁创造新词。
4. 不要使用 Markdown 代码块或括号，**仅使用 && 符号**。
{self._persona_marker}
"""

            req.system_prompt += emotion_instruction

        except Exception as e:
            logger.error(f"[Stealer] 注入情绪选择指令失败: {e}", exc_info=True)

    @filter.on_decorating_result(priority=100)
    async def _prepare_emoji_response(self, event: AstrMessageEvent):
        """清理情绪标签并异步发送表情包（不阻塞回复）。"""
        result = event.get_result()
        if result is None:
            return False
        if not result.is_llm_result():
            return False
        turn_state = self._emoji_turn_state(event)
        if turn_state.is_active_sent():
            text = result.get_plain_text() or ""
            if text.strip():
                _, cleaned_text = await self._extract_emotions_from_text(event, text)
                if cleaned_text != text:
                    self._update_result_with_cleaned_text_safe(event, result, cleaned_text)
            return False
        text = result.get_plain_text() or ""
        if not text.strip():
            return False
        explicit_emojis = []
        text_without_explicit = re.sub(
            r"\[ast_emoji:(.*?)\]", lambda m: explicit_emojis.append(m.group(1)) or "", text
        )

        emotions = []
        cleaned_text = text_without_explicit
        if not self.plugin_config.enable_natural_emotion_analysis:
            emotions, cleaned_text = await self._extract_emotions_from_text(event, text_without_explicit)
            if cleaned_text != text:
                self._update_result_with_cleaned_text_safe(event, result, cleaned_text)

        turn_allowed = await self._resolve_auto_emoji_turn_permission(event)
        if explicit_emojis:
            if not turn_allowed:
                return cleaned_text != text
            if not self._claim_auto_emoji_turn(event):
                return cleaned_text != text
            sent = await self._send_explicit_emojis(event, explicit_emojis, cleaned_text)
            if sent:
                await self._mark_auto_emoji_sent(event)
                turn_state.mark_active_sent()
                return True
            return cleaned_text != text
        if not turn_allowed:
            return cleaned_text != text
        if self._should_skip_auto_emoji_by_gate(text_without_explicit):
            return cleaned_text != text
        if not self.plugin_config.enable_natural_emotion_analysis and not emotions:
            return cleaned_text != text

        if not self._claim_auto_emoji_turn(event):
            return cleaned_text != text
        user_query = ""
        try:
            user_query = event.get_message_str() or ""
        except Exception:
            pass
        task = self._safe_create_task(
            self._async_analyze_and_send_emoji(event, cleaned_text, emotions, user_query=user_query),
            name="emoji_analyze_passive",
        )
        self._schedule_auto_emoji_task(event, task)
        return True

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
            self.plugin_config.ensure_category_dirs(
                list(self.plugin_config.categories or []) or list(
                    self.plugin_config.DEFAULT_CATEGORIES
                )
            )
            await self.image_processor_service._auto_migrate_categories()
            self._auto_merge_existing_categories()
            try:
                plugin_dir = Path(__file__).parent
                prompts_path = plugin_dir / "prompts.json"
                if prompts_path.exists():
                    if aiofiles:
                        async with aiofiles.open(prompts_path, encoding="utf-8") as f:
                            content = await f.read()
                        prompts = json.loads(content)
                    else:
                        with open(prompts_path, encoding="utf-8") as f:
                            prompts = json.load(f)
                    self._apply_prompts(prompts)
                    self._ensure_default_prompts_in_config(prompts)
            except Exception as e:
                logger.error(f"初始化提示词失败: {e}")
            await self.index_manager.load_index()
            await self._migrate_blacklist_to_db()
            await self.maintenance.run_startup_cleanup()
            self._sync_image_processor_from_runtime()
            self.maintenance.start_periodic_tasks()

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
        if self.cache_service:
            try:
                await self.cache_service.cleanup()
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
            except Exception:
                pass
        if self.command_handler:
            try:
                self.command_handler.cleanup()
            except Exception:
                pass
        if self.event_handler:
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

    async def _migrate_blacklist_to_db(self) -> None:
        """将旧的 blacklist_cache.json 迁移到数据库 blacklist 表。

        幂等：DB 已有同样 hash 时跳过。迁移后保留 JSON 文件直到下一次写黑名单不再写它，
        避免在长期运行实例中破坏现有读取链路。
        """
        try:
            db = getattr(self, "db_service", None)
            if db is None or not hasattr(db, "add_blacklist_batch"):
                return
            cached = self.cache_service.get_cache("blacklist_cache") or {}
            if not isinstance(cached, dict) or not cached:
                return
            hashes: dict[str, int] = {}
            for h, ts in cached.items():
                try:
                    hashes[str(h)] = int(ts) if ts else int(time.time())
                except Exception:
                    hashes[str(h)] = 0
            imported = await db.add_blacklist_batch(hashes)
            if imported > 0:
                logger.info(f"[DB] 黑名单从缓存迁移完成，新增 {imported} 条")
        except Exception as e:
            logger.warning(f"[DB] 黑名单迁移失败: {e}")
