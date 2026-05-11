import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import (
    EventMessageType,
    PermissionType,
    PlatformAdapterType,
)
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import TextPart

from .cache_service import CacheService
from .core.commands.command_handler import CommandHandler
from .core.config.config import PluginConfig
from .core.db.database_service import DatabaseService
from .core.search.emoji_selector import EmojiSelector
from .core.events.event_handler import EventHandler
from .core.events.emoji_sender_engine import EmojiSenderEngine
from .core.db.index_manager import IndexManager
from .core.processing.natural_emotion_analyzer import SmartEmotionMatcher
from .core.processing.image_processor_service import ImageProcessorService
from .task_scheduler import TaskScheduler
from .plugin_api import PluginAPI

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

        # 同步配置到实例属性（纯属性赋值，无IO）
        self._sync_all_config()

        # 初始化核心服务类
        self.cache_service = CacheService(self.cache_dir)
        self.db_service = DatabaseService(self.cache_dir / "emoji.db")
        self.command_handler = CommandHandler(self)
        self.web_server = None
        self.plugin_api = PluginAPI(self)
        self.plugin_api.register(context)

        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.emoji_selector = EmojiSelector(self)
        self.task_scheduler = TaskScheduler()

        # 初始化自然语言情绪分析器（新增）
        self.smart_emotion_matcher = SmartEmotionMatcher(self)

        self.index_manager = IndexManager(self)
        self._emoji_sender_engine = EmojiSenderEngine(self)

        # 运行时属性
        self._terminated: bool = False  # 终止标志位，防止重复清理
        # 强制捕获窗口已迁移到 EventHandler

    def _sync_all_config(self) -> None:
        """从配置服务同步所有配置到实例属性。"""
        self.auto_send = self.plugin_config.auto_send
        self.emoji_chance = self.plugin_config.emoji_chance
        self.steal_mode = self.plugin_config.steal_mode
        self.steal_chance = self.plugin_config.steal_chance
        self.send_emoji_as_gif = self.plugin_config.send_emoji_as_gif
        self.emoji_send_delay = self.plugin_config.emoji_send_delay
        self.emoji_send_delay_random = self.plugin_config.emoji_send_delay_random
        self.emoji_send_delay_max = self.plugin_config.emoji_send_delay_max
        self.max_reg_num = self.plugin_config.max_reg_num
        self.content_filtration = self.plugin_config.content_filtration
        self.smart_emoji_selection = self.plugin_config.smart_emoji_selection
        self.steal_emoji = self.plugin_config.steal_emoji
        self.categories = list(self.plugin_config.categories or []) or list(
            self.plugin_config.DEFAULT_CATEGORIES
        )
        self.vision_provider_id = self._load_vision_provider_id()
        self.napcat_token = self._load_napcat_token()
        self.enable_natural_emotion_analysis = self.plugin_config.enable_natural_emotion_analysis
        self.emotion_analysis_provider_id = self.plugin_config.emotion_analysis_provider_id
        self.image_processing_cooldown = self.plugin_config.image_processing_cooldown

    def _load_vision_provider_id(self) -> str:
        """加载视觉模型提供商ID。"""
        provider_id = getattr(self.plugin_config, "vision_provider_id", "")
        return str(provider_id).strip() if provider_id else ""

    def _load_napcat_token(self) -> str:
        """加载 NapCat 访问令牌。"""
        user_token = getattr(self.plugin_config, "napcat_token", "")
        if user_token:
            return str(user_token).strip()
        return ""

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
        current_prompt = getattr(self.plugin_config, "custom_emoji_classification_prompt", "")
        if not current_prompt or not current_prompt.strip():
            default_prompt = prompts.get("EMOJI_CLASSIFICATION_PROMPT", "")
            if default_prompt:
                updates["custom_emoji_classification_prompt"] = default_prompt
        current_filter_prompt = getattr(
            self.plugin_config, "custom_emoji_classification_with_filter_prompt", ""
        )
        if not current_filter_prompt or not current_filter_prompt.strip():
            default_filter_prompt = prompts.get("EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", "")
            if default_filter_prompt:
                updates["custom_emoji_classification_with_filter_prompt"] = default_filter_prompt
        if updates:
            self._update_config_from_dict(updates)
            logger.info(f"已将默认提示词写入配置: {list(updates.keys())}")

    def _auto_merge_existing_categories(self) -> None:
        """自动合并已存在的分类目录到配置中。"""
        current = list(getattr(self.plugin_config, "DEFAULT_CATEGORIES", []) or [])
        current_set = set(current)
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
            if not index:
                index = self.cache_service.get_index_cache_readonly()
            for meta in index.values():
                if not isinstance(meta, dict):
                    continue
                cat = str(meta.get("category", "")).strip()
                if not cat or cat == "unknown":
                    continue
                discovered.add(cat)
        except Exception as e:
            logger.warning(f"[Config] 从索引合并分类时出错: {e}")
        to_add = sorted(discovered - current_set)
        if not to_add:
            return
        merged_categories = current + to_add
        self._update_config_from_dict({"categories": merged_categories})
        self.plugin_config.ensure_category_dirs(to_add)

    def _validate_config(self) -> bool:
        """验证配置参数的有效性。"""
        errors = []
        fixed = []
        fixed_values = {}
        if not isinstance(self.max_reg_num, int) or self.max_reg_num <= 0:
            errors.append("最大表情数量必须大于0的整数")
            self.max_reg_num = 100
            fixed.append("最大表情数量已重置为100")
            fixed_values["max_reg_num"] = 100
        if not isinstance(self.emoji_chance, (int, float)) or not (0 <= self.emoji_chance <= 1):
            errors.append("表情发送概率必须在0-1之间")
            self.emoji_chance = 0.4
            fixed.append("表情发送概率已重置为0.4")
            fixed_values["emoji_chance"] = 0.4
        if self.steal_mode not in ("probability", "cooldown"):
            errors.append(f"偷图模式 '{self.steal_mode}' 无效，必须为 probability 或 cooldown")
            self.steal_mode = "probability"
            fixed.append("偷图模式已重置为 probability")
            fixed_values["steal_mode"] = "probability"
        if not isinstance(self.steal_chance, (int, float)) or not (0 <= self.steal_chance <= 1):
            errors.append("偷图概率必须在0-1之间")
            self.steal_chance = 0.6
            fixed.append("偷图概率已重置为0.6")
            fixed_values["steal_chance"] = 0.6
        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
        if fixed:
            logger.info(f"配置已自动修复: {'; '.join(fixed)}")
            try:
                self._update_config_from_dict(fixed_values)
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
        """将更新字典写回 PluginConfig（包含 webui 嵌套字段兼容）。"""
        for k, v in config_dict.items():
            if k == "webui" and isinstance(v, dict):
                current_webui = self.plugin_config.webui
                for wk, wv in v.items():
                    setattr(current_webui, wk, wv)
                self.plugin_config.save_webui_config()
            elif k.startswith("webui_"):
                wk = k[6:]
                if hasattr(self.plugin_config.webui, wk):
                    setattr(self.plugin_config.webui, wk, v)
                    self.plugin_config.save_webui_config()
            else:
                setattr(self.plugin_config, k, v)

    def _sync_image_processor_from_runtime(self) -> None:
        final_prompts = self.plugin_config.get_prompts(
            {
                "EMOJI_CLASSIFICATION_PROMPT": getattr(self, "EMOJI_CLASSIFICATION_PROMPT", None),
                "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT": getattr(
                    self, "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
                ),
            }
        )
        self.image_processor_service.update_config(
            categories=self.categories,
            content_filtration=self.content_filtration,
            vision_provider_id=self.vision_provider_id,
            emoji_classification_prompt=final_prompts.get("emoji_classification_prompt"),
            emoji_classification_with_filter_prompt=final_prompts.get(
                "emoji_classification_with_filter_prompt"
            ),
        )

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return
        try:
            if self.plugin_config:
                self._apply_plugin_config_updates(config_dict)
                self._sync_all_config()
                self._sync_image_processor_from_runtime()
                try:
                    self.plugin_config.ensure_category_dirs(self.categories)
                except Exception as e:
                    logger.warning(f"[Config] 创建分类目录失败: {e}")
                logger.debug("[Config] 配置已更新，下次 LLM 请求将使用新分类")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")

    # ===== 门面委托：EmojiSenderEngine =====
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
    _normalize_auto_emoji_chance = (  # noqa: E731
        lambda self: self._emoji_sender_engine.normalize_auto_emoji_chance()
    )
    _resolve_auto_emoji_turn_permission = (  # noqa: E731
        lambda self, event: self._emoji_sender_engine.resolve_auto_emoji_turn_permission(event)
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
    _try_send_emoji = lambda self, event, emotions, text: self._emoji_sender_engine.try_send_emoji(  # noqa: E731
        event, emotions, text
    )
    _get_emoji_send_delay = lambda self: self._emoji_sender_engine.get_emoji_send_delay()  # noqa: E731
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
        """委托给 EmojiSelector.smart_search。"""
        if idx is None:
            idx = (
                self.db_service.get_index_cache_readonly()
                if self.db_service.count_total() > 0
                else {}
            )
            if not idx:
                idx = self.cache_service.get_index_cache_readonly()

        return await self.emoji_selector.smart_search(query, limit=limit, idx=idx, event=event)

    def _find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的多个分类，委托给 EmojiSelector。"""
        return self.emoji_selector.find_similar_categories(query, top_n)

    @filter.llm_tool(name="search_emoji")
    async def search_emoji(self, event: AstrMessageEvent, query: str):
        """搜索表情包候选，并优先按你当前心情词进行匹配。

        Args:
            query(string): 你当前心情的代表词（也支持描述词、场景词）

        使用建议：
        - 先判断你此刻最能代表自己的心情词（例如：开心、无语、尴尬、感谢）
        - 再用该心情词调用本工具搜索候选
        - 若无结果，可换同义词再搜索（如“无语”->"dumb/尴尬"）

        返回值：
        返回候选表情包列表，每个包含：
        - 编号：用于调用 send_emoji_by_id
        - 分类：表情包的情绪分类
        - 描述：表情包的详细描述（这是你选择时的重要参考）

        请先锁定“当前心情词”，再仔细阅读候选描述，选择最能代表你当前心情与语气的一张。
        """
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
            elif self.cache_service.get_index_cache_readonly():
                idx = self.cache_service.get_index_cache_readonly()
            else:
                logger.debug("索引未加载，正在加载...")
                await self._load_index()
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
                suggestion += "\n\n可用分类：" + ", ".join(self.categories[:10])
                if len(self.categories) > 10:
                    suggestion += f" 等共{len(self.categories)}个分类"
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
                        }
                    )
                    result_lines.append(f"\n[{i + 1}] 分类：{emotion}")
                    if tags:
                        result_lines.append(f"    标签：{tags}")
                    if scenes_str:
                        result_lines.append(f"    场景：{scenes_str}")
                    else:
                        result_lines.append("    场景：无")
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

    @filter.llm_tool(name="send_emoji_by_id")
    async def send_emoji_by_id(self, event: AstrMessageEvent, emoji_id: int):
        """发送你选择的表情包。必须先调用 search_emoji 获取候选列表。

        选择原则：优先发送能代表你“当前心情词”的候选项。

        Args:
            emoji_id(number): 表情包编号（从 search_emoji 返回的列表中选择）

        """
        logger.info(f"[Tool] LLM 选择发送表情包编号: {emoji_id}")
        turn_state = self._emoji_turn_state(event)

        try:
            if not self.is_send_enabled_for_event(event):
                yield "发送失败：当前群聊已禁用表情包功能"
                return

            if emoji_id is None:
                yield "发送失败：缺少 emoji_id 参数。请先调用 search_emoji，再传入候选编号。"
                return

            try:
                emoji_id = int(emoji_id)
            except Exception:
                yield f"发送失败：编号 {emoji_id} 无法解析为整数，请输入有效的数字编号"
                return

            candidates = turn_state.get_candidates()
            if not candidates:
                yield "发送失败：没有可用的候选列表。请先调用 search_emoji 搜索表情包。"
                return

            if emoji_id < 1 or emoji_id > len(candidates):
                yield f"发送失败：编号 {emoji_id} 无效。可选编号范围：1-{len(candidates)}，请重新选择。"
                return

            selected = candidates[emoji_id - 1]
            path = selected["path"]
            desc = selected["desc"]
            emotion = selected["emotion"]

            if not os.path.exists(path):
                yield f"发送失败：表情包文件已丢失。\n你选择的是：编号 {emoji_id}，分类 {emotion}，描述 {desc}\n请重新搜索并选择其他表情包。"
                return

            if not self.emoji_selector.is_path_allowed_for_event(path, event):
                yield "发送失败：该表情包被限制为仅来源群可发送，请选择其他表情包。"
                return

            if not await self._resolve_auto_emoji_turn_permission(event):
                yield "发送失败：当前轮次未触发表情包发送条件"
                return

            if not self._claim_auto_emoji_turn(event):
                yield "发送失败：当前轮次已经有表情包发送流程"
                return

            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            send_mode = await self.emoji_selector.send_emoji_message(event, path)
            if not send_mode:
                yield "发送失败：表情包编码或发送失败，请重试。"
                return
            sent_as_sticker = send_mode == "telegram_sticker"

            await self.emoji_selector.record_emoji_usage(path, trigger="llm_tool")
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

    async def _save_index(self, idx: dict[str, Any]):
        """将当前权威索引同步到数据库与缓存。"""
        await self.db_service.sync_index(idx)
        await self.cache_service.set_cache("index_cache", idx, persist=False)

    async def _rebuild_index_from_files(self) -> dict[str, Any]:
        """从文件重建基础索引（不保存到数据库，等待合并后保存）。"""
        return await self.cache_service.rebuild_index_from_files(self.base_dir, self.categories_dir)

    async def _process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        is_platform_emoji: bool = False,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片的方法，包括过滤、分类、存储和索引更新。"""
        try:
            success, updated_idx = await asyncio.wait_for(
                self.image_processor_service.process_image(
                    event=event,
                    file_path=file_path,
                    is_temp=is_temp,
                    idx=idx,
                    categories=self.categories,
                    content_filtration=self.content_filtration,
                    is_platform_emoji=is_platform_emoji,
                    extra_meta=extra_meta,
                ),
                timeout=self.IMAGE_PROCESSING_TIMEOUT_SECONDS,
            )
            if idx is None and updated_idx is not None:
                full_idx = await self._load_index()
                full_idx.update(updated_idx)
                return success, full_idx
            return success, updated_idx
        except asyncio.TimeoutError:
            logger.warning(f"图片处理超时: {file_path}")
            if is_temp:
                await self._safe_remove_file(file_path)
            return False, idx if idx is not None else {}
        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            if is_temp:
                await self._safe_remove_file(file_path)
            return False, idx if idx is not None else {}

    async def _safe_remove_file(self, file_path: str) -> bool:
        """安全删除文件。"""
        try:
            return await self.image_processor_service.safe_remove_file(file_path)
        except Exception as e:
            logger.error(f"安全删除文件失败: {e}")
            return False

    async def _extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            return await self.emoji_selector.extract_emotions_from_text(event, text)
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 每条新消息到达时重置回合状态，防止上一轮的标记影响当前对话
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
        """在 LLM 请求时动态注入情绪选择指令。

        使用 extra_user_content_parts 追加指令，避免修改 system_prompt
        破坏 LLM 提供商的提示词缓存。
        """
        try:
            if not self.auto_send:
                return

            turn_state = self._emoji_turn_state(event)
            if turn_state.is_active_sent():
                return

            if turn_state.is_auto_claimed():
                return

            if not await self._resolve_auto_emoji_turn_permission(event):
                return

            if self.enable_natural_emotion_analysis:
                return

            if not self.categories:
                return

            categories_str = ", ".join(self.categories)

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

            req.extra_user_content_parts.append(TextPart(text=emotion_instruction))

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
        turn_allowed = await self._resolve_auto_emoji_turn_permission(event)
        explicit_emojis = []
        text_without_explicit = re.sub(
            r"\[ast_emoji:(.*?)\]", lambda m: explicit_emojis.append(m.group(1)) or "", text
        )
        if explicit_emojis:
            _, cleaned_text = await self._extract_emotions_from_text(event, text_without_explicit)
            if cleaned_text != text:
                self._update_result_with_cleaned_text_safe(event, result, cleaned_text)
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
            return False
        if self._should_skip_auto_emoji_by_gate(text_without_explicit):
            return False

        emotions = []
        cleaned_text = text_without_explicit
        if not self.enable_natural_emotion_analysis:
            emotions, cleaned_text = await self._extract_emotions_from_text(event, text_without_explicit)
            if cleaned_text != text:
                self._update_result_with_cleaned_text_safe(event, result, cleaned_text)
            if not emotions:
                return cleaned_text != text

        if not self._claim_auto_emoji_turn(event):
            return False
        user_query = ""
        try:
            user_query = event.get_message_str() or ""
        except Exception:
            pass
        self._safe_create_task(
            self._async_analyze_and_send_emoji(event, cleaned_text, emotions, user_query=user_query),
            name="emoji_analyze_passive",
        )
        return True

    async def initialize(self):
        """初始化插件运行时资源。

        加载情绪映射和提示词等运行时需要的资源。
        __init__ 仅做属性赋值，IO/目录/密码等操作统一在此执行。
        """
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
            self._sync_all_config()
            self.plugin_config.ensure_base_dirs()
            self.plugin_config.ensure_category_dirs(self.categories)
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
            await self._load_index()
            self._sync_all_config()
            self._sync_image_processor_from_runtime()
            self.task_scheduler.create_task("raw_cleanup_loop", self._raw_cleanup_loop())
            self.task_scheduler.create_task("capacity_control_loop", self._capacity_control_loop())
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
        logger.info("[Stealer] 插件资源清理完成")

    async def _load_index(self) -> dict[str, Any]:
        """加载索引，优先从数据库加载。"""
        try:
            idx: dict[str, Any] = {}
            db_count = self.db_service.count_total()
            if db_count > 0:
                idx = self.db_service.get_index_cache_readonly()
            if idx:
                await self.cache_service.set_cache("index_cache", idx, persist=False)
            return idx
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return {}

    async def _raw_cleanup_loop(self):
        """raw目录清理循环。"""
        while True:
            try:
                await asyncio.sleep(self.RAW_CLEANUP_INTERVAL_SECONDS)
                if self.event_handler:
                    await self.event_handler._clean_raw_directory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"raw清理循环出错: {e}")

    async def _capacity_control_loop(self):
        """容量控制循环。"""
        while True:
            try:
                await asyncio.sleep(self.CAPACITY_CONTROL_INTERVAL_SECONDS)
                idx = await self._load_index()
                if len(idx) > self.max_reg_num:
                    await self.event_handler._enforce_capacity(idx)
                    await self._save_index(idx)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"容量控制循环出错: {e}")
