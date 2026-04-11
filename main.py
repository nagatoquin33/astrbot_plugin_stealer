import asyncio
import json
import os
import random
import re
import secrets
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import (
    EventMessageType,
    PermissionType,
    PlatformAdapterType,
)
from astrbot.api.message_components import Image as ImageComponent
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star

from .cache_service import CacheService
from .core.command_handler import CommandHandler
from .core.config import PluginConfig
from .core.database_service import DatabaseService
from .core.emoji_selector import EmojiSelector
from .core.event_handler import EventHandler
from .core.image_processor_service import ImageProcessorService
from .core.natural_emotion_analyzer import SmartEmotionMatcher
from .task_scheduler import TaskScheduler
from .web_server import WebServer

try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None


class _EmojiTurnState:
    """Wrap per-turn emoji state stored on the event object."""

    ACTIVE_SENT_KEY = "stealer_active_sent"
    AUTO_DECIDED_KEY = "stealer_auto_emoji_turn_decided"
    AUTO_ALLOWED_KEY = "stealer_auto_emoji_turn_allowed"
    AUTO_REASON_KEY = "stealer_auto_emoji_turn_reason"
    AUTO_CLAIMED_KEY = "stealer_auto_emoji_turn_claimed"
    AUTO_SENT_KEY = "stealer_auto_emoji_sent"
    CANDIDATES_KEY = "stealer_emoji_candidates"

    def __init__(self, event: AstrMessageEvent):
        self.event = event

    def is_active_sent(self) -> bool:
        return bool(self.event.get_extra(self.ACTIVE_SENT_KEY, False))

    def mark_active_sent(self) -> None:
        self.event.set_extra(self.ACTIVE_SENT_KEY, True)

    def is_auto_decided(self) -> bool:
        return bool(self.event.get_extra(self.AUTO_DECIDED_KEY, False))

    def get_auto_allowed(self) -> bool:
        return bool(self.event.get_extra(self.AUTO_ALLOWED_KEY, False))

    def set_auto_decision(self, *, allowed: bool, reason: str) -> None:
        self.event.set_extra(self.AUTO_DECIDED_KEY, True)
        self.event.set_extra(self.AUTO_ALLOWED_KEY, allowed)
        self.event.set_extra(self.AUTO_REASON_KEY, reason)

    def get_auto_reason(self) -> str:
        return str(self.event.get_extra(self.AUTO_REASON_KEY, "unknown"))

    def is_auto_claimed(self) -> bool:
        return bool(self.event.get_extra(self.AUTO_CLAIMED_KEY, False))

    def claim_auto_send(self) -> bool:
        if self.is_auto_claimed():
            return False
        self.event.set_extra(self.AUTO_CLAIMED_KEY, True)
        return True

    def set_candidates(self, candidates: list[dict[str, Any]]) -> None:
        self.event.set_extra(self.CANDIDATES_KEY, candidates)

    def get_candidates(self) -> list[dict[str, Any]] | None:
        return self.event.get_extra(self.CANDIDATES_KEY)

    def is_auto_sent(self) -> bool:
        return bool(self.event.get_extra(self.AUTO_SENT_KEY, False))

    def mark_auto_sent(self) -> bool:
        if self.is_auto_sent():
            return False
        self.event.set_extra(self.AUTO_SENT_KEY, True)
        return True


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

        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.emoji_selector = EmojiSelector(self)
        self.task_scheduler = TaskScheduler()

        # 初始化自然语言情绪分析器（新增）
        self.smart_emotion_matcher = SmartEmotionMatcher(self)

        # 运行时属性
        self.backend_tag: str = self.BACKEND_TAG
        self._migration_done: bool = False  # 迁移只执行一次
        self._auto_emoji_cooldowns: dict[str, float] = {}
        self._auto_emoji_cooldowns_max = 1000  # 最大条目数，防止内存泄漏
        self._auto_emoji_cooldowns_lock = asyncio.Lock()
        self._terminated: bool = False  # 终止标志位，防止重复清理
        # 强制捕获窗口已迁移到 EventHandler

    def _emoji_turn_state(self, event: AstrMessageEvent) -> _EmojiTurnState:
        return _EmojiTurnState(event)

    def _load_vision_provider_id(self) -> str:
        """加载视觉模型提供商ID。

        Returns:
            str: 视觉模型提供商ID，如果未配置则返回空字符串
        """
        provider_id = getattr(self.plugin_config, "vision_provider_id", "")
        return str(provider_id).strip() if provider_id else ""

    def _load_napcat_token(self) -> str:
        """加载 NapCat 访问令牌。

        优先从用户配置获取，如果未配置则尝试从 OneBot 适配器配置中自动获取。

        Returns:
            str: NapCat 访问令牌，如果未配置则返回空字符串
        """
        user_token = getattr(self.plugin_config, "napcat_token", "")
        if user_token:
            logger.debug("使用用户手动配置的 NapCat token")
            return str(user_token).strip()

        try:
            if hasattr(self, "context") and self.context:
                adapter_config = getattr(self.context, "_adapter_config", None)
                if not adapter_config:
                    adapter_config = getattr(self.context, "adapter_config", None)

                if adapter_config and isinstance(adapter_config, dict):
                    for key, value in adapter_config.items():
                        if "onebot" in key.lower() or "napcat" in key.lower():
                            if isinstance(value, dict):
                                token = value.get("token") or value.get("access_token")
                                if token:
                                    logger.debug(f"从适配器配置 '{key}' 中自动获取 NapCat token")
                                    return str(token).strip()

                adapters = getattr(self.context, "adapters", [])
                for adapter in adapters:
                    adapter_name = getattr(adapter, "name", "").lower()
                    if "onebot" in adapter_name or "napcat" in adapter_name:
                        token = getattr(adapter, "token", None) or getattr(adapter, "access_token", None)
                        if not token:
                            cfg = getattr(adapter, "config", None)
                            if cfg:
                                token = getattr(cfg, "token", None) or getattr(cfg, "access_token", None)
                        if token:
                            logger.debug(f"从适配器 '{adapter_name}' 中自动获取 NapCat token")
                            return str(token).strip()
        except Exception as e:
            logger.debug(f"自动获取 NapCat token 失败：{e}")

        logger.debug("未配置 NapCat token，将不使用认证")
        return ""


    def _sync_all_config(self) -> None:
        """从配置服务同步所有配置到实例属性。

        统一的配置同步方法，避免重复代码。
        """
        # 同步基础配置
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

        # 同步模型相关配置
        self.vision_provider_id = self._load_vision_provider_id()
        self.napcat_token = self._load_napcat_token()

        # 同步自然语言分析配置
        self.enable_natural_emotion_analysis = (
            self.plugin_config.enable_natural_emotion_analysis
        )
        self.emotion_analysis_provider_id = (
            self.plugin_config.emotion_analysis_provider_id
        )

        # 同步图片处理节流配置
        self.image_processing_cooldown = self.plugin_config.image_processing_cooldown

        # 同步 WebUI 配置
        self.webui_enabled = self.plugin_config.webui.enabled
        self.webui_host = self.plugin_config.webui.host
        self.webui_port = self.plugin_config.webui.port
        self.webui_auth_enabled = self.plugin_config.webui.auth_enabled
        self.webui_password = self.plugin_config.webui.password
        self.webui_session_timeout = self.plugin_config.webui.session_timeout

    def _ensure_webui_password(self) -> bool:
        """确保 WebUI 密码已设置，自动生成时返回明文密码供用户查看。

        Returns:
            bool: 是否生成了新密码
        """
        if (
            self.plugin_config.webui.enabled
            and self.plugin_config.webui.auth_enabled
            and not str(self.plugin_config.webui.password or "").strip()
        ):
            # 生成随机密码（仅在首次生成时返回明文）
            generated = f"{secrets.randbelow(1000000):06d}"
            # 存储密码（配置文件中的密码由用户自行管理安全性）
            # 注意：配置文件本身应设置适当的文件权限
            self.plugin_config.webui.password = generated
            self.plugin_config.save_webui_config()
            logger.info("WebUI 访问密码已自动生成，请在配置中查看")
            return True
        return False

    def _apply_prompts(self, prompts: dict) -> None:
        for key, value in prompts.items():
            setattr(self, key, value)

        # 使用配置服务获取提示词
        final_prompts = self.plugin_config.get_prompts(prompts)

        self.image_processor_service.update_config(
            emoji_classification_prompt=final_prompts.get("emoji_classification_prompt"),
            combined_analysis_prompt=prompts.get("COMBINED_ANALYSIS_PROMPT"),
            emoji_classification_with_filter_prompt=final_prompts.get(
                "emoji_classification_with_filter_prompt"
            ),
        )

    def _ensure_default_prompts_in_config(self, prompts: dict) -> None:
        """如果配置中的提示词字段为空，将 prompts.json 内容写入配置作为默认显示值。

        这样用户在配置界面就能直接看到默认提示词内容，并在此基础上编辑。
        """
        updates = {}

        # 检查普通分类提示词
        current_prompt = getattr(self.plugin_config, "custom_emoji_classification_prompt", "")
        if not current_prompt or not current_prompt.strip():
            default_prompt = prompts.get("EMOJI_CLASSIFICATION_PROMPT", "")
            if default_prompt:
                updates["custom_emoji_classification_prompt"] = default_prompt

        # 检查带审核的分类提示词
        current_filter_prompt = getattr(
            self.plugin_config, "custom_emoji_classification_with_filter_prompt", ""
        )
        if not current_filter_prompt or not current_filter_prompt.strip():
            default_filter_prompt = prompts.get("EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", "")
            if default_filter_prompt:
                updates["custom_emoji_classification_with_filter_prompt"] = default_filter_prompt

        # 如果有更新，写入配置
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
            index = self.db_service.get_index_cache_readonly() if self.db_service.count_total() > 0 else {}
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

        # 为新增的分类创建对应的目录
        self.plugin_config.ensure_category_dirs(to_add)

    def _validate_config(self) -> bool:
        """验证配置参数的有效性。

        Returns:
            bool: 配置是否有效（修复后的配置也算有效）
        """
        errors = []
        fixed = []
        fixed_values = {}  # 需要持久化的修复值

        # 验证最大表情数量
        if not isinstance(self.max_reg_num, int) or self.max_reg_num <= 0:
            errors.append("最大表情数量必须大于0的整数")
            self.max_reg_num = 100
            fixed.append("最大表情数量已重置为100")
            fixed_values["max_reg_num"] = 100

        # 验证表情发送概率
        if not isinstance(self.emoji_chance, int | float) or not (
            0 <= self.emoji_chance <= 1
        ):
            errors.append("表情发送概率必须在0-1之间")
            self.emoji_chance = 0.4
            fixed.append("表情发送概率已重置为0.4")
            fixed_values["emoji_chance"] = 0.4

        # 验证偷图模式
        if self.steal_mode not in ("probability", "cooldown"):
            errors.append(
                f"偷图模式 '{self.steal_mode}' 无效，必须为 probability 或 cooldown"
            )
            self.steal_mode = "probability"
            fixed.append("偷图模式已重置为 probability")
            fixed_values["steal_mode"] = "probability"

        # 验证偷图概率
        if not isinstance(self.steal_chance, int | float) or not (
            0 <= self.steal_chance <= 1
        ):
            errors.append("偷图概率必须在0-1之间")
            self.steal_chance = 0.6
            fixed.append("偷图概率已重置为0.6")
            fixed_values["steal_chance"] = 0.6

        # 记录问题和修复
        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
        if fixed:
            logger.info(f"配置已自动修复: {'; '.join(fixed)}")
            # 持久化修复的值到配置文件
            try:
                self._update_config_from_dict(fixed_values)
            except Exception as e:
                logger.error(f"持久化配置修复失败: {e}")

        return True  # 即使有问题也返回True，因为已经修复

    @staticmethod
    def _safe_create_task(coro, *, name: str = "") -> asyncio.Task:
        """创建 asyncio task 并自动记录未处理异常，避免 fire-and-forget 静默吞异常。"""
        task = asyncio.create_task(coro, name=name or None)

        def _on_done(t: asyncio.Task):
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                logger.error(f"后台任务 '{t.get_name()}' 异常: {exc}", exc_info=exc)

        task.add_done_callback(_on_done)
        return task

    def _get_group_id(self, event: AstrMessageEvent) -> str | None:
        """委托给 PluginConfig。"""
        return self.plugin_config.get_group_id(event)

    def get_event_target(self, event: AstrMessageEvent) -> tuple[str, str]:
        if self.plugin_config is None:
            return "", ""
        try:
            return self.plugin_config.get_event_target(event)
        except Exception:
            return "", ""

    def _is_action_enabled_for_event(
        self, action: str, event: AstrMessageEvent
    ) -> bool:
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

    def is_meme_enabled_for_event(self, event: AstrMessageEvent) -> bool:
        return self.is_send_enabled_for_event(event)

    def begin_force_capture(self, event: AstrMessageEvent, seconds: int) -> None:
        """委托给 EventHandler。"""
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法进入强制接收模式"
        )
        if event_handler is None:
            return
        event_handler.begin_force_capture(event, seconds)

    def get_force_capture_entry(
        self, event: AstrMessageEvent
    ) -> dict[str, object] | None:
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

    def _get_event_handler(
        self,
        *,
        log_message: str | None = None,
        log_level: str = "warning",
    ) -> EventHandler | None:
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

    def _snapshot_webui_runtime(self) -> tuple[bool, str, int, str, int]:
        """获取当前 WebUI 运行态配置快照。"""
        return (
            getattr(self, "webui_enabled", True),
            getattr(self, "webui_host", "0.0.0.0"),
            getattr(self, "webui_port", 9191),
            getattr(self, "webui_password", ""),
            getattr(self, "webui_session_timeout", 3600),
        )

    def _is_webui_runtime_changed(
        self, old_state: tuple[bool, str, int, str, int]
    ) -> bool:
        return old_state != self._snapshot_webui_runtime()

    async def _restart_webui(self) -> None:
        logger.info("检测到 WebUI 配置变更，正在重启 WebUI...")

        if not self.webui_enabled:
            # WebUI 已禁用，停止旧服务即可
            if self.web_server:
                await self.web_server.stop()
                self.web_server = None
            return

        # 保存旧服务引用，在新服务启动成功后再停止
        old_server = self.web_server

        try:
            new_server = WebServer(self, host=self.webui_host, port=self.webui_port)
            await new_server.start()
            # 新服务启动成功，更新引用并停止旧服务
            self.web_server = new_server
            if old_server:
                try:
                    await old_server.stop()
                except Exception as e:
                    logger.warning(f"停止旧 WebUI 服务时出错: {e}")
            logger.info("WebUI 重启成功")
        except Exception as e:
            logger.error(f"重启 WebUI 失败: {e}", exc_info=True)
            # 启动失败，恢复旧服务引用
            if old_server and self.web_server != old_server:
                self.web_server = old_server
                logger.info("已恢复旧的 WebUI 服务")

    def _apply_plugin_config_updates(self, config_dict: dict) -> None:
        """将更新字典写回 PluginConfig（包含 webui 嵌套字段兼容）。"""
        for k, v in config_dict.items():
            if k == "webui" and isinstance(v, dict):
                current_webui = self.plugin_config.webui
                for wk, wv in v.items():
                    setattr(current_webui, wk, wv)
                self.plugin_config.save_webui_config()
            elif k.startswith("webui_"):
                # 兼容旧版扁平 key：webui_enabled -> webui.enabled
                wk = k[6:]
                if hasattr(self.plugin_config.webui, wk):
                    setattr(self.plugin_config.webui, wk, v)
                    self.plugin_config.save_webui_config()
            else:
                setattr(self.plugin_config, k, v)

    def _sync_image_processor_from_runtime(self) -> None:
        # 使用配置服务获取提示词
        final_prompts = self.plugin_config.get_prompts({
            "EMOJI_CLASSIFICATION_PROMPT": getattr(self, "EMOJI_CLASSIFICATION_PROMPT", None),
            "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT": getattr(
                self, "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
            ),
        })

        self.image_processor_service.update_config(
            categories=self.categories,
            content_filtration=self.content_filtration,
            vision_provider_id=self.vision_provider_id,
            emoji_classification_prompt=final_prompts.get("emoji_classification_prompt"),
            combined_analysis_prompt=getattr(self, "COMBINED_ANALYSIS_PROMPT", None),
            emoji_classification_with_filter_prompt=final_prompts.get(
                "emoji_classification_with_filter_prompt"
            ),
        )

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            # 使用配置服务更新配置
            if self.plugin_config:
                old_webui_state = self._snapshot_webui_runtime()
                self._apply_plugin_config_updates(config_dict)

                # 统一同步所有配置
                self._sync_all_config()

                if self._ensure_webui_password():
                    self._sync_all_config()

                # 检查 WebUI 配置是否变化并重启
                # 注意：on_config_update 可能是同步调用，重启操作涉及IO，使用 create_task 异步执行
                if self._is_webui_runtime_changed(old_webui_state):
                    self._safe_create_task(self._restart_webui(), name="restart_webui")

                # 更新其他服务的配置
                self._sync_image_processor_from_runtime()

                # 为新增的分类创建对应的目录
                try:
                    self.plugin_config.ensure_category_dirs(self.categories)
                except Exception as e:
                    logger.warning(f"[Config] 创建分类目录失败: {e}")

                # 注意：不再调用 _reload_personas，因为已改用 LLM 钩子
                # LLM 钩子会在下次请求时自动使用更新后的分类列表
                logger.debug("[Config] 配置已更新，下次 LLM 请求将使用新分类")
        except Exception as e:
            logger.error(f"更新配置失败: {e}")

    async def initialize(self):
        """初始化插件运行时资源。

        加载情绪映射和提示词等运行时需要的资源。
        __init__ 仅做属性赋值，IO/目录/密码等操作统一在此执行。
        """
        try:
            # ── 从 __init__ 移入的IO操作 ──
            self._ensure_webui_password()
            self._validate_config()

            if (
                self._get_event_handler(
                    log_message="[Stealer] event_handler 未初始化，插件无法启动",
                    log_level="error",
                )
                is None
            ):
                raise RuntimeError("event_handler 未初始化")

            # 密码可能被自动生成，立即同步到实例属性
            self._sync_all_config()

            self.plugin_config.ensure_base_dirs()
            self.plugin_config.ensure_category_dirs(self.categories)

            # 异步分类迁移（ImageProcessorService 的旧版迁移）
            await self.image_processor_service._auto_migrate_categories()

            self._auto_merge_existing_categories()

            # 加载提示词文件
            try:
                # 使用__file__获取当前脚本所在目录，即插件安装目录
                plugin_dir = Path(__file__).parent
                prompts_path = plugin_dir / "prompts.json"
                if prompts_path.exists():
                    if aiofiles:
                        async with aiofiles.open(prompts_path, encoding="utf-8") as f:
                            content = await f.read()
                        prompts = json.loads(content)
                    else:
                        # aiofiles不可用，回退到同步方式
                        logger.debug("aiofiles不可用，使用同步文件读取")
                        with open(prompts_path, encoding="utf-8") as f:
                            prompts = json.load(f)
                    logger.info(f"已加载提示词文件: {prompts_path}")
                    self._apply_prompts(prompts)

                    # 如果配置为空，将 prompts.json 内容写入配置作为默认显示值
                    self._ensure_default_prompts_in_config(prompts)
                else:
                    logger.warning(f"提示词文件不存在: {prompts_path}")
            except Exception as e:
                logger.error(f"初始化提示词失败: {e}")

            # 启动WebUI（如果启用）
            if self.webui_enabled:
                try:
                    self.web_server = WebServer(
                        self, host=self.webui_host, port=self.webui_port
                    )
                    await self.web_server.start()
                except Exception as e:
                    logger.error(f"启动WebUI失败: {e}")
            else:
                logger.info("WebUI 已禁用，跳过启动")

            # 加载索引缓存
            await self._load_index()

            # 统一同步所有配置
            self._sync_all_config()

            # 将关键配置同步到子服务（_sync_all_config 只更新 main 实例属性）
            self._sync_image_processor_from_runtime()

            # 启动独立的后台任务
            # raw目录清理任务
            self.task_scheduler.create_task(
                "raw_cleanup_loop", self._raw_cleanup_loop()
            )
            logger.info("已启动raw目录清理任务，周期: 30分钟")

            # 容量控制任务
            self.task_scheduler.create_task(
                "capacity_control_loop", self._capacity_control_loop()
            )
            logger.info("已启动容量控制任务，周期: 60分钟")

            # 注意：人格注入已改为使用 LLM 钩子（@filter.on_llm_request），
            # 不再需要在 initialize() 中注入人格配置
            logger.info("[Stealer] 插件初始化完成，情绪注入将通过 LLM 请求钩子实现")

        except Exception as e:
            logger.error(f"初始化插件失败: {e}")
            raise

    async def terminate(self):
        """插件销毁生命周期钩子。清理任务。"""
        # 防止重复清理
        if self._terminated:
            return
        self._terminated = True

        # 停止WebUI
        if getattr(self, "web_server", None):
            try:
                await self.web_server.stop()
            except Exception as e:
                logger.error(f"停止WebUI失败: {e}")

        logger.info("[Stealer] 开始清理插件资源...")

        # 停止后台任务（每个任务独立 try-except）
        try:
            await self.task_scheduler.cancel_task("raw_cleanup_loop")
        except Exception as e:
            logger.error(f"取消 raw_cleanup_loop 任务失败: {e}")

        try:
            await self.task_scheduler.cancel_task("capacity_control_loop")
        except Exception as e:
            logger.error(f"取消 capacity_control_loop 任务失败: {e}")

        # 清理各服务资源（每个服务独立 try-except）
        if self.cache_service:
            try:
                await self.cache_service.cleanup()
            except Exception as e:
                logger.error(f"清理缓存服务失败: {e}")

        if self.task_scheduler:
            try:
                await self.task_scheduler.cleanup()
            except Exception as e:
                logger.error(f"清理任务调度器失败: {e}")

        if self.image_processor_service:
            try:
                self.image_processor_service.cleanup()
            except Exception as e:
                logger.error(f"清理图片处理服务失败: {e}")

        if self.command_handler:
            try:
                self.command_handler.cleanup()
            except Exception as e:
                logger.error(f"清理命令处理器失败: {e}")

        if self.event_handler:
            try:
                await self.event_handler.cleanup_async()
            except Exception as e:
                logger.error(f"Async event handler cleanup failed: {e}")
            try:
                self.event_handler.cleanup()
            except Exception as e:
                logger.error(f"清理事件处理器失败: {e}")

        logger.info("[Stealer] 插件资源清理完成")

    async def _load_index(self) -> dict[str, Any]:
        """加载索引，优先从数据库加载，必要时从旧 JSON 迁移。

        Returns:
            dict[str, Any]: 索引数据（兼容旧接口的字典格式）
        """
        try:
            idx: dict[str, Any] = {}

            # 优先从数据库加载
            db_count = self.db_service.count_total()

            if db_count > 0:
                logger.debug(f"[DB] 从数据库加载 {db_count} 条索引")
                idx = self.db_service.get_index_cache_readonly()
            elif not self._migration_done:
                # 数据库为空，尝试迁移旧数据
                # 1. 尝试从旧版 JSON 文件迁移
                old_json_path = self.cache_dir / "index_cache.json"
                if old_json_path.exists():
                    migrated = await self.db_service.migrate_from_json(old_json_path)
                    if migrated > 0:
                        self._migration_done = True
                        logger.info(f"[DB] 迁移了 {migrated} 条旧记录到数据库")
                        idx = self.db_service.get_index_cache_readonly()

                # 2. 尝试从其他可能的旧位置迁移
                if not idx:
                    legacy_data = await self.cache_service.migrate_legacy_data(self.base_dir)
                    if legacy_data:
                        await self.db_service.save_index(legacy_data)
                        self._migration_done = True
                        logger.info("[DB] 迁移旧数据到数据库完成")
                        idx = self.db_service.get_index_cache_readonly()

                self._migration_done = True

            # 同步到 cache_service 内存缓存（供 WebUI 等模块使用）
            if idx:
                await self.cache_service.set_cache("index_cache", idx, persist=False)

            return idx

        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def _rebuild_index_from_files(self) -> dict[str, Any]:
        """重建索引，委托给 CacheService 后保存到数据库。"""
        rebuilt = await self.cache_service.rebuild_index_from_files(
            self.base_dir, self.categories_dir
        )
        if rebuilt:
            await self.db_service.save_index(rebuilt)
            await self.cache_service.set_cache("index_cache", rebuilt, persist=False)
        return rebuilt

    async def _save_index(self, idx: dict[str, Any]):
        """保存索引到数据库（智能增量更新）。

        根据传入索引与数据库的差异，选择最优更新策略：
        - 全量替换：删除场景或首次初始化
        - 增量插入：新增场景
        """
        db_count = self.db_service.count_total()
        idx_count = len(idx)

        # 如果数据库为空或有删除操作（条目数减少），使用全量替换
        if db_count == 0 or idx_count < db_count:
            await self.db_service.save_index(idx)
        else:
            # 增量插入：只处理新增的条目
            existing_paths = set(self.db_service.get_all_paths())
            new_entries = [
                {"path": path, **meta}
                for path, meta in idx.items()
                if path not in existing_paths and isinstance(meta, dict)
            ]
            if new_entries:
                await self.db_service.insert_batch(new_entries)
                logger.debug(f"[DB] 增量插入 {len(new_entries)} 条新记录")

        # 同步到 cache_service 内存缓存（供 WebUI 等模块使用）
        await self.cache_service.set_cache("index_cache", idx, persist=False)

    async def _process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        is_platform_emoji: bool = False,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片的方法，包括过滤、分类、存储和索引更新

        Args:
            event: 消息事件对象，可为None
            file_path: 图片文件路径
            is_temp: 是否为临时文件，处理后需要删除
            idx: 可选的索引字典，如果提供则直接使用，否则加载新的
            is_platform_emoji: 是否为平台标记的表情包（用于优化处理）

        Returns:
            (成功与否, 更新后的索引字典)
        """
        try:
            # 添加超时控制，防止长时间阻塞
            success, updated_idx = await asyncio.wait_for(
                self.image_processor_service.process_image(
                    event=event,
                    file_path=file_path,
                    is_temp=is_temp,
                    idx=idx,
                    categories=self.categories,
                    content_filtration=self.content_filtration,
                    backend_tag=self.backend_tag,
                    is_platform_emoji=is_platform_emoji,
                    extra_meta=extra_meta,
                ),
                timeout=self.IMAGE_PROCESSING_TIMEOUT_SECONDS,
            )

            # 如果没有提供索引，我们需要加载完整的索引
            if idx is None and updated_idx is not None:
                # 加载完整索引
                full_idx = await self._load_index()
                # 合并更新
                full_idx.update(updated_idx)
                return success, full_idx

            return success, updated_idx
        except asyncio.TimeoutError:
            logger.warning(f"图片处理超时: {file_path}")
            if is_temp:
                await self._safe_remove_file(file_path)
            return False, idx if idx is not None else {}
        except FileNotFoundError as e:
            logger.error(f"文件不存在错误: {e}")
        except PermissionError as e:
            logger.error(f"权限错误: {e}")
        except OSError as e:
            logger.error(f"文件操作错误: {e}")
        except Exception as e:
            logger.error(f"处理图片失败: {e}", exc_info=True)  # 记录完整堆栈信息

        # 异常情况下的清理和返回
        if is_temp:
            await self._safe_remove_file(file_path)
        # 确保返回有效的索引字典
        return False, idx if idx is not None else {}

    async def _safe_remove_file(self, file_path: str) -> bool:
        """委托给 ImageProcessorService。"""
        return await self.image_processor_service.safe_remove_file(file_path)

    async def _extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。
        委托给 EmojiSelector 类处理
        """
        try:
            return await self.emoji_selector.extract_emotions_from_text(event, text)

        except ValueError as e:
            logger.error(f"情绪提取参数错误: {e}")
            return [], text
        except TypeError as e:
            logger.error(f"情绪提取类型错误: {e}")
            return [], text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}", exc_info=True)
            return [], text

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        event_handler = self._get_event_handler(
            log_message="[Stealer] event_handler 未初始化，跳过消息处理",
            log_level="debug",
        )
        if event_handler is None:
            return

        try:
            # 委托给 EventHandler 类处理
            await event_handler.on_message(event)
        except Exception as e:
            logger.error(f"[Stealer] 处理消息时发生错误: {e}", exc_info=True)

    async def _raw_cleanup_loop(self):
        """raw目录清理循环任务。"""
        # 启动时立即执行一次清理
        try:
            if self.steal_emoji:
                logger.info("启动时执行初始raw目录清理")
                event_handler = self._get_event_handler(
                    log_message="event_handler 未初始化，跳过初始清理"
                )
                if event_handler is None:
                    return
                await event_handler._clean_raw_directory()
                logger.info("初始raw目录清理完成")
        except Exception as e:
            logger.error(f"初始raw目录清理失败: {e}", exc_info=True)

        while True:
            try:
                await asyncio.sleep(self.RAW_CLEANUP_INTERVAL_SECONDS)

                if self.steal_emoji:
                    logger.debug("开始执行raw目录清理任务")

                    event_handler = self._get_event_handler(
                        log_message="event_handler 未初始化，跳过清理任务"
                    )
                    if event_handler is None:
                        continue
                    await event_handler._clean_raw_directory()
                    logger.debug("raw目录清理任务完成")

            except asyncio.CancelledError:
                logger.info("raw目录清理任务已取消")
                break
            except Exception as e:
                logger.error(f"raw目录清理任务发生错误: {e}", exc_info=True)
                # 发生错误后继续循环
                continue

    async def _capacity_control_loop(self):
        """容量控制循环任务。"""
        while True:
            try:
                await asyncio.sleep(self.CAPACITY_CONTROL_INTERVAL_SECONDS)

                if self.steal_emoji:
                    logger.debug("开始执行容量控制任务")

                    # 一次性加载索引，用于无效清理和容量控制
                    image_index = await self._load_index()
                    logger.debug(f"当前索引条目数: {len(image_index)}")

                    # 1) 收集无效路径
                    invalid_paths: list[str] = []
                    for path, info in image_index.items():
                        if not isinstance(info, dict):
                            invalid_paths.append(path)
                            continue
                        actual_path = info.get("path", path)
                        if not os.path.exists(actual_path):
                            invalid_paths.append(path)

                    # 在 updater 内原子地删除无效条目，避免丢失并发新增条目
                    if invalid_paths:
                        removed_count = 0

                        def cleanup_updater(current: dict):
                            nonlocal removed_count
                            for p in invalid_paths:
                                if p in current:
                                    del current[p]
                                    removed_count += 1

                        # 使用 cache_service 更新内存缓存（不持久化到JSON），然后同步到数据库
                        if self.cache_service:
                            updated_idx = await self.cache_service.update_index(cleanup_updater, persist=False)
                            # 同步到数据库
                            await self.db_service.save_index(updated_idx)
                        if removed_count > 0:
                            logger.info(f"已清理 {removed_count} 个无效索引条目")

                    # 2) 容量控制（使用原子更新器避免竞态条件）
                    event_handler = self._get_event_handler(
                        log_message="event_handler 未初始化，跳过容量控制"
                    )
                    if event_handler is None:
                        continue

                    removed_count = 0
                    files_to_delete: list[str] = []

                    def capacity_updater(current: dict):
                        nonlocal removed_count, files_to_delete
                        old_count = len(current)
                        # 直接在 current 上执行容量控制，返回要删除的文件
                        files_to_delete = event_handler._enforce_capacity_sync(current)
                        removed_count = old_count - len(current)

                    # 使用 cache_service 更新内存缓存（不持久化到JSON），然后同步到数据库
                    if self.cache_service:
                        updated_idx = await self.cache_service.update_index(capacity_updater, persist=False)
                        # 同步到数据库
                        await self.db_service.save_index(updated_idx)
                        if removed_count > 0:
                            logger.info(
                                f"容量控制：已删除 {removed_count} 个最旧条目"
                            )
                            # 删除实际文件
                            for file_path in files_to_delete:
                                try:
                                    if os.path.exists(file_path):
                                        await self._safe_remove_file(file_path)
                                except Exception as e:
                                    logger.warning(
                                        f"删除文件失败: {file_path}, {e}"
                                    )

                    logger.debug("容量控制任务完成")

            except asyncio.CancelledError:
                logger.info("容量控制任务已取消")
                break
            except Exception as e:
                logger.error(f"容量控制任务发生错误: {e}", exc_info=True)
                continue

    async def _clean_raw_directory(self) -> int:
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法执行raw目录清理"
        )
        if event_handler is None:
            return 0
        return await event_handler._clean_raw_directory()

    async def _enforce_capacity(self, idx: dict):
        """执行容量控制，删除最旧的图片。"""
        # 委托给 EventHandler 类处理
        event_handler = self._get_event_handler(
            log_message="event_handler 未初始化，无法执行容量控制"
        )
        if event_handler is None:
            return
        await event_handler._enforce_capacity(idx)

    @filter.on_llm_request()
    async def _inject_emotion_instruction(self, event: AstrMessageEvent, request):
        """在 LLM 请求时动态注入情绪选择指令。

            模式切换逻辑：
        - LLM模式（enable_natural_emotion_analysis=True）：不注入提示词，由轻量模型分析
        - 被动标签模式（enable_natural_emotion_analysis=False）：注入提示词，LLM插入标签
        """
        try:
            # 检查是否启用自动发送
            if not self.auto_send:
                logger.debug("[Stealer] 自动发送已禁用，跳过情绪注入")
                return

            # LLM模式：启用自然语言分析时，不注入提示词
            turn_state = self._emoji_turn_state(event)
            if turn_state.is_active_sent():
                logger.debug("[Stealer] 当前轮次已转入主动发表情流程，跳过情绪注入")
                return

            if turn_state.is_auto_claimed():
                logger.debug("[Stealer] 当前轮次已完成自动发表情判定，跳过重复注入")
                return

            if not await self._resolve_auto_emoji_turn_permission(event):
                logger.debug("[Stealer] 当前轮次未触发表情包发送条件，跳过情绪注入")
                return

            if self.enable_natural_emotion_analysis:
                logger.debug(
                    "[Stealer] LLM模式已启用，跳过提示词注入，将使用轻量模型分析"
                )
                return

            # 被动标签模式：注入提示词让LLM插入标签
            logger.debug("[Stealer] 被动标签模式：注入提示词让LLM插入情绪标签")

            # 检查分类列表是否为空
            if not self.categories:
                logger.debug("[Stealer] 分类列表为空，跳过情绪注入")
                return

            # 构建情绪分类字符串
            categories_str = ", ".join(self.categories)

            # 生成情绪选择指令
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

            # 将指令添加到系统提示词
            if hasattr(request, "system_prompt"):
                request.system_prompt = (
                    request.system_prompt or ""
                ) + emotion_instruction
                logger.debug(
                    f"[Stealer] 被动标签模式：已注入情绪选择指令 (categories: {len(self.categories)})"
                )
            else:
                logger.warning("[Stealer] LLM 请求对象没有 system_prompt 属性")

        except Exception as e:
            logger.error(f"[Stealer] 注入情绪选择指令失败: {e}", exc_info=True)

    @filter.on_decorating_result(priority=100)
    async def _prepare_emoji_response(self, event: AstrMessageEvent):
        """清理情绪标签并异步发送表情包（不阻塞回复）"""

        # 首先检查是否为 LLM 回复（过滤命令输出、系统消息等）
        result = event.get_result()
        if result is None:
            return False

        # 只处理 LLM 生成的回复，跳过命令/插件输出
        if not result.is_llm_result():
            logger.debug("[Stealer] 非 LLM 回复，跳过表情包处理")
            return False

        logger.info("[Stealer] _prepare_emoji_response 被调用")

        turn_state = self._emoji_turn_state(event)

        # 检查是否为主动发送（工具已发送表情包）
        if turn_state.is_active_sent():
            # 清理回复中的标签，但不发送表情包
            result = event.get_result()
            if result:
                text = result.get_plain_text() or ""
                if text.strip():
                    # 复用 _extract_emotions_from_text 的清理逻辑
                    _, cleaned_text = await self._extract_emotions_from_text(
                        event, text
                    )
                    if cleaned_text != text:
                        self._update_result_with_cleaned_text_safe(
                            event, result, cleaned_text
                        )
                        logger.debug("[Stealer] 已清理主动发送后的情绪标签")
            return False

        try:
            # 1. 验证结果对象
            result = event.get_result()
            if not self._validate_result(result):
                logger.debug("[Stealer] 结果对象无效，跳过处理")
                return False

            # 2. 提取纯文本
            text = result.get_plain_text() or ""
            if not text.strip():
                logger.debug("[Stealer] 没有可处理的文本内容，跳过")
                return False

            # 3. 检查并处理显式的表情包标记 (来自 Tool 调用)
            turn_allowed = await self._resolve_auto_emoji_turn_permission(event)

            explicit_emojis = []

            def tag_replacer(match):
                explicit_emojis.append(match.group(1))
                return ""

            text_without_explicit = re.sub(r"\[ast_emoji:(.*?)\]", tag_replacer, text)
            has_explicit = len(explicit_emojis) > 0

            # 6. 处理显式表情包（同步处理）
            if has_explicit:
                _, cleaned_text = await self._extract_emotions_from_text(
                    event, text_without_explicit
                )
                if cleaned_text != text:
                    self._update_result_with_cleaned_text_safe(
                        event, result, cleaned_text
                    )

                if not self._claim_auto_emoji_turn(event):
                    logger.debug("[Stealer] Current turn already handled explicit emoji")
                    return cleaned_text != text

                await self._send_explicit_emojis(event, explicit_emojis, cleaned_text)
                logger.info(f"[Stealer] Sent {len(explicit_emojis)} explicit emojis")
                return True


            # 7. 模式判断：LLM模式 vs 被动标签模式
            is_intelligent_mode = self.enable_natural_emotion_analysis

            if is_intelligent_mode:
                # LLM模式：不修改消息链，直接异步分析
                # 提取用户原始消息作为上下文（QA 分析）
                if not turn_allowed:
                    logger.debug("[Stealer] Current turn is not allowed to send auto emoji")
                    return False

                if self._should_skip_auto_emoji_by_gate(text_without_explicit):
                    logger.debug("[Stealer] Skip auto emoji by content gate")
                    return False

                if not self._claim_auto_emoji_turn(event):
                    logger.debug("[Stealer] Current turn already handled auto emoji")
                    return False

                user_query = ""
                try:
                    user_query = event.get_message_str() or ""
                except Exception as e:
                    logger.debug(f"获取用户消息失败: {e}")
                logger.debug("[Stealer] LLM模式：保持消息链不变，异步分析语义")
                self._safe_create_task(
                    self._async_analyze_and_send_emoji(
                        event,
                        text_without_explicit,
                        [],
                        user_query=user_query,
                    ),
                    name="emoji_analyze_intelligent",
                )
                return False  # 不修改消息链
            else:
                # 被动标签模式：提取并清理标签，修改消息链
                logger.debug("[Stealer] 被动标签模式：提取标签并清理消息链")

                # 提取情绪标签
                all_emotions, cleaned_text = await self._extract_emotions_from_text(
                    event, text_without_explicit
                )

                # 判断是否需要更新文本
                need_update = cleaned_text != text_without_explicit

                # 清理标签（修改消息链）
                if need_update:
                    self._update_result_with_cleaned_text_safe(
                        event, result, cleaned_text
                    )
                    logger.debug("[Stealer] 被动标签模式：已清理情绪标签")

                # 异步发送表情包
                if not turn_allowed:
                    logger.debug("[Stealer] Current turn is not allowed to send auto emoji")
                    return need_update

                if not all_emotions:
                    logger.debug("[Stealer] No extracted emotions, skip auto emoji")
                    return need_update

                if self._should_skip_auto_emoji_by_gate(cleaned_text):
                    logger.debug("[Stealer] Skip auto emoji by content gate")
                    return need_update

                if not self._claim_auto_emoji_turn(event):
                    logger.debug("[Stealer] Current turn already handled auto emoji")
                    return need_update

                self._safe_create_task(
                    self._async_analyze_and_send_emoji(
                        event, cleaned_text, all_emotions
                    ),
                    name="emoji_analyze_passive",
                )

                return need_update  # 返回是否修改了消息链

        except Exception as e:
            logger.error(f"[Stealer] 处理表情包响应时发生错误: {e}", exc_info=True)
            return False

    async def _send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ):
        """委托给 EmojiSelector。"""
        await self.emoji_selector.send_explicit_emojis(event, emoji_paths, cleaned_text)

    def _get_auto_emoji_session_key(self, event: AstrMessageEvent) -> str:
        try:
            session_id = event.get_session_id()
            if session_id:
                return str(session_id)
        except Exception:
            pass

        scope, target_id = self.get_event_target(event)
        if scope and target_id:
            return f"{scope}:{target_id}"
        return ""

    def _should_skip_auto_emoji_by_gate(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return True

        if len(cleaned) > 180:
            return True

        if cleaned.count("\n") + 1 >= 6:
            return True

        lowered = cleaned.lower()
        if "```" in cleaned:
            return True

        skip_tokens = (
            "import ",
            "def ",
            "class ",
            "traceback",
            "exception:",
            "error:",
            "warning:",
            "pip install",
            "http://",
            "https://",
            "/meme ",
        )
        if any(token in lowered for token in skip_tokens):
            return True

        punctuation_count = sum(cleaned.count(ch) for ch in (":", ";", "`", "/", "\\"))
        if punctuation_count >= 10:
            return True

        return False

    async def _is_auto_emoji_cooldown_ready(self, event: AstrMessageEvent) -> bool:
        session_key = self._get_auto_emoji_session_key(event)
        if not session_key:
            return True

        now = asyncio.get_running_loop().time()
        async with self._auto_emoji_cooldowns_lock:
            self._prune_auto_emoji_cooldowns(now)
            last_sent_at = self._auto_emoji_cooldowns.get(session_key, 0.0)
        return now - last_sent_at >= self.AUTO_EMOJI_COOLDOWN_SECONDS

    def _normalize_auto_emoji_chance(self) -> float:
        try:
            chance = float(self.emoji_chance)
        except Exception:
            logger.warning("[Stealer] 表情包发送概率配置无效，按 0 处理")
            return 0.0

        if chance <= 0:
            return 0.0
        if chance >= 1:
            return 1.0
        return chance

    async def _resolve_auto_emoji_turn_permission(
        self, event: AstrMessageEvent
    ) -> bool:
        turn_state = self._emoji_turn_state(event)
        if turn_state.is_auto_decided():
            return turn_state.get_auto_allowed()

        allowed = False
        reason = "unknown"

        if not self.auto_send:
            reason = "auto_send_disabled"
        elif not self.is_meme_enabled_for_event(event):
            reason = "meme_disabled"
        elif not await self._is_auto_emoji_cooldown_ready(event):
            reason = "cooldown"
        else:
            chance = self._normalize_auto_emoji_chance()
            if chance <= 0:
                reason = "chance_zero"
            elif chance >= 1.0:
                allowed = True
                reason = "chance_hit"
            elif random.random() < chance:
                allowed = True
                reason = "chance_hit"
            else:
                reason = "chance_miss"

        turn_state.set_auto_decision(allowed=allowed, reason=reason)

        logger.debug(
            f"[Stealer] 当前轮次自动发表情判定: allowed={allowed}, reason={reason}"
        )
        return allowed

    def _claim_auto_emoji_turn(self, event: AstrMessageEvent) -> bool:
        return self._emoji_turn_state(event).claim_auto_send()

    def _prune_auto_emoji_cooldowns(self, now: float) -> None:
        # 1. 过期清理
        expire_after = self.AUTO_EMOJI_COOLDOWN_SECONDS * 3
        expired_keys = [
            key
            for key, timestamp in self._auto_emoji_cooldowns.items()
            if now - timestamp >= expire_after
        ]
        for key in expired_keys:
            self._auto_emoji_cooldowns.pop(key, None)

        # 2. 数量限制：超过上限时删除最旧的一半
        if len(self._auto_emoji_cooldowns) > self._auto_emoji_cooldowns_max:
            sorted_items = sorted(
                self._auto_emoji_cooldowns.items(),
                key=lambda x: x[1]
            )
            to_remove = len(self._auto_emoji_cooldowns) - self._auto_emoji_cooldowns_max // 2
            for key, _ in sorted_items[:to_remove]:
                self._auto_emoji_cooldowns.pop(key, None)
            logger.debug(f"冷却记录数量超限，已清理 {to_remove} 个最旧记录")

    async def _mark_auto_emoji_sent(self, event: AstrMessageEvent) -> None:
        session_key = self._get_auto_emoji_session_key(event)
        if not session_key:
            return
        now = asyncio.get_running_loop().time()
        async with self._auto_emoji_cooldowns_lock:
            self._prune_auto_emoji_cooldowns(now)
            self._auto_emoji_cooldowns[session_key] = now

    async def _try_send_emoji(
        self,
        event: AstrMessageEvent,
        emotions: list[str],
        cleaned_text: str,
    ) -> bool:
        """委托给 EmojiSelector。

        注意：概率判定由 _resolve_auto_emoji_turn_permission 在调用前完成。
        """
        return await self.emoji_selector.try_send_emoji(
            event,
            emotions,
            cleaned_text,
        )

    def _get_emoji_send_delay(self) -> float:
        """根据配置计算表情包发送延迟时间（秒）。

        Returns:
            float: 延迟秒数，0 表示无延迟
        """
        import random

        min_delay = max(0.0, self.emoji_send_delay)

        if not self.emoji_send_delay_random:
            return min_delay

        max_delay = max(min_delay, self.emoji_send_delay_max)
        return min_delay + random.random() * (max_delay - min_delay)

    async def _async_analyze_and_send_emoji(
        self,
        event: AstrMessageEvent,
        text: str,
        emotions: list[str],
        *,
        user_query: str = "",
    ):
        """分析情绪并发送表情包

        Args:
            event: 消息事件
            text: LLM 回复文本内容
            emotions: 已提取的情绪列表（被动模式使用，智能模式忽略）
            user_query: 用户原始消息（智能模式下与 text 组成 QA 上下文）
        """
        try:
            turn_state = self._emoji_turn_state(event)

            # 双重检查：防止多段回复时重复发送表情包
            if turn_state.is_auto_sent():
                logger.debug("[Stealer] 当前轮次已发送过表情包，跳过重复发送")
                return

            # 检查是否启用自动发送
            if not self.auto_send:
                logger.debug("[Stealer] 自动发送已禁用，跳过表情包发送")
                return

            # 检查群聊是否允许
            if not self.is_meme_enabled_for_event(event):
                logger.debug("[Stealer] 当前群聊已禁用表情包功能")
                return

            if not turn_state.get_auto_allowed():
                logger.debug("[Stealer] 当前轮次未触发表情包发送条件，跳过自动发送")
                return

            if self._should_skip_auto_emoji_by_gate(text):
                logger.debug("[Stealer] Skip auto emoji by content gate")
                return

            # cooldown is decided before llm analysis for the whole turn



            # 判断模式
            is_intelligent_mode = self.enable_natural_emotion_analysis

            final_emotions = []

            if is_intelligent_mode:
                # 智能模式：使用轻量模型分析（传入完整对话作为上下文）
                logger.debug("[Stealer] 智能模式：使用轻量模型分析情绪")
                try:
                    analyzed_emotion = (
                        await self.smart_emotion_matcher.analyze_and_match_emotion(
                            event,
                            text,
                            use_natural_analysis=True,
                            user_query=user_query,
                        )
                    )
                    if analyzed_emotion:
                        final_emotions = [analyzed_emotion]
                        logger.info(f"[Stealer] 智能分析结果: {analyzed_emotion}")
                    else:
                        logger.debug("[Stealer] 智能分析未识别出情绪")
                        return
                except Exception as e:
                    logger.error(f"[Stealer] 智能情绪分析失败: {e}", exc_info=True)
                    return
            else:
                # 被动模式：使用已提取的情绪标签
                logger.debug("[Stealer] 被动模式：使用已提取的情绪标签")
                final_emotions = emotions

            # 如果没有情绪，跳过
            if not final_emotions:
                logger.debug("[Stealer] 没有可用的情绪标签，跳过发送")
                return

            # 根据配置应用延迟（避免与分段插件冲突）
            delay_seconds = self._get_emoji_send_delay()
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            # 尝试发送表情包
            sent = await self._try_send_emoji(
                event,
                final_emotions,
                text,
            )
            if sent:
                turn_state.mark_auto_sent()
                await self._mark_auto_emoji_sent(event)

        except Exception as e:
            logger.error(f"[Stealer] 异步发送表情包失败: {e}", exc_info=True)

    def _validate_result(self, result) -> bool:
        """验证结果对象是否有效。"""
        return (
            result is not None
            and hasattr(result, "chain")
            and hasattr(result, "get_plain_text")
        )

    def _update_result_with_cleaned_text_safe(
        self, event: AstrMessageEvent, result, cleaned_text: str
    ):
        """安全更新结果文本，保留其他组件。

        策略：找到所有 Plain 组件，将清理后的文本写入第一个非空 Plain，
        其余 Plain 清空，从而保留非文本组件（图片、引用等）的位置。
        """
        plain_components = [
            comp
            for comp in result.chain
            if isinstance(comp, Plain) and hasattr(comp, "text")
        ]

        if not plain_components:
            logger.debug("[Stealer] 未找到 Plain 组件，添加新的文本组件")
            result.message(cleaned_text)
            return

        # 将清理后的文本写入第一个 Plain，其余 Plain 置空
        first_set = False
        for comp in plain_components:
            if not first_set:
                comp.text = cleaned_text
                first_set = True
                logger.debug(f"[Stealer] 已更新 Plain 组件: {cleaned_text[:50]}...")
            else:
                comp.text = ""

    @filter.command_group("meme")
    def meme(self):
        """指令组占位符（按官方文档：指令组函数无需实现逻辑）。"""
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
        async for result in self.command_handler.set_image_scope(
            event, identifier, scope_mode
        ):
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
            idx = self.db_service.get_index_cache_readonly() if self.db_service.count_total() > 0 else {}
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

        # Mark this turn as an explicit emoji flow and suppress auto-send hooks.
        turn_state = self._emoji_turn_state(event)
        turn_state.mark_active_sent()

        try:
            if not query:
                yield "搜索失败：缺少 query 参数。请传入你当前心情词，例如：开心、无语、尴尬、感谢。"
                return

            if not self.is_meme_enabled_for_event(event):
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

                    scenes_items: list[str] = []
                    if isinstance(raw_scenes, str):
                        scenes_items = [
                            s.strip()
                            for s in re.split(r"[，,、;；]", raw_scenes)
                            if s.strip()
                        ]
                    elif isinstance(raw_scenes, list):
                        scenes_items = [str(s).strip() for s in raw_scenes if str(s).strip()]

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
        # Mark this turn as an explicit emoji flow and suppress auto-send hooks.
        turn_state = self._emoji_turn_state(event)
        turn_state.mark_active_sent()

        try:
            if not self.is_meme_enabled_for_event(event):
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

            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            sent_as_sticker = False
            try:
                sent_as_sticker = await self.emoji_selector._try_send_telegram_sticker(
                    event, path
                )
            except Exception:
                sent_as_sticker = False

            if not sent_as_sticker:
                b64 = await self.image_processor_service._file_to_gif_base64(path)
                await event.send(MessageChain([ImageComponent.fromBase64(b64)]))
            await self.emoji_selector.record_emoji_usage(path, trigger="llm_tool")

            mode_desc = "Telegram贴纸" if sent_as_sticker else "图片"
            success_msg = (
                f"发送成功（{mode_desc}）。\n\n你发送的表情包：\n- 编号：{emoji_id}\n- 分类：{emotion}\n- 描述：{desc}"
            )
            logger.info(f"[Tool] {success_msg}")
            yield success_msg
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield f"发送出错：{e}"
            return
