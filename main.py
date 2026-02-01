import asyncio
import inspect
import json
import math
import os
import random
import re
import shutil
import time
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
from astrbot.core.file_token_service import FileTokenService
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .cache_service import CacheService
from .command_handler import CommandHandler

# 导入原有服务类 - 使用标准的相对导入
from .config_service import ConfigService
from .emotion_analyzer_service import EmotionAnalyzerService
from .event_handler import EventHandler
from .image_processor_service import ImageProcessorService
from .natural_emotion_analyzer import SmartEmotionMatcher
from .task_scheduler import TaskScheduler
from .text_similarity import calculate_hybrid_similarity
from .web_server import WebServer

try:
    import aiofiles  # type: ignore
except ImportError:
    aiofiles = None

# ================= Monkey Patch Start =================
# 修复 AstrBot Token 一次性销毁导致部分客户端无法预览/下载图片的问题
#
# 问题背景：
# 部分客户端（QQ/NapCat、Lagrange）在下载图片前会先发送 HEAD 请求探测文件，
# 导致 Token 被提前消费，后续 GET 请求失败返回 404。
#
# 优化方案：
# 1. 使用访问计数而非全局禁用 pop()，减少对其他功能的影响
# 2. Token 有效期缩短至 60 秒（表情包场景足够使用）
# 3. 仅对插件注册的 Token 启用多次访问保护

# 保存原始方法引用
_original_handle_file = FileTokenService.handle_file
_original_register_file = FileTokenService.register_file

# 插件专用的 Token 标记集合（使用 set 提高查询效率）
_plugin_reusable_tokens = set()


async def patched_register_file(
    self, file_path: str, timeout: float | None = None
) -> str:
    """重写注册方法，为插件的 Token 添加标记"""
    # 调用原始注册方法
    file_token = await _original_register_file(self, file_path, timeout)

    # 检查调用栈，判断是否来自本插件
    frame = inspect.currentframe()
    try:
        # 向上查找调用栈
        caller_frame = frame.f_back
        while caller_frame:
            caller_file = caller_frame.f_code.co_filename
            # 如果调用者是本插件，标记为可重复使用
            if "astrbot_plugin_stealer" in caller_file:
                _plugin_reusable_tokens.add(file_token)
                # 为表情包场景设置更短的超时（60秒足够）
                if timeout is None:
                    # 重新注册，使用短超时
                    async with self.lock:
                        if file_token in self.staged_files:
                            file_path_stored, _ = self.staged_files[file_token]
                            expire_time = time.time() + 60  # 60秒超时
                            self.staged_files[file_token] = (
                                file_path_stored,
                                expire_time,
                            )
                break
            caller_frame = caller_frame.f_back
    finally:
        del frame

    return file_token


async def patched_handle_file(self, file_token: str) -> str:
    """优化的 handle_file，仅对插件 Token 启用多次访问"""
    async with self.lock:
        await self._cleanup_expired_tokens()

        if file_token not in self.staged_files:
            raise KeyError(f"无效或过期的文件 token: {file_token}")

        # 判断是否为插件的可重复使用 Token
        if file_token in _plugin_reusable_tokens:
            # 插件 Token：保留不删除，支持多次访问（HEAD + GET）
            file_path, _ = self.staged_files[file_token]
        else:
            # 其他 Token：保持原有的一次性行为
            file_path, _ = self.staged_files.pop(file_token)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return file_path


# 应用补丁
FileTokenService.register_file = patched_register_file
FileTokenService.handle_file = patched_handle_file


# 清理函数（在插件卸载时调用）
def _cleanup_monkey_patch():
    """恢复原始方法并清理标记集合"""
    FileTokenService.handle_file = _original_handle_file
    FileTokenService.register_file = _original_register_file
    _plugin_reusable_tokens.clear()


# ================= Monkey Patch End =================


class Main(Star):
    """表情包偷取与发送插件。

    功能：
    - 监听消息中的图片并自动保存到插件数据目录
    - 使用当前会话的多模态模型进行情绪分类与标签生成
    - 建立分类索引，支持自动与手动在合适时机发送表情包
    """

    # 常量定义
    BACKEND_TAG = "emoji_stealer"

    # 提示词常量
    IMAGE_FILTER_PROMPT = (
        "根据以下审核准则判断图片是否符合: {filtration_rule}。只返回是或否。"
    )
    TEXT_EMOTION_PROMPT_TEMPLATE = """请基于这段文本的情绪选择一个最匹配的类别: {categories}。
请使用&&emotion&&格式返回，例如&&happy&&、&&sad&&。
只返回表情标签，不要添加任何其他内容。文本: {text}"""

    # 从外部文件加载的提示词（已迁移到ImageProcessorService）

    # 缓存相关常量和方法已迁移到CacheService类

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)

        # 初始化基础路径 - 遵循 AstrBot 插件存储大文件规范
        # 大文件应存储于 data/plugin_data/{plugin_name}/ 目录下
        # self.name 在 v4.9.2 及以上版本可用
        plugin_name = getattr(self, "name", "astrbot_plugin_stealer")
        self.base_dir: Path = (
            Path(get_astrbot_data_path()) / "plugin_data" / plugin_name
        )

        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir: Path = self.base_dir / "raw"
        self.categories_dir: Path = self.base_dir / "categories"
        self.cache_dir: Path = self.base_dir / "cache"

        # 情绪选择标记（用于识别注入的内容）
        self._persona_marker = "<!-- STEALER_PLUGIN_EMOTION_MARKER_v3 -->"  # 更新版本号

        # 初始化服务类
        self.config_service = ConfigService(
            base_dir=self.base_dir, astrbot_config=config
        )
        self.config_service.initialize()

        # 从配置服务获取初始配置
        self.auto_send = self.config_service.auto_send
        self.emoji_chance = self.config_service.emoji_chance
        self.max_reg_num = self.config_service.max_reg_num
        self.do_replace = self.config_service.do_replace
        self.steal_emoji = self.config_service.steal_emoji
        self.content_filtration = self.config_service.content_filtration

        # 同步所有配置
        self._sync_all_config()

        # 创建必要的目录
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.categories_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        for category in self.categories:
            (self.categories_dir / category).mkdir(parents=True, exist_ok=True)

        # 初始化核心服务类
        self.cache_service = CacheService(self.cache_dir)
        self.command_handler = CommandHandler(self)
        self.web_server = None

        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.emotion_analyzer_service = EmotionAnalyzerService(self)
        self.task_scheduler = TaskScheduler()

        # 初始化自然语言情绪分析器（新增）
        self.smart_emotion_matcher = SmartEmotionMatcher(self)

        # 运行时属性
        self.backend_tag: str = self.BACKEND_TAG
        self._scanner_task: asyncio.Task | None = None
        self._migration_done: bool = False  # 迁移只执行一次
        self._force_capture_windows: dict[str, dict[str, object]] = {}

        # 验证配置
        self._validate_config()

    def _load_vision_provider_id(self) -> str | None:
        """加载视觉模型提供商ID。

        Returns:
            str | None: 视觉模型提供商ID，如果未配置则返回None
        """
        provider_id = self.config_service.get_config("vision_provider_id")
        return str(provider_id) if provider_id else None

    def _sync_all_config(self) -> None:
        """从配置服务同步所有配置到实例属性。

        统一的配置同步方法，避免重复代码。
        """
        # 同步基础配置
        self.auto_send = self.config_service.auto_send
        self.emoji_chance = self.config_service.emoji_chance
        self.smart_emoji_selection = self.config_service.smart_emoji_selection
        self.send_emoji_as_gif = self.config_service.send_emoji_as_gif
        self.max_reg_num = self.config_service.max_reg_num
        self.do_replace = self.config_service.do_replace
        self.raw_cleanup_interval = self.config_service.raw_cleanup_interval
        self.capacity_control_interval = self.config_service.capacity_control_interval
        self.enable_raw_cleanup = self.config_service.enable_raw_cleanup
        self.enable_capacity_control = self.config_service.enable_capacity_control
        self.steal_emoji = self.config_service.steal_emoji
        self.content_filtration = self.config_service.content_filtration
        self.raw_retention_minutes = self.config_service.raw_retention_minutes
        self.categories = self.config_service.categories

        # 同步模型相关配置
        self.vision_provider_id = self._load_vision_provider_id()
        self.enable_natural_emotion_analysis = (
            self.config_service.enable_natural_emotion_analysis
        )
        self.emotion_analysis_provider_id = (
            self.config_service.emotion_analysis_provider_id
        )

        # 同步图片处理节流配置
        self.image_processing_mode = self.config_service.image_processing_mode
        self.image_processing_probability = (
            self.config_service.image_processing_probability
        )
        self.image_processing_interval = self.config_service.image_processing_interval
        self.image_processing_cooldown = self.config_service.image_processing_cooldown

        # 同步 WebUI 配置
        self.webui_enabled = self.config_service.webui_enabled
        self.webui_host = self.config_service.webui_host
        self.webui_port = self.config_service.webui_port

    def _auto_merge_existing_categories(self) -> None:
        current = list(getattr(self.config_service, "categories", []) or [])
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
        except Exception:
            pass

        try:
            index = self.cache_service.get_cache("index_cache") or {}
            for meta in index.values():
                if not isinstance(meta, dict):
                    continue
                cat = str(meta.get("category", "")).strip()
                if not cat or cat == "unknown":
                    continue
                discovered.add(cat)
        except Exception:
            pass

        to_add = sorted(discovered - current_set)
        if not to_add:
            return

        merged_categories = current + to_add
        self._update_config_from_dict({"categories": merged_categories})

        for category in to_add:
            try:
                (self.categories_dir / category).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # 同步视觉模型配置
        self.vision_provider_id = self._load_vision_provider_id()

        # 同步自然语言情绪分析配置（新增）
        self.enable_natural_emotion_analysis = (
            self.config_service.enable_natural_emotion_analysis
        )
        self.emotion_analysis_provider_id = (
            self.config_service.emotion_analysis_provider_id
        )

    def _validate_config(self) -> bool:
        """验证配置参数的有效性。

        Returns:
            bool: 配置是否有效（修复后的配置也算有效）
        """
        errors = []
        fixed = []

        # 验证最大表情数量
        if not isinstance(self.max_reg_num, int) or self.max_reg_num <= 0:
            errors.append("最大表情数量必须大于0的整数")
            self.max_reg_num = 100
            fixed.append("最大表情数量已重置为100")

        # 验证表情发送概率
        if not isinstance(self.emoji_chance, (int, float)) or not (
            0 <= self.emoji_chance <= 1
        ):
            errors.append("表情发送概率必须在0-1之间")
            self.emoji_chance = 0.4
            fixed.append("表情发送概率已重置为0.4")

        # 验证清理周期
        if (
            not isinstance(self.raw_cleanup_interval, int)
            or self.raw_cleanup_interval < 1
        ):
            errors.append("raw清理周期必须至少为1分钟")
            self.raw_cleanup_interval = 30
            fixed.append("raw清理周期已重置为30分钟")

        # 验证容量控制周期
        if (
            not isinstance(self.capacity_control_interval, int)
            or self.capacity_control_interval < 1
        ):
            errors.append("容量控制周期必须至少为1分钟")
            self.capacity_control_interval = 60
            fixed.append("容量控制周期已重置为60分钟")

        # 验证保留期限（如果存在）
        if hasattr(self, "raw_retention_minutes") and (
            not isinstance(self.raw_retention_minutes, int)
            or self.raw_retention_minutes < 1
        ):
            errors.append("raw目录保留期限必须至少为1分钟")
            self.raw_retention_minutes = 60
            fixed.append("raw目录保留期限已重置为60分钟")

        # 记录问题和修复
        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
        if fixed:
            logger.info(f"配置已自动修复: {'; '.join(fixed)}")

        return True  # 即使有问题也返回True，因为已经修复

    def _get_force_capture_key(self, event: AstrMessageEvent) -> str:
        if hasattr(event, "get_session_id"):
            try:
                session_id = event.get_session_id()
                if session_id:
                    return str(session_id)
            except Exception:
                pass

        if hasattr(event, "unified_msg_origin"):
            try:
                return str(event.unified_msg_origin)
            except Exception:
                pass

        return "global"

    def _get_force_capture_sender_id(self, event: AstrMessageEvent) -> str | None:
        for attr in ("sender_id", "user_id"):
            value = getattr(event, attr, None)
            if value:
                return str(value)

        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            for attr in ("sender_id", "user_id"):
                value = getattr(message_obj, attr, None)
                if value:
                    return str(value)

        return None

    def _get_group_id(self, event: AstrMessageEvent) -> str | None:
        try:
            if hasattr(event, "get_group_id"):
                gid = event.get_group_id()
                if gid:
                    return str(gid)
        except Exception:
            pass

        message_obj = getattr(event, "message_obj", None)
        if message_obj is not None:
            try:
                gid = getattr(message_obj, "group_id", "") or ""
                gid = str(gid).strip()
                return gid or None
            except Exception:
                return None
        return None

    def is_meme_enabled_for_event(self, event: AstrMessageEvent) -> bool:
        group_id = self._get_group_id(event)
        config_service = getattr(self, "config_service", None)
        if config_service is None:
            return True
        try:
            return bool(config_service.is_group_allowed(group_id))
        except Exception:
            return True

    def begin_force_capture(self, event: AstrMessageEvent, seconds: int) -> None:
        key = self._get_force_capture_key(event)
        sender_id = self._get_force_capture_sender_id(event)
        until = time.time() + max(1, int(seconds))
        self._force_capture_windows[key] = {"until": until, "sender_id": sender_id}

    def get_force_capture_entry(
        self, event: AstrMessageEvent
    ) -> dict[str, object] | None:
        key = self._get_force_capture_key(event)
        entry = self._force_capture_windows.get(key)
        if not entry:
            return None

        try:
            until = float(entry.get("until", 0))
        except Exception:
            self._force_capture_windows.pop(key, None)
            return None

        if time.time() > until:
            self._force_capture_windows.pop(key, None)
            return None

        expected_sender_id = entry.get("sender_id")
        if expected_sender_id:
            current_sender_id = self._get_force_capture_sender_id(event)
            if current_sender_id and str(current_sender_id) != str(expected_sender_id):
                return None

        return entry

    def consume_force_capture(self, event: AstrMessageEvent) -> None:
        key = self._get_force_capture_key(event)
        self._force_capture_windows.pop(key, None)

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            # 使用配置服务更新配置
            if self.config_service:
                # 记录旧的 WebUI 配置，用于判断是否需要重启 Web Server
                old_webui_enabled = getattr(self, "webui_enabled", True)
                old_webui_host = getattr(self, "webui_host", "0.0.0.0")
                old_webui_port = getattr(self, "webui_port", 8899)

                old_enable_raw_cleanup = getattr(self, "enable_raw_cleanup", True)
                old_raw_cleanup_interval = getattr(self, "raw_cleanup_interval", 30)
                old_enable_capacity_control = getattr(
                    self, "enable_capacity_control", True
                )
                old_capacity_control_interval = getattr(
                    self, "capacity_control_interval", 60
                )

                self.config_service.update_config_from_dict(config_dict)

                # 统一同步所有配置
                self._sync_all_config()

                try:
                    old_raw_cleanup_interval_i = int(old_raw_cleanup_interval)
                except Exception:
                    old_raw_cleanup_interval_i = int(
                        getattr(self, "raw_cleanup_interval", 30)
                    )

                try:
                    old_capacity_control_interval_i = int(old_capacity_control_interval)
                except Exception:
                    old_capacity_control_interval_i = int(
                        getattr(self, "capacity_control_interval", 60)
                    )

                background_task_toggle_changed = (
                    old_enable_raw_cleanup != self.enable_raw_cleanup
                    or old_enable_capacity_control != self.enable_capacity_control
                )
                background_task_interval_changed = old_raw_cleanup_interval_i != int(
                    self.raw_cleanup_interval
                ) or old_capacity_control_interval_i != int(
                    self.capacity_control_interval
                )
                if background_task_toggle_changed or background_task_interval_changed:

                    async def reconcile_background_tasks():
                        if not getattr(self, "task_scheduler", None):
                            return

                        raw_running = self.task_scheduler.is_task_running(
                            "raw_cleanup_loop"
                        )
                        if not self.enable_raw_cleanup:
                            if raw_running:
                                await self.task_scheduler.cancel_task(
                                    "raw_cleanup_loop"
                                )
                        else:
                            if raw_running and old_raw_cleanup_interval_i != int(
                                self.raw_cleanup_interval
                            ):
                                await self.task_scheduler.cancel_task(
                                    "raw_cleanup_loop"
                                )
                                raw_running = False
                            if not raw_running:
                                self.task_scheduler.create_task(
                                    "raw_cleanup_loop", self._raw_cleanup_loop()
                                )

                        capacity_running = self.task_scheduler.is_task_running(
                            "capacity_control_loop"
                        )
                        if not self.enable_capacity_control:
                            if capacity_running:
                                await self.task_scheduler.cancel_task(
                                    "capacity_control_loop"
                                )
                        else:
                            if (
                                capacity_running
                                and old_capacity_control_interval_i
                                != int(self.capacity_control_interval)
                            ):
                                await self.task_scheduler.cancel_task(
                                    "capacity_control_loop"
                                )
                                capacity_running = False
                            if not capacity_running:
                                self.task_scheduler.create_task(
                                    "capacity_control_loop",
                                    self._capacity_control_loop(),
                                )

                    asyncio.create_task(reconcile_background_tasks())

                # 检查 WebUI 配置是否变化并重启
                # 注意：on_config_update 可能是同步调用，重启操作涉及IO，使用 create_task 异步执行
                if (
                    old_webui_enabled != self.webui_enabled
                    or old_webui_host != self.webui_host
                    or old_webui_port != self.webui_port
                ):

                    async def restart_webui():
                        logger.info("检测到 WebUI 配置变更，正在重启 WebUI...")
                        if self.web_server:
                            await self.web_server.stop()
                            self.web_server = None

                        if self.webui_enabled:
                            try:
                                self.web_server = WebServer(
                                    self, host=self.webui_host, port=self.webui_port
                                )
                                await self.web_server.start()
                            except Exception as e:
                                logger.error(f"重启 WebUI 失败: {e}")

                    asyncio.create_task(restart_webui())

                # 更新其他服务的配置
                self.image_processor_service.update_config(
                    categories=self.categories,
                    content_filtration=self.content_filtration,
                    vision_provider_id=self.vision_provider_id,
                    emoji_classification_prompt=getattr(
                        self, "EMOJI_CLASSIFICATION_PROMPT", None
                    ),
                    combined_analysis_prompt=getattr(
                        self, "COMBINED_ANALYSIS_PROMPT", None
                    ),
                )

                self.emotion_analyzer_service.update_config(categories=self.categories)

                # 为新增的分类创建对应的目录
                try:
                    for category in self.categories:
                        category_path = self.categories_dir / category
                        if not category_path.exists():
                            category_path.mkdir(parents=True, exist_ok=True)
                            logger.info(f"[Config] 已创建新分类目录: {category}")
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
        """
        try:
            # 创建必要的数据目录结构
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            self.categories_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            try:
                self._auto_merge_existing_categories()
            except Exception:
                pass

            # 加载提示词文件
            try:
                # 使用__file__获取当前脚本所在目录，即插件安装目录
                plugin_dir = Path(__file__).parent
                prompts_path = plugin_dir / "prompts.json"
                if prompts_path.exists():
                    # 使用异步文件读取
                    try:
                        if aiofiles:
                            async with aiofiles.open(
                                prompts_path, encoding="utf-8"
                            ) as f:
                                content = await f.read()
                            prompts = json.loads(content)
                            logger.info(f"已加载提示词文件: {prompts_path}")
                            # 将加载的提示词赋值给插件实例属性
                            for key, value in prompts.items():
                                setattr(self, key, value)
                            # 更新图片处理器的提示词
                            self.image_processor_service.update_config(
                                emoji_classification_prompt=prompts.get(
                                    "EMOJI_CLASSIFICATION_PROMPT", None
                                ),
                                combined_analysis_prompt=prompts.get(
                                    "COMBINED_ANALYSIS_PROMPT", None
                                ),
                                emoji_classification_with_filter_prompt=prompts.get(
                                    "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
                                ),
                            )
                    except ImportError:
                        # 如果aiofiles不可用，回退到同步方式
                        logger.debug("aiofiles不可用，使用同步文件读取")
                        with open(prompts_path, encoding="utf-8") as f:
                            prompts = json.load(f)
                            logger.info(f"已加载提示词文件: {prompts_path}")
                            # 将加载的提示词赋值给插件实例属性
                            for key, value in prompts.items():
                                setattr(self, key, value)
                            # 更新图片处理器的提示词
                            self.image_processor_service.update_config(
                                emoji_classification_prompt=prompts.get(
                                    "EMOJI_CLASSIFICATION_PROMPT", None
                                ),
                                combined_analysis_prompt=prompts.get(
                                    "COMBINED_ANALYSIS_PROMPT", None
                                ),
                                emoji_classification_with_filter_prompt=prompts.get(
                                    "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
                                ),
                            )
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
            self._load_index()

            # 统一同步所有配置
            self._sync_all_config()

            # 初始化子目录
            for category in self.categories:
                (self.categories_dir / category).mkdir(parents=True, exist_ok=True)

            # 启动独立的后台任务
            # raw目录清理任务
            if self.enable_raw_cleanup:
                self.task_scheduler.create_task(
                    "raw_cleanup_loop", self._raw_cleanup_loop()
                )
                logger.info(
                    f"已启动raw目录清理任务，周期: {self.raw_cleanup_interval}分钟"
                )

            # 容量控制任务
            if self.enable_capacity_control:
                self.task_scheduler.create_task(
                    "capacity_control_loop", self._capacity_control_loop()
                )
                logger.info(
                    f"已启动容量控制任务，周期: {self.capacity_control_interval}分钟"
                )

            # 注意：人格注入已改为使用 LLM 钩子（@filter.on_llm_request），
            # 不再需要在 initialize() 中注入人格配置
            logger.info("[Stealer] 插件初始化完成，情绪注入将通过 LLM 请求钩子实现")

        except Exception as e:
            logger.error(f"初始化插件失败: {e}")
            raise

    async def terminate(self):
        """插件销毁生命周期钩子。清理任务。"""

        # 清理 Monkey Patch（优先执行，避免影响其他清理流程）
        try:
            _cleanup_monkey_patch()
            logger.info("已恢复 FileTokenService 原始行为")
        except Exception as e:
            logger.error(f"清理 Monkey Patch 失败: {e}")

        # 停止WebUI
        if getattr(self, "web_server", None):
            try:
                await self.web_server.stop()
            except Exception as e:
                logger.error(f"停止WebUI失败: {e}")

        try:
            # 注意：由于改用 LLM 钩子注入，不再需要清理人格配置
            # LLM 钩子会在每次请求时独立注入，插件卸载后自动失效
            logger.info("[Stealer] 使用 LLM 钩子注入，无需清理人格配置")

            # 使用任务调度器停止所有后台任务
            await self.task_scheduler.cancel_task("raw_cleanup_loop")
            await self.task_scheduler.cancel_task("capacity_control_loop")

            # 清理各服务资源
            if hasattr(self, "cache_service") and self.cache_service:
                self.cache_service.cleanup()

            if hasattr(self, "task_scheduler") and self.task_scheduler:
                await self.task_scheduler.cleanup()

            if hasattr(self, "config_service") and self.config_service:
                self.config_service.cleanup()

            if (
                hasattr(self, "image_processor_service")
                and self.image_processor_service
            ):
                # ImageProcessorService没有cleanup方法，但可以清理缓存
                if hasattr(self.image_processor_service, "_image_cache"):
                    self.image_processor_service._image_cache.clear()

            if (
                hasattr(self, "emotion_analyzer_service")
                and self.emotion_analyzer_service
            ):
                self.emotion_analyzer_service.cleanup()

            if hasattr(self, "command_handler") and self.command_handler:
                # CommandHandler没有cleanup方法，清理引用即可
                self.command_handler = None

            if hasattr(self, "event_handler") and self.event_handler:
                # EventHandler没有cleanup方法，清理引用即可
                self.event_handler = None

        except Exception as e:
            logger.error(f"终止插件失败: {e}")

        return

    async def _load_index(self) -> dict[str, Any]:
        """加载分类索引文件。

        Returns:
            Dict[str, Any]: 键为文件路径，值为包含 category 与 tags 的字典。
        """
        try:
            cache_data = self.cache_service.get_cache("index_cache")

            logger.debug(
                f"[_load_index] raw cache type: {type(cache_data)}, keys: {list(cache_data.keys())[:5] if cache_data else 'empty'}"
            )

            index_data = dict(cache_data) if cache_data else {}

            logger.debug(f"[_load_index] converted to dict, {len(index_data)} items")

            if not index_data and not self._migration_done:
                logger.debug("[_load_index] cache empty, attempting migration...")
                index_data = await self._migrate_legacy_data()
                self._migration_done = True
                logger.debug(
                    f"[_load_index] migration returned {len(index_data)} items"
                )
                if index_data:
                    self.cache_service.set_cache(
                        "index_cache", index_data, persist=True
                    )

            return index_data
        except OSError as e:
            logger.error(f"索引文件IO错误: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"索引文件格式错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def _migrate_legacy_data(self) -> dict[str, Any]:
        """迁移旧版本数据到新版本。

        Returns:
            Dict[str, Any]: 迁移后的索引数据
        """
        try:
            logger.info("开始检查和迁移旧版本数据...")

            # 可能的旧版本数据路径
            possible_paths = [
                # 旧版本可能使用的路径
                self.base_dir / "index.json",
                self.base_dir / "image_index.json",
                self.base_dir / "cache" / "index.json",
                # 其他可能的路径
                Path("data/plugin_data/astrbot_plugin_stealer/index.json"),
                Path("data/plugin_data/astrbot_plugin_stealer/image_index.json"),
            ]

            migrated_data = {}

            for old_path in possible_paths:
                if old_path.exists():
                    try:
                        logger.info(f"发现旧版本索引文件: {old_path}")
                        with open(old_path, encoding="utf-8") as f:
                            old_data = json.load(f)

                        if isinstance(old_data, dict) and old_data:
                            logger.info(
                                f"从 {old_path} 加载了 {len(old_data)} 条旧记录"
                            )
                            migrated_data.update(old_data)

                            # 备份旧文件
                            backup_path = old_path.with_suffix(".json.backup")
                            shutil.copy2(old_path, backup_path)
                            logger.info(f"已备份旧索引文件到: {backup_path}")

                    except Exception as e:
                        logger.error(f"迁移文件 {old_path} 失败: {e}")
                        continue

            # 如果没有找到任何旧数据，直接返回
            if not migrated_data:
                logger.info("未发现需要迁移的旧版本数据文件")
                return {}

            # --- 智能合并逻辑 ---
            # 加载当前索引
            try:
                current_index = dict(self.cache_service.get_cache("index_cache") or {})
            except Exception:
                current_index = {}

            # 建立当前索引的哈希映射
            current_hash_map = {}
            for k, v in current_index.items():
                if isinstance(v, dict) and v.get("hash"):
                    current_hash_map[v["hash"]] = k  # hash -> path

            merged_count = 0

            # 遍历旧数据，尝试合并到当前索引
            for old_path, old_info in migrated_data.items():
                if not isinstance(old_info, dict):
                    continue

                target_path = None

                # 1. 路径完全匹配
                if old_path in current_index:
                    target_path = old_path
                # 2. 哈希匹配（处理路径变更）
                elif old_info.get("hash") in current_hash_map:
                    target_path = current_hash_map[old_info["hash"]]

                # 如果找到了对应的目标记录，且旧数据有描述/标签，保留之
                if target_path:
                    target_info = current_index[target_path]
                    updated = False

                    if old_info.get("desc") and not target_info.get("desc"):
                        target_info["desc"] = old_info["desc"]
                        updated = True

                    if old_info.get("tags") and not target_info.get("tags"):
                        target_info["tags"] = old_info["tags"]
                        updated = True

                    if updated:
                        merged_count += 1

            # 保存合并后的索引
            if merged_count > 0:
                logger.info(f"成功从旧数据中恢复了 {merged_count} 条记录的元数据")
                await self._save_index(current_index)
            else:
                logger.info("旧数据已加载，但没有新的元数据需要合并")

            return migrated_data

        except Exception as e:
            logger.error(f"数据迁移失败: {e}", exc_info=True)
            return {}

    async def _rebuild_index_from_files(self) -> dict[str, Any]:
        """从现有的分类文件重建索引。

        Returns:
            Dict[str, Any]: 重建的索引数据
        """
        try:
            rebuilt_index = {}

            if not self.categories_dir.exists():
                return rebuilt_index

            # 遍历所有分类目录
            for category_dir in self.categories_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name
                logger.info(f"重建分类 '{category_name}' 的索引...")

                # 遍历分类目录中的图片文件
                for img_file in category_dir.iterdir():
                    if not img_file.is_file():
                        continue

                    # 检查是否是图片文件
                    if img_file.suffix.lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".webp",
                    ]:
                        continue

                    # 尝试找到对应的raw文件
                    raw_path = None
                    if self.raw_dir.exists():
                        # 查找同名文件
                        potential_raw = self.raw_dir / img_file.name
                        if potential_raw.exists():
                            raw_path = str(potential_raw)
                        else:
                            # 查找相似名称的文件
                            for raw_file in self.raw_dir.iterdir():
                                if (
                                    raw_file.is_file()
                                    and raw_file.stem == img_file.stem
                                ):
                                    raw_path = str(raw_file)
                                    break

                    # 如果没找到raw文件，使用categories中的文件路径
                    if not raw_path:
                        raw_path = str(img_file)

                    # 计算文件哈希
                    try:
                        file_hash = await self.image_processor_service._compute_hash(
                            str(img_file)
                        )
                    except Exception as e:
                        logger.debug(f"计算文件哈希失败: {e}")
                        file_hash = ""

                    # 创建索引记录
                    rebuilt_index[raw_path] = {
                        "hash": file_hash,
                        "category": category_name,
                        "created_at": int(img_file.stat().st_mtime),
                        "migrated": True,  # 标记为迁移数据
                    }

            logger.info(f"从文件重建了 {len(rebuilt_index)} 条索引记录")
            return rebuilt_index

        except Exception as e:
            logger.error(f"从文件重建索引失败: {e}", exc_info=True)
            return {}

    async def _save_index(self, idx: dict[str, Any]):
        """保存分类索引文件。"""
        try:
            # 使用缓存服务保存索引
            self.cache_service.set_cache("index_cache", idx)
        except OSError as e:
            logger.error(f"索引文件IO错误: {e}")
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}", exc_info=True)

    async def _load_aliases(self) -> dict[str, str]:
        """加载分类别名文件。

        Returns:
            Dict[str, str]: 别名映射字典。
        """
        try:
            return self.config_service.get_aliases()
        except OSError as e:
            logger.error(f"别名文件IO错误: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"别名文件格式错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"加载别名失败: {e}", exc_info=True)
            return {}

    async def _save_aliases(self, aliases: dict[str, str]):
        """保存分类别名文件。"""
        try:
            self.config_service.update_aliases(aliases)
        except OSError as e:
            logger.error(f"别名文件IO错误: {e}")
        except Exception as e:
            logger.error(f"保存别名文件失败: {e}", exc_info=True)

        # _normalize_category 方法已迁移到 EmotionAnalyzer 类

    async def _process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        is_platform_emoji: bool = False,
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
                ),
                timeout=60,  # 60秒超时
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
            return False, idx
        except FileNotFoundError as e:
            logger.error(f"文件不存在错误: {e}")
        except PermissionError as e:
            logger.error(f"权限错误: {e}")
        except shutil.Error as e:
            logger.error(f"文件操作错误: {e}")
        except Exception as e:
            logger.error(f"处理图片失败: {e}", exc_info=True)  # 记录完整堆栈信息

        # 异常情况下的清理和返回
        if is_temp:
            await self._safe_remove_file(file_path)
        return False, idx

    async def _safe_remove_file(self, file_path: str) -> bool:
        """安全删除文件。

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否删除成功
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"已删除文件: {file_path}")
                return True
            logger.debug(f"文件不存在，无需删除: {file_path}")
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False

    def _is_in_parentheses(self, text: str, index: int) -> bool:
        """判断字符串中指定索引位置是否在括号内。

        支持圆括号()和方括号[]。
        """
        parentheses_count = 0
        bracket_count = 0

        for i in range(index):
            if text[i] == "(":
                parentheses_count += 1
            elif text[i] == ")":
                parentheses_count -= 1
            elif text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1

        return parentheses_count > 0 or bracket_count > 0

    async def _extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词，本地提取不到时使用 LLM。

        委托给 EmotionAnalyzerService 类处理
        """
        try:
            return await self.emotion_analyzer_service.extract_emotions_from_text(
                event, text
            )
        except ValueError as e:
            logger.error(f"情绪提取参数错误: {e}")
            return [], text
        except TypeError as e:
            logger.error(f"情绪提取类型错误: {e}")
            return [], text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}", exc_info=True)
            return [], text

    async def _pick_vision_provider(self, event: AstrMessageEvent | None) -> str | None:
        if self.vision_provider_id:
            return self.vision_provider_id
        if event is None:
            return None
        try:
            return await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        except Exception as e:
            logger.error(f"获取视觉模型提供者失败: {e}")
            return None

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 防护检查：确保 event_handler 已初始化且可用
        if not hasattr(self, "event_handler") or self.event_handler is None:
            logger.debug("[Stealer] event_handler 未初始化，跳过消息处理")
            return

        try:
            # 委托给 EventHandler 类处理
            await self.event_handler.on_message(event)
        except Exception as e:
            logger.error(f"[Stealer] 处理消息时发生错误: {e}", exc_info=True)

    async def _raw_cleanup_loop(self):
        """raw目录清理循环任务。"""
        # 启动时立即执行一次清理
        try:
            if self.steal_emoji and self.enable_raw_cleanup:
                logger.info("启动时执行初始raw目录清理")
                if hasattr(self, "event_handler") and self.event_handler:
                    await self.event_handler._clean_raw_directory()
                    logger.info("初始raw目录清理完成")
                else:
                    logger.warning("event_handler 未初始化，跳过初始清理")
        except Exception as e:
            logger.error(f"初始raw目录清理失败: {e}", exc_info=True)

        while True:
            try:
                # 等待指定的清理周期
                await asyncio.sleep(max(1, int(self.raw_cleanup_interval)) * 60)

                # 只有当偷图功能开启且清理功能启用时才执行
                if self.steal_emoji and self.enable_raw_cleanup:
                    logger.info("开始执行raw目录清理任务")
                    if hasattr(self, "event_handler") and self.event_handler:
                        await self.event_handler._clean_raw_directory()
                        logger.info("raw目录清理任务完成")
                    else:
                        logger.warning("event_handler 未初始化，跳过清理任务")

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
                await asyncio.sleep(max(1, int(self.capacity_control_interval)) * 60)

                if self.steal_emoji and self.enable_capacity_control:
                    logger.info("开始执行容量控制任务")
                    image_index = await self._load_index()

                    logger.info(f"当前索引条目数: {len(image_index)}")

                    removed_invalid = 0
                    valid_paths = []

                    for path, info in list(image_index.items()):
                        if not isinstance(info, dict):
                            logger.debug(f"跳过无效索引条目（非字典）: {path}")
                            del image_index[path]
                            continue

                        actual_path = info.get("path", path)
                        path_exists = os.path.exists(actual_path)

                        if not path_exists:
                            logger.debug(f"文件不存在，将删除索引: {actual_path}")
                            del image_index[path]
                            removed_invalid += 1
                        else:
                            valid_paths.append(path)

                    if removed_invalid > 0:
                        logger.info(f"已清理 {removed_invalid} 个无效索引条目")

                    logger.info(
                        f"清理后索引条目数: {len(image_index)}，有效文件: {len(valid_paths)}"
                    )

                    if hasattr(self, "event_handler") and self.event_handler:
                        await self.event_handler._enforce_capacity(image_index)
                    else:
                        logger.warning("event_handler 未初始化，跳过容量控制")

                    if self.cache_service:
                        self.cache_service.set_cache(
                            "index_cache", image_index, persist=True
                        )

                    logger.info("容量控制任务完成")

            except asyncio.CancelledError:
                logger.info("容量控制任务已取消")
                break
            except Exception as e:
                logger.error(f"容量控制任务发生错误: {e}", exc_info=True)
                continue

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        if hasattr(self, "event_handler") and self.event_handler:
            await self.event_handler._clean_raw_directory()
        else:
            logger.warning("event_handler 未初始化，无法执行raw目录清理")

    async def _enforce_capacity(self, idx: dict):
        """执行容量控制，删除最旧的图片。"""
        # 委托给 EventHandler 类处理
        if hasattr(self, "event_handler") and self.event_handler:
            await self.event_handler._enforce_capacity(idx)
        else:
            logger.warning("event_handler 未初始化，无法执行容量控制")

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
            if getattr(self, "enable_natural_emotion_analysis", True):
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
                logger.info(
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

        # 检查是否为主动发送（工具已发送表情包）
        if event.get_extra("stealer_active_sent"):
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
            explicit_emojis = []

            def tag_replacer(match):
                explicit_emojis.append(match.group(1))
                return ""

            text_without_explicit = re.sub(r"\[ast_emoji:(.*?)\]", tag_replacer, text)
            has_explicit = len(explicit_emojis) > 0

            # 6. 处理显式表情包（同步处理）
            if has_explicit:
                if not self.is_meme_enabled_for_event(event):
                    _, cleaned_text = await self._extract_emotions_from_text(
                        event, text_without_explicit
                    )
                    if cleaned_text != text_without_explicit:
                        self._update_result_with_cleaned_text_safe(
                            event, result, cleaned_text
                        )
                    logger.debug("[Stealer] 群聊已禁用表情包发送，跳过显式表情包发送")
                    return cleaned_text != text_without_explicit
                await self._send_explicit_emojis(
                    event, explicit_emojis, text_without_explicit
                )
                logger.info(f"[Stealer] 已发送 {len(explicit_emojis)} 张显式表情包")
                return True

            # 7. 模式判断：LLM模式 vs 被动标签模式
            is_intelligent_mode = getattr(self, "enable_natural_emotion_analysis", True)

            if is_intelligent_mode:
                # LLM模式：不修改消息链，直接异步分析
                logger.debug("[Stealer] LLM模式：保持消息链不变，异步分析语义")
                asyncio.create_task(
                    self._async_analyze_and_send_emoji(event, text_without_explicit, [])
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
                asyncio.create_task(
                    self._async_analyze_and_send_emoji(
                        event, cleaned_text, all_emotions
                    )
                )

                return need_update  # 返回是否修改了消息链

        except Exception as e:
            logger.error(f"[Stealer] 处理表情包响应时发生错误: {e}", exc_info=True)
            return False

    async def _async_analyze_and_send_emoji(
        self, event: AstrMessageEvent, cleaned_text: str, extracted_emotions: list
    ):
        """后台异步分析情绪并发送表情包"""
        try:
            if not self.is_meme_enabled_for_event(event):
                return

            all_emotions = []

            # 模式切换：LLM模式 vs 被动标签模式
            if getattr(self, "enable_natural_emotion_analysis", True):
                # LLM模式：使用轻量模型分析，忽略标签
                logger.debug("[Stealer] LLM模式：后台使用轻量模型分析LLM回复")

                # 使用智能情绪匹配器分析LLM回复的真实情绪
                analyzed_emotion = (
                    await self.smart_emotion_matcher.analyze_and_match_emotion(
                        event, cleaned_text, use_natural_analysis=True
                    )
                )

                if analyzed_emotion:
                    all_emotions = [analyzed_emotion]
                    logger.debug(
                        f"[Stealer] LLM模式：轻量模型识别情绪 {analyzed_emotion}"
                    )
                else:
                    logger.debug(
                        "[Stealer] LLM模式：轻量模型未识别到情绪，跳过表情包发送"
                    )
                    return
            else:
                # 被动标签模式：依赖LLM插入的标签
                if not extracted_emotions:
                    logger.debug(
                        "[Stealer] 被动标签模式：未提取到LLM插入的情绪标签，跳过表情包发送"
                    )
                    return
                else:
                    all_emotions = extracted_emotions
                    logger.debug(
                        f"[Stealer] 被动标签模式：检测到LLM插入的情绪标签 {all_emotions}"
                    )

            # 尝试发送表情包
            # 注意：在发送前短暂等待，给 tool loop 留出设置 stealer_active_sent 标记的时间
            # 这是因为 LLM 可能在第一轮回复后决定调用 tool，而我们的异步任务可能在 tool 执行前就完成了分析
            await asyncio.sleep(0.5)

            # 再次检查标记（tool 可能在等待期间被调用）
            if event.get_extra("stealer_active_sent"):
                logger.debug("[Stealer] 检测到 tool 已主动发送表情包，跳过自动发送")
                return

            await self._try_send_emoji(event, all_emotions, cleaned_text)

        except Exception as e:
            logger.error(f"[Stealer] 异步情绪分析和表情包发送失败: {e}", exc_info=True)

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
        """安全更新结果文本，保留其他组件"""
        # 查找并更新 Plain 组件
        text_updated = False
        for comp in result.chain:
            if isinstance(comp, Plain) and hasattr(comp, "text"):
                # 只更新包含情绪标签的 Plain 组件
                if comp.text and comp.text.strip():
                    # 简单替换：将组件文本设置为清理后的文本
                    # 这样可以保留其他非文本组件（如图片、引用等）
                    comp.text = cleaned_text
                    text_updated = True
                    logger.debug(f"[Stealer] 已更新 Plain 组件: {cleaned_text[:50]}...")
                    break  # 通常只有一个主文本组件，找到后跳出

        if not text_updated:
            # 如果没有找到 Plain 组件，添加一个新的
            logger.debug("[Stealer] 未找到 Plain 组件，添加新的文本组件")
            result.message(cleaned_text)

    def _update_result_with_cleaned_text(
        self, event: AstrMessageEvent, result, cleaned_text: str
    ):
        """更新结果文本（重建消息链，不推荐使用）"""
        new_result = event.make_result().set_result_content_type(
            result.result_content_type
        )

        # 添加除了Plain文本外的其他组件
        for comp in result.chain:
            if not isinstance(comp, Plain):
                new_result.chain.append(comp)

        # 添加清理后的文本
        if cleaned_text.strip():
            new_result.message(cleaned_text.strip())

        event.set_result(new_result)

    async def _try_send_emoji(
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str
    ) -> bool:
        """尝试发送表情包。"""
        if not self.is_meme_enabled_for_event(event):
            return False

        # 如果本轮已经通过 LLM 工具主动发送过表情包，则跳过自动发送（避免重复）
        # 典型场景：LLM模式下 LLM 先调用 send_emoji_by_id 发送了一张，但后台自然语言分析仍可能触发概率发送。
        if event.get_extra("stealer_active_sent"):
            logger.debug("[Stealer] 检测到 stealer_active_sent=True，跳过自动表情发送")
            return False

        # 1. 检查发送概率
        if not self._check_send_probability():
            return False

        # 2. 智能选择表情包（传入上下文）
        emoji_path = await self._select_emoji(emotions[0], cleaned_text)
        if not emoji_path:
            return False

        # 3. 发送表情包
        await self._send_emoji_with_text(event, emoji_path, cleaned_text)

        logger.debug("已发送表情包")
        return True

    def _check_send_probability(self) -> bool:
        """检查表情包发送概率。

            说明：
            - 被动标签模式（LLM 插入 &&emotion&&）
        - LLM模式（自然语言分析 / 智能 LLM 模式）

            以上两种模式共享同一个概率配置：self.emoji_chance。
        """
        try:
            chance = float(self.emoji_chance)
            if chance <= 0:
                logger.debug("表情包自动发送概率为0，未触发图片发送")
                return False
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return False

            logger.debug("表情包自动发送概率检查通过")
            return True
        except Exception as e:
            logger.error(f"解析表情包自动发送概率配置失败: {e}")
            return False

    async def _select_emoji(self, category: str, context_text: str = "") -> str | None:
        """选择表情包（智能或随机）"""
        # 检查是否启用智能选择
        use_smart = getattr(self, "smart_emoji_selection", True)

        # 如果启用智能选择且提供了上下文，使用智能选择
        if use_smart and context_text and len(context_text.strip()) > 5:
            smart_path = await self._select_emoji_smart(category, context_text)
            if smart_path:
                return smart_path

        # 降级到随机选择（原有逻辑）
        cat_dir = self.base_dir / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在")
            return None

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                logger.debug(f"情绪'{category}'对应的图片目录为空")
                return None

            logger.debug(
                f"从'{category}'目录中找到 {len(files)} 张图片（{'智能选择失败，' if use_smart else ''}随机选择）"
            )

            # 改进的随机选择：避免重复
            recent_usage_key = f"recent_usage_{category}"
            recent_usage = getattr(self, recent_usage_key, [])

            # 过滤最近使用的文件
            available_files = [f for f in files if f.as_posix() not in recent_usage]

            # 如果过滤后没有可用文件，清空历史
            if not available_files:
                available_files = files
                recent_usage.clear()

            # 随机选择
            picked_image = random.choice(available_files)

            # 更新最近使用历史
            picked_path = picked_image.as_posix()
            if picked_path in recent_usage:
                recent_usage.remove(picked_path)
            recent_usage.append(picked_path)

            # 保持历史队列大小
            max_recent = min(5, max(2, len(files) // 2))
            if len(recent_usage) > max_recent:
                recent_usage.pop(0)

            setattr(self, recent_usage_key, recent_usage)

            return picked_path
        except Exception as e:
            logger.error(f"选择表情包失败: {e}")
            return None

    async def _select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """智能选择表情包（多样性+匹配度+文本相似度）"""
        try:
            # 1. 加载索引，获取该分类下的所有表情包
            idx = await self._load_index()
            candidates = []
            current_time = time.time()

            # 获取最近使用历史（避免重复）
            recent_usage_key = f"recent_usage_{category}"
            recent_usage = getattr(self, recent_usage_key, [])

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                # 匹配分类
                file_category = data.get("category", data.get("emotion", ""))
                if file_category != category:
                    continue

                # 检查文件是否存在
                if not os.path.exists(file_path):
                    continue

                candidates.append(
                    {
                        "path": file_path,
                        "data": data,
                        "last_used": data.get("last_used", 0),
                        "use_count": data.get("use_count", 0),
                        "desc": str(data.get("desc", "")).lower(),
                        "tags": [str(t).lower() for t in data.get("tags", [])],
                        "scenes": [str(s).lower() for s in data.get("scenes", [])],
                    }
                )

            if not candidates:
                logger.debug(f"分类 '{category}' 下没有可用的表情包")
                return None

            # 2. 强制避重：过滤最近使用的表情包
            available_candidates = [
                c for c in candidates if c["path"] not in recent_usage
            ]

            # 如果过滤后候选太少，只避免最近3个
            if len(available_candidates) < max(2, len(candidates) * 0.3):
                recent_3 = recent_usage[-3:] if len(recent_usage) >= 3 else recent_usage
                available_candidates = [
                    c for c in candidates if c["path"] not in recent_3
                ]

            # 如果还是没有候选，清空历史重新开始
            if not available_candidates:
                available_candidates = candidates
                recent_usage.clear()

            # 3. 计算多样性分数（权重70%）
            for candidate in available_candidates:
                diversity_score = 100  # 基础分数

                # 时间多样性（使用时间越久分数越高）
                time_since_last_use = current_time - candidate["last_used"]
                if time_since_last_use < 300:  # 5分钟内大幅减分
                    diversity_score -= 60
                elif time_since_last_use < 1800:  # 30分钟内
                    diversity_score -= 30
                elif time_since_last_use < 3600:  # 1小时内
                    diversity_score -= 10
                else:
                    # 超过1小时给予奖励
                    hours_passed = time_since_last_use / 3600
                    diversity_score += min(hours_passed * 5, 30)

                # 频率多样性（使用次数越少分数越高）
                use_count = candidate["use_count"]
                if use_count == 0:
                    diversity_score += 20  # 从未使用过的优先
                elif use_count < 3:
                    diversity_score += 10
                elif use_count < 10:
                    diversity_score += 0
                else:
                    diversity_score -= min(use_count * 2, 30)

                candidate["diversity_score"] = max(diversity_score, 10)

            # 4. 计算匹配分数（权重30%）
            has_context = context_text and len(context_text.strip()) > 5

            for candidate in available_candidates:
                match_score = 10  # 基础匹配分数

                if has_context:
                    context_lower = context_text.lower()

                    # 场景匹配
                    for scene in candidate["scenes"]:
                        if len(scene) > 2 and scene in context_lower:
                            match_score += 25

                    # 描述匹配
                    desc = candidate["desc"]
                    if desc and desc in context_lower:
                        match_score += 20
                    elif desc:
                        desc_words = [w for w in desc.split() if len(w) > 1]
                        matched_words = sum(
                            1 for word in desc_words if word in context_lower
                        )
                        match_score += matched_words * 5

                    # 标签匹配
                    for tag in candidate["tags"]:
                        if len(tag) > 1 and tag in context_lower:
                            match_score += 8

                    # 新增：文本相似度加成（类似MaiBot的编辑距离匹配）
                    # 计算上下文与描述的整体相似度
                    if desc and len(desc) > 3:
                        similarity = calculate_hybrid_similarity(context_text, desc)
                        # 相似度转换为分数加成（最多+15分）
                        similarity_bonus = similarity * 15
                        match_score += similarity_bonus

                        if similarity > 0.3:  # 如果相似度较高，记录日志
                            logger.debug(
                                f"[相似度匹配] '{context_text[:20]}...' vs '{desc[:20]}...' = {similarity:.2f}"
                            )

                candidate["match_score"] = match_score

            # 5. 计算综合分数：多样性70% + 匹配度30%
            max_diversity = max(c["diversity_score"] for c in available_candidates)
            max_match = max(c["match_score"] for c in available_candidates)

            for candidate in available_candidates:
                # 标准化到0-100
                norm_diversity = (candidate["diversity_score"] / max_diversity) * 100
                norm_match = (candidate["match_score"] / max_match) * 100

                # 多样性权重更高
                candidate["final_score"] = norm_diversity * 0.7 + norm_match * 0.3

            # 6. 选择策略：从前40%候选中加权随机选择
            available_candidates.sort(key=lambda x: x["final_score"], reverse=True)
            top_40_percent = max(1, int(len(available_candidates) * 0.4))
            top_candidates = available_candidates[:top_40_percent]

            # 使用指数衰减权重，保持随机性
            weights = [math.exp(-i * 0.3) for i in range(len(top_candidates))]
            selected = random.choices(top_candidates, weights=weights, k=1)[0]

            # 7. 更新使用历史和统计
            selected_path = selected["path"]

            # 更新索引中的使用统计
            idx[selected_path]["last_used"] = int(current_time)
            idx[selected_path]["use_count"] = idx[selected_path].get("use_count", 0) + 1
            await self._save_index(idx)

            # 更新最近使用历史
            if selected_path in recent_usage:
                recent_usage.remove(selected_path)
            recent_usage.append(selected_path)

            # 保持历史队列大小（最多记录8个）
            max_recent = min(8, max(3, len(candidates) // 2))
            if len(recent_usage) > max_recent:
                recent_usage.pop(0)

            setattr(self, recent_usage_key, recent_usage)

            logger.info(
                f"选择表情包: 多样性={selected['diversity_score']:.1f}, "
                f"匹配度={selected['match_score']:.1f}, "
                f"综合={selected['final_score']:.1f}"
            )

            return selected_path

        except Exception as e:
            logger.error(f"智能选择表情包失败: {e}", exc_info=True)
            return None

    async def _send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ):
        """发送表情包（异步场景下直接发送新消息）"""
        try:
            if not self.is_meme_enabled_for_event(event):
                return
            b64 = await self.image_processor_service._file_to_gif_base64(emoji_path)

            # 创建结果并发送
            # 使用 event.send() 方法，传入 MessageChain 对象
            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))

            logger.debug(f"[Stealer] 已发送表情包: {emoji_path}")

        except Exception as e:
            logger.error(f"发送表情包失败: {e}", exc_info=True)

    async def _send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ):
        """发送显式指定的表情包列表和文本。"""
        try:
            # 获取当前结果
            result = event.get_result()

            # 创建新的结果对象
            new_result = event.make_result().set_result_content_type(
                result.result_content_type
            )

            # 添加除了Plain文本外的其他组件
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            # 添加清理后的文本
            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            # 依次添加图片
            for path in emoji_paths:
                try:
                    if os.path.exists(path):
                        b64 = await self.image_processor_service._file_to_gif_base64(
                            path
                        )
                        new_result.base64_image(b64)
                    else:
                        logger.warning(f"显式表情包文件不存在: {path}")
                except Exception as e:
                    logger.error(f"加载显式表情包失败: {path}, {e}")

            # 设置新的结果对象
            event.set_result(new_result)
        except Exception as e:
            logger.error(f"发送显式表情包失败: {e}", exc_info=True)

    @filter.command("meme on")
    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        async for result in self.command_handler.meme_on(event):
            yield result

    @filter.command("meme off")
    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        async for result in self.command_handler.meme_off(event):
            yield result

    @filter.command("meme auto_on")
    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        async for result in self.command_handler.auto_on(event):
            yield result

    @filter.command("meme auto_off")
    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        async for result in self.command_handler.auto_off(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme group")
    async def group_filter(
        self,
        event: AstrMessageEvent,
        list_name: str = "",
        action: str = "",
        group_id: str = "",
    ):
        async for result in self.command_handler.group_filter(
            event, list_name, action, group_id
        ):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme 偷")
    async def capture(self, event: AstrMessageEvent):
        async for result in self.command_handler.capture(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme natural_analysis")
    async def toggle_natural_analysis(self, event: AstrMessageEvent, action: str = ""):
        async for result in self.command_handler.toggle_natural_analysis(event, action):
            yield result

    @filter.command("meme emotion_stats")
    async def emotion_analysis_stats(self, event: AstrMessageEvent):
        """显示情绪分析统计信息。"""
        async for result in self.command_handler.emotion_analysis_stats(event):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme clear_emotion_cache")
    async def clear_emotion_cache(self, event: AstrMessageEvent):
        """清空情绪分析缓存。"""
        async for result in self.command_handler.clear_emotion_cache(event):
            yield result

    @filter.command("meme status")
    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        async for result in self.command_handler.status(event):
            yield result

    @filter.command("meme clean", priority=-100)
    async def clean(self, event: AstrMessageEvent, mode: str = ""):
        """手动清理raw目录中的原始图片文件（不影响已分类的表情包）。

        用法:
        /meme clean - 清理所有raw文件
        /meme clean expired - 只清理过期文件（按保留期限）
        """
        async for result in self.command_handler.clean(event, mode):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme capacity")
    async def enforce_capacity(self, event: AstrMessageEvent):
        """手动执行容量控制，删除最旧的表情包以控制总数量。"""
        async for result in self.command_handler.enforce_capacity(event):
            yield result

    async def get_count(self) -> int:
        idx = await self._load_index()
        return len(idx)

    async def get_info(self) -> dict:
        idx = await self._load_index()
        return {
            "current_count": len(idx),
            "max_count": self.max_reg_num,
            "available_emojis": len(idx),
        }

    async def get_emotions(self) -> list[str]:
        idx = await self._load_index()
        s = set()
        for v in idx.values():
            if isinstance(v, dict):
                emo = v.get("emotion")
                if isinstance(emo, str) and emo:
                    s.add(emo)
        return sorted(s)

    async def get_descriptions(self) -> list[str]:
        image_index = await self._load_index()
        descriptions = []
        for record in image_index.values():
            if isinstance(record, dict):
                description = record.get("desc")
                if isinstance(description, str) and description:
                    descriptions.append(description)
        return descriptions

    async def _load_all_records(self) -> list[tuple[str, dict]]:
        idx = await self._load_index()
        return [
            (k, v) for k, v in idx.items() if isinstance(v, dict) and os.path.exists(k)
        ]

    async def get_random_paths(
        self, count: int | None = 1
    ) -> list[tuple[str, str, str]]:
        all_records = await self._load_all_records()
        if not all_records:
            return []
        sample_count = max(1, int(count or 1))
        picked_records = random.sample(all_records, min(sample_count, len(all_records)))
        results = []
        for image_path, record_dict in picked_records:
            description = str(record_dict.get("desc", ""))
            emotion = str(
                record_dict.get(
                    "emotion",
                    record_dict.get(
                        "category", self.categories[0] if self.categories else "开心"
                    ),
                )
            )
            results.append((image_path, description, emotion))
        return results

    async def get_by_emotion_path(self, emotion: str) -> tuple[str, str, str] | None:
        all_records = await self._load_all_records()
        candidates = []
        for image_path, record_dict in all_records:
            record_emotion = str(
                record_dict.get("emotion", record_dict.get("category", ""))
            )
            record_tags = record_dict.get("tags", [])
            if emotion and (
                emotion == record_emotion
                or (
                    isinstance(record_tags, list)
                    and emotion in [str(tag) for tag in record_tags]
                )
            ):
                candidates.append((image_path, record_dict))
        if not candidates:
            return None
        picked_path, picked_record = random.choice(candidates)
        return (
            picked_path,
            str(picked_record.get("desc", "")),
            str(
                picked_record.get(
                    "emotion",
                    picked_record.get(
                        "category", self.categories[0] if self.categories else "开心"
                    ),
                )
            ),
        )

    async def get_by_description_path(
        self, description: str
    ) -> tuple[str, str, str] | None:
        all_records = await self._load_all_records()
        candidates = []
        for image_path, record_dict in all_records:
            desc_text = str(record_dict.get("desc", ""))
            if description and description in desc_text:
                candidates.append((image_path, record_dict))
        if not candidates:
            for image_path, record_dict in all_records:
                record_tags = record_dict.get("tags", [])
                if isinstance(record_tags, list):
                    if any(str(description) in str(tag) for tag in record_tags):
                        candidates.append((image_path, record_dict))
        if not candidates:
            return None
        picked_path, picked_record = random.choice(candidates)
        return (
            picked_path,
            str(picked_record.get("desc", "")),
            str(
                picked_record.get(
                    "emotion",
                    picked_record.get(
                        "category", self.categories[0] if self.categories else "开心"
                    ),
                )
            ),
        )

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme push")
    async def push(self, event: AstrMessageEvent, category: str = "", alias: str = ""):
        """手动推送指定分类的表情包。"""
        async for result in self.command_handler.push(event, category, alias):
            yield result

    @filter.command("meme list")
    async def list_images(
        self, event: AstrMessageEvent, category: str = "", limit: str = "10"
    ):
        """列出表情包。用法: /meme list [分类] [数量]"""
        async for result in self.command_handler.list_images(event, category, limit):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme delete")
    async def delete_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定表情包。用法: /meme delete <序号|文件名>"""
        async for result in self.command_handler.delete_image(event, identifier):
            yield result

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme rebuild_index")
    async def rebuild_index(self, event: AstrMessageEvent):
        """重建索引，用于迁移旧版本或修复索引。用法: /meme rebuild_index"""
        async for result in self.command_handler.rebuild_index(event):
            yield result

    async def _search_emoji_candidates(
        self,
        query: str,
        *,
        limit: int = 5,
        idx: dict | None = None,
    ):
        """统一的表情包搜索逻辑（给 LLM 工具复用）。

        委托给 ImageProcessorService.smart_search 实现。

        Returns:
            list[tuple[path, desc, emotion]]
        """
        if idx is None:
            idx = self.cache_service.get_cache("index_cache") or {}

        return await self.image_processor_service.smart_search(
            query, limit=limit, idx=idx
        )

    @filter.llm_tool(name="search_emoji")
    async def search_emoji(self, event: AstrMessageEvent, query: str):
        """搜索表情包，返回候选列表供你选择。

        Args:
            query(string): 搜索关键词

        推荐分类词汇：
        - confused: 困惑, 疑问, 不懂, 啥情况
        - dumb: 无语, 尴尬, 呆住
        - happy: 开心, 高兴, 大笑, 兴奋
        - sad: 难过, 伤心, 哭了
        - angry: 生气, 愤怒, 恼火
        - surprised: 惊讶, 震惊, 卧槽
        - troll: 嘲讽, 搞怪, 呵呵, 发癫
        - tired: 累, 瘫倒, 躺平
        - disgust: 嫌弃, 鄙视, 恶心
        - thank: 感谢, 谢谢, 牛逼, 赞
        - cry: 大哭, 泪崩
        - shy: 害羞, 脸红
        - love: 喜欢, 爱了, 么么哒
        - fear: 害怕, 瑟瑟发抖
        - embarrassed: 尴尬, 社死

        返回值：
        - 成功：返回候选表情包列表（包含编号、描述、分类），请选择一个编号调用 send_emoji_by_id 发送
        - 失败：返回错误提示
        """
        logger.info(f"[Tool] LLM 搜索表情包: {query}")

        # 标记为主动发送流程开始，避免自动发送重复触发
        # 注意：必须在 tool 执行最开始就设置，因为 on_decorating_result 可能在 tool loop 中途触发
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.is_meme_enabled_for_event(event):
                yield "搜索失败：当前群聊已禁用表情包功能"
                return

            if not self.cache_service.get_cache("index_cache"):
                logger.debug("索引未加载，正在加载...")
                await self._load_index()

            idx = self.cache_service.get_cache("index_cache") or {}

            results = await self._search_emoji_candidates(query, limit=5, idx=idx)

            # 4. 如果仍然没结果
            if not results:
                logger.warning(f"未找到匹配的表情包: {query}")
                yield f"搜索失败：未找到与'{query}'匹配的表情包。建议尝试：happy, sad, angry, confused, troll 等分类词"
                return

            # 5. 构建候选列表，存入缓存供后续发送
            candidates = []
            result_lines = [f"找到 {len(results)} 个匹配的表情包：\n"]

            for i, (path, desc, emotion) in enumerate(results):
                if os.path.exists(path):
                    candidate_id = f"emoji_{i + 1}"
                    candidates.append(
                        {
                            "id": candidate_id,
                            "path": path,
                            "desc": desc,
                            "emotion": emotion,
                        }
                    )
                    # 截断描述，避免太长
                    short_desc = desc[:50] + "..." if len(desc) > 50 else desc
                    result_lines.append(f"  [{i + 1}] [{emotion}] {short_desc}")

            if not candidates:
                yield "搜索失败：找到的表情包文件均已丢失"
                return

            # 存入实例属性，供 send_emoji_by_id 使用（临时存储，不持久化）
            self._emoji_candidates = candidates

            result_lines.append(
                f"\n请调用 send_emoji_by_id 并传入编号(1-{len(candidates)})来发送你选择的表情包。"
            )

            result_text = "\n".join(result_lines)
            logger.info(f"[Tool] 搜索完成，返回 {len(candidates)} 个候选")
            yield result_text

        except Exception as e:
            logger.error(f"[Tool] 搜索表情包失败: {e}", exc_info=True)
            yield f"搜索出错：{e}"

    @filter.llm_tool(name="send_emoji_by_id")
    async def send_emoji_by_id(self, event: AstrMessageEvent, emoji_id: int):
        """发送指定编号的表情包。必须先调用 search_emoji 获取候选列表。

        Args:
            emoji_id(number): 表情包编号（1-5），从 search_emoji 返回的列表中选择

        返回值：
        - 成功：返回已发送的表情包描述
        - 失败：返回错误提示
        """
        logger.info(f"[Tool] LLM 选择发送表情包编号: {emoji_id}")

        # 标记为主动发送，避免被动标签模式重复触发
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.is_meme_enabled_for_event(event):
                yield "发送失败：当前群聊已禁用表情包功能"
                return

            tool_text: str | None = None
            # LLM 可能会传入 2.0 / "2" 等，统一转为 int 处理
            try:
                emoji_id = int(emoji_id)
            except Exception:
                tool_text = (
                    f"发送失败：编号 {emoji_id} 无法解析为整数，请选择 1-5 之间的编号"
                )
                yield tool_text
                return

            # 从实例属性获取候选列表
            candidates = getattr(self, "_emoji_candidates", None)

            if not candidates:
                tool_text = (
                    "发送失败：没有可用的候选列表，请先调用 search_emoji 搜索表情包"
                )
                yield tool_text
                return

            # 验证编号范围
            if emoji_id < 1 or emoji_id > len(candidates):
                tool_text = f"发送失败：编号 {emoji_id} 无效，请选择 1-{len(candidates)} 之间的编号"
                yield tool_text
                return

            # 获取选中的表情包
            selected = candidates[emoji_id - 1]
            path = selected["path"]
            desc = selected["desc"]
            emotion = selected["emotion"]

            if not os.path.exists(path):
                tool_text = "发送失败：表情包文件已丢失"
                yield tool_text
                return

            # 发送表情包
            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            b64 = await self.image_processor_service._file_to_gif_base64(path)

            # 使用 event.send() 直接发送图片，而不是 yield
            # 这样可以确保后续的 yield tool_text 能正常返回给 LLM
            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))

            # 返回成功信息给 LLM
            tool_text = f"已发送表情包：{desc} (分类：{emotion})"
            logger.info(f"[Tool] {tool_text}")
            yield tool_text
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            tool_text = f"发送出错：{e}"
            yield tool_text
            return

    @filter.llm_tool(name="send_emoji")
    async def send_emoji(self, event: AstrMessageEvent, query: str):
        """快速发送表情包（自动选择最佳匹配）。如果你想自己选择，请改用 search_emoji + send_emoji_by_id。

        Args:
            query(string): 搜索关键词（如：开心、难过、无语、生气）

        返回值：
        - 成功：返回已发送的表情包描述
        - 失败：返回错误提示
        """
        logger.info(f"[Tool] LLM 快速发送表情包: {query}")

        # 标记为主动发送，避免被动标签模式重复触发
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.is_meme_enabled_for_event(event):
                yield "发送失败：当前群聊已禁用表情包功能"
                return

            tool_text: str | None = None
            if not self.cache_service.get_cache("index_cache"):
                await self._load_index()

            idx = self.cache_service.get_cache("index_cache") or {}

            results = await self._search_emoji_candidates(query, limit=5, idx=idx)

            if not results:
                tool_text = f"发送失败：未找到与'{query}'匹配的表情包"
                yield tool_text
                return

            # 发送最佳匹配
            best_path, best_desc, best_emotion = results[0]
            if not os.path.exists(best_path):
                tool_text = "发送失败：文件丢失"
                yield tool_text
                return

            logger.info(f"[Tool] 快速发送表情包: {best_path} (emotion={best_emotion})")
            b64 = await self.image_processor_service._file_to_gif_base64(best_path)

            # 使用 event.send() 直接发送图片，而不是 yield
            # 这样可以确保后续的 yield tool_text 能正常返回给 LLM
            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))

            tool_text = f"已发送表情包：{best_desc} (分类：{best_emotion})"
            logger.info(f"[Tool] {tool_text}")
            yield tool_text
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            tool_text = f"发送出错：{e}"
            yield tool_text
            return
