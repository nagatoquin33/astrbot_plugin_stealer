import asyncio
import json
import os
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import (
    EventMessageType,
    PermissionType,
    PlatformAdapterType,
)
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star

from .cache_service import CacheService
from .core.command_handler import CommandHandler
from .core.config import PluginConfig
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

        # 情绪选择标记（用于识别注入的内容）
        self._persona_marker = "<!-- STEALER_PLUGIN_EMOTION_MARKER_v3 -->"  # 更新版本号

        # 初始化插件配置
        self.plugin_config = PluginConfig(config, context)

        self.base_dir: Path = self.plugin_config.data_dir
        self.raw_dir: Path = self.plugin_config.raw_dir
        self.categories_dir: Path = self.plugin_config.categories_dir
        self.cache_dir: Path = self.plugin_config.cache_dir

        self._ensure_webui_password()
        self._sync_all_config()

        self.plugin_config.ensure_category_dirs(self.categories)

        # 初始化核心服务类
        self.cache_service = CacheService(self.cache_dir)
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
        self._scanner_task: asyncio.Task | None = None
        self._migration_done: bool = False  # 迁移只执行一次
        # 强制捕获窗口已迁移到 EventHandler

        # 验证配置
        self._validate_config()

    def _load_vision_provider_id(self) -> str | None:
        """加载视觉模型提供商ID。

        Returns:
            str | None: 视觉模型提供商ID，如果未配置则返回None
        """
        provider_id = getattr(self.plugin_config, "vision_provider_id", "")
        return str(provider_id) if provider_id else None

    def _sync_all_config(self) -> None:
        """从配置服务同步所有配置到实例属性。

        统一的配置同步方法，避免重复代码。
        """
        # 同步基础配置
        self.auto_send = self.plugin_config.auto_send
        self.emoji_chance = self.plugin_config.emoji_chance
        # 强制使用智能模式
        # self.smart_emoji_selection = True
        self.send_emoji_as_gif = self.plugin_config.send_emoji_as_gif
        self.max_reg_num = self.plugin_config.max_reg_num
        self.do_replace = self.plugin_config.do_replace

        # 清理与容量控制配置（内部常量）
        self.raw_cleanup_interval = getattr(
            self.plugin_config, "raw_cleanup_interval", 30
        )
        self.capacity_control_interval = getattr(
            self.plugin_config, "capacity_control_interval", 60
        )
        self.enable_raw_cleanup = getattr(
            self.plugin_config, "enable_raw_cleanup", True
        )
        self.enable_capacity_control = getattr(
            self.plugin_config, "enable_capacity_control", True
        )

        self.steal_emoji = self.plugin_config.steal_emoji

        # 内容过滤（默认开启，通过 Prompt 控制）
        # self.content_filtration = True

        self.raw_retention_minutes = getattr(
            self.plugin_config, "raw_retention_minutes", 60
        )
        self.categories = list(
            getattr(self.plugin_config, "categories", []) or []
        ) or list(self.plugin_config.DEFAULT_CATEGORIES)

        # 同步模型相关配置
        self.vision_provider_id = self._load_vision_provider_id()

        # 强制启用自然语言分析（智能模式）
        self.enable_natural_emotion_analysis = True
        # 使用默认 Provider 或 Vision Provider
        self.emotion_analysis_provider_id = ""

        # 同步图片处理节流配置
        # self.image_processing_mode = self.plugin_config.image_processing_mode
        # self.image_processing_probability = (
        #     self.plugin_config.image_processing_probability
        # )
        # self.image_processing_interval = self.plugin_config.image_processing_interval
        # self.image_processing_cooldown = self.plugin_config.image_processing_cooldown
        self.image_processing_cooldown = getattr(
            self.plugin_config, "image_processing_cooldown", 10
        )

        # 同步 WebUI 配置
        self.webui_enabled = self.plugin_config.webui.enabled
        self.webui_host = self.plugin_config.webui.host
        self.webui_port = self.plugin_config.webui.port
        self.webui_auth_enabled = self.plugin_config.webui.auth_enabled
        self.webui_password = self.plugin_config.webui.password
        self.webui_session_timeout = self.plugin_config.webui.session_timeout

    def _ensure_webui_password(self) -> bool:
        if (
            self.plugin_config.webui.enabled
            and self.plugin_config.webui.auth_enabled
            and not str(self.plugin_config.webui.password or "").strip()
        ):
            generated = f"{secrets.randbelow(1000000):06d}"
            # self.config_service.update_config_from_dict({"webui_password": generated})
            self.plugin_config.webui.password = generated
            # 手动触发保存 WebUI 配置
            self.plugin_config.save_webui_config()
            logger.info("WebUI 访问密码已自动生成，请在配置中查看")
            return True
        return False

    def _apply_prompts(self, prompts: dict) -> None:
        for key, value in prompts.items():
            setattr(self, key, value)
        self.image_processor_service.update_config(
            emoji_classification_prompt=prompts.get(
                "EMOJI_CLASSIFICATION_PROMPT", None
            ),
            combined_analysis_prompt=prompts.get("COMBINED_ANALYSIS_PROMPT", None),
            emoji_classification_with_filter_prompt=prompts.get(
                "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT", None
            ),
        )

    def _auto_merge_existing_categories(self) -> None:
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

        # 为新增的分类创建对应的目录
        self.plugin_config.ensure_category_dirs(to_add)

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
        if not isinstance(self.emoji_chance, int | float) or not (
            0 <= self.emoji_chance <= 1
        ):
            errors.append("表情发送概率必须在0-1之间")
            self.emoji_chance = 0.4
            fixed.append("表情发送概率已重置为0.4")

        # 记录问题和修复

        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
        if fixed:
            logger.info(f"配置已自动修复: {'; '.join(fixed)}")

        return True  # 即使有问题也返回True，因为已经修复

    def _get_force_capture_key(self, event: AstrMessageEvent) -> str:
        """委托给 EventHandler。"""
        return self.event_handler._get_force_capture_key(event)

    def _get_force_capture_sender_id(self, event: AstrMessageEvent) -> str | None:
        """委托给 EventHandler。"""
        return self.event_handler._get_force_capture_sender_id(event)

    def _get_group_id(self, event: AstrMessageEvent) -> str | None:
        """委托给 PluginConfig。"""
        return self.plugin_config.get_group_id(event)

    def is_meme_enabled_for_event(self, event: AstrMessageEvent) -> bool:
        group_id = self._get_group_id(event)
        if self.plugin_config is None:
            return True
        try:
            return bool(self.plugin_config.is_group_allowed(group_id))
        except Exception:
            return True

    def begin_force_capture(self, event: AstrMessageEvent, seconds: int) -> None:
        """委托给 EventHandler。"""
        self.event_handler.begin_force_capture(event, seconds)

    def get_force_capture_entry(
        self, event: AstrMessageEvent
    ) -> dict[str, object] | None:
        """委托给 EventHandler。"""
        return self.event_handler.get_force_capture_entry(event)

    def consume_force_capture(self, event: AstrMessageEvent) -> None:
        """委托给 EventHandler。"""
        self.event_handler.consume_force_capture(event)

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            # 使用配置服务更新配置
            if self.plugin_config:
                # 记录旧的 WebUI 配置，用于判断是否需要重启 Web Server
                old_webui_enabled = getattr(self, "webui_enabled", True)
                old_webui_host = getattr(self, "webui_host", "0.0.0.0")
                old_webui_port = getattr(self, "webui_port", 8899)
                old_webui_password = getattr(self, "webui_password", "")
                old_webui_session_timeout = getattr(self, "webui_session_timeout", 3600)

                # self.config_service.update_config_from_dict(config_dict)
                # 直接更新 PluginConfig 属性
                # 注意：config_dict 可能包含扁平的 key，也可能包含嵌套的 webui dict
                for k, v in config_dict.items():
                    if k == "webui" and isinstance(v, dict):
                        # 处理嵌套 webui 更新
                        current_webui = self.plugin_config.webui
                        for wk, wv in v.items():
                            setattr(current_webui, wk, wv)
                        # 触发 webui 保存
                        self.plugin_config.save_webui_config()
                    elif k.startswith("webui_"):
                        # 兼容旧的扁平 key (webui_enabled -> webui.enabled)
                        wk = k[6:]  # remove 'webui_' prefix
                        if hasattr(self.plugin_config.webui, wk):
                            setattr(self.plugin_config.webui, wk, v)
                            self.plugin_config.save_webui_config()
                    else:
                        setattr(self.plugin_config, k, v)

                # 统一同步所有配置
                self._sync_all_config()

                if self._ensure_webui_password():
                    self._sync_all_config()

                # 后台任务管理已简化，不再需要在此处动态协调
                # 任务会自动使用新的配置参数（通过 internal constants）
                pass

                # 检查 WebUI 配置是否变化并重启
                # 注意：on_config_update 可能是同步调用，重启操作涉及IO，使用 create_task 异步执行
                if (
                    old_webui_enabled != self.webui_enabled
                    or old_webui_host != self.webui_host
                    or old_webui_port != self.webui_port
                    or old_webui_password != getattr(self, "webui_password", "")
                    or old_webui_session_timeout
                    != getattr(self, "webui_session_timeout", 3600)
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
                    # content_filtration=self.content_filtration,
                    vision_provider_id=self.vision_provider_id,
                    emoji_classification_prompt=getattr(
                        self, "EMOJI_CLASSIFICATION_PROMPT", None
                    ),
                    combined_analysis_prompt=getattr(
                        self, "COMBINED_ANALYSIS_PROMPT", None
                    ),
                )

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
        """
        try:
            # 创建必要的数据目录结构
            self.plugin_config.ensure_base_dirs()

            self._auto_merge_existing_categories()

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
                            self._apply_prompts(prompts)
                    except ImportError:
                        # 如果aiofiles不可用，回退到同步方式
                        logger.debug("aiofiles不可用，使用同步文件读取")
                        with open(prompts_path, encoding="utf-8") as f:
                            prompts = json.load(f)
                            logger.info(f"已加载提示词文件: {prompts_path}")
                            self._apply_prompts(prompts)
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
            self.plugin_config.ensure_category_dirs(self.categories)

            # 启动独立的后台任务
            # raw目录清理任务
            # if self.enable_raw_cleanup:
            self.task_scheduler.create_task(
                "raw_cleanup_loop", self._raw_cleanup_loop()
            )
            logger.info("已启动raw目录清理任务，周期: 30分钟")

            # 容量控制任务
            # if self.enable_capacity_control:
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
                await self.cache_service.cleanup()

            if hasattr(self, "task_scheduler") and self.task_scheduler:
                await self.task_scheduler.cleanup()

            if hasattr(self, "config_service") and self.config_service:
                # self.config_service.cleanup()
                pass

            if (
                hasattr(self, "image_processor_service")
                and self.image_processor_service
            ):
                # ImageProcessorService没有cleanup方法，但可以清理缓存
                if hasattr(self.image_processor_service, "_image_cache"):
                    self.image_processor_service._image_cache.clear()

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
        """委托给 CacheService。"""
        try:
            index_data = await self.cache_service.load_index()

            if not index_data and not self._migration_done:
                logger.debug("[_load_index] cache empty, attempting migration...")
                index_data = await self.cache_service.migrate_legacy_data(self.base_dir)
                self._migration_done = True
                if index_data:
                    await self.cache_service.save_index(index_data)

            return index_data
        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def _migrate_legacy_data(self) -> dict[str, Any]:
        """委托给 CacheService。"""
        return await self.cache_service.migrate_legacy_data(self.base_dir)

    async def _rebuild_index_from_files(self) -> dict[str, Any]:
        """委托给 CacheService。"""
        return await self.cache_service.rebuild_index_from_files(
            self.base_dir, self.categories_dir
        )

    async def _save_index(self, idx: dict[str, Any]):
        """委托给 CacheService。"""
        await self.cache_service.save_index(idx)

    async def _load_aliases(self) -> dict[str, str]:
        """加载分类别名文件。

        Returns:
            Dict[str, str]: 别名映射字典。
        """
        try:
            return self.plugin_config.get_keyword_map()
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
            # self.config_service.update_aliases(aliases)
            # 暂时不支持动态修改别名并持久化，因为已改为常量
            pass
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
                    # content_filtration=self.content_filtration,
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
        """委托给 ImageProcessorService。"""
        return await self.image_processor_service.safe_remove_file(file_path)

    def _is_in_parentheses(self, text: str, index: int) -> bool:
        """委托给 EmojiSelector (原 EmotionAnalyzerService) 不再支持此方法，或需要迁移"""
        # EmojiSelector 中没有 is_in_parentheses，如果需要，应该加进去
        # 暂时返回 False 或迁移逻辑
        return False

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

    async def _pick_vision_provider(self, event: AstrMessageEvent | None) -> str | None:
        """委托给 ImageProcessorService。"""
        return await self.image_processor_service.pick_vision_provider(event)

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
            # if self.steal_emoji and self.enable_raw_cleanup:
            if self.steal_emoji:
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
                # await asyncio.sleep(max(1, int(self.raw_cleanup_interval)) * 60)
                await asyncio.sleep(self.RAW_CLEANUP_INTERVAL_SECONDS)

                # 只有当偷图功能开启且清理功能启用时才执行
                # if self.steal_emoji and self.enable_raw_cleanup:
                if self.steal_emoji:
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
                # await asyncio.sleep(max(1, int(self.capacity_control_interval)) * 60)
                await asyncio.sleep(self.CAPACITY_CONTROL_INTERVAL_SECONDS)

                # if self.steal_emoji and self.enable_capacity_control:
                if self.steal_emoji:
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

                        def updater(current: dict):
                            current.clear()
                            current.update(image_index)

                        await self.cache_service.update_index(updater)

                    logger.info("容量控制任务完成")

            except asyncio.CancelledError:
                logger.info("容量控制任务已取消")
                break
            except Exception as e:
                logger.error(f"容量控制任务发生错误: {e}", exc_info=True)
                continue

    async def _clean_raw_directory(self) -> int:
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        if hasattr(self, "event_handler") and self.event_handler:
            return await self.event_handler._clean_raw_directory()
        logger.warning("event_handler 未初始化，无法执行raw目录清理")
        return 0

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

    def _check_send_probability(self) -> bool:
        """委托给 EmojiSelector。"""
        return self.emoji_selector.check_send_probability()

    async def _select_emoji(self, category: str, context_text: str = "") -> str | None:
        """委托给 EmojiSelector。"""
        return await self.emoji_selector.select_emoji(category, context_text)

    async def _select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """委托给 EmojiSelector。"""
        return await self.emoji_selector.select_emoji_smart(category, context_text)

    async def _send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ):
        """委托给 EmojiSelector。"""
        await self.emoji_selector.send_explicit_emojis(event, emoji_paths, cleaned_text)

    async def _try_send_emoji(
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str
    ) -> bool:
        """委托给 EmojiSelector。"""
        return await self.emoji_selector.try_send_emoji(event, emotions, cleaned_text)

    async def _async_analyze_and_send_emoji(
        self, event: AstrMessageEvent, text: str, emotions: list[str]
    ):
        """异步分析情绪并发送表情包（不阻塞主流程）。

        Args:
            event: 消息事件
            text: 文本内容
            emotions: 已提取的情绪列表（被动模式使用，智能模式忽略）
        """
        try:
            # 检查是否启用自动发送
            if not self.auto_send:
                logger.debug("[Stealer] 自动发送已禁用，跳过表情包发送")
                return

            # 检查群聊是否允许
            if not self.is_meme_enabled_for_event(event):
                logger.debug("[Stealer] 当前群聊已禁用表情包功能")
                return

            # 判断模式
            is_intelligent_mode = getattr(self, "enable_natural_emotion_analysis", True)

            final_emotions = []

            if is_intelligent_mode:
                # 智能模式：使用轻量模型分析
                logger.debug("[Stealer] 智能模式：使用轻量模型分析情绪")
                try:
                    analyzed_emotion = (
                        await self.smart_emotion_matcher.analyze_and_match_emotion(
                            event, text, use_natural_analysis=True
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

            # 尝试发送表情包
            await self._try_send_emoji(event, final_emotions, text)

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
        """委托给 CommandHandler。"""
        return await self.command_handler.get_emoji_count()

    async def get_info(self) -> dict:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_emoji_info()

    async def get_emotions(self) -> list[str]:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_available_emotions()

    async def get_descriptions(self) -> list[str]:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_all_descriptions()

    async def _load_all_records(self) -> list[tuple[str, dict]]:
        """委托给 CommandHandler。"""
        return await self.command_handler.load_all_emoji_records()

    async def get_random_paths(
        self, count: int | None = 1
    ) -> list[tuple[str, str, str]]:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_random_emojis(count)

    async def get_by_emotion_path(self, emotion: str) -> tuple[str, str, str] | None:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_emoji_by_emotion(emotion)

    async def get_by_description_path(
        self, description: str
    ) -> tuple[str, str, str] | None:
        """委托给 CommandHandler。"""
        return await self.command_handler.get_emoji_by_description(description)

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
        """委托给 ImageProcessorService.smart_search。"""
        if idx is None:
            idx = self.cache_service.get_cache("index_cache") or {}

        return await self.image_processor_service.smart_search(
            query, limit=limit, idx=idx
        )

    @filter.llm_tool(name="search_emoji")
    async def search_emoji(self, event: AstrMessageEvent, query: str):
        """搜索表情包，返回候选列表供你选择。

        Args:
            query(string): 搜索关键词（支持情绪词、描述词、场景词）

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
        返回候选表情包列表，每个包含：
        - 编号：用于调用 send_emoji_by_id
        - 分类：表情包的情绪分类
        - 描述：表情包的详细描述（这是你选择时的重要参考）

        请仔细阅读每个表情包的描述，选择最符合当前对话情境的一个。
        """
        logger.info(f"[Tool] LLM 搜索表情包: {query}")

        # 标记为主动发送流程开始
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.is_meme_enabled_for_event(event):
                yield "搜索失败：当前群聊已禁用表情包功能"
                return

            if not self.cache_service.get_cache("index_cache"):
                logger.debug("索引未加载，正在加载...")
                await self._load_index()

            idx = self.cache_service.get_cache("index_cache") or {}

            # 增加搜索结果数量，给LLM更多选择
            results = await self._search_emoji_candidates(query, limit=8, idx=idx)

            if not results:
                logger.warning(f"未找到匹配的表情包: {query}")
                yield f"搜索失败：未找到与'{query}'匹配的表情包。\n\n建议尝试：\n- 情绪词：happy, sad, angry, confused, troll\n- 描述词：大笑, 无语, 哭了, 震惊\n- 场景词：尴尬, 社死, 躺平"
                return

            # 构建候选列表
            candidates = []
            result_lines = [f"找到 {len(results)} 个匹配的表情包：\n"]

            for i, (path, desc, emotion, tags) in enumerate(results):
                if os.path.exists(path):
                    candidate_id = f"emoji_{i + 1}"
                    candidates.append(
                        {
                            "id": candidate_id,
                            "path": path,
                            "desc": desc,
                            "emotion": emotion,
                            "tags": tags,
                        }
                    )
                    # 格式化输出
                    result_lines.append(f"\n[{i + 1}] 分类：{emotion}")
                    if tags:
                        result_lines.append(f"    标签：{tags}")
                    result_lines.append(f"    描述：{desc}")

            if not candidates:
                yield "搜索失败：找到的表情包文件均已丢失"
                return

            # 存入实例属性，供 send_emoji_by_id 使用
            self._emoji_candidates = candidates

            result_lines.append(
                "\n\n请根据描述选择最合适的表情包，然后调用 send_emoji_by_id(编号) 发送。"
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

        Args:
            emoji_id(number): 表情包编号（从 search_emoji 返回的列表中选择）

        返回值：
        返回你发送的表情包的完整信息，包括：
        - 分类：表情包的情绪分类
        - 描述：表情包的详细描述
        - 状态：发送成功/失败

        这样你就能清楚地知道自己发送了什么表情包。
        """
        logger.info(f"[Tool] LLM 选择发送表情包编号: {emoji_id}")

        # 标记为主动发送
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.is_meme_enabled_for_event(event):
                yield "发送失败：当前群聊已禁用表情包功能"
                return

            # 统一转为 int
            try:
                emoji_id = int(emoji_id)
            except Exception:
                yield f"发送失败：编号 {emoji_id} 无法解析为整数，请输入有效的数字编号"
                return

            # 获取候选列表
            candidates = getattr(self, "_emoji_candidates", None)

            if not candidates:
                yield "发送失败：没有可用的候选列表。请先调用 search_emoji 搜索表情包。"
                return

            # 验证编号范围
            if emoji_id < 1 or emoji_id > len(candidates):
                yield f"发送失败：编号 {emoji_id} 无效。可选编号范围：1-{len(candidates)}，请重新选择。"
                return

            # 获取选中的表情包
            selected = candidates[emoji_id - 1]
            path = selected["path"]
            desc = selected["desc"]
            emotion = selected["emotion"]

            if not os.path.exists(path):
                yield f"发送失败：表情包文件已丢失。\n你选择的是：编号 {emoji_id}，分类 {emotion}，描述 {desc}\n请重新搜索并选择其他表情包。"
                return

            # 发送表情包
            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            b64 = await self.image_processor_service._file_to_gif_base64(path)

            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))

            # 返回详细的成功信息
            success_msg = f"发送成功。\n\n你发送的表情包：\n- 编号：{emoji_id}\n- 分类：{emotion}\n- 描述：{desc}"
            logger.info(f"[Tool] {success_msg}")
            yield success_msg
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield f"发送出错：{e}"
            return
