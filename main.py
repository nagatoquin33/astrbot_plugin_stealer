import asyncio
import copy
import json
import os
import random
import shutil
import time
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
from astrbot.api.star import Context, Star, StarTools

from .cache_service import CacheService
from .command_handler import CommandHandler

# 导入原有服务类 - 使用标准的相对导入
from .config_service import ConfigService
from .emotion_analyzer_service import EmotionAnalyzerService
from .event_handler import EventHandler
from .image_processor_service import ImageProcessorService
from .task_scheduler import TaskScheduler
from .web_server import WebServer

try:
    # 可选依赖，用于通过图片尺寸/比例进行快速过滤，未安装时自动降级
    from PIL import Image as PILImage  # type: ignore[import]
except Exception:  # pragma: no cover - 仅作为兼容分支
    PILImage = None


# ================= Monkey Patch Start =================
# 修复 AstrBot Token 一次性销毁导致部分客户端无法预览/下载图片的问题
from astrbot.core.file_token_service import FileTokenService
import os

# 定义一个新的 handle_file 方法，不删除 Token
async def patched_handle_file(self, file_token: str) -> str:
    async with self.lock:
        await self._cleanup_expired_tokens()

        if file_token not in self.staged_files:
            raise KeyError(f"无效或过期的文件 token: {file_token}")

        # 修改点：使用 [] 读取而不是 pop() 删除
        # file_path, _ = self.staged_files.pop(file_token) 
        file_path, _ = self.staged_files[file_token]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        return file_path

# 将核心类的方法替换为新方法
FileTokenService.handle_file = patched_handle_file
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

    # 情绪分类列表（英文标签）
    # 注意: 实际使用的分类列表由 ImageProcessorService.VALID_CATEGORIES 定义
    CATEGORIES = [
        "happy",       # 开心
        "sad",         # 难过
        "angry",       # 生气
        "cry",         # 大哭
        "shy",         # 害羞
        "surprised",   # 惊讶
        "love",        # 喜爱
        "fear",        # 害怕
        "tired",       # 疲惫
        "disgust",     # 厌恶
        "excitement",  # 兴奋
        "embarrassed", # 尴尬
        "sigh",        # 叹气
        "thank",       # 感谢
        "confused",    # 困惑
        "dumb",        # 无语/呆
        "troll",       # 发癫/搞怪
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)

        # 初始化基础路径 - 遵循 AstrBot 插件存储大文件规范
        # 大文件应存储于 data/plugin_data/{plugin_name}/ 目录下
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path
        # self.name 在 v4.9.2 及以上版本可用
        plugin_name = getattr(self, "name", "astrbot_plugin_stealer")
        self.base_dir: Path = Path(get_astrbot_data_path()) / "plugin_data" / plugin_name

        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.config_path: Path = self.base_dir / "config.json"
        self.raw_dir: Path = self.base_dir / "raw"
        self.categories_dir: Path = self.base_dir / "categories"
        self.cache_dir: Path = self.base_dir / "cache"

        # 设置PILImage实例属性，供ImageProcessorService使用
        self.PILImage = PILImage

        # 初始化人格注入相关属性
        # 直接集成提示词，不在设置界面显示
        # 优化后的结构化 Prompt，提升约束力
        self.prompt_head: str = (
            "\n\n# 角色指令：情绪表达\n"
            "你需要根据对话的上下文和你当前的回复态度，从以下列表中选择一个最匹配的情绪：\n"
            "[{categories}]\n"
        )
        self.prompt_tail: str = (
            "\n# 输出格式严格要求\n"
            "1. 必须在回复的**最开头**，使用双浮点号 '&&' 包裹情绪标签。\n"
            "2. 格式示例：\n"
            "   &&happy&& 哈哈，这个太有意思了！\n"
            "   &&sad&& 唉，怎么会这样...\n"
            "3. 只能使用列表中的情绪词，严禁创造新词。\n"
            "4. 不要使用 Markdown 代码块或括号，**仅使用 && 符号**。\n"
        )
        self.persona_backup: list = []
        self._persona_injected: bool = False  # 标记是否已注入
        self._persona_marker: str = "<!-- STEALER_PLUGIN_EMOTION_MARKER_v2 -->"  # 更唯一的标记

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

        # 运行时属性
        self.backend_tag: str = self.BACKEND_TAG
        self._scanner_task: asyncio.Task | None = None
        self._migration_done: bool = False  # 迁移只执行一次

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

        # 同步视觉模型配置
        self.vision_provider_id = self._load_vision_provider_id()

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
        if not isinstance(self.emoji_chance, (int, float)) or not (0 <= self.emoji_chance <= 1):
            errors.append("表情发送概率必须在0-1之间")
            self.emoji_chance = 0.4
            fixed.append("表情发送概率已重置为0.4")

        # 验证清理周期
        if not isinstance(self.raw_cleanup_interval, int) or self.raw_cleanup_interval < 1:
            errors.append("raw清理周期必须至少为1分钟")
            self.raw_cleanup_interval = 30
            fixed.append("raw清理周期已重置为30分钟")

        # 验证容量控制周期
        if not isinstance(self.capacity_control_interval, int) or self.capacity_control_interval < 1:
            errors.append("容量控制周期必须至少为1分钟")
            self.capacity_control_interval = 60
            fixed.append("容量控制周期已重置为60分钟")

        # 验证保留期限（如果存在）
        if hasattr(self, "raw_retention_minutes") and (
            not isinstance(self.raw_retention_minutes, int) or self.raw_retention_minutes < 1
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

                self.config_service.update_config_from_dict(config_dict)

                # 统一同步所有配置
                self._sync_all_config()

                # 检查 WebUI 配置是否变化并重启
                # 注意：on_config_update 可能是同步调用，重启操作涉及IO，使用 create_task 异步执行
                if (
                    old_webui_enabled != self.webui_enabled or
                    old_webui_host != self.webui_host or
                    old_webui_port != self.webui_port
                ):
                    async def restart_webui():
                        logger.info("检测到 WebUI 配置变更，正在重启 WebUI...")
                        if self.web_server:
                            await self.web_server.stop()
                            self.web_server = None

                        if self.webui_enabled:
                            try:
                                self.web_server = WebServer(
                                    self,
                                    host=self.webui_host,
                                    port=self.webui_port
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

                # 重新加载人格注入
                asyncio.create_task(self._reload_personas())
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

            # 加载提示词文件
            try:
                # 使用__file__获取当前脚本所在目录，即插件安装目录
                plugin_dir = Path(__file__).parent
                prompts_path = plugin_dir / "prompts.json"
                if prompts_path.exists():
                    # 使用异步文件读取
                    import aiofiles
                    try:
                        async with aiofiles.open(prompts_path, encoding="utf-8") as f:
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
                            )
                else:
                    logger.warning(f"提示词文件不存在: {prompts_path}")
            except Exception as e:
                logger.error(f"初始化提示词失败: {e}")

            # 启动WebUI（如果启用）
            if self.webui_enabled:
                try:
                    self.web_server = WebServer(
                        self,
                        host=self.webui_host,
                        port=self.webui_port
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

            # 加载并注入人格
            personas = self.context.provider_manager.personas
            self.persona_backup = [copy.deepcopy(personas[0])] if personas else []
            await self._reload_personas()

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
            # 清理人格注入 - 只移除我们注入的部分，不影响其他插件的修改
            personas = self.context.provider_manager.personas
            marker = self._persona_marker

            for persona in personas:
                try:
                    current_prompt = persona.get("prompt", "")

                    # 检查是否包含我们的标记
                    if marker in current_prompt:
                        # 使用正则移除标记包裹的注入内容
                        import re
                        # 移除：\n标记\n内容\n标记\n
                        pattern = rf"\n\s*{re.escape(marker)}\n.*?\n\s*{re.escape(marker)}\n"
                        new_prompt = re.sub(pattern, "", current_prompt, flags=re.DOTALL)
                        persona["prompt"] = new_prompt
                        logger.debug(f"已清理人格中的情绪注入")

                except Exception as e:
                    logger.error(f"清理人格时出错: {e}")
                    continue

            # 重置注入状态
            self._persona_injected = False

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

    async def _reload_personas(self, force: bool = False):
        """重新加载人格配置并注入情绪选择提醒。

        Args:
            force: 是否强制重新注入（即使已注入过）

        优化后的注入逻辑，更加兼容其他插件：
        1. 使用可识别的标记包裹注入内容
        2. 恢复时只移除我们注入的部分，不影响其他插件的修改
        3. 支持通过 force 参数强制重新注入
        """
        try:
            from astrbot.api import logger

            personas = self.context.provider_manager.personas

            if not personas:
                logger.debug("未配置人格，跳过注入")
                return

            persona = personas[0]
            current_prompt = persona.get("prompt", "")

            # 如果已注入且不是强制重新注入，则跳过
            if self._persona_injected and not force:
                logger.debug("情绪注入已存在，跳过重复注入")
                return

            # 构建情绪分类字符串
            categories_str = ", ".join(self.categories)

            # 生成系统提示添加内容
            head_with_categories = self.prompt_head.replace(
                "{categories}", categories_str
            )
            sys_prompt_add = f"\n\n{head_with_categories}\n{self.prompt_tail}"

            # 如果已有旧注入，先清理
            if self._persona_marker in current_prompt:
                import re
                pattern = rf"\n?{re.escape(self._persona_marker)}\n.*?{re.escape(self._persona_marker)}\n?"
                current_prompt = re.sub(pattern, "", current_prompt, flags=re.DOTALL)
                logger.debug("已清理旧的情绪注入")

            # 备份原始prompt（只备份第一个）
            if not self.persona_backup:
                self.persona_backup = [copy.deepcopy(persona)]
            else:
                self.persona_backup[0]["prompt"] = current_prompt

            # 注入情绪提醒，使用标记包裹
            injected_content = f"\n{self._persona_marker}\n{sys_prompt_add}\n{self._persona_marker}\n"
            persona["prompt"] = current_prompt + injected_content

            self._persona_injected = True
            logger.info(f"已注入情绪选择提醒到人格配置中（force={force}）")

        except Exception as e:
            logger.error(f"注入情绪选择提醒失败: {e}")

    def _persist_config(self):
        """持久化插件运行配置。"""
        try:
            # 使用配置服务更新并保存配置
            config_updates = {
                "auto_send": self.auto_send,
                "categories": self.categories,
                "backend_tag": self.backend_tag,
                "emoji_chance": self.emoji_chance,
                "max_reg_num": self.max_reg_num,
                "do_replace": self.do_replace,
                "raw_cleanup_interval": self.raw_cleanup_interval,
                "capacity_control_interval": self.capacity_control_interval,
                "enable_raw_cleanup": self.enable_raw_cleanup,
                "enable_capacity_control": self.enable_capacity_control,
                "steal_emoji": self.steal_emoji,
                "content_filtration": self.content_filtration,
                "vision_provider_id": self.vision_provider_id,
                "raw_retention_minutes": self.raw_retention_minutes,
            }

            # 使用ConfigService的update_config方法确保配置同步
            self.config_service.update_config(config_updates)

        except Exception as e:
            logger.error(f"保存配置失败: {e}")

    async def _load_index(self) -> dict[str, Any]:
        """加载分类索引文件。

        Returns:
            Dict[str, Any]: 键为文件路径，值为包含 category 与 tags 的字典。
        """
        try:
            cache_data = self.cache_service.get_cache("index_cache")
            
            logger.debug(f"[_load_index] raw cache type: {type(cache_data)}, keys: {list(cache_data.keys())[:5] if cache_data else 'empty'}")
            
            index_data = dict(cache_data) if cache_data else {}

            logger.debug(f"[_load_index] converted to dict, {len(index_data)} items")

            if not index_data and not self._migration_done:
                logger.debug(f"[_load_index] cache empty, attempting migration...")
                index_data = await self._migrate_legacy_data()
                self._migration_done = True
                logger.debug(f"[_load_index] migration returned {len(index_data)} items")

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
                            logger.info(f"从 {old_path} 加载了 {len(old_data)} 条旧记录")
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
            current_index = await self._load_index()

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
                    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
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
                                if raw_file.is_file() and raw_file.stem == img_file.stem:
                                    raw_path = str(raw_file)
                                    break

                    # 如果没找到raw文件，使用categories中的文件路径
                    if not raw_path:
                        raw_path = str(img_file)

                    # 计算文件哈希
                    try:
                        file_hash = await self.image_processor_service._compute_hash(str(img_file))
                    except Exception as e:
                        logger.debug(f"计算文件哈希失败: {e}")
                        file_hash = ""

                    # 创建索引记录
                    rebuilt_index[raw_path] = {
                        "hash": file_hash,
                        "category": category_name,
                        "created_at": int(img_file.stat().st_mtime),
                        "migrated": True  # 标记为迁移数据
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

    def _is_safe_path(self, path: str) -> tuple[bool, str]:
        """检查路径是否安全，防止路径遍历攻击。

        Args:
            path: 要检查的文件路径

        Returns:
            tuple[bool, str]: (是否安全, 规范化后的安全路径)
        """
        try:
            # 定义允许的基准目录
            # 优先使用公开 API，如果失败则使用内部 API 作为备选
            try:
                # 使用 StarTools.get_data_dir() 的父目录来获取 data 目录路径
                astrbot_data_path = StarTools.get_data_dir("").parent.resolve()
            except Exception:
                # 备选方案：使用内部 API
                logger.warning("使用内部API获取数据路径，建议检查StarTools.get_data_dir()的使用")
                from astrbot.core.utils.astrbot_path import get_astrbot_data_path

                astrbot_data_path = Path(get_astrbot_data_path()).resolve()

            plugin_base_dir = Path(self.base_dir).resolve()

            # 调试日志
            logger.debug(f"路径安全检查 - 输入路径: {path}")
            logger.debug(f"astrbot_data_path: {astrbot_data_path}")
            logger.debug(f"plugin_base_dir: {plugin_base_dir}")

            path_obj = Path(path)
            normalized_path = None

            if path_obj.is_absolute():
                # 绝对路径：直接检查是否在允许的目录内
                normalized_path = path_obj.resolve()
            else:
                # 相对路径：根据路径前缀解析到安全的基准目录
                lower_path = path.lower()

                # 检查是否包含路径遍历攻击
                if ".." in str(path_obj):
                    # 直接检查解析后的路径是否仍在预期的父目录内
                    if lower_path.startswith(("data/", "data\\")):
                        # 以 data/ 开头的相对路径解析到 astrbot_data_path
                        relative_part = path[5:]  # 移除 "data/" 前缀
                        normalized_path = (astrbot_data_path / relative_part).resolve()
                        # 确保解析后的路径仍在 astrbot_data_path 内
                        if not normalized_path.is_relative_to(astrbot_data_path):
                            logger.error(
                                f"路径遍历攻击检测: {path} -> {normalized_path}"
                            )
                            return False, path
                    elif lower_path.startswith(("astrbot/", "astrbot\\")):
                        # 以 AstrBot/ 开头的相对路径解析到 astrbot_data_path
                        relative_part = path[8:]  # 移除 "AstrBot/" 前缀
                        normalized_path = (astrbot_data_path / relative_part).resolve()
                        # 确保解析后的路径仍在 astrbot_data_path 内
                        if not normalized_path.is_relative_to(astrbot_data_path):
                            logger.error(
                                f"路径遍历攻击检测: {path} -> {normalized_path}"
                            )
                            return False, path
                    else:
                        # 其他相对路径解析到 plugin_base_dir
                        normalized_path = (plugin_base_dir / path).resolve()
                        # 确保解析后的路径仍在 plugin_base_dir 内
                        if not normalized_path.is_relative_to(plugin_base_dir):
                            logger.error(
                                f"路径遍历攻击检测: {path} -> {normalized_path}"
                            )
                            return False, path
                else:
                    # 不包含路径遍历的相对路径，正常处理
                    if lower_path.startswith(("data/", "data\\")):
                        # 以 data/ 开头的相对路径解析到 astrbot_data_path
                        relative_part = path[5:]  # 移除 "data/" 前缀
                        normalized_path = (astrbot_data_path / relative_part).resolve()
                    elif lower_path.startswith(("astrbot/", "astrbot\\")):
                        # 以 AstrBot/ 开头的相对路径解析到 astrbot_data_path
                        relative_part = path[8:]  # 移除 "AstrBot/" 前缀
                        normalized_path = (astrbot_data_path / relative_part).resolve()
                    else:
                        # 其他相对路径解析到 plugin_base_dir
                        normalized_path = (plugin_base_dir / path).resolve()

            # 检查路径是否在允许的目录内
            is_safe = False
            # 添加临时目录到允许列表，因为 convert_to_file_path() 返回临时文件路径
            temp_dir = astrbot_data_path / "temp"
            allowed_parents = [astrbot_data_path, plugin_base_dir, temp_dir]

            for parent in allowed_parents:
                try:
                    normalized_path.relative_to(parent)
                    is_safe = True
                    break
                except ValueError:
                    pass

            if not is_safe:
                logger.error(f"路径超出允许范围: {path} -> {normalized_path}")
                logger.error(f"允许的父目录: {[str(p) for p in allowed_parents]}")
                # 检查是否是临时文件目录的问题
                if "temp" in str(normalized_path):
                    logger.error("检测到临时文件路径，可能需要添加 temp 目录到允许列表")
                return False, path

            return True, str(normalized_path)
        except Exception as e:
            logger.error(f"路径安全检查失败: {e}")
            return False, path

    @filter.event_message_type(EventMessageType.ALL)
    @filter.platform_adapter_type(PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 委托给 EventHandler 类处理
        await self.event_handler.on_message(event)

    async def _raw_cleanup_loop(self):
        """raw目录清理循环任务。"""
        # 启动时立即执行一次清理
        try:
            if self.steal_emoji and self.enable_raw_cleanup:
                logger.info("启动时执行初始raw目录清理")
                await self.event_handler._clean_raw_directory()
                logger.info("初始raw目录清理完成")
        except Exception as e:
            logger.error(f"初始raw目录清理失败: {e}", exc_info=True)

        while True:
            try:
                # 等待指定的清理周期
                await asyncio.sleep(max(1, int(self.raw_cleanup_interval)) * 60)

                # 只有当偷图功能开启且清理功能启用时才执行
                if self.steal_emoji and self.enable_raw_cleanup:
                    logger.info("开始执行raw目录清理任务")
                    await self.event_handler._clean_raw_directory()
                    logger.info("raw目录清理任务完成")

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
                    
                    logger.info(f"清理后索引条目数: {len(image_index)}，有效文件: {len(valid_paths)}")
                    
                    await self.event_handler._enforce_capacity(image_index)
                    
                    if self.cache_service:
                        self.cache_service.set_cache("index_cache", image_index, persist=True)
                    
                    logger.info("容量控制任务完成")

            except asyncio.CancelledError:
                logger.info("容量控制任务已取消")
                break
            except Exception as e:
                logger.error(f"容量控制任务发生错误: {e}", exc_info=True)
                continue

    # 已移除_scan_register_emoji_folder方法（扫描系统表情包目录功能，无实际用途）
    # 已移除_persona_maintenance_loop方法（不再需要定期维护，新注入逻辑只在首次加载时执行一次）

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        await self.event_handler._clean_raw_directory()

    async def _enforce_capacity(self, idx: dict):
        """执行容量控制，删除最旧的图片。"""
        # 委托给 EventHandler 类处理
        await self.event_handler._enforce_capacity(idx)

    # 情绪标签正则模式 - 支持两种格式
    EMOTION_TAG_PATTERNS = [
        r"&&(\w+)&&",           # &&happy&& 格式（简短）
        r"\[emoji:\s*(\w+)\]",  # [emoji: happy] 格式（可读性好）
    ]

    @filter.on_decorating_result(priority=-100)
    async def _prepare_emoji_response(self, event: AstrMessageEvent):
        """准备表情包响应的公共逻辑。

        使用负优先级(-100)确保最高优先级执行（数值越小优先级越高），
        在所有其他插件之前处理标签清理，避免标签被其他插件误处理。

        支持的标签格式：
        - &&happy&& 格式
        - [emoji: happy] 格式
        """
        logger.info("[Stealer] _prepare_emoji_response 被调用")

        # 检查是否为主动发送（工具已发送表情包）
        if event.get_extra("stealer_active_sent"):
            # 清理回复中的标签，但不发送表情包
            result = event.get_result()
            if result:
                text = result.get_plain_text() or ""
                if text.strip():
                    cleaned_text = self._clean_emotion_tags(text)
                    if cleaned_text != text:
                        self._update_result_with_cleaned_text(event, result, cleaned_text)
                        logger.debug("[Stealer] 已清理主动发送后的情绪标签")
            return False

        try:
            # 1. 验证结果对象
            result = event.get_result()
            if not self._validate_result(result):
                logger.debug("[Stealer] 结果对象无效，跳过处理")
                return False

            # 2. 提取和清理文本
            text = result.get_plain_text() or event.get_message_str() or ""
            if not text.strip():
                logger.debug("没有可处理的文本内容，未触发图片发送")
                return False

            # 2.5 检查并处理显式的表情包标记 (来自 Tool 调用)
            import re
            explicit_emojis = []

            def tag_replacer(match):
                explicit_emojis.append(match.group(1))
                return "" # 从文本中移除标记

            # 标记格式: [ast_emoji:path]
            # 使用非贪婪匹配
            text_without_tags = re.sub(r"\[ast_emoji:(.*?)\]", tag_replacer, text)
            has_explicit = len(explicit_emojis) > 0

            # 2.6 检查并处理情绪标签 (&&happy&& 或 [发送表情包: happy] 等)
            emotion_from_tags = self._extract_emotions_from_tags(text_without_tags)
            text_without_tags = self._clean_emotion_tags(text_without_tags)

            # 3. 委托给情绪分析服务处理情绪提取和标签清理
            # 传入已移除显式标记和情绪标签的文本
            emotions_from_llm, cleaned_text = await self._extract_emotions_from_text(event, text_without_tags)
            text_updated = cleaned_text != text

            # 合并情绪来源：优先使用显式标签中的情绪
            all_emotions = list(set(emotion_from_tags + emotions_from_llm))

            # 4. 更新结果
            if has_explicit:
                # 如果有显式表情包，优先发送显式表情包
                # 这时不再触发自动表情包发送，避免重复
                await self._send_explicit_emojis(event, explicit_emojis, cleaned_text)
                logger.info(f"已发送 {len(explicit_emojis)} 张显式表情包")
                return True

            # 4.1 更新结果对象（清理标签）仅当没有显式发送时（显式发送已包含文本更新逻辑）
            if text_updated:
                self._update_result_with_cleaned_text(event, result, cleaned_text)
                logger.debug("已清理情绪标签")

            # 5. 检查是否需要发送表情包
            if not all_emotions:
                logger.debug("未从文本中提取到情绪关键词，未触发图片发送")
                return text_updated

            # 6. 委托给事件处理器检查发送条件和发送表情包
            emoji_sent = await self._try_send_emoji(event, all_emotions, cleaned_text)

            return text_updated or emoji_sent

        except Exception as e:
            logger.error(f"[Stealer] 处理表情包响应时发生错误: {e}", exc_info=True)
            # 即使出错也要返回text_updated，确保标签清理生效
            return text_updated if "text_updated" in locals() else False

    def _extract_emotions_from_tags(self, text: str) -> list[str]:
        """从文本中提取情绪标签中的情绪类别。

        支持的格式：
        - &&happy&&
        - [emoji: happy]

        Args:
            text: 原始文本

        Returns:
            list[str]: 提取到的情绪类别列表
        """
        import re
        emotions = []

        for pattern in self.EMOTION_TAG_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower()
                # 验证是否是有效的情绪类别
                if match_lower in [c.lower() for c in self.categories]:
                    emotions.append(match_lower)
                    logger.debug(f"[Stealer] 从标签中提取到情绪: {match_lower}")

        return emotions

    def _clean_emotion_tags(self, text: str) -> str:
        """从文本中清理所有情绪标签。

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        import re
        cleaned = text

        for pattern in self.EMOTION_TAG_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 清理多余的空白字符
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _validate_result(self, result) -> bool:
        """验证结果对象是否有效。"""
        return (
            result is not None
            and hasattr(result, "chain")
            and hasattr(result, "get_plain_text")
        )

    def _update_result_with_cleaned_text(
        self, event: AstrMessageEvent, result, cleaned_text: str
    ):
        """更新结果对象，使用清理后的文本。"""
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
        """检查表情包发送概率。"""
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
        """智能选择表情包文件，根据上下文匹配最相关的表情包。

        Args:
            category: 情绪分类
            context_text: 上下文文本（可选），用于智能匹配

        Returns:
            表情包文件路径，如果没有则返回None
        """
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

            logger.debug(f"从'{category}'目录中找到 {len(files)} 张图片（{'智能选择失败，' if use_smart else ''}随机选择）")
            picked_image = random.choice(files)
            return picked_image.as_posix()
        except Exception as e:
            logger.error(f"选择表情包失败: {e}")
            return None

    async def _select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """智能选择表情包，根据上下文匹配描述和标签,并考虑使用频率避免重复。

        Args:
            category: 情绪分类
            context_text: 上下文文本

        Returns:
            最匹配的表情包路径，如果没有则返回None
        """
        try:
            # 1. 加载索引，获取该分类下的所有表情包
            idx = await self._load_index()
            candidates = []

            # 获取当前时间戳，用于计算使用频率衰减
            current_time = time.time()

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

                # 获取描述、标签和适用场景
                desc = str(data.get("desc", "")).lower()
                tags = [str(t).lower() for t in data.get("tags", [])]
                scenes = [str(s).lower() for s in data.get("scenes", [])]

                # 计算匹配分数
                score = 0
                context_lower = context_text.lower()

                # 1. 适用场景匹配（新增，优先级最高）
                for scene in scenes:
                    if len(scene) > 2 and scene in context_lower:
                        score += 25  # 场景匹配25分（最高）

                # 2. 描述匹配
                if desc:
                    if desc in context_lower:
                        # 描述完整包含在上下文中
                        score += 20
                    else:
                        # 词汇匹配
                        desc_words = [w for w in desc.split() if len(w) > 1]
                        matched_words = 0
                        for word in desc_words:
                            if word in context_lower:
                                matched_words += 1

                        if matched_words > 0:
                            score += matched_words * 5  # 每个匹配词5分

                # 3. 标签匹配
                for tag in tags:
                    if len(tag) > 1 and tag in context_lower:
                        score += 8  # 标签匹配8分

                # 4. 计算使用频率惩罚（新增）
                last_used = data.get("last_used", 0)
                use_count = data.get("use_count", 0)

                # 时间衰减：最近使用的减分更多
                time_since_last_use = current_time - last_used
                if time_since_last_use < 300:  # 5分钟内
                    score -= 15
                elif time_since_last_use < 1800:  # 30分钟内
                    score -= 10
                elif time_since_last_use < 3600:  # 1小时内
                    score -= 5

                # 使用频率惩罚：使用次数越多，减分越多
                if use_count > 10:
                    score -= min(use_count * 0.5, 10)  # 最多减10分

                # 即使没有匹配也加入候选（score可能为负）
                candidates.append({
                    "path": file_path,
                    "score": score,
                    "desc": desc,
                    "last_used": last_used,
                    "use_count": use_count
                })

            if not candidates:
                logger.debug(f"分类 '{category}' 下没有可用的表情包")
                return None

            # 2. 根据分数选择
            candidates.sort(key=lambda x: x["score"], reverse=True)

            if candidates[0]["score"] > 0:
                # 有匹配：从高分候选中随机选择（增加多样性）
                max_score = candidates[0]["score"]
                top_candidates = [c for c in candidates if c["score"] >= max_score * 0.7]

                # 使用加权随机：分数越高，被选中概率越大
                weights = [c["score"] for c in top_candidates]
                selected = random.choices(top_candidates, weights=weights, k=1)[0]

                logger.info(f"智能匹配表情包: 分数={selected['score']}, 描述={selected['desc'][:30]}")
            else:
                # 无匹配：使用反向加权随机（使用越少越容易选中）
                # 为所有候选赋予基础分数，减去使用频率惩罚
                for c in candidates:
                    # 基础分10分，减去使用相关惩罚
                    time_bonus = min((current_time - c["last_used"]) / 3600, 10)  # 时间越久加分越多，最多10分
                    use_penalty = min(c["use_count"] * 0.3, 5)  # 使用次数惩罚，最多5分
                    c["adjusted_score"] = 10 + time_bonus - use_penalty

                # 确保所有分数为正
                min_score = min(c["adjusted_score"] for c in candidates)
                if min_score < 1:
                    for c in candidates:
                        c["adjusted_score"] += (1 - min_score)

                weights = [c["adjusted_score"] for c in candidates]
                selected = random.choices(candidates, weights=weights, k=1)[0]

                logger.debug(f"未找到匹配，加权随机选择表情包（调整分数={selected['adjusted_score']:.1f}）")

            # 3. 更新使用统计
            selected_path = selected["path"]
            idx[selected_path]["last_used"] = int(current_time)
            idx[selected_path]["use_count"] = idx[selected_path].get("use_count", 0) + 1
            await self._save_index(idx)

            return selected_path

        except Exception as e:
            logger.error(f"智能选择表情包失败: {e}", exc_info=True)
            return None

    async def _send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ):
        """发送表情包和文本。"""
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

            # 添加图片
            b64 = await self.image_processor_service._file_to_base64(emoji_path)
            new_result.base64_image(b64)

            # 设置新的结果对象
            event.set_result(new_result)
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
                        b64 = await self.image_processor_service._file_to_base64(path)
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

    @filter.command("meme set_vision")
    async def set_vision(self, event: AstrMessageEvent, provider_id: str = ""):
        """设置视觉模型。"""
        async for result in self.command_handler.set_vision(event, provider_id):
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



    @filter.command("meme throttle")
    async def throttle_config(self, event: AstrMessageEvent, action: str = "", value: str = ""):
        """配置图片处理节流。用法: /meme throttle <mode|probability|interval|cooldown> <值>"""
        if not action:
            # 显示当前节流状态
            async for result in self.command_handler.throttle_status(event):
                yield result
        elif action == "mode":
            async for result in self.command_handler.set_throttle_mode(event, value):
                yield result
        elif action == "probability":
            async for result in self.command_handler.set_throttle_probability(event, value):
                yield result
        elif action == "interval":
            async for result in self.command_handler.set_throttle_interval(event, value):
                yield result
        elif action == "cooldown":
            async for result in self.command_handler.set_throttle_cooldown(event, value):
                yield result
        else:
            yield event.plain_result("用法: /meme throttle <mode|probability|interval|cooldown> <值>\n或 /meme throttle 查看状态")



    @filter.command("meme task")
    async def task_config(self, event: AstrMessageEvent, task_type: str = "", action: str = "", value: str = ""):
        """配置后台任务。用法: /meme task <cleanup|capacity> <on|off|interval> [值]"""
        if not task_type:
            # 显示任务状态（已合并到 status 中）
            yield event.plain_result("使用 /meme status 查看任务状态")
            return

        if task_type == "cleanup":
            if action == "on" or action == "off":
                async for result in self.command_handler.toggle_raw_cleanup(event, action):
                    yield result
            elif action == "interval":
                async for result in self.command_handler.set_raw_cleanup_interval(event, value):
                    yield result
            else:
                yield event.plain_result("用法: /meme task cleanup <on|off|interval> [分钟数]")
        elif task_type == "capacity":
            if action == "on" or action == "off":
                async for result in self.command_handler.toggle_capacity_control(event, action):
                    yield result
            elif action == "interval":
                async for result in self.command_handler.set_capacity_control_interval(event, value):
                    yield result
            else:
                yield event.plain_result("用法: /meme task capacity <on|off|interval> [分钟数]")
        elif task_type == "status":
            # 显示任务状态 - 重定向到主状态命令
            yield event.plain_result("请使用 /meme status 查看完整状态信息")
        else:
            yield event.plain_result("用法: /meme task <cleanup|capacity> <on|off|interval> [值]")

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

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme debug_image")
    async def debug_image(self, event: AstrMessageEvent):
        """调试命令：处理当前消息中的图片并显示详细信息"""
        async for result in self.command_handler.debug_image(event):
            yield result

    @filter.command("meme list")
    async def list_images(self, event: AstrMessageEvent, category: str = "", limit: str = "10"):
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

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme reload_persona")
    async def reload_persona(self, event: AstrMessageEvent):
        """重新注入人格情绪选择提醒。用法: /meme reload_persona"""
        try:
            await self._reload_personas()
            yield event.plain_result("✅ 已重新注入人格情绪选择提醒")
        except Exception as e:
            logger.error(f"重新注入人格失败: {e}")
            yield event.plain_result(f"❌ 重新注入人格失败: {e}")

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme persona_status")
    async def persona_status(self, event: AstrMessageEvent):
        """检查人格注入状态。用法: /meme persona_status"""
        try:
            personas = self.context.provider_manager.personas

            status_text = "人格注入状态:\n"

            for i, persona in enumerate(personas):
                current_prompt = persona.get("prompt", "")
                persona_name = persona.get("name", f"人格{i+1}")

                if self._persona_marker in current_prompt:
                    status_text += f"✅ {persona_name}: 已注入\n"
                else:
                    status_text += f"❌ {persona_name}: 未注入\n"

            status_text += f"\n备份状态: {'✅ 正常' if self.persona_backup else '❌ 无备份'}"
            status_text += f"\n注入标记: `{self._persona_marker}`"

            yield event.plain_result(status_text)
        except Exception as e:
            logger.error(f"检查人格状态失败: {e}")
            yield event.plain_result(f"❌ 检查人格状态失败: {e}")

    @filter.llm_tool(name="send_emoji")
    async def send_emoji(self, event: AstrMessageEvent, query: str):
        """发送匹配的表情包图片。

        Args:
            query (str): 表情包搜索关键词（情绪词或描述）

        使用规则：
        - 查询词必须是情绪词或具体描述（如：开心、无奈、困了、大笑）
        - 工具会直接发送最佳匹配的表情包
        """
        logger.info(f"[Tool] LLM 请求发送表情包: {query}")

        # 标记为主动发送，避免被动模式重复触发
        event.set_extra("stealer_active_sent", True)

        try:
            if not self.cache_service.get_cache("index_cache"):
                logger.debug("索引未加载，正在加载...")
                await self._load_index()

            idx = self.cache_service.get_cache("index_cache") or {}
            results = await self.image_processor_service.search_images(query, limit=5, idx=idx)

            if not results:
                logger.warning(f"未找到匹配的表情包: {query}")
                yield event.plain_result(f"💡 图库中暂无关于'{query}'的表情包")
                return

            # 发送最佳匹配（第一个）
            best_path, best_desc, best_emotion = results[0]
            if not os.path.exists(best_path):
                logger.warning(f"最佳匹配表情包文件不存在: {best_path}")
                yield event.plain_result(f"💡 表情包文件丢失，请检查图库")
                return

            logger.info(f"[Tool] 直接发送表情包: {best_path} (emotion={best_emotion})")

            # 发送表情包（不带文本，LLM 的回复单独发送）
            b64 = await self.image_processor_service._file_to_base64(best_path)
            yield event.make_result().message("").base64_image(b64)

            # 打印候选列表供调试/LLM 参考
            for i, (path, desc, emotion) in enumerate(results[:5]):
                if os.path.exists(path):
                    logger.debug(f"[Tool] 候选{i+1}: [{emotion}] {desc[:20]}")

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield event.plain_result("⚠️ 发送表情包时出错")
