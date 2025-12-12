import asyncio
import copy
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import (
    EventMessageType,
    PermissionType,
    PlatformAdapterType,
)
from astrbot.api.message_components import Image, Plain
from astrbot.api.star import Context, Star, StarTools, register


from .cache_service import CacheService

# 导入新创建的服务类
from .command_handler import CommandHandler

# 导入原有服务类 - 使用标准的相对导入
from .config_service import ConfigService
from .emotion_analyzer_service import EmotionAnalyzerService
from .event_handler import EventHandler
from .image_processor_service import ImageProcessorService
from .task_scheduler import TaskScheduler

try:
    # 可选依赖，用于通过图片尺寸/比例进行快速过滤，未安装时自动降级
    from PIL import Image as PILImage  # type: ignore[import]
except Exception:  # pragma: no cover - 仅作为兼容分支
    PILImage = None


@register(
    "astrbot_plugin_stealer",
    "nagatoquin33",
    "自动偷取并分类表情包，在合适时机发送",
    "1.0.0",
)
class StealerPlugin(Star):
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
    CATEGORIES = [
        "happy",
        "sad",
        "angry",
        "shy",
        "surprised",
        "smirk",
        "cry",
        "confused",
        "embarrassed",
        "love",
        "disgust",
        "fear",
        "excitement",
        "tired",
        "sigh",  # 叹气分类
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)

        # 初始化基础路径
        self.base_dir: Path = StarTools.get_data_dir("astrbot_plugin_stealer")
        self.config_path: Path = self.base_dir / "config.json"
        self.raw_dir: Path = self.base_dir / "raw"
        self.categories_dir: Path = self.base_dir / "categories"
        self.cache_dir: Path = self.base_dir / "cache"

        # 设置PILImage实例属性，供ImageProcessorService使用
        self.PILImage = PILImage

        # 初始化人格注入相关属性
        # 直接集成提示词，不在设置界面显示
        self.prompt_head: str = (
            "你需要根据用户的情绪选择不同的回复方式，情绪分类包括：{categories}。"
        )
        self.prompt_tail: str = "请根据对话内容选择最合适的情绪标签，并在回复内容前添加情绪标签，标签使用&&包裹，例如：&&happy&&你好啊！"
        self.persona_backup: list = []

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
        self.event_handler = EventHandler(self)
        self.image_processor_service = ImageProcessorService(self)
        self.emotion_analyzer_service = EmotionAnalyzerService(self)
        self.task_scheduler = TaskScheduler()

        # 运行时属性
        self.backend_tag: str = self.BACKEND_TAG
        self._scanner_task: asyncio.Task | None = None

        # 验证配置
        self._validate_config()

    # _clean_cache方法已迁移到CacheService类

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

        # 同步视觉模型配置
        self.vision_provider_id = self._load_vision_provider_id()

    def _validate_config(self):
        """验证配置参数的有效性。"""
        errors = []

        if self.max_reg_num <= 0:
            errors.append("最大表情数量必须大于0")

        if not (0 <= self.emoji_chance <= 1):
            errors.append("表情发送概率必须在0-1之间")

        if self.raw_cleanup_interval < 1:
            errors.append("raw清理周期必须至少为1分钟")

        if self.capacity_control_interval < 1:
            errors.append("容量控制周期必须至少为1分钟")

        if self.raw_retention_minutes < 1:
            errors.append("raw目录保留期限必须至少为1分钟")

        if errors:
            logger.warning(f"配置验证发现问题: {'; '.join(errors)}")
            # 不抛出异常，而是使用默认值
            if self.max_reg_num <= 0:
                self.max_reg_num = 100
            if not (0 <= self.emoji_chance <= 1):
                self.emoji_chance = 0.4
            if self.raw_cleanup_interval < 1:
                self.raw_cleanup_interval = 30
            if self.capacity_control_interval < 1:
                self.capacity_control_interval = 60
            if self.raw_retention_minutes < 1:
                self.raw_retention_minutes = 60

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            # 使用配置服务更新配置
            if self.config_service:
                self.config_service.update_config_from_dict(config_dict)

                # 统一同步所有配置
                self._sync_all_config()

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
                logger.error(f"加载提示词文件失败: {e}")

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
            self.persona_backup = copy.deepcopy(personas)
            await self._reload_personas()

        except Exception as e:
            logger.error(f"初始化插件失败: {e}")
            raise

    async def terminate(self):
        """插件销毁生命周期钩子。清理任务。"""

        try:
            # 恢复人格
            personas = self.context.provider_manager.personas
            for persona, persona_backup in zip(personas, self.persona_backup):
                persona["prompt"] = persona_backup["prompt"]

            # 使用任务调度器停止所有后台任务
            self.task_scheduler.cancel_task("raw_cleanup_loop")
            self.task_scheduler.cancel_task("capacity_control_loop")

            # 清理各服务资源
            if hasattr(self, "cache_service") and self.cache_service:
                self.cache_service.cleanup()

            if hasattr(self, "task_scheduler") and self.task_scheduler:
                self.task_scheduler.cleanup()

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

    async def _reload_personas(self):
        """重新加载人格配置并注入情绪选择提醒。

        该方法会获取当前的人格配置，在原始人格的基础上添加情绪选择提醒，
        并保存原始人格的备份以便在插件终止时恢复。
        """
        try:
            from astrbot.api import logger

            # 构建情绪分类字符串
            categories_str = ", ".join(self.categories)

            # 生成系统提示添加内容
            # 替换提示词中的占位符
            head_with_categories = self.prompt_head.replace(
                "{categories}", categories_str
            )
            sys_prompt_add = f"\n\n{head_with_categories}\n{self.prompt_tail}"

            # 获取当前人格配置并注入
            personas = self.context.provider_manager.personas
            for persona, persona_backup in zip(personas, self.persona_backup):
                # 每次都从备份恢复原始状态，避免人格被多次叠加修改
                persona["prompt"] = persona_backup["prompt"]
                # 再添加自定义提示词
                persona["prompt"] += sys_prompt_add

            logger.info("已成功注入情绪选择提醒到人格配置中")
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
            # 使用缓存服务加载索引
            return self.cache_service.get_cache("index_cache") or {}
        except OSError as e:
            logger.error(f"索引文件IO错误: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"索引文件格式错误: {e}")
            return {}
        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
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
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片的方法，包括过滤、分类、存储和索引更新

        Args:
            event: 消息事件对象，可为None
            file_path: 图片文件路径
            is_temp: 是否为临时文件，处理后需要删除
            idx: 可选的索引字典，如果提供则直接使用，否则加载新的

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
                astrbot_data_path = StarTools.get_data_dir().parent.resolve()
            except Exception:
                # 备选方案：使用内部 API
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
    async def on_message(self, event: AstrMessageEvent, *args, **kwargs):
        """消息监听：偷取消息中的图片并分类存储。"""
        # 委托给 EventHandler 类处理
        await self.event_handler.on_message(event, *args, **kwargs)

    async def _raw_cleanup_loop(self):
        """raw目录清理循环任务。"""
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
                # 等待指定的控制周期
                await asyncio.sleep(max(1, int(self.capacity_control_interval)) * 60)

                # 只有当偷图功能开启且容量控制启用时才执行
                if self.steal_emoji and self.enable_capacity_control:
                    logger.info("开始执行容量控制任务")
                    image_index = await self._load_index()
                    await self.event_handler._enforce_capacity(image_index)
                    await self._save_index(image_index)
                    logger.info("容量控制任务完成")

            except asyncio.CancelledError:
                logger.info("容量控制任务已取消")
                break
            except Exception as e:
                logger.error(f"容量控制任务发生错误: {e}", exc_info=True)
                # 发生错误后继续循环
                continue

    # 已移除_scan_register_emoji_folder方法（扫描系统表情包目录功能，无实际用途）

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        await self.event_handler._clean_raw_directory()

    async def _enforce_capacity(self, idx: dict):
        """执行容量控制，删除低使用频率/旧文件。"""
        # 委托给 EventHandler 类处理
        await self.event_handler._enforce_capacity(idx)

    @filter.on_decorating_result(priority=100000)
    async def _prepare_emoji_response(self, event: AstrMessageEvent):
        """准备表情包响应的公共逻辑。

        使用高优先级(100000)确保在分段插件之前执行，
        避免标签被分段插件处理后无法识别的问题。
        """
        logger.info("[Stealer] _prepare_emoji_response 被调用")

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

            # 3. 委托给情绪分析服务处理情绪提取和标签清理
            emotions, cleaned_text = await self._extract_emotions_from_text(event, text)
            text_updated = cleaned_text != text

            # 4. 更新结果对象（清理标签）
            if text_updated:
                self._update_result_with_cleaned_text(event, result, cleaned_text)
                logger.debug("已清理情绪标签")

            # 5. 检查是否需要发送表情包
            if not emotions:
                logger.debug("未从文本中提取到情绪关键词，未触发图片发送")
                return text_updated

            # 6. 委托给事件处理器检查发送条件和发送表情包
            emoji_sent = await self._try_send_emoji(event, emotions, cleaned_text)

            return text_updated or emoji_sent

        except Exception as e:
            logger.error(f"[Stealer] 处理表情包响应时发生错误: {e}", exc_info=True)
            # 即使出错也要返回text_updated，确保标签清理生效
            return text_updated if "text_updated" in locals() else False

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

        # 2. 选择表情包
        emoji_path = await self._select_emoji(emotions[0])
        if not emoji_path:
            return False

        # 3. 更新使用次数
        await self._update_usage_count(emoji_path)

        # 4. 发送表情包
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

    async def _select_emoji(self, category: str) -> str | None:
        """选择表情包文件。"""
        cat_dir = self.base_dir / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在")
            return None

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                logger.debug(f"情绪'{category}'对应的图片目录为空")
                return None

            logger.debug(f"从'{category}'目录中找到 {len(files)} 张图片")
            picked_image = random.choice(files)
            return picked_image.as_posix()
        except Exception as e:
            logger.error(f"选择表情包失败: {e}")
            return None

    async def _update_usage_count(self, emoji_path: str):
        """更新表情包使用次数。"""
        try:
            image_index = await self._load_index()
            image_record = image_index.get(emoji_path)
            if isinstance(image_record, dict):
                old_count = int(image_record.get("usage_count", 0))
                image_record["usage_count"] = old_count + 1
                image_record["last_used"] = int(asyncio.get_event_loop().time())
                image_index[emoji_path] = image_record
                await self._save_index(image_index)
                logger.info(
                    f"已更新表情包使用次数: {Path(emoji_path).name} ({old_count} -> {image_record['usage_count']})"
                )
        except Exception as e:
            logger.error(f"更新使用次数失败: {e}")

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
        if not provider_id:
            yield event.plain_result("请提供视觉模型的 provider_id")
            return
        # 同时更新实例属性和配置服务中的值，确保同步
        self.vision_provider_id = provider_id
        self.config_service.vision_provider_id = provider_id
        self._persist_config()
        yield event.plain_result(f"已设置视觉模型: {provider_id}")

    @filter.command("meme status")
    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        st_on = "开启" if self.steal_emoji else "关闭"
        st_auto = "开启" if self.auto_send else "关闭"

        idx = await self._load_index()
        # 添加视觉模型信息
        vision_model = self.vision_provider_id or "未设置（将使用当前会话默认模型）"

        status_text = "插件状态:\n"
        status_text += f"偷取: {st_on}\n"
        status_text += f"自动发送: {st_auto}\n"
        status_text += f"已注册数量: {len(idx)}\n"
        status_text += f"概率: {self.emoji_chance}\n"
        status_text += f"上限: {self.max_reg_num}\n"
        status_text += f"替换: {self.do_replace}\n"
        status_text += f"审核: {self.content_filtration}\n"
        status_text += f"视觉模型: {vision_model}\n\n"
        status_text += "后台任务:\n"
        status_text += f"Raw清理: {'启用' if self.enable_raw_cleanup else '禁用'} ({self.raw_cleanup_interval}min)\n"
        status_text += f"容量控制: {'启用' if self.enable_capacity_control else '禁用'} ({self.capacity_control_interval}min)\n\n"
        status_text += "使用 /meme task_status 查看详细任务状态"

        yield event.plain_result(status_text)


    @filter.command("meme clean")
    async def clean(self, event: AstrMessageEvent):
        """手动清理过期的原始图片文件。"""
        async for result in self.command_handler.clean(event):
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
        if not self.base_dir:
            yield event.plain_result("插件未正确配置，缺少图片存储目录")
            return
        if alias:
            aliases = await self._load_aliases()
            if alias in aliases:
                category = aliases[alias]
            else:
                yield event.plain_result("别名不存在")
                return
        cat = category or (self.categories[0] if self.categories else "happy")
        cat_dir = self.base_dir / "categories" / cat
        if not cat_dir.exists():
            yield event.plain_result("分类不存在")
            return
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            yield event.plain_result("该分类暂无表情包")
            return
        pick = random.choice(files)
        b64 = await self.image_processor_service._file_to_base64(pick.as_posix())
        # 直接发送base64图片
        yield event.make_result().base64_image(b64)

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("meme debug_image")
    async def debug_image(self, event: AstrMessageEvent):
        """调试命令：处理当前消息中的图片并显示详细信息"""

        # 收集所有图片组件
        imgs = [comp for comp in event.message_obj.message if isinstance(comp, Image)]

        if not imgs:
            yield event.plain_result("当前消息中没有图片")
            return

        for i, img in enumerate(imgs):
            try:
                # 转换图片到临时文件路径
                temp_path = await img.convert_to_file_path()
                yield event.plain_result(f"图片 {i + 1}: 临时路径: {temp_path}")

                # 临时文件由框架创建，无需安全检查
                # 安全检查会在 process_image 中处理最终存储路径时进行

                # 确保临时文件存在且可访问
                if not os.path.exists(temp_path):
                    yield event.plain_result(f"图片 {i + 1}: 临时文件不存在，跳过处理")
                    continue

                # 使用统一的图片处理方法
                yield event.plain_result(f"图片 {i + 1}: 开始处理...")
                success, idx = await self._process_image(event, temp_path, is_temp=True)

                if success:
                    if idx:
                        await self._save_index(idx)
                        yield event.plain_result(
                            f"图片 {i + 1}: 处理成功！已保存到索引"
                        )
                        # 显示处理结果
                        for img_path, img_info in idx.items():
                            if os.path.exists(img_path):
                                yield event.plain_result(
                                    f"图片 {i + 1}: 保存路径: {img_path}"
                                )
                                yield event.plain_result(
                                    f"图片 {i + 1}: 分类: {img_info.get('category', '未知')}"
                                )
                                yield event.plain_result(
                                    f"图片 {i + 1}: 情绪: {img_info.get('emotion', '未知')}"
                                )
                                yield event.plain_result(
                                    f"图片 {i + 1}: 描述: {img_info.get('desc', '无')}"
                                )
                    else:
                        yield event.plain_result(
                            f"图片 {i + 1}: 处理成功，但没有生成索引"
                        )
                else:
                    yield event.plain_result(f"图片 {i + 1}: 处理失败")
            except Exception as e:
                yield event.plain_result(f"图片 {i + 1}: 处理出错: {str(e)}")
                logger.error(f"调试图片处理失败: {e}", exc_info=True)
