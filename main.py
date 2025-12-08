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
from astrbot.core.utils.astrbot_path import get_astrbot_data_path, get_astrbot_root

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
        "speechless",  # 无语分类
    ]

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)

        # 初始化基础路径
        self.base_dir: Path = StarTools.get_data_dir("astrbot_plugin_stealer")
        self.config_path: Path = self.base_dir / "config.json"
        self.raw_dir: Path = self.base_dir / "raw"
        self.categories_dir: Path = self.base_dir / "categories"
        self.cache_dir: Path = self.base_dir / "cache"

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
        self.maintenance_interval = self.config_service.maintenance_interval
        self.steal_emoji = self.config_service.steal_emoji
        self.content_filtration = self.config_service.content_filtration

        self.vision_provider_id = self.config_service.vision_provider_id
        self.raw_retention_hours = self.config_service.raw_retention_hours
        self.raw_clean_interval = self.config_service.raw_clean_interval

        # 添加缺失的兼容mainv2的配置项

        self.max_raw_emoji_size = getattr(
            self.config_service, "max_raw_emoji_size", 3 * 1024 * 1024
        )
        self.steal_type = getattr(self.config_service, "steal_type", "both")

        # 获取分类列表
        self.categories = self.config_service.categories

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

    # _clean_cache方法已迁移到CacheService类

    def _update_config_from_dict(self, config_dict: dict):
        """从配置字典更新插件配置。"""
        if not config_dict:
            return

        try:
            # 使用配置服务更新配置
            if self.config_service:
                self.config_service.update_config_from_dict(config_dict)

                # 同步更新实例属性以保持兼容性
                self.auto_send = self.config_service.get_config("auto_send")
                self.emoji_chance = self.config_service.get_config("emoji_chance")
                self.max_reg_num = self.config_service.get_config("max_reg_num")
                self.do_replace = self.config_service.get_config("do_replace")
                self.maintenance_interval = self.config_service.get_config(
                    "maintenance_interval"
                )
                self.content_filtration = self.config_service.get_config(
                    "content_filtration"
                )

                self.vision_provider_id = (
                    str(self.config_service.get_config("vision_provider_id"))
                    if self.config_service.get_config("vision_provider_id")
                    else None
                )
                self.raw_retention_hours = self.config_service.get_config(
                    "raw_retention_hours"
                )
                self.raw_clean_interval = self.config_service.get_config(
                    "raw_clean_interval"
                )

                # 更新兼容mainv2的配置属性

                self.max_raw_emoji_size = config_dict.get(
                    "max_raw_emoji_size",
                    getattr(self.config_service, "max_raw_emoji_size", 3 * 1024 * 1024),
                )
                self.steal_type = config_dict.get(
                    "steal_type", getattr(self.config_service, "steal_type", "both")
                )

                # 更新分类列表
                self.categories = (
                    self.config_service.get_config("categories") or self.CATEGORIES
                )

                # 更新其他服务的配置
                self.image_processor_service.update_config(
                    categories=self.categories,
                    content_filtration=self.content_filtration,
                    vision_provider_id=self.vision_provider_id,
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
                else:
                    logger.warning(f"提示词文件不存在: {prompts_path}")
            except Exception as e:
                logger.error(f"加载提示词文件失败: {e}")

            # 加载配置
            self.auto_send = self.config_service.get_config("auto_send")
            self.emoji_chance = self.config_service.get_config("emoji_chance")
            self.max_reg_num = self.config_service.get_config("max_reg_num")
            self.do_replace = self.config_service.get_config("do_replace")
            self.maintenance_interval = self.config_service.get_config(
                "maintenance_interval"
            )
            self.content_filtration = self.config_service.get_config(
                "content_filtration"
            )

            self.vision_provider_id = (
                str(self.config_service.get_config("vision_provider_id"))
                if self.config_service.get_config("vision_provider_id")
                else None
            )
            self.raw_retention_hours = self.config_service.get_config(
                "raw_retention_hours"
            )
            self.raw_clean_interval = self.config_service.get_config(
                "raw_clean_interval"
            )
            self.categories = (
                self.config_service.get_config("categories") or self.CATEGORIES
            )

            # 初始化子目录
            for category in self.categories:
                (self.categories_dir / category).mkdir(parents=True, exist_ok=True)

            # 启动扫描任务
            self.task_scheduler.create_task("scanner_loop", self._scanner_loop())

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

            # 使用任务调度器停止扫描任务
            self.task_scheduler.cancel_task("scanner_loop")

            # 清理各服务资源
            self.cache_service.cleanup()
            self.task_scheduler.cleanup()
            self.config_service.cleanup()

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
                "maintenance_interval": self.maintenance_interval,
                "content_filtration": self.content_filtration,
                "vision_provider_id": self.vision_provider_id,
                "raw_retention_hours": self.raw_retention_hours,
                "raw_clean_interval": self.raw_clean_interval,
                "enabled": self.enabled,
            }

            self.config_service.update_config(config_updates)
            self.config_service.save_config()

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
            # 委托给ImageProcessorService类处理
            success, updated_idx = await self.image_processor_service.process_image(
                event=event,
                file_path=file_path,
                is_temp=is_temp,
                idx=idx,
                categories=self.categories,
                content_filtration=self.content_filtration,
                backend_tag=self.backend_tag,
            )

            # 如果没有提供索引，我们需要加载完整的索引
            if idx is None and updated_idx is not None:
                # 加载完整索引
                full_idx = await self._load_index()
                # 合并更新
                full_idx.update(updated_idx)
                return success, full_idx

            return success, updated_idx
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

    async def _classify_text_category(
        self, event: AstrMessageEvent | None, text: str
    ) -> str:
        """调用文本模型判断文本情绪并映射到插件分类。"""
        try:
            # 委托给EmotionAnalyzerService类进行文本情绪分类
            result = await self.emotion_analyzer_service.classify_text_emotion(
                event, text
            )
            return result
        except ValueError as e:
            logger.error(f"文本分类参数错误: {e}")
            return ""
        except TypeError as e:
            logger.error(f"文本分类类型错误: {e}")
            return ""
        except Exception as e:
            logger.error(f"文本情绪分类失败: {e}", exc_info=True)
            return ""

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
            astrbot_data_path = Path(get_astrbot_data_path()).resolve()
            plugin_base_dir = Path(self.base_dir).resolve()

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
            allowed_parents = [astrbot_data_path, plugin_base_dir]

            for parent in allowed_parents:
                try:
                    normalized_path.relative_to(parent)
                    is_safe = True
                    break
                except ValueError:
                    pass

            if not is_safe:
                logger.error(f"路径超出允许范围: {path} -> {normalized_path}")
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

    async def _scanner_loop(self):
        """扫描循环：定期清理文件和执行维护任务。"""
        # 委托给 EventHandler 类处理
        await self.event_handler._scanner_loop()

    # 已移除_scan_register_emoji_folder方法（扫描系统表情包目录功能，无实际用途）

    async def _clean_raw_directory(self):
        """按时间定时清理raw目录中的原始图片"""
        # 委托给 EventHandler 类处理
        await self.event_handler._clean_raw_directory()

    async def _enforce_capacity(self, idx: dict):
        """执行容量控制，删除低使用频率/旧文件。"""
        # 委托给 EventHandler 类处理
        await self.event_handler._enforce_capacity(idx)

    @filter.on_decorating_result()
    async def before_send(self, event: AstrMessageEvent, *args, **kwargs):
        if not self.auto_send or not self.base_dir:
            return
        result = event.get_result()
        # 只在有文本结果时尝试匹配表情包
        if result is None:
            return

        # 文本仅用于本地规则提取情绪关键字，不再请求额外的 LLM
        text = result.get_plain_text() or event.get_message_str()
        if not text or not text.strip():
            logger.debug("没有可处理的文本内容，未触发图片发送")
            return

        emotions, cleaned_text = await self._extract_emotions_from_text(event, text)

        # 先执行标签清理，无论是否发送表情包都需要清理标签
        if cleaned_text != text:
            # 创建新的结果对象并更新内容
            new_result = event.make_result().set_result_content_type(
                result.result_content_type
            )

            # 添加除了Plain文本外的其他组件
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            # 添加清除标签后的文本
            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            # 设置新的结果对象
            event.set_result(new_result)

            # 更新result和text变量，使用清理后的结果
            result = new_result
            text = cleaned_text

        # 如果没有情绪标签，不需要继续处理图片发送
        if not emotions:
            logger.debug("未从文本中提取到情绪关键词，未触发图片发送")
            return

        # 只有在有情绪标签时才检查发送概率
        try:
            chance = float(self.emoji_chance)
            # 兜底保护，防止配置错误导致永远/从不触发
            if chance <= 0:
                logger.debug("表情包自动发送概率为0，未触发图片发送")
                return
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return
        except Exception:
            logger.error("解析表情包自动发送概率配置失败，未触发图片发送")
            return

        logger.debug("表情包自动发送概率检查通过，开始处理图片发送")

        logger.debug(f"提取到情绪关键词: {emotions}")

        # 目前只取第一个识别到的情绪类别
        category = emotions[0]
        cat_dir = self.base_dir / "categories" / category
        if not cat_dir.exists():
            logger.debug(f"情绪'{category}'对应的图片目录不存在，未触发图片发送")
            # 目录不存在时，仍需使用清理后的文本
            if cleaned_text != text:
                # 创建新的结果对象并更新内容
                new_result = event.make_result().set_result_content_type(
                    result.result_content_type
                )

                # 添加除了Plain文本外的其他组件
                for comp in result.chain:
                    if not isinstance(comp, Plain):
                        new_result.chain.append(comp)

                # 添加清除标签后的文本
                if cleaned_text.strip():
                    new_result.message(cleaned_text.strip())

                # 设置新的结果对象
                event.set_result(new_result)
            return

        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            logger.debug(f"情绪'{category}'对应的图片目录为空，未触发图片发送")
            # 目录为空时，仍需使用清理后的文本
            if cleaned_text != text:
                # 创建新的结果对象并更新内容
                new_result = event.make_result().set_result_content_type(
                    result.result_content_type
                )

                # 添加除了Plain文本外的其他组件
                for comp in result.chain:
                    if not isinstance(comp, Plain):
                        new_result.chain.append(comp)

                # 添加清除标签后的文本
                if cleaned_text.strip():
                    new_result.message(cleaned_text.strip())

                # 设置新的结果对象
                event.set_result(new_result)
            return

        logger.debug(f"从'{category}'目录中找到 {len(files)} 张图片")
        picked_image = random.choice(files)
        image_index = await self._load_index()
        image_record = image_index.get(picked_image.as_posix())
        if isinstance(image_record, dict):
            image_record["usage_count"] = int(image_record.get("usage_count", 0)) + 1
            image_record["last_used"] = int(asyncio.get_event_loop().time())
            image_index[picked_image.as_posix()] = image_record
            await self._save_index(image_index)
        # 创建新的结果对象并更新内容
        new_result = event.make_result().set_result_content_type(
            result.result_content_type
        )

        # 添加除了Plain文本外的其他组件
        for comp in result.chain:
            if not isinstance(comp, Plain):
                new_result.chain.append(comp)

        # 添加清除标签后的文本
        if cleaned_text.strip():
            new_result.message(cleaned_text.strip())

        # 添加图片
        base64_data = await self._file_to_base64(picked_image.as_posix())
        new_result.base64_image(base64_data)

        # 设置新的结果对象
        event.set_result(new_result)

    @filter.command("meme on")
    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        return await self.command_handler.meme_on(event)

    @filter.command("meme off")
    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        return await self.command_handler.meme_off(event)

    @filter.command("meme auto_on")
    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        return await self.command_handler.auto_on(event)

    @filter.command("meme auto_off")
    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        return await self.command_handler.auto_off(event)

    @filter.command("meme set_vision")
    async def set_vision(self, event: AstrMessageEvent, provider_id: str = ""):
        if not provider_id:
            yield event.plain_result("请提供视觉模型的 provider_id")
            return
        self.vision_provider_id = provider_id
        self._persist_config()
        yield event.plain_result(f"已设置视觉模型: {provider_id}")

    @filter.command("meme show_providers")
    async def show_providers(self, event: AstrMessageEvent):
        vp = self.vision_provider_id or "当前会话"
        yield event.plain_result(f"视觉模型: {vp}")

    @filter.command("meme status")
    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        st_on = "开启" if self.enabled else "关闭"
        st_auto = "开启" if self.auto_send else "关闭"

        idx = await self._load_index()
        # 添加视觉模型信息
        vision_model = self.vision_provider_id or "未设置（将使用当前会话默认模型）"
        yield event.plain_result(
            f"偷取: {st_on}\n自动发送: {st_auto}\n已注册数量: {len(idx)}\n概率: {self.emoji_chance}\n上限: {self.max_reg_num}\n替换: {self.do_replace}\n维护周期: {self.maintenance_interval}min\n审核: {self.content_filtration}\n视觉模型: {vision_model}"
        )

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
            record_emotion = str(record_dict.get("emotion", record_dict.get("category", "")))
            record_tags = record_dict.get("tags", [])
            if emotion and (
                emotion == record_emotion
                or (isinstance(record_tags, list) and emotion in [str(tag) for tag in record_tags])
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
            return
        if alias:
            aliases = await self._load_aliases()
            if alias in aliases:
                aliases[alias]
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
        b64 = await self._file_to_base64(pick.as_posix())
        chain = MessageChain().base64_image(b64)
        # 统一使用yield返回结果，保持交互体验一致
        yield event.result_with_message_chain(chain)

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

                # 检查路径安全性
                is_safe, safe_path = self._is_safe_path(temp_path)
                if not is_safe:
                    yield event.plain_result(f"图片 {i + 1}: 路径不安全，跳过处理")
                    continue

                temp_path = safe_path
                yield event.plain_result(f"图片 {i + 1}: 安全路径: {temp_path}")

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
