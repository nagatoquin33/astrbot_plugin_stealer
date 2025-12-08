import asyncio
import base64
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from astrbot.core.utils.astrbot_path import get_astrbot_root
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class ImageProcessorService:
    """图片处理服务类，负责处理所有与图片相关的操作。"""

    # 有效分类列表作为类常量
    VALID_CATEGORIES = ["happy", "neutral", "sad", "angry", "shy", "surprised", "smirk", "cry", "confused", "embarrassed", "sigh", "speechless"]

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        # 确保base_dir始终是字符串
        if hasattr(plugin_instance, "base_dir") and plugin_instance.base_dir is not None:
            self.base_dir = str(plugin_instance.base_dir)
        else:
            # 如果没有base_dir，尝试从context获取数据目录
            self.base_dir = None

        # 尝试从插件实例获取提示词配置，如果不存在则使用默认值
        # 表情包识别和分类的合并提示词（不包含内容过滤）
        self.emoji_classification_prompt = getattr(plugin_instance, "EMOJI_CLASSIFICATION_PROMPT",
            """你是一个图片分析专家，需要完成表情包判断和情绪分类任务。请按照以下要求进行分析：

1. 表情包判断：判断这张图片是否为聊天表情包（emoji/meme/sticker）。
   - 表情包通常具有以下特征：尺寸相对较小，画面主体清晰突出，可能包含少量文字、夸张表情或动作，宽高比通常在1:2到2:1之间，风格简洁明了。
   - 如果不是表情包，返回'非表情包'。
2. 情绪分类：如果是表情包，从以下情感标签中选择最准确的一项：{emotion_list}。

请按照以下格式返回结果：
是否为表情包|情绪分类

示例：
是|happy
否|非表情包""")
        
        # 内容过滤+表情包识别+分类的三合一合并提示词
        self.combined_analysis_prompt = getattr(plugin_instance, "COMBINED_ANALYSIS_PROMPT",
            """你是一个图片分析专家，需要同时完成内容过滤和表情包分类任务。请按照以下要求进行分析：

1. 内容过滤：请判断这张图片是否包含违反规定的内容。
   如果包含裸露、暴力、敏感或违法内容，返回'过滤不通过'。否则返回'通过'。

2. 表情包判断：如果内容过滤通过，进一步判断这张图片是否为聊天表情包（emoji/meme/sticker）。
   - 表情包通常具有以下特征：尺寸相对较小，画面主体清晰突出，可能包含少量文字、夸张表情或动作，宽高比通常在1:2到2:1之间，风格简洁明了。
   - 如果不是表情包，返回'非表情包'。
3. 情绪分类：如果是表情包，从以下情感标签中选择最准确的一项：{emotion_list}。

请按照以下格式返回结果：
过滤结果|是否为表情包|情绪分类

示例：
通过|是|happy
过滤不通过|否|none
通过|否|非表情包""")
        
        # 配置参数
        self.categories = []
        self.content_filtration = False
        self.filtration_prompt = ""
        self.vision_provider_id = ""

    def update_config(self, categories=None, content_filtration=None, filtration_prompt=None, vision_provider_id=None):
        """更新图片处理器配置。

        Args:
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            filtration_prompt: 内容过滤提示
            vision_provider_id: 视觉模型提供者ID
        """
        if categories is not None:
            self.categories = categories
        if content_filtration is not None:
            self.content_filtration = content_filtration
        if filtration_prompt is not None:
            self.filtration_prompt = filtration_prompt
        if vision_provider_id is not None:
            self.vision_provider_id = vision_provider_id



    async def process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        content_filtration: bool | None = None,
        backend_tag: str | None = None
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片：存储、分类、过滤。

        Args:
            event: 消息事件
            file_path: 图片路径
            is_temp: 是否为临时文件
            idx: 索引字典
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            backend_tag: 后端标签

        Returns:
            tuple: (是否成功, 图片索引)
        """
        # 使用传入的索引或创建空索引
        if idx is None:
            idx = {}

        base_path = Path(file_path)
        if not base_path.exists():
            logger.error(f"图片文件不存在: {file_path}")
            return False, None

        # 计算图片哈希作为唯一标识符
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        hash_val = hasher.hexdigest()

        # 检查图片是否已存在
        for k, v in idx.items():
            if isinstance(v, dict) and v.get("hash") == hash_val:
                logger.debug(f"图片已存在: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await self._safe_remove_file(file_path)
                return False, None

        # 存储图片到raw目录
        if self.base_dir:
            raw_dir = os.path.join(self.base_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
            filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
            raw_path = os.path.join(raw_dir, filename)
            if is_temp:
                shutil.move(file_path, raw_path)
            else:
                shutil.copy2(file_path, raw_path)
        else:
            raw_path = file_path

        # 过滤和分类图片（合并为一次VLM调用）
        try:
            # 调用classify_image方法，该方法已实现：
            # 1. 当开启内容过滤时，先进行内容过滤
            # 2. 然后进行表情包判断和情绪分类（合并一步完成）
            category, tags, desc, emotion = await self.classify_image(
                event=event, 
                file_path=raw_path, 
                categories=categories, 
                content_filtration=content_filtration
            )
            logger.debug(f"图片分类结果: category={category}, emotion={emotion}")

            # 处理过滤不通过的情况
            if category == "过滤不通过" or emotion == "过滤不通过":
                logger.debug(f"图片内容过滤不通过，跳过存储: {raw_path}")
                if is_temp:
                    await self._safe_remove_file(raw_path)
                return False, None

            # 处理非表情包的情况
            if category == "非表情包" or emotion == "非表情包":
                logger.debug(f"图片非表情包，跳过存储: {raw_path}")
                if is_temp:
                    await self._safe_remove_file(raw_path)
                return False, None

            # 处理有效分类结果
            if category and category in self.VALID_CATEGORIES:
                logger.debug(f"图片分类结果有效: {category}")

                # 复制图片到对应分类目录
                if self.base_dir:
                    cat_dir = os.path.join(self.base_dir, "categories", category)
                    os.makedirs(cat_dir, exist_ok=True)
                    cat_path = os.path.join(cat_dir, os.path.basename(raw_path))
                    shutil.copy2(raw_path, cat_path)

                # 更新索引
                idx[raw_path] = {
                    "hash": hash_val,
                    "category": category,
                    "created_at": int(time.time()),
                    "usage_count": 0,
                    "last_used": 0
                }

                return True, idx
            else:
                logger.warning(f"图片分类结果无效: {category}，图片将留在raw目录等待清理")
                # 分类失败时，图片留在raw目录，不添加到索引，不占用配额
                return False, None
        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            await self._safe_remove_file(raw_path)
            return False, None

    def _is_likely_emoji_by_metadata(self, file_path: str) -> bool:
        """根据图片元数据判断是否可能是表情包。

        Args:
            file_path: 图片文件路径

        Returns:
            是否可能是表情包
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = None

        if PILImage is None:
            return True  # 没有PIL时默认通过

        try:
            with PILImage.open(file_path) as img:
                width, height = img.size
                # 检查图片尺寸是否符合表情包特征
                if max(width, height) > 2000 or min(width, height) < 20:  # 降低最小尺寸限制从50到20
                    return False
                # 检查图片宽高比
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 5:
                    return False
                return True
        except Exception:
            return True

    async def classify_image(self, event: AstrMessageEvent, file_path: str, categories=None, backend_tag=None, content_filtration=None) -> tuple[str, list[str], str, str]:
        """使用视觉模型对图片进行分类并返回详细信息。

        Args:
            event: 消息事件
            file_path: 图片路径
            categories: 分类列表
            backend_tag: 后端标签
            content_filtration: 是否进行内容过滤（可选）

        Returns:
            tuple: (category, tags, desc, emotion)，其中：
                  - category: 主要分类（emotion类别）
                  - tags: 图片内容标签列表
                  - desc: 图片内容描述
                  - emotion: 情绪标签（与category相同）
        """
        try:
            # 确保file_path是绝对路径
            file_path = os.path.abspath(file_path)
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                logger.error(f"分类图片时文件不存在: {file_path}")
                fallback = categories[0] if categories else "speechless"
                return fallback, [], "", fallback

            # 先用元数据做一次快速过滤，明显不是表情图片的直接跳过
            is_likely_emoji = self._is_likely_emoji_by_metadata(file_path)
            logger.debug(f"元数据判断是否为表情包: {is_likely_emoji}")
            if not is_likely_emoji:
                return "非表情包", [], "", "非表情包"

            # 确定是否进行内容过滤
            should_filter = content_filtration if content_filtration is not None else getattr(self.plugin, "content_filtration", True)
            
            # 构建情绪类别列表字符串
            emotion_list_str = ", ".join(self.VALID_CATEGORIES)
            
            # 如果不进行内容过滤，直接进行表情包识别和分类
            if not should_filter:
                # 使用表情包识别和分类的合并提示词
                prompt = self.emoji_classification_prompt.format(emotion_list=emotion_list_str)
                
                # 调用视觉模型进行分析
                response = await self._call_vision_model(file_path, prompt)
                logger.debug(f"表情包分析原始响应: {response}")
                
                # 解析响应结果
                parts = response.strip().split('|')
                if len(parts) < 2:
                    logger.error(f"表情包分析响应格式错误: {response}")
                    fallback = categories[0] if categories else "speechless"
                    return fallback, [], "", fallback
                
                filter_result = "通过"
                is_emoji_result = parts[0].strip()
                emotion_result = parts[1].strip()
            else:
                # 使用内容过滤和表情包分析的合并提示词
                prompt = self.combined_analysis_prompt.format(
                    emotion_list=emotion_list_str
                )
                
                # 调用视觉模型进行一次性分析
                response = await self._call_vision_model(file_path, prompt)
                logger.debug(f"内容过滤和表情包分析原始响应: {response}")
                
                # 解析响应结果
                parts = response.strip().split('|')
                if len(parts) < 3:
                    logger.error(f"内容过滤和表情包分析响应格式错误: {response}")
                    fallback = categories[0] if categories else "speechless"
                    return fallback, [], "", fallback
                
                filter_result = parts[0].strip()
                is_emoji_result = parts[1].strip()
                emotion_result = parts[2].strip()


            # 处理过滤结果
            if filter_result == "过滤不通过":
                return "过滤不通过", [], "", "过滤不通过"

            # 处理表情包判断结果
            if is_emoji_result.lower() != "是" and emotion_result != "非表情包":
                return "非表情包", [], "", "非表情包"

            # 处理情绪分类结果
            if emotion_result in self.VALID_CATEGORIES:
                category = emotion_result
            else:
                # 尝试从响应中提取有效类别
                for valid_cat in self.VALID_CATEGORIES:
                    if valid_cat in response.lower():
                        category = valid_cat
                        break
                else:
                    category = "speechless"

                # 如果不是表情包，返回特定标识
                if is_emoji_result.lower() != "是":
                    return "非表情包", [], "", "非表情包"

            # 不使用VLM进行详细描述，直接返回空的描述和标签
            desc = ""
            tags = []

            # 确保返回格式一致，情绪标签与分类相同
            return category, tags, desc, category
        except Exception as e:
            logger.error(f"图片分类失败: {e}")
            fallback = categories[0] if categories else "speechless"
            return fallback, [], "", fallback

    async def _call_vision_model(self, img_path: str, prompt: str) -> str:
        """调用视觉模型的共享辅助方法。

        Args:
            img_path: 图片路径
            prompt: 提示词

        Returns:
            str: LLM响应文本
        """
        try:
            # 简单的路径处理：确保使用绝对路径
            if not os.path.isabs(img_path):
                # 如果是相对路径，检查是否有base_dir
                if self.base_dir:
                    img_path = os.path.abspath(os.path.join(self.base_dir, img_path))
                else:
                    img_path = os.path.abspath(img_path)

            if not os.path.exists(img_path):
                logger.error(f"图片文件不存在: {img_path}")
                return ""

            # 调用LLM
            # 使用getattr获取配置，避免直接访问不存在的config属性
            max_retries = int(getattr(self.plugin, "vision_max_retries", 3))
            retry_delay = float(getattr(self.plugin, "vision_retry_delay", 1.0))

            for attempt in range(max_retries):
                try:
                    # 获取当前使用的聊天提供商实例
                    from astrbot.core.provider.manager import ProviderType
                    provider = self.plugin.context.provider_manager.get_using_provider(ProviderType.CHAT_COMPLETION)
                    chat_provider_id = provider.meta().id

                    # 使用插件配置的视觉模型，如果没有则不指定模型
                    model = self.plugin.vision_model if hasattr(self.plugin, "vision_model") else None

                    logger.debug(f"使用视觉模型 {model} 处理图片")

                    # 将本地文件路径转换为file:///格式的URL
                    file_url = f"file:///{os.path.abspath(img_path)}"
                    logger.debug(f"构建的图片URL: {file_url}")

                    result = await self.plugin.context.llm_generate(
                        chat_provider_id=chat_provider_id,
                        prompt=prompt,
                        image_urls=[file_url],
                        model=model
                    )
                    if result:
                        # 获取实际的文本结果
                        llm_response_text = ""
                        if hasattr(result, "result_chain") and result.result_chain:
                            llm_response_text = result.result_chain.get_plain_text()
                        elif hasattr(result, "completion_text") and result.completion_text:
                            llm_response_text = result.completion_text
                        else:
                            llm_response_text = str(result)

                        logger.debug(f"原始LLM响应: {llm_response_text}")
                        return llm_response_text.strip()
                except Exception as e:
                    error_msg = str(e)
                    # 检查是否为限流错误
                    is_rate_limit = "429" in error_msg or "RateLimit" in error_msg or "exceeded your current request limit" in error_msg
                    if is_rate_limit:
                        logger.warning(f"视觉模型请求被限流，正在重试 ({attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"视觉模型调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                        else:
                            break
            logger.error("视觉模型调用失败，达到最大重试次数")
            return ""
        except Exception as e:
            logger.error(f"视觉模型调用失败: {e}")
            return ""





    async def _compute_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值。

        Args:
            file_path: 文件路径

        Returns:
            str: MD5哈希值
        """
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"计算哈希值失败: {e}")
            return ""

    async def _file_to_base64(self, file_path: str) -> str:
        """将文件转换为base64编码。

        Args:
            file_path: 文件路径

        Returns:
            str: base64编码
        """
        try:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"文件转换为base64失败: {e}")
            return ""

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

    async def _store_image(self, src_path: str, category: str) -> str:
        """将图片存储到指定分类目录。

        Args:
            src_path: 源图片路径
            category: 分类名称

        Returns:
            str: 存储后的图片路径
        """
        try:
            if not self.base_dir:
                logger.error("base_dir未设置，无法存储图片")
                return src_path

            # 确保分类目录存在
            cat_dir = os.path.join(self.base_dir, "categories", category)
            os.makedirs(cat_dir, exist_ok=True)

            # 复制图片到分类目录
            filename = os.path.basename(src_path)
            dest_path = os.path.join(cat_dir, filename)

            # 如果文件已存在，生成新文件名
            if os.path.exists(dest_path):
                base_name, ext = os.path.splitext(filename)
                dest_path = os.path.join(cat_dir, f"{base_name}_{int(time.time())}{ext}")

            shutil.copy2(src_path, dest_path)
            logger.debug(f"图片已存储到分类目录: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"存储图片失败: {e}")
            return src_path


