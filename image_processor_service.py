import asyncio
import base64
import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class ImageProcessorService:
    """图片处理服务类，负责处理所有与图片相关的操作。"""

    # 有效分类集合作为类常量
    VALID_CATEGORIES = {
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
        "sigh",
    }

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        # 从插件实例获取PILImage引用，避免重复导入逻辑
        self.PILImage = getattr(plugin_instance, "PILImage", None)
        # 确保base_dir始终是字符串
        if (
            hasattr(plugin_instance, "base_dir")
            and plugin_instance.base_dir is not None
        ):
            self.base_dir = str(plugin_instance.base_dir)
        else:
            # 如果没有base_dir，尝试从context获取数据目录
            self.base_dir = None

        # 图片分类结果缓存，key为图片哈希，value为分类结果元组
        self._image_cache = {}
        # 缓存过期时间（秒），默认1小时
        self._cache_expire_time = getattr(
            plugin_instance, "image_cache_expire_time", 3600
        )

        # 尝试从插件实例获取提示词配置，如果不存在则使用默认值
        # 表情包识别和分类的合并提示词（不包含内容过滤）
        self.emoji_classification_prompt = getattr(
            plugin_instance, "EMOJI_CLASSIFICATION_PROMPT",
            """# Role
你是一名资深的视觉符号学专家和互联网迷因（Meme/Emoji）分析师。通过分析图像的视觉特征、文字内容和文化语境，你能精准判断图片属性并解读其核心情绪。

# Task
请对输入的图片进行两步分析：
1. **属性判断**：判断是否为“聊天表情包”。
2. **情绪分类**：如果是表情包，从指定列表中选出最匹配的情绪。

---

# Step 1: 表情包属性判断 (Binary Classification)
请基于以下标准严格筛选。

**[判定为“是”的强特征]**：
- **视觉风格**：画风简单（线稿/卡通）、有白边（Sticker风格）、低分辨率或明显的压缩痕迹（电子包浆）、夸张的面部特写。
- **文字特征**：图片上叠加了文字（特别是粗体/描边字体），且文字旨在表达情绪或吐槽。
- **功能意图**：该图片明显是为了在IM聊天（微信/TG/Discord）中代替文字表达情感、反应或状态。
- **特殊情况**：
    - 二次元/动漫截图，如果带有字幕或表情夸张，**是**表情包。
    - 带有文字的动物/人物照片（Meme图），**是**表情包。
- **表情包通常具有以下特征**：
    - 尺寸相对较小，主要用于聊天中快速表达情绪或态度；
    - 画面主体清晰突出，通常集中在人物/卡通形象/动物或简洁抽象图案上；
    - 可能包含少量文字、夸张表情或动作来强化情绪表达；
    - 常以方图或接近方图的比例出现（宽高比通常在1:2到2:1之间）；
    - 风格简洁明了，能在短时间内传达情绪。

**[判定为“否”的排除项]**：
- **普通摄影**：风景照、无明显情绪导向的普通自拍、证件照。
- **信息截屏**：纯粹的软件界面截图、文档截图、电商商品图。
- **复杂插画**：用于展示艺术而非沟通的高清壁纸/艺术画作。
- **人物语录**：只是群友的发言记录截图，不包含任何表情或动作，通常为头像后面跟着文字。
- **注意**：如果无法确认其具有社交沟通功能，一律判定为“非表情包”。

---

# Step 2: 情绪精准分类 (Emotion Classification)
仅当Step 1判定为“是”时执行。请综合分析**面部表情**、**肢体动作**和**图片文字**。

**[分析逻辑]**：
1. **图文一致**：表情和文字情绪相同 -> 直接分类。
2. **图文冲突（重点）**：如果表情是笑脸，但文字是“想死”、“无语”，请以**整体表达的含义**为准）。
3. **模糊匹配原则**：
      - 必须从列表 `{emotion_list}` 中选择唯一标签。
      - 识别表情时，除了观察画面中人物或者动漫人物（动物）的表情之外，还需要注意图中出现的其他元素，如文字、动物、人物的动作等，这些都可能对情绪产生影响。

**[强制约束]**：
- 即使图片情绪复杂，也必须强制归类到 `{emotion_list}` 中最接近的一项，严禁输出空值。

---

请严格按照以下格式返回结果，不要添加任何额外内容：
是否为表情包|情绪分类

示例：
是|happy
否|非表情包""",
        )

        # 内容过滤+表情包识别+分类的三合一合并提示词
        self.combined_analysis_prompt = getattr(
            plugin_instance, "COMBINED_ANALYSIS_PROMPT",
            """你是一个专业的图片分析专家，特别擅长内容过滤和表情包识别分类。请按照以下要求进行分析：

1. 内容过滤：请判断这张图片是否包含违反规定的内容。
   如果包含裸露、暴力、敏感或违法内容，返回'过滤不通过'。否则返回'通过'。

2. 表情包判断：如果内容过滤通过，进一步判断这张图片是否为聊天表情包。
   # Role
   你是一名资深的视觉符号学专家和互联网迷因（Meme/Emoji）分析师。通过分析图像的视觉特征、文字内容和文化语境，你能精准判断图片属性并解读其核心情绪。

   **[判定为“是”的强特征]**：
   - **视觉风格**：画风简单（线稿/卡通）、有白边（Sticker风格）、低分辨率或明显的压缩痕迹（电子包浆）、夸张的面部特写。
   - **文字特征**：图片上叠加了文字（特别是粗体/描边字体），且文字旨在表达情绪或吐槽。
   - **功能意图**：该图片明显是为了在IM聊天（微信/TG/Discord）中代替文字表达情感、反应或状态。
   - **特殊情况**：
       - 二次元/动漫截图，如果带有字幕或表情夸张，**是**表情包。
       - 带有文字的动物/人物照片（Meme图），**是**表情包。
   - **表情包通常具有以下特征**：
       - 尺寸相对较小，主要用于聊天中快速表达情绪或态度；
       - 画面主体清晰突出，通常集中在人物/卡通形象/动物或简洁抽象图案上；
       - 可能包含少量文字、夸张表情或动作来强化情绪表达；
       - 常以方图或接近方图的比例出现（宽高比通常在1:2到2:1之间）；
       - 风格简洁明了，能在短时间内传达情绪。

   **[判定为“否”的排除项]**：
   - **普通摄影**：风景照、无明显情绪导向的普通自拍、证件照。
   - **信息截屏**：纯粹的软件界面截图、文档截图、电商商品图。
   - **复杂插画**：用于展示艺术而非沟通的高清壁纸/艺术画作。
   - **注意**：如果无法确认其具有社交沟通功能，一律判定为“非表情包”。

3. 情绪分类：如果是表情包，请进行以下操作：
   # Step 2: 情绪精准分类 (Emotion Classification)
   仅当Step 1判定为“是”时执行。请综合分析**面部表情**、**肢体动作**和**图片文字**。

   **[分析逻辑]**：
   1. **图文一致**：表情和文字情绪相同 -> 直接分类。
   2. **图文冲突（重点）**：如果表情是笑脸，但文字是“想死”、“无语”，请以**整体表达的含义**为准）。
   3. **模糊匹配原则**：
      - 必须从列表 `{emotion_list}` 中选择唯一标签。
      - 识别表情时，除了观察画面中人物或者动漫人物（动物）的表情之外，还需要注意图中出现的其他元素，如文字、动物、人物的动作等，这些都可能对情绪产生影响。

   **[强制约束]**：
   - 即使图片情绪复杂，也必须强制归类到 `{emotion_list}` 中最接近的一项，严禁输出空值。

请严格按照以下格式返回结果，不要添加任何额外内容：
过滤结果|是否为表情包|情绪分类

示例：
通过|是|happy
过滤不通过|否|none
通过|否|非表情包""",
        )

        # 配置参数
        self.categories = []
        self.content_filtration = False
        self.vision_provider_id = ""

    def update_config(
        self, categories=None, content_filtration=None, vision_provider_id=None,
        emoji_classification_prompt=None, combined_analysis_prompt=None
    ):
        """更新图片处理器配置。

        Args:
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            vision_provider_id: 视觉模型提供者ID
            emoji_classification_prompt: 表情包分类提示词
            combined_analysis_prompt: 综合分析提示词
        """
        if categories is not None:
            self.categories = categories
        if content_filtration is not None:
            self.content_filtration = content_filtration
        if vision_provider_id is not None:
            self.vision_provider_id = vision_provider_id
        if emoji_classification_prompt is not None:
            self.emoji_classification_prompt = emoji_classification_prompt
        if combined_analysis_prompt is not None:
            self.combined_analysis_prompt = combined_analysis_prompt

    async def process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        content_filtration: bool | None = None,
        backend_tag: str | None = None,
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
        hash_val = await self._compute_hash(file_path)

        # 检查图片是否已存在（索引中）
        for k, v in idx.items():
            if isinstance(v, dict) and v.get("hash") == hash_val:
                logger.debug(f"图片已存在于索引中: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await self.plugin._safe_remove_file(file_path)
                return False, None

        # 检查图片是否已存在于持久化索引中
        if hasattr(self.plugin, "cache_service"):
            persistent_idx = self.plugin.cache_service.get_cache("index_cache")
            for k, v in persistent_idx.items():
                if isinstance(v, dict) and v.get("hash") == hash_val:
                    logger.debug(f"图片已存在于持久化索引中: {hash_val}")
                    if is_temp and os.path.exists(file_path):
                        await self.plugin._safe_remove_file(file_path)
                    return False, None

        # 检查图片是否已存在于缓存中
        if hash_val in self._image_cache:
            cached_result = self._image_cache[hash_val]
            # 检查缓存是否过期
            if time.time() - cached_result["timestamp"] < self._cache_expire_time:
                logger.debug(
                    f"图片分类结果已缓存: {hash_val} -> {cached_result['category']}"
                )

                category = cached_result["category"]
                tags = cached_result["tags"]
                desc = cached_result["desc"]
                emotion = cached_result["emotion"]

                # 缓存结果处理：跳过VLM调用，直接使用缓存
                if category == "过滤不通过" or emotion == "过滤不通过":
                    logger.debug(f"图片内容过滤不通过（缓存），跳过存储: {hash_val}")
                    if is_temp and os.path.exists(file_path):
                        await self.plugin._safe_remove_file(file_path)  # 清理临时文件
                    return False, None

                # 处理非表情包的缓存结果
                if category == "非表情包" or emotion == "非表情包":
                    logger.debug(f"图片非表情包（缓存），跳过存储: {hash_val}")
                    if is_temp and os.path.exists(file_path):
                        await self.plugin._safe_remove_file(file_path)  # 清理临时文件
                    return False, None

                # 处理有效分类的缓存结果
                if category and category in self.VALID_CATEGORIES:
                    logger.debug(f"图片分类结果有效（缓存）: {category}")

                    # 存储图片到raw目录（如果还没有存储的话）
                    if self.base_dir:
                        raw_dir = os.path.join(self.base_dir, "raw")
                        os.makedirs(raw_dir, exist_ok=True)  # 确保目录存在
                        ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
                        filename = (
                            f"{int(time.time())}_{hash_val[:8]}{ext}"  # 生成唯一文件名
                        )
                        raw_path = os.path.join(raw_dir, filename)
                        if is_temp:
                            shutil.move(file_path, raw_path)  # 临时文件直接移动
                        else:
                            shutil.copy2(
                                file_path, raw_path
                            )  # 非临时文件复制（保留元数据）
                    else:
                        raw_path = file_path

                    # 复制图片到对应分类目录
                    if self.base_dir:
                        cat_dir = os.path.join(self.base_dir, "categories", category)
                        os.makedirs(cat_dir, exist_ok=True)
                        cat_path = os.path.join(cat_dir, os.path.basename(raw_path))
                        shutil.copy2(raw_path, cat_path)  # 复制到分类目录

                    # 更新图片索引（用于管理和检索）
                    idx[raw_path] = {
                        "hash": hash_val,
                        "category": category,
                        "created_at": int(time.time()),
                        "usage_count": 0,
                        "last_used": 0,
                    }
                    return True, idx
                else:
                    # 处理无法分类的缓存结果
                    logger.debug(f"图片无法分类（缓存），留在raw目录: {hash_val}")
                    if is_temp:
                        # 已经存储到raw目录，无需删除
                        pass
                    return False, None
            else:
                # 缓存过期，从缓存中移除
                del self._image_cache[hash_val]

        # 首次处理：将图片存储到raw目录
        if self.base_dir:
            raw_dir = os.path.join(self.base_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)  # 确保raw目录存在
            ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
            filename = f"{int(time.time())}_{hash_val[:8]}{ext}"  # 生成包含时间戳和哈希前缀的唯一文件名
            raw_path = os.path.join(raw_dir, filename)
            if is_temp:
                shutil.move(file_path, raw_path)  # 移动临时文件
            else:
                shutil.copy2(file_path, raw_path)  # 复制非临时文件
        else:
            raw_path = file_path

        # 过滤和分类图片（合并为一次VLM调用以提高效率）
        try:
            # 调用分类方法：包含内容过滤、表情包判断和情绪分类
            category, tags, desc, emotion = await self.classify_image(
                event=event,
                file_path=raw_path,
                categories=categories,
                content_filtration=content_filtration,
            )
            logger.debug(f"图片分类结果: category={category}, emotion={emotion}")

            # 处理内容过滤不通过的情况
            if category == "过滤不通过" or emotion == "过滤不通过":
                logger.debug(f"图片内容过滤不通过，跳过存储: {raw_path}")
                if is_temp:
                    await self.plugin._safe_remove_file(raw_path)  # 清理临时文件
                return False, None

            # 处理非表情包的情况
            if category == "非表情包" or emotion == "非表情包":
                logger.debug(f"图片非表情包，跳过存储: {raw_path}")
                if is_temp:
                    await self.plugin._safe_remove_file(raw_path)  # 清理临时文件
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

                # 更新图片索引
                idx[raw_path] = {
                    "hash": hash_val,
                    "category": category,
                    "created_at": int(time.time()),
                    "usage_count": 0,
                    "last_used": 0,
                }

                # 将结果存入缓存，避免重复处理
                self._image_cache[hash_val] = {
                    "category": category,
                    "tags": tags,
                    "desc": desc,
                    "emotion": emotion,
                    "timestamp": time.time(),
                }

                return True, idx
            else:
                logger.warning(
                    f"图片分类结果无效: {category}，图片将留在raw目录等待清理"
                )

                # 将无法分类的结果也存入缓存，避免重复处理
                self._image_cache[hash_val] = {
                    "category": category,
                    "tags": tags,
                    "desc": desc,
                    "emotion": emotion,
                    "timestamp": time.time(),
                }

                # 分类失败时，图片留在raw目录，不添加到索引，不占用配额
                return False, None
        except Exception as e:
            # 异常处理：记录详细上下文并确保资源正确释放
            error_msg = f"处理图片失败 [图片路径: {raw_path}]: {e}"
            logger.error(error_msg)
            # 确保临时文件被正确清理
            if is_temp:
                await self.plugin._safe_remove_file(raw_path)
            # 重新抛出异常，添加更多上下文信息
            raise Exception(error_msg) from e

    def _is_likely_emoji_by_metadata(self, file_path: str) -> bool:
        """根据图片元数据判断是否可能是表情包。

        Args:
            file_path: 图片文件路径

        Returns:
            是否可能是表情包
        """
        # 使用从插件实例获取的PILImage引用，避免重复导入
        if self.PILImage is None:
            return False  # 没有PIL时默认不通过，避免处理过多非表情包

        try:
            with self.PILImage.open(file_path) as img:
                width, height = img.size
                # 检查图片尺寸是否符合表情包特征
                # 表情包通常是中等大小，太小或太大都不太可能
                if (
                    max(width, height) > 1000 or min(width, height) < 50
                ):  # 提高最小尺寸限制到50，降低最大尺寸到1000
                    return False
                # 检查图片宽高比，表情包通常接近正方形
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio > 3:  # 提高宽高比限制到3，更严格筛选
                    return False
                return True
        except Exception:
            return False  # 异常时默认不通过，避免处理损坏的图片

    async def classify_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        categories=None,
        backend_tag=None,
        content_filtration=None,
    ) -> tuple[str, list[str], str, str]:
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
                error_msg = f"分类图片时文件不存在: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 先用元数据做一次快速过滤，明显不是表情图片的直接跳过
            is_likely_emoji = self._is_likely_emoji_by_metadata(file_path)
            logger.debug(f"元数据判断是否为表情包: {is_likely_emoji}")
            if not is_likely_emoji:
                return "非表情包", [], "", "非表情包"

            # 确定是否进行内容过滤
            should_filter = (
                content_filtration
                if content_filtration is not None
                else getattr(self.plugin, "content_filtration", True)
            )

            # 构建情绪类别列表字符串
            emotion_list_str = ", ".join(self.VALID_CATEGORIES)

            # 如果不进行内容过滤，直接进行表情包识别和分类
            if not should_filter:
                # 使用表情包识别和分类的合并提示词
                prompt = self.emoji_classification_prompt.format(
                    emotion_list=emotion_list_str
                )

                # 调用视觉模型进行分析
                response = await self._call_vision_model(event, file_path, prompt)
                logger.debug(f"表情包分析原始响应: {response}")

                # 解析响应结果 - 使用正则表达式提高健壮性
                pattern = r"^(是|否)\s*\|\s*([^|]+)$"
                match = re.match(pattern, response.strip())
                if not match:
                    error_msg = f"表情包分析响应格式错误: {response}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                filter_result = "通过"
                is_emoji_result = match.group(1).strip()
                emotion_result = match.group(2).strip()
            else:
                # 使用内容过滤和表情包分析的合并提示词
                prompt = self.combined_analysis_prompt.format(
                    emotion_list=emotion_list_str
                )

                # 调用视觉模型进行一次性分析
                response = await self._call_vision_model(event, file_path, prompt)
                logger.debug(f"内容过滤和表情包分析原始响应: {response}")

                # 解析响应结果 - 使用正则表达式提高健壮性
                pattern = r"^([^|]+)\s*\|\s*(是|否)\s*\|\s*([^|]+)$"
                match = re.match(pattern, response.strip())
                if not match:
                    error_msg = f"内容过滤和表情包分析响应格式错误: {response}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                filter_result = match.group(1).strip()
                is_emoji_result = match.group(2).strip()
                emotion_result = match.group(3).strip()

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
                    # 无法提取有效分类，返回空结果
                    logger.debug(f"无法从响应中提取有效分类: {response}")
                    return "", [], "", ""

                # 如果不是表情包，返回特定标识
                if is_emoji_result.lower() != "是":
                    return "非表情包", [], "", "非表情包"

            # 不使用VLM进行详细描述，直接返回空的描述和标签
            desc = ""
            tags = []

            # 确保返回格式一致，情绪标签与分类相同
            return category, tags, desc, category
        except Exception as e:
            # 添加更多上下文信息
            error_msg = f"图片分类失败 [图片路径: {file_path}]: {e}"
            logger.error(error_msg)

            # 检查错误类型，添加更明确的错误提醒
            if "未配置视觉模型" in str(e) or "vision_model" in str(e).lower():
                logger.error("请检查插件配置，确保已正确设置视觉模型(vision_model)参数")
            elif "429" in str(e) or "RateLimit" in str(e):
                logger.error("视觉模型请求被限流，请稍后再试或调整vision_max_retries和vision_retry_delay配置")
            elif "图片文件不存在" in str(e):
                logger.error("图片文件不存在，可能是文件路径错误或文件已被删除")
            else:
                logger.error("视觉模型调用失败，可能是模型配置错误或API密钥问题")

            # 根据测试要求，无法分类时返回空字符串
            return "", [], "", ""

    async def _call_vision_model(self, event: AstrMessageEvent | None, img_path: str, prompt: str) -> str:
        """调用视觉模型的共享辅助方法。

        Args:
            event: 消息事件
            img_path: 图片路径
            prompt: 提示词

        Returns:
            str: LLM响应文本

        Raises:
            Exception: 当视觉模型调用失败时抛出，包含详细的错误信息和上下文
        """
        try:
            # 路径处理和验证：使用pathlib确保跨平台兼容性
            img_path_obj = Path(img_path)
            if not img_path_obj.is_absolute():
                # 如果是相对路径，根据base_dir构建绝对路径
                if self.base_dir:
                    img_path_obj = Path(self.base_dir) / img_path
                    img_path_obj = img_path_obj.absolute()
                else:
                    img_path_obj = img_path_obj.absolute()

            img_path = str(img_path_obj)
            if not os.path.exists(img_path):
                error_msg = f"图片文件不存在: {img_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 获取重试配置：使用getattr避免直接访问不存在的属性
            max_retries = int(getattr(self.plugin, "vision_max_retries", 3))
            retry_delay = float(getattr(self.plugin, "vision_retry_delay", 1.0))

            # 实现指数退避重试机制
            for attempt in range(max_retries):
                try:
                    # 获取当前会话使用的聊天模型ID
                    chat_provider_id = None
                    if event:
                        if hasattr(event, "unified_msg_origin"):
                            umo = event.unified_msg_origin
                            chat_provider_id = await self.plugin.context.get_current_chat_provider_id(umo=umo)
                            logger.debug(f"从事件获取的聊天模型ID: {chat_provider_id}")

                    # 获取配置的视觉模型 provider_id
                    vision_provider_id = getattr(self.plugin.config_service, "vision_provider_id", None)
                    
                    # 如果配置了视觉模型，使用它；否则使用当前会话的 provider
                    if vision_provider_id:
                        chat_provider_id = vision_provider_id
                        logger.debug(f"使用配置的视觉模型 provider_id: {chat_provider_id}")
                    elif not chat_provider_id:
                        # 如果既没有配置视觉模型，也没有从事件获取到 provider，使用默认配置
                        chat_provider_id = getattr(self.plugin, "default_chat_provider_id", None)
                        logger.debug(f"使用默认聊天模型ID: {chat_provider_id}")

                    # 检查是否有可用的 provider
                    if not chat_provider_id:
                        error_msg = "未配置视觉模型(vision_provider_id)，无法进行图片分析"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # 构建图片的file:///格式URL供LLM访问
                    file_url = f"file:///{img_path}"
                    logger.debug(f"构建的图片URL: {file_url}")

                    # 调用LLM生成服务
                    # 注意：不传递 model 参数，让 provider 使用其默认模型
                    result = await self.plugin.context.llm_generate(
                        chat_provider_id=chat_provider_id,
                        prompt=prompt,
                        image_urls=[file_url],
                    )

                    if result:
                        # 提取响应文本：支持不同的响应格式
                        llm_response_text = ""
                        if hasattr(result, "result_chain") and result.result_chain:
                            # 优先从result_chain获取格式化文本
                            llm_response_text = result.result_chain.get_plain_text()
                            logger.debug(f"从result_chain获取的响应文本: {llm_response_text}")
                        elif hasattr(result, "completion_text") and result.completion_text:
                            # 其次从completion_text获取
                            llm_response_text = result.completion_text
                            logger.debug(f"从completion_text获取的响应文本: {llm_response_text}")
                        else:
                            # 最后使用字符串转换作为兜底
                            llm_response_text = str(result)
                            logger.debug(f"使用字符串转换获取的响应文本: {llm_response_text}")

                        logger.debug(f"最终处理的LLM响应: {llm_response_text}")
                        return llm_response_text.strip()
                except Exception as e:
                    error_msg = str(e)
                    # 检查是否为限流错误（HTTP 429或包含限流关键词）
                    is_rate_limit = (
                        "429" in error_msg
                        or "RateLimit" in error_msg
                        or "exceeded your current request limit" in error_msg
                    )
                    if is_rate_limit:
                        logger.warning(
                            f"视觉模型请求被限流，正在重试 ({attempt + 1}/{max_retries})"
                        )
                    else:
                        logger.error(
                            f"视觉模型调用失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                        )

                    # 指数退避策略：每次重试延迟时间翻倍
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2**attempt))
                    else:
                        # 最后一次尝试失败，抛出异常并保留原始异常上下文
                        raise Exception(
                            f"视觉模型调用失败（已重试{max_retries}次）: {e}"
                        ) from e

            # 达到最大重试次数（理论上不会到达这里，因为最后一次尝试会抛出异常）
            error_msg = f"视觉模型调用失败，达到最大重试次数（{max_retries}次）"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            # 添加上下文信息并重新抛出异常
            error_msg = f"视觉模型调用失败 [图片路径: {img_path}]: {e}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

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
        except FileNotFoundError as e:
            logger.error(f"文件不存在: {e}")
            return ""
        except PermissionError as e:
            logger.error(f"文件权限错误: {e}")
            return ""
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
                dest_path = os.path.join(
                    cat_dir, f"{base_name}_{int(time.time())}{ext}"
                )

            shutil.copy2(src_path, dest_path)
            logger.debug(f"图片已存储到分类目录: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"存储图片失败: {e}")
            return src_path

    def cleanup(self):
        """清理资源。"""
        # 清理图片缓存
        if hasattr(self, '_image_cache'):
            self._image_cache.clear()
        logger.debug("ImageProcessorService 资源已清理")
