import asyncio
import base64
import hashlib
import os
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class ImageProcessorService:
    """图片处理服务类，负责处理所有与图片相关的操作。"""

    # 分类迁移映射表（用于自动迁移旧版本数据）
    # 从前前版本迁移到新版本(17分类)
    CATEGORY_MIGRATION_MAP = {
        "smirk": "troll",  # 坏笑 -> 发癫
    }

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self.plugin_config = getattr(plugin_instance, "plugin_config", None)

        self.raw_dir = self.plugin_config.raw_dir if self.plugin_config else None
        self.categories_dir = (
            self.plugin_config.categories_dir if self.plugin_config else None
        )

        # 增强存储系统组件
        self.lifecycle_manager = None
        self.statistics_tracker = None

        # 图片分类结果缓存，key为图片哈希，value为分类结果元组
        self._image_cache = {}
        # 缓存过期时间（秒），默认1小时
        self._cache_expire_time = getattr(
            plugin_instance, "image_cache_expire_time", 3600
        )
        self._gif_base64_cache: dict[str, tuple[float, str]] = {}
        self._gif_base64_cache_expire_time = getattr(
            plugin_instance, "gif_base64_cache_expire_time", 3600
        )

        # 尝试从插件实例获取提示词配置，如果不存在则使用默认值
        # 表情包含义与场景分析提示词（统一使用）
        # 注意：正常情况下会使用 prompts.json 中的优化版本，这里只是备用
        self.emoji_classification_prompt = getattr(
            plugin_instance,
            "EMOJI_CLASSIFICATION_PROMPT",
            """# 表情包含义与场景分析专家

你是中文互联网文化专家，精通贴吧、微博、小红书等平台的梗文化和表情包使用习惯。

## 任务目标
分析表情包的情绪、语义标签和画面内容。

## 分析要求

### 1. 情绪识别
从以下情绪列表中选择最匹配的分类：`{emotion_list}`

**分析重点：**
- 面部表情和肢体语言（最重要）
- 图片中的文字内容和网络梗
- 整体传达的情绪氛围
- 幽默和讽刺意味

### 2. 语义标签
提取表情包的关键语义标签，体现网络文化特色：
- 使用网络流行语和梗
- 突出幽默、讽刺、调侃等元素
- 每个标签简洁有力
- 多个标签用逗号分隔

### 3. 画面描述
用一句话描述画面内容：
- 简洁明了，突出重点
- 描述主要人物、动作、表情
- 体现表情包的核心特征

## 输出格式
严格按照以下格式返回，使用竖线'|'分隔，不要添加任何其他内容：
情绪分类|语义标签(用逗号分隔)|画面描述(一句话)

**示例：**
happy|大笑,熊猫人,指人|熊猫人指着屏幕大笑
troll|小丑,嘲讽,阴阳怪气|卡通人物做鬼脸嘲笑

## 重要提醒
- 直接输出结果，不要解释过程
- 必须体现网络文化和梗的特色
- 语义标签要准确反映表情包含义
- 画面描述要简洁生动""",
        )

        # 带审核的表情包分析提示词（合并版本）
        self.emoji_classification_with_filter_prompt = getattr(
            plugin_instance,
            "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT",
            """# 表情包内容审核与分析专家

你是中文互联网文化专家，精通贴吧、微博、小红书等平台的梗文化和表情包使用习惯。

## 第一步：内容审核
首先判断图片是否包含不当内容：
- 如果包含裸露、暴力、血腥、政治敏感或违法内容，直接返回'过滤不通过'
- 否则继续进行表情包分析

## 第二步：表情包分析（仅在通过审核后）

### 1. 情绪识别
从以下情绪列表中选择最匹配的分类：`{emotion_list}`

### 2. 语义标签
提取表情包的关键语义标签，体现网络文化特色：
- 使用网络流行语和梗
- 突出幽默、讽刺、调侃等元素
- 每个标签简洁有力
- 多个标签用逗号分隔

### 3. 画面描述
用一句话描述画面内容：
- 简洁明了，突出重点
- 描述主要人物、动作、表情
- 体现表情包的核心特征

## 输出格式
- 如果内容审核不通过，只返回：过滤不通过
- 如果审核通过，严格按照以下格式返回，使用竖线'|'分隔：
情绪分类|语义标签(用逗号分隔)|画面描述(一句话)

**示例：**
happy|大笑,熊猫人,指人|熊猫人指着屏幕大笑
troll|小丑,嘲讽,阴阳怪气|卡通人物做鬼脸嘲笑

## 重要提醒
- 直接输出结果，不要解释过程
- 必须体现网络文化和梗的特色
- 语义标签要准确反映表情包含义
- 画面描述要简洁生动""",
        )

        # 保留combined_analysis_prompt作为备用（现在基本不使用）
        self.combined_analysis_prompt = self.emoji_classification_prompt

        # 配置参数
        self.categories = list(
            getattr(self.plugin_config, "categories", []) or []
            if self.plugin_config
            else []
        )
        self.content_filtration = False
        self.vision_provider_id = ""

        # 执行自动迁移检查（在插件启动时运行一次）
        self._auto_migrate_categories()

    def _auto_migrate_categories(self):
        """自动迁移旧版本分类到新分类系统。

        该方法会：
        1. 扫描 categories 目录下的所有旧分类文件夹
        2. 根据 CATEGORY_MIGRATION_MAP 迁移文件和索引数据
        3. 删除空的旧分类文件夹
        4. 确保新分类文件夹存在
        """
        categories_dir = self.categories_dir
        if not categories_dir or not categories_dir.exists():
            return

        migrated_files = 0

        migrated_indices = 0

        # 遍历所有旧分类，执行迁移
        for old_category, new_category in self.CATEGORY_MIGRATION_MAP.items():
            old_dir = categories_dir / old_category
            if not old_dir.exists():
                continue

            new_dir = self.plugin_config.ensure_category_dir(new_category)

            # 迁移图片文件
            for img_file in old_dir.glob("*"):
                if img_file.is_file() and img_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".webp",
                ]:
                    target_path = new_dir / img_file.name
                    # 避免文件名冲突
                    if target_path.exists():
                        stem = img_file.stem
                        suffix = img_file.suffix
                        counter = 1
                        while target_path.exists():
                            target_path = new_dir / f"{stem}_migrated{counter}{suffix}"
                            counter += 1

                    try:
                        shutil.move(str(img_file), str(target_path))
                        migrated_files += 1
                    except Exception as e:
                        logger.error(f"迁移文件失败 {img_file} -> {target_path}: {e}")

            # 迁移索引数据
            old_index = old_dir / "index.json"
            if old_index.exists():
                try:
                    import json

                    with open(old_index, encoding="utf-8") as f:
                        old_data = json.load(f)

                    # 更新分类字段
                    for item in old_data:
                        if (
                            isinstance(item, dict)
                            and item.get("category") == old_category
                        ):
                            item["category"] = new_category

                    # 合并到新索引
                    new_index = new_dir / "index.json"
                    if new_index.exists():
                        with open(new_index, encoding="utf-8") as f:
                            new_data = json.load(f)
                        new_data.extend(old_data)
                    else:
                        new_data = old_data

                    with open(new_index, "w", encoding="utf-8") as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=2)

                    old_index.unlink()
                    migrated_indices += len(old_data)
                except Exception as e:
                    logger.error(f"迁移索引失败 {old_index}: {e}")

            # 删除空的旧文件夹
            try:
                if old_dir.exists() and not any(old_dir.iterdir()):
                    old_dir.rmdir()
                    logger.info(f"已删除空分类文件夹: {old_category}")
            except Exception as e:
                logger.warning(f"删除文件夹失败 {old_dir}: {e}")

        # 确保所有新分类文件夹存在
        for category in self.categories:
            self.plugin_config.ensure_category_dir(category)

        if migrated_files > 0 or migrated_indices > 0:
            logger.info(
                f"分类迁移完成: 迁移 {migrated_files} 个文件, {migrated_indices} 条索引记录"
            )

    def update_config(
        self,
        categories=None,
        content_filtration=None,
        vision_provider_id=None,
        emoji_classification_prompt=None,
        combined_analysis_prompt=None,
        emoji_classification_with_filter_prompt=None,
    ):
        """更新图片处理器配置。

        Args:
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            vision_provider_id: 视觉模型提供者ID
            emoji_classification_prompt: 表情包分类提示词
            combined_analysis_prompt: 综合分析提示词
            emoji_classification_with_filter_prompt: 带审核的表情包分析提示词
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
        if emoji_classification_with_filter_prompt is not None:
            self.emoji_classification_with_filter_prompt = (
                emoji_classification_with_filter_prompt
            )

    async def process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        content_filtration: bool | None = None,
        backend_tag: str | None = None,
        is_platform_emoji: bool = False,
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
            is_platform_emoji: 是否为平台标记的表情包

        Returns:
            tuple: (是否成功, 图片索引)
        """
        # 使用传入的索引或创建空索引
        if idx is None:
            idx = {}

        base_path = Path(file_path)
        if not base_path.exists():
            logger.warning(f"图片文件不存在: {file_path}")
            return False, None

        # 计算图片哈希作为唯一标识符
        hash_val = await self._compute_hash(file_path)

        # 使用简化的处理流程
        return await self._process_image_legacy(
            event,
            file_path,
            is_temp,
            idx,
            categories,
            content_filtration,
            backend_tag,
            hash_val,
            is_platform_emoji,
        )

    async def _process_image_legacy(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool,
        idx: dict[str, Any],
        categories: list[str] | None,
        content_filtration: bool | None,
        backend_tag: str | None,
        hash_val: str,
        is_platform_emoji: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        """使用原有系统处理图片（保持向后兼容）

        Args:
            is_platform_emoji: 是否为平台标记的表情包，如果是则跳过元数据过滤
        """
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

            # 检查图片是否在黑名单中
            blacklist = self.plugin.cache_service.get_cache("blacklist_cache")
            if blacklist and hash_val in blacklist:
                logger.debug(f"图片在黑名单中，跳过存储: {hash_val}")
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
                        await self.plugin._safe_remove_file(file_path)
                    return False, None

                # 处理非表情包的缓存结果
                if category == "非表情包" or emotion == "非表情包":
                    logger.debug(f"图片非表情包（缓存），跳过存储: {hash_val}")
                    if is_temp and os.path.exists(file_path):
                        await self.plugin._safe_remove_file(file_path)
                    return False, None

                # 处理有效分类的缓存结果
                if category and category in self.categories:
                    logger.debug(f"图片分类结果有效（缓存）: {category}")

                    # 存储图片到raw目录（如果还没有存储的话）
                    raw_dir = self.plugin_config.ensure_raw_dir()
                    if raw_dir:
                        base_path = Path(file_path)
                        ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
                        filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
                        raw_path = str(raw_dir / filename)
                        if is_temp:
                            shutil.move(file_path, raw_path)
                        else:
                            shutil.copy2(file_path, raw_path)
                    else:
                        raw_path = file_path

                    # 复制图片到对应分类目录
                    cat_dir = self.plugin_config.ensure_category_dir(category)
                    if cat_dir:
                        cat_path = str(cat_dir / os.path.basename(raw_path))

                        # 检查文件是否仍然存在
                        if not os.path.exists(raw_path):
                            logger.warning(
                                f"原始文件已不存在（缓存分支），可能被清理: {raw_path}"
                            )
                            return False, None

                        try:
                            shutil.copy2(raw_path, cat_path)
                        except FileNotFoundError:
                            logger.warning(
                                f"复制文件时发现文件已被删除（缓存分支）: {raw_path}"
                            )
                            return False, None

                    # 图片已成功分类，立即删除raw目录中的原始文件
                    try:
                        if os.path.exists(raw_path):
                            await self.plugin._safe_remove_file(raw_path)
                            logger.debug(f"已删除已分类的原始文件（缓存）: {raw_path}")
                    except Exception as e:
                        logger.warning(
                            f"删除已分类的原始文件失败（缓存）: {raw_path}, 错误: {e}"
                        )

                    # 更新图片索引（使用分类文件路径）
                    idx[cat_path] = {
                        "hash": hash_val,
                        "category": category,
                        "created_at": int(time.time()),
                    }
                    return True, idx
                else:
                    # 处理无法分类的缓存结果
                    logger.debug(f"图片无法分类（缓存），留在raw目录: {hash_val}")
                    return False, None
            else:
                # 缓存过期，从缓存中移除
                del self._image_cache[hash_val]

        # 首次处理：将图片存储到raw目录
        raw_dir = self.plugin_config.ensure_raw_dir()
        if raw_dir:
            base_path = Path(file_path)
            ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
            filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
            raw_path = str(raw_dir / filename)
            if is_temp:
                shutil.move(file_path, raw_path)
            else:
                shutil.copy2(file_path, raw_path)
        else:
            raw_path = file_path

        # 过滤和分类图片（合并为一次VLM调用以提高效率）
        try:
            # 统一调用VLM分类，不再区分平台标记
            # 移除元数据预过滤后，所有图片都直接进入VLM分析
            category, tags, desc, emotion, scenes = await self.classify_image(
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
                    await self.plugin._safe_remove_file(raw_path)
                return False, None

            # 处理非表情包的情况
            if category == "非表情包" or emotion == "非表情包":
                logger.debug(f"图片非表情包，跳过存储: {raw_path}")
                if is_temp:
                    await self.plugin._safe_remove_file(raw_path)
                return False, None

            # 处理有效分类结果
            if category and category in self.categories:
                logger.debug(f"图片分类结果有效: {category}")

                # 检查文件是否仍然存在（可能被清理任务删除）
                if not os.path.exists(raw_path):
                    logger.warning(f"原始文件已不存在，可能被清理任务删除: {raw_path}")
                    return False, None

                # 复制图片到对应分类目录
                cat_dir = self.plugin_config.ensure_category_dir(category)
                if cat_dir:
                    cat_path = str(cat_dir / os.path.basename(raw_path))

                    try:
                        shutil.copy2(raw_path, cat_path)
                    except FileNotFoundError:
                        logger.warning(f"复制文件时发现文件已被删除: {raw_path}")
                        return False, None

                # 图片已成功分类，立即删除raw目录中的原始文件
                # 这样可以避免raw目录积压大量文件
                try:
                    if os.path.exists(raw_path):
                        await self.plugin._safe_remove_file(raw_path)
                        logger.debug(f"已删除已分类的原始文件: {raw_path}")
                except Exception as e:
                    logger.warning(f"删除已分类的原始文件失败: {raw_path}, 错误: {e}")

                # 更新图片索引（使用分类文件路径而不是raw路径）
                idx[cat_path] = {
                    "hash": hash_val,
                    "category": category,
                    "tags": tags,
                    "desc": desc,
                    "scenes": scenes,  # 新增：适用场景
                    "created_at": int(time.time()),
                }

                # 将结果存入缓存，避免重复处理
                self._image_cache[hash_val] = {
                    "category": category,
                    "tags": tags,
                    "desc": desc,
                    "emotion": emotion,
                    "scenes": scenes,  # 新增：适用场景
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
                    "scenes": scenes,  # 新增：适用场景
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

    def _build_emotion_list_str(self, categories: list[str] | None = None) -> str:
        categories = categories if categories is not None else (self.categories or [])
        categories = [c for c in categories if isinstance(c, str) and c.strip()]
        info_map = {}
        try:
            if self.plugin_config:
                info_map = getattr(self.plugin_config, "category_info", {}) or {}
            else:
                info_map = {}
        except Exception:
            info_map = {}

        lines = []
        for raw_key in categories:
            key = raw_key.strip()
            info = info_map.get(key)
            if isinstance(info, dict):
                name = str(info.get("name", "")).strip()
                desc = str(info.get("desc", "")).strip()
            else:
                name = ""
                desc = ""

            if name and name != key:
                if desc:
                    lines.append(f"{key} - {name}：{desc}")
                else:
                    lines.append(f"{key} - {name}")
            else:
                if desc:
                    lines.append(f"{key}：{desc}")
                else:
                    lines.append(key)

        if lines:
            return "\n".join(lines)
        return ", ".join(categories)

    async def classify_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        categories=None,
        backend_tag=None,
        content_filtration=None,
    ) -> tuple[str, list[str], str, str, list[str]]:
        """使用视觉模型对图片进行分类并返回详细信息。

        Args:
            event: 消息事件
            file_path: 图片路径
            categories: 分类列表
            backend_tag: 后端标签
            content_filtration: 是否进行内容过滤（可选）

        Returns:
            tuple: (category, tags, desc, emotion, scenes)，其中：
                  - category: 主要分类（emotion类别）
                  - tags: 语义标签列表
                  - desc: 画面描述（一句话）
                  - emotion: 情绪标签（与category相同）
                  - scenes: 适用场景列表（新格式下为空列表）
        """
        try:
            # 确保file_path是绝对路径
            file_path = os.path.abspath(file_path)

            # 检查文件是否存在
            if not os.path.exists(file_path):
                error_msg = f"分类图片时文件不存在: {file_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            # 移除元数据预过滤，直接使用VLM进行准确判断
            # 原因：现在有了更准确的表情包识别方法，不需要基于图片尺寸的粗糙过滤
            logger.debug("跳过元数据过滤，直接使用VLM进行表情包判断")

            # 确定是否进行内容过滤
            should_filter = (
                content_filtration
                if content_filtration is not None
                else getattr(self.plugin, "content_filtration", True)
            )

            prompt_categories = categories if isinstance(categories, list) else None
            emotion_list_str = self._build_emotion_list_str(prompt_categories)

            # 根据是否开启审核选择合适的提示词
            if should_filter:
                # 使用合并的审核+分析提示词，一次调用完成
                prompt = self.emoji_classification_with_filter_prompt.format(
                    emotion_list=emotion_list_str
                )
            else:
                # 使用纯分析提示词
                prompt = self.emoji_classification_prompt.format(
                    emotion_list=emotion_list_str
                )

            # 调用视觉模型进行分析
            response = await self._call_vision_model(event, file_path, prompt)
            logger.debug(f"VLM响应: {response}")

            # 处理审核不通过的情况
            if "过滤不通过" in response:
                logger.warning(f"图片内容审核不通过: {file_path}")
                return "过滤不通过", [], "", "过滤不通过", []

            # 统一的响应格式: 情绪分类|语义标签(用逗号分隔)|画面描述(一句话)
            parts = [p.strip() for p in response.strip().split("|")]

            emotion_result = (
                parts[0]
                if len(parts) > 0
                else (self.categories[0] if self.categories else "happy")
            )

            # 语义标签作为tags
            tags_str = parts[1] if len(parts) > 1 else ""
            tags_result = [t.strip() for t in tags_str.split(",") if t.strip()]

            # 画面描述作为desc
            desc_result = parts[2] if len(parts) > 2 else "表情包"

            # 适用场景设为空（新格式不再包含场景）
            scenes_result = []

            # 新逻辑：既然移除了元数据过滤，假设输入的都是表情包
            # 只需要处理情绪分类结果
            normalized = ""
            if self.plugin_config and hasattr(
                self.plugin_config, "normalize_category_strict"
            ):
                try:
                    normalized = self.plugin_config.normalize_category_strict(
                        emotion_result
                    )
                except Exception:
                    normalized = ""

            if normalized and normalized in self.categories:
                category = normalized
            else:
                logger.warning(
                    f"无法从响应中提取有效情绪分类: {emotion_result}，使用默认分类"
                )
                category = self.categories[0] if self.categories else "happy"

            return category, tags_result, desc_result, category, scenes_result

        except Exception as e:
            # 添加更多上下文信息
            error_msg = f"图片分类失败 [图片路径: {file_path}]: {e}"
            logger.error(error_msg)

            # 检查错误类型，添加更明确的错误提醒
            if "未配置视觉模型" in str(e) or "vision_model" in str(e).lower():
                logger.error("请检查插件配置，确保已正确设置视觉模型(vision_model)参数")
            elif "429" in str(e) or "RateLimit" in str(e):
                logger.error(
                    "视觉模型请求被限流，请稍后再试或调整vision_max_retries和vision_retry_delay配置"
                )
            elif "图片文件不存在" in str(e):
                logger.error("图片文件不存在，可能是文件路径错误或文件已被删除")
            else:
                logger.error("视觉模型调用失败，可能是模型配置错误或API密钥问题")

            # 根据测试要求，无法分类时返回空字符串
            return "", [], "", "", []

    async def _call_vision_model(
        self, event: AstrMessageEvent | None, img_path: str, prompt: str
    ) -> str:
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
                if self.plugin_config and getattr(self.plugin_config, "data_dir", None):
                    img_path_obj = (
                        Path(self.plugin_config.data_dir) / img_path
                    ).absolute()
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
                            chat_provider_id = (
                                await self.plugin.context.get_current_chat_provider_id(
                                    umo=umo
                                )
                            )
                            logger.debug(f"从事件获取的聊天模型ID: {chat_provider_id}")

                    # 获取配置的视觉模型 provider_id
                    vision_provider_id = (
                        getattr(self.plugin_config, "vision_provider_id", None)
                        if self.plugin_config
                        else None
                    )

                    # 如果配置了视觉模型，使用它；否则使用当前会话的 provider
                    if vision_provider_id:
                        chat_provider_id = vision_provider_id
                        logger.debug(
                            f"使用配置的视觉模型 provider_id: {chat_provider_id}"
                        )
                    elif not chat_provider_id:
                        # 如果既没有配置视觉模型，也没有从事件获取到 provider，使用默认配置
                        chat_provider_id = getattr(
                            self.plugin, "default_chat_provider_id", None
                        )
                        logger.debug(f"使用默认聊天模型ID: {chat_provider_id}")

                    # 检查是否有可用的 provider
                    if not chat_provider_id:
                        error_msg = (
                            "未配置视觉模型(vision_provider_id)，无法进行图片分析"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # 根据AstrBot开发文档，使用正确的VLM调用方式
                    logger.debug(f"准备调用VLM，图片路径: {img_path}")
                    logger.debug(f"准备调用VLM，provider_id: {chat_provider_id}")

                    # 根据开发文档，构建包含文本和图片的消息
                    from astrbot.api.message_components import Image, Plain

                    # 构建消息链
                    message_items = [Plain(text=prompt), Image.fromFileSystem(img_path)]

                    # 方法1：尝试使用Context的AI服务调用
                    try:
                        # 检查是否有直接的AI调用方法
                        if hasattr(self.plugin.context, "llm_generate"):
                            logger.debug("使用context.llm_generate方法")
                            # 使用file://协议传递本地图片路径
                            file_url = f"file:///{img_path.replace(chr(92), '/')}"  # 处理Windows路径
                            result = await self.plugin.context.llm_generate(
                                chat_provider_id=chat_provider_id,
                                prompt=prompt,
                                image_urls=[file_url],
                            )
                            logger.debug("context.llm_generate调用成功")
                        else:
                            raise AttributeError("context.llm_generate方法不存在")

                    except Exception as context_error:
                        logger.warning(f"Context方法调用失败: {context_error}")

                        # 方法2：尝试直接使用provider
                        try:
                            logger.debug("尝试直接使用provider")
                            provider_manager = self.plugin.context.provider_manager

                            # 检查provider是否存在
                            if (
                                not hasattr(provider_manager, "providers")
                                or chat_provider_id not in provider_manager.providers
                            ):
                                raise ValueError(
                                    f"Provider {chat_provider_id} 不存在。请检查配置。"
                                )

                            provider = provider_manager.providers[chat_provider_id]

                            # 检查provider是否支持文本聊天
                            if not hasattr(provider, "text_chat"):
                                raise ValueError(
                                    f"Provider {chat_provider_id} 不支持文本聊天功能。"
                                )

                            # 创建模拟消息对象
                            class MockMessage:
                                def __init__(self, text, image_path):
                                    self.message_str = text
                                    self.message_chain = message_items
                                    self.sender_id = "vision_analysis"
                                    self.session_id = "vision_analysis"
                                    self.unified_msg_origin = None

                            mock_message = MockMessage(prompt, img_path)

                            # 调用provider的text_chat方法
                            result = await provider.text_chat.text_chat(
                                message=mock_message, session_id="vision_analysis"
                            )

                            logger.debug("Provider直接调用成功")

                        except Exception as provider_error:
                            logger.error(f"Provider直接调用也失败: {provider_error}")
                            raise Exception(
                                f"所有VLM调用方法都失败。Context错误: {context_error}，Provider错误: {provider_error}"
                            )

                    if result:
                        # 处理不同类型的响应
                        llm_response_text = ""

                        if isinstance(result, str):
                            # 如果result是字符串，直接使用
                            llm_response_text = result
                            logger.debug(f"直接获取的字符串响应: {llm_response_text}")
                        elif hasattr(result, "get_plain_text"):
                            # 如果result是MessageChain，直接获取纯文本
                            llm_response_text = result.get_plain_text()
                            logger.debug(
                                f"从MessageChain获取的响应文本: {llm_response_text}"
                            )
                        elif hasattr(result, "result_chain") and result.result_chain:
                            # 如果是旧API返回的结果格式
                            llm_response_text = result.result_chain.get_plain_text()
                            logger.debug(
                                f"从result_chain获取的响应文本: {llm_response_text}"
                            )
                        elif (
                            hasattr(result, "completion_text")
                            and result.completion_text
                        ):
                            # 从completion_text获取
                            llm_response_text = result.completion_text
                            logger.debug(
                                f"从completion_text获取的响应文本: {llm_response_text}"
                            )
                        else:
                            # 兜底处理：尝试转换为字符串
                            llm_response_text = str(result)
                            logger.debug(
                                f"使用字符串转换获取的响应文本: {llm_response_text}"
                            )

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

    def invalidate_cache(self, image_hash: str):
        """失效指定图片的缓存。"""
        if hasattr(self, "_image_cache") and image_hash in self._image_cache:
            del self._image_cache[image_hash]
            logger.debug(f"已失效缓存: {image_hash}")

    def cleanup(self):
        """清理资源。"""
        # 清理图片缓存
        if hasattr(self, "_image_cache"):
            self._image_cache.clear()
        logger.debug("ImageProcessorService 资源已清理")

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

    async def _file_to_gif_base64(self, file_path: str) -> str:
        """将文件转换为 GIF 格式的 base64 编码（用于发送侧强制 GIF）。"""
        if not getattr(self.plugin, "send_emoji_as_gif", True):
            return await self._file_to_base64(file_path)
        try:
            stat_mtime = os.path.getmtime(file_path)
        except Exception:
            stat_mtime = None

        cache_key = f"{file_path}:{stat_mtime}"
        now = time.time()
        cached = self._gif_base64_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_b64 = cached
            if now - cached_at <= self._gif_base64_cache_expire_time and cached_b64:
                return cached_b64

        try:
            if file_path.lower().endswith(".gif"):
                b64 = await self._file_to_base64(file_path)
                if b64:
                    self._gif_base64_cache[cache_key] = (now, b64)
                return b64

            if PILImage is None:
                return await self._file_to_base64(file_path)

            with PILImage.open(file_path) as im:
                buf = BytesIO()
                is_animated = bool(getattr(im, "is_animated", False))
                n_frames = int(getattr(im, "n_frames", 1) or 1)

                if is_animated and n_frames > 1:
                    frames = []
                    durations = []
                    for frame_idx in range(n_frames):
                        im.seek(frame_idx)
                        frame = im.convert("RGBA")
                        frames.append(frame)
                        durations.append(int(im.info.get("duration", 100) or 100))
                    frames[0].save(
                        buf,
                        format="GIF",
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0,
                        disposal=2,
                    )
                else:
                    frame = im.convert("RGBA")
                    frame.save(buf, format="GIF")

                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                if b64:
                    self._gif_base64_cache[cache_key] = (now, b64)
                return b64
        except Exception as e:
            logger.error(f"文件转换为GIF base64失败: {e}")
            return await self._file_to_base64(file_path)

    async def _store_image(self, src_path: str, category: str) -> str:
        """将图片存储到指定分类目录。

        Args:
            src_path: 源图片路径
            category: 分类名称

        Returns:
            str: 存储后的图片路径
        """
        try:
            cat_dir = self.plugin_config.ensure_category_dir(category)
            if not cat_dir:
                logger.error("分类目录未设置，无法存储图片")
                return src_path

            # 复制图片到分类目录
            filename = os.path.basename(src_path)
            dest_path = str(cat_dir / filename)

            # 如果文件已存在，生成新文件名
            if os.path.exists(dest_path):
                base_name, ext = os.path.splitext(filename)
                dest_path = str(cat_dir / f"{base_name}_{int(time.time())}{ext}")

            shutil.copy2(src_path, dest_path)
            logger.debug(f"图片已存储到分类目录: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"存储图片失败: {e}")
            return src_path

    async def safe_remove_file(self, file_path: str) -> bool:
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

    async def pick_vision_provider(self, event) -> str | None:
        """选择视觉模型提供商。

        Args:
            event: 消息事件对象

        Returns:
            str | None: 提供商ID
        """
        if self.vision_provider_id:
            return self.vision_provider_id
        if event is None:
            return None
        try:
            if hasattr(self.plugin, "context"):
                return await self.plugin.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
        except Exception as e:
            logger.error(f"获取视觉模型提供者失败: {e}")
            return None
