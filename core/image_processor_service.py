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

    # 分类结果常量
    CATEGORY_FILTERED = "过滤不通过"
    CATEGORY_NOT_EMOJI = "非表情包"

    # 缓存常量
    IMAGE_CACHE_MAX_SIZE = 500  # 最大缓存条目数
    GIF_CACHE_MAX_SIZE = 50  # 最大 GIF base64 缓存条目数（减小以降低内存占用）
    GIF_CACHE_MAX_SIZE_BYTES = 10 * 1024 * 1024  # GIF 缓存最大总字节数 (10MB)
    CACHE_EXPIRE_TIME = 3600  # 缓存过期时间（秒）

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self.plugin_config = plugin_instance.plugin_config

        self.raw_dir = self.plugin_config.raw_dir if self.plugin_config else None
        self.categories_dir = (
            self.plugin_config.categories_dir if self.plugin_config else None
        )

        # 图片分类结果缓存，key为图片哈希，value为分类结果元组
        self._image_cache: dict[str, dict] = {}
        self._image_cache_max_size = self.IMAGE_CACHE_MAX_SIZE
        self._cache_expire_time = self.CACHE_EXPIRE_TIME
        self._gif_base64_cache: dict[str, tuple[float, str]] = {}
        self._gif_base64_cache_max_size = self.GIF_CACHE_MAX_SIZE
        self._gif_base64_cache_expire_time = self.CACHE_EXPIRE_TIME

        # 提示词配置：正常运行时由 prompts.json 加载并通过 update_config 注入，
        # 以下仅为 prompts.json 缺失时的最小化 fallback
        _FALLBACK_PROMPT = (
            "分析表情包：从 `{emotion_list}` 中选择情绪分类。"
            "返回格式：情绪分类|语义标签(逗号分隔)|画面描述(一句话)"
        )
        _FALLBACK_FILTER_PROMPT = (
            "审核图片是否含不当内容，不当则返回'过滤不通过'。"
            "否则从 `{emotion_list}` 中选择情绪分类。"
            "返回格式：情绪分类|语义标签(逗号分隔)|画面描述(一句话)"
        )

        self.emoji_classification_prompt = getattr(
            plugin_instance, "EMOJI_CLASSIFICATION_PROMPT", _FALLBACK_PROMPT
        )
        self.emoji_classification_with_filter_prompt = getattr(
            plugin_instance,
            "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT",
            _FALLBACK_FILTER_PROMPT,
        )

        # 保留combined_analysis_prompt作为备用
        self.combined_analysis_prompt = self.emoji_classification_prompt

        # 配置参数（初始值从 plugin_config 读取，后续通过 update_config 更新）
        self.categories = (
            list(self.plugin_config.categories or []) if self.plugin_config else []
        )
        self.content_filtration = (
            bool(getattr(self.plugin_config, "content_filtration", False))
            if self.plugin_config
            else False
        )
        self.vision_provider_id = (
            str(self.plugin_config.vision_provider_id or "")
            if self.plugin_config
            else ""
        )
        # 框架 VLM provider 缓存，None 表示未查询过
        self._cached_framework_vlm_id: str | None = None

        # 执行自动迁移检查（在插件启动时运行一次）
        # 注意：_auto_migrate_categories 是异步方法，需在 initialize() 中调用
        # self._auto_migrate_categories() 已移至 initialize()

    async def _auto_migrate_categories(self):
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
                        await asyncio.to_thread(
                            shutil.move, str(img_file), str(target_path)
                        )
                        migrated_files += 1
                    except Exception as e:
                        logger.error(f"迁移文件失败 {img_file} -> {target_path}: {e}")

            # 迁移索引数据
            old_index = old_dir / "index.json"
            if old_index.exists():
                try:
                    import json

                    def _migrate_index(old_index_path, new_dir_path, old_cat, new_cat):
                        """同步迁移索引（在线程中执行）"""
                        with open(old_index_path, encoding="utf-8") as f:
                            old_data = json.load(f)
                        for item in old_data:
                            if (
                                isinstance(item, dict)
                                and item.get("category") == old_cat
                            ):
                                item["category"] = new_cat
                        new_index_path = new_dir_path / "index.json"
                        if new_index_path.exists():
                            with open(new_index_path, encoding="utf-8") as f:
                                new_data = json.load(f)
                            new_data.extend(old_data)
                        else:
                            new_data = old_data
                        with open(new_index_path, "w", encoding="utf-8") as f:
                            json.dump(new_data, f, ensure_ascii=False, indent=2)
                        old_index_path.unlink()
                        return len(old_data)

                    migrated_indices += await asyncio.to_thread(
                        _migrate_index, old_index, new_dir, old_category, new_category
                    )
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
            # 插件 provider 变更时，重置框架缓存以便重新解析
            self._cached_framework_vlm_id = None
        if emoji_classification_prompt is not None:
            self.emoji_classification_prompt = emoji_classification_prompt
        if combined_analysis_prompt is not None:
            self.combined_analysis_prompt = combined_analysis_prompt
        if emoji_classification_with_filter_prompt is not None:
            self.emoji_classification_with_filter_prompt = (
                emoji_classification_with_filter_prompt
            )

    async def _store_and_index_image(
        self,
        file_path: str,
        is_temp: bool,
        category: str,
        hash_val: str,
        idx: dict[str, Any],
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
        already_in_raw: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        """将图片存储到 raw → 复制到分类目录 → 删除 raw → 更新索引。

        Args:
            already_in_raw: 若为 True，则 file_path 已在 raw 目录中，
                            跳过 move/copy-to-raw 步骤，直接作为 raw_path 使用。

        Returns:
            (成功与否, 更新后的索引)
        """
        if already_in_raw:
            raw_path = file_path
        else:
            # 存储图片到raw目录
            raw_dir = self.plugin_config.ensure_raw_dir()
            if raw_dir:
                base_path = Path(file_path)
                ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
                filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
                raw_path = str(raw_dir / filename)
                if is_temp:
                    await asyncio.to_thread(shutil.move, file_path, raw_path)
                else:
                    await asyncio.to_thread(shutil.copy2, file_path, raw_path)
            else:
                raw_path = file_path

        # 复制图片到对应分类目录
        cat_dir = self.plugin_config.ensure_category_dir(category)
        cat_path = str(cat_dir / os.path.basename(raw_path)) if cat_dir else raw_path

        if not os.path.exists(raw_path):
            logger.warning(f"原始文件已不存在，可能被清理: {raw_path}")
            return False, None

        try:
            if cat_dir:
                await asyncio.to_thread(shutil.copy2, raw_path, cat_path)
        except FileNotFoundError:
            logger.warning(f"复制文件时发现文件已被删除: {raw_path}")
            return False, None

        # 立即删除raw目录中的原始文件
        try:
            if os.path.exists(raw_path):
                await self.plugin._safe_remove_file(raw_path)
                logger.debug(f"已删除已分类的原始文件: {raw_path}")
        except Exception as e:
            logger.warning(f"删除已分类的原始文件失败: {raw_path}, 错误: {e}")

        # 更新图片索引
        entry: dict[str, Any] = {
            "hash": hash_val,
            "category": category,
            "created_at": int(time.time()),
        }
        if tags:
            entry["tags"] = tags
        if desc:
            entry["desc"] = desc
        if scenes:
            entry["scenes"] = scenes
        idx[cat_path] = entry
        return True, idx

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
        """处理图片：去重 → 缓存检查 → VLM 分类 → 存储索引。"""

        # ── 去重：持久化索引 / 黑名单 / 传入索引 ──
        if await self._is_duplicate_or_blacklisted(hash_val, idx, file_path, is_temp):
            return False, None

        # ── 内存缓存命中 ──
        cached = self._get_valid_cache(hash_val)
        if cached is not None:
            return await self._handle_classification_result(
                cached["category"],
                cached["emotion"],
                cached.get("tags", []),
                cached.get("desc", ""),
                cached.get("scenes", []),
                file_path,
                is_temp,
                hash_val,
                idx,
                from_cache=True,
            )

        # ── 存入 raw 目录 ──
        raw_path = await self._move_to_raw(file_path, hash_val, is_temp)

        # ── VLM 分类 ──
        try:
            category, tags, desc, emotion, scenes = await self.classify_image(
                event=event,
                file_path=raw_path,
                categories=categories,
                content_filtration=content_filtration,
            )
        except Exception as e:
            logger.error(f"处理图片失败 [{raw_path}]: {e}")
            if is_temp:
                await self.plugin._safe_remove_file(raw_path)
            raise

        # ── 缓存结果 ──
        self._put_image_cache(hash_val, category, tags, desc, emotion, scenes)

        # ── 处理分类结果 ──
        return await self._handle_classification_result(
            category,
            emotion,
            tags,
            desc,
            scenes,
            raw_path,
            False,
            hash_val,
            idx,
            from_cache=False,
            already_in_raw=True,
        )

    # ── _process_image_legacy 的辅助方法 ──

    async def _is_duplicate_or_blacklisted(
        self,
        hash_val: str,
        idx: dict,
        file_path: str,
        is_temp: bool,
    ) -> bool:
        """检查图片是否已存在于索引或黑名单中。"""

        async def _cleanup_temp():
            if is_temp and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)

        # 持久化索引
        if hasattr(self.plugin, "cache_service"):
            persistent_idx = self.plugin.cache_service.get_index_cache()
            if any(
                isinstance(v, dict) and v.get("hash") == hash_val
                for v in persistent_idx.values()
            ):
                logger.debug(f"图片已存在于持久化索引中: {hash_val}")
                await _cleanup_temp()
                return True

            blacklist = self.plugin.cache_service.get_cache("blacklist_cache")
            if blacklist and hash_val in blacklist:
                logger.debug(f"图片在黑名单中: {hash_val}")
                await _cleanup_temp()
                return True
        else:
            if any(
                isinstance(v, dict) and v.get("hash") == hash_val for v in idx.values()
            ):
                logger.debug(f"图片已存在于索引中: {hash_val}")
                await _cleanup_temp()
                return True

        return False

    def _get_valid_cache(self, hash_val: str) -> dict | None:
        """获取有效（未过期）的分类缓存，过期则清除。"""
        cached = self._image_cache.get(hash_val)
        if cached is None:
            return None
        if time.time() - cached.get("timestamp", 0) < self._cache_expire_time:
            return cached
        self._image_cache.pop(hash_val, None)
        return None

    def _put_image_cache(
        self,
        hash_val: str,
        category: str,
        tags: list,
        desc: str,
        emotion: str,
        scenes: list,
    ) -> None:
        """写入分类缓存并淘汰过期条目。"""
        self._image_cache[hash_val] = {
            "category": category,
            "tags": tags,
            "desc": desc,
            "emotion": emotion,
            "scenes": scenes,
            "timestamp": time.time(),
        }
        self._evict_image_cache()

    async def _move_to_raw(self, file_path: str, hash_val: str, is_temp: bool) -> str:
        """将图片移动/复制到 raw 目录，返回 raw 路径。"""
        raw_dir = self.plugin_config.ensure_raw_dir()
        if not raw_dir:
            return file_path
        ext = Path(file_path).suffix.lower() or ".jpg"
        filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
        raw_path = str(raw_dir / filename)
        if is_temp:
            await asyncio.to_thread(shutil.move, file_path, raw_path)
        else:
            await asyncio.to_thread(shutil.copy2, file_path, raw_path)
        return raw_path

    async def _handle_classification_result(
        self,
        category: str,
        emotion: str,
        tags: list,
        desc: str,
        scenes: list,
        file_path: str,
        is_temp: bool,
        hash_val: str,
        idx: dict,
        from_cache: bool = False,
        already_in_raw: bool = False,
    ) -> tuple[bool, dict[str, Any] | None]:
        """根据分类结果决定存储、跳过或清理。"""
        source = "缓存" if from_cache else "VLM"

        # 过滤不通过
        if category == self.CATEGORY_FILTERED or emotion == self.CATEGORY_FILTERED:
            logger.debug(f"图片过滤不通过（{source}）: {hash_val}")
            if is_temp and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)
            elif not from_cache and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)
            return False, None

        # 非表情包
        if category == self.CATEGORY_NOT_EMOJI or emotion == self.CATEGORY_NOT_EMOJI:
            logger.debug(f"非表情包（{source}）: {hash_val}")
            if is_temp and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)
            elif not from_cache and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)
            return False, None

        # 有效分类
        if category and category in self.categories:
            logger.debug(f"分类有效（{source}）: {category}")
            return await self._store_and_index_image(
                file_path,
                is_temp,
                category,
                hash_val,
                idx,
                tags=tags,
                desc=desc,
                scenes=scenes,
                already_in_raw=already_in_raw,
            )

        # 无效分类
        logger.warning(f"分类无效（{source}）: {category!r}，图片留在raw目录等待清理")
        return False, None

    def _build_emotion_list_str(self, categories: list[str] | None = None) -> str:
        categories = categories if categories is not None else (self.categories or [])
        categories = [c for c in categories if isinstance(c, str) and c.strip()]
        info_map = getattr(self.plugin_config, "category_info", None) or {}

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
            file_path: 图片绝对路径
            categories: 分类列表（可选，默认使用 self.categories）
            backend_tag: 后端标签（保留接口兼容）
            content_filtration: 是否进行内容过滤（可选，默认使用 self.content_filtration）

        Returns:
            tuple: (category, tags, desc, emotion, scenes)
        """
        # 路径验证（单一入口，_call_vision_model 不再重复）
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"分类图片时文件不存在: {file_path}")

        try:
            # 确定是否进行内容过滤
            should_filter = (
                content_filtration
                if content_filtration is not None
                else self.content_filtration
            )

            prompt_categories = categories if isinstance(categories, list) else None
            emotion_list_str = self._build_emotion_list_str(prompt_categories)

            # 根据是否开启审核选择合适的提示词
            prompt_template = (
                self.emoji_classification_with_filter_prompt
                if should_filter
                else self.emoji_classification_prompt
            )
            prompt = prompt_template.format(emotion_list=emotion_list_str)

            # 调用视觉模型
            response = await self._call_vision_model(event, file_path, prompt)

            # 处理审核不通过
            if self.CATEGORY_FILTERED in response:
                logger.warning(f"图片内容审核不通过: {file_path}")
                return self.CATEGORY_FILTERED, [], "", self.CATEGORY_FILTERED, []

            # 解析响应：情绪分类|语义标签(逗号分隔)|画面描述(一句话)
            parts = [p.strip() for p in response.strip().split("|")]
            emotion_result = parts[0] if parts else ""
            tags_str = parts[1] if len(parts) > 1 else ""
            tags_result = [t.strip() for t in tags_str.split(",") if t.strip()]
            desc_result = parts[2] if len(parts) > 2 else "表情包"

            # 规范化分类
            category = self._normalize_category(emotion_result)
            return category, tags_result, desc_result, category, []

        except (FileNotFoundError, ValueError):
            # 配置错误 / 文件不存在，直接抛出不吞异常
            raise
        except Exception as e:
            logger.error(f"图片分类失败 [{file_path}]: {e}")
            return "", [], "", "", []

    def _normalize_category(self, raw: str) -> str:
        """将 VLM 返回的分类文本规范化为有效分类名。"""
        if self.plugin_config:
            try:
                normalized = self.plugin_config.normalize_category_strict(raw)
                if normalized and normalized in self.categories:
                    return normalized
            except Exception as e:
                logger.debug(f"[分类规范化] 异常: {e}")

        # 获取默认分类
        default_categories = (
            list(self.plugin_config.DEFAULT_CATEGORIES)
            if self.plugin_config
            else ["happy"]
        )
        fallback = self.categories[0] if self.categories else default_categories[0]
        logger.warning(f"无法识别情绪分类: {raw!r}，使用默认分类: {fallback}")
        return fallback

    async def _call_vision_model(
        self, event: AstrMessageEvent | None, img_path: str, prompt: str
    ) -> str:
        """调用视觉模型分析图片。

        使用 context.llm_generate 调用指定的视觉模型 provider，
        支持指数退避重试。

        Args:
            event: 消息事件（用于 provider 解析）
            img_path: 图片绝对路径（调用方需保证已验证）
            prompt: 提示词

        Returns:
            str: 模型响应文本

        Raises:
            ValueError: 未配置视觉模型
            FileNotFoundError: 图片文件不存在
            Exception: 模型调用失败（已重试）
        """
        # 路径规范化
        img_path_obj = Path(img_path)
        if not img_path_obj.is_absolute():
            data_dir = getattr(self.plugin_config, "data_dir", None)
            img_path_obj = (
                (Path(data_dir) / img_path).absolute()
                if data_dir
                else img_path_obj.absolute()
            )
        img_path = str(img_path_obj)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在: {img_path}")

        # 解析 provider
        provider_id = await self._resolve_vision_provider(event)
        if not provider_id:
            raise ValueError(
                "未配置视觉模型(vision_provider_id)，无法进行图片分析。"
                "请在插件配置或 AstrBot 全局配置中设置。"
            )

        # 构建图片 URL（file:// 协议）
        file_url = f"file:///{img_path.replace(chr(92), '/')}"

        # 重试配置
        max_retries = int(getattr(self.plugin, "vision_max_retries", 3))
        retry_delay = float(getattr(self.plugin, "vision_retry_delay", 1.0))
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"调用VLM (尝试 {attempt + 1}/{max_retries}), "
                    f"provider={provider_id}, 图片={img_path}"
                )
                result = await self.plugin.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    image_urls=[file_url],
                )

                # LLMResponse.completion_text 是 @property，自动处理 result_chain
                text = (result.completion_text or "").strip() if result else ""
                if text:
                    logger.debug(f"VLM响应: {text[:200]}")
                    return text

                logger.warning("VLM返回空响应")
                last_error = Exception("VLM返回空响应")

            except Exception as e:
                last_error = e
                error_msg = str(e)
                is_rate_limit = any(
                    kw in error_msg
                    for kw in (
                        "429",
                        "RateLimit",
                        "exceeded your current request limit",
                    )
                )
                is_provider_error = "Provider" in error_msg or "提供商" in error_msg
                if is_rate_limit:
                    logger.warning(f"VLM请求被限流 ({attempt + 1}/{max_retries})")
                elif is_provider_error:
                    logger.error(
                        f"VLM模型提供商错误 ({attempt + 1}/{max_retries}): {e}\n"
                        f"  当前provider_id: {provider_id}\n"
                        f"  提示: 请检查插件配置中的'视觉模型'是否有效，"
                        f"  或尝试清空该配置使用框架全局的图片描述模型"
                    )
                else:
                    logger.error(f"VLM调用失败 ({attempt + 1}/{max_retries}): {e}")

            # 指数退避
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))

        raise Exception(
            f"视觉模型调用失败（已重试{max_retries}次）: {last_error}"
        ) from last_error

    async def _compute_hash(self, file_path: str) -> str:
        """计算文件的SHA256哈希值。

        Args:
            file_path: 文件路径

        Returns:
            str: SHA256哈希值
        """

        def _sync_hash(fp: str) -> str:
            hasher = hashlib.sha256()
            with open(fp, "rb") as f:
                hasher.update(f.read())
            return hasher.hexdigest()

        try:
            return await asyncio.to_thread(_sync_hash, file_path)
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
        if hasattr(self, "_image_cache"):
            self._image_cache.pop(image_hash, None)
            logger.debug(f"已失效缓存: {image_hash}")

    def _evict_image_cache(self) -> None:
        """淘汰 _image_cache 中最旧的条目，保持在最大容量以内。"""
        if len(self._image_cache) <= self._image_cache_max_size:
            return
        # 按 timestamp 排序，保留最新的一半
        sorted_items = sorted(
            self._image_cache.items(),
            key=lambda kv: kv[1].get("timestamp", 0),
        )
        keep = sorted_items[len(sorted_items) // 2 :]
        self._image_cache.clear()
        self._image_cache.update(keep)
        logger.debug(f"_image_cache 淘汰完成，当前 {len(self._image_cache)} 条")

    def _evict_gif_base64_cache(self) -> None:
        """淘汰 _gif_base64_cache 中最旧的条目，保持在最大容量以内。"""
        # 检查条目数量
        if len(self._gif_base64_cache) <= self._gif_base64_cache_max_size:
            # 还需要检查总字节数
            total_bytes = sum(len(v[1]) for v in self._gif_base64_cache.values())
            if total_bytes <= self.GIF_CACHE_MAX_SIZE_BYTES:
                return

        # 按时间排序，淘汰旧条目
        sorted_items = sorted(
            self._gif_base64_cache.items(),
            key=lambda kv: kv[1][0],  # cached_at timestamp
        )

        # 计算需要保留的条目
        target_count = self._gif_base64_cache_max_size // 2
        target_bytes = self.GIF_CACHE_MAX_SIZE_BYTES // 2

        # 从新到旧保留，直到满足条件
        keep_items = []
        current_bytes = 0
        for key, value in reversed(sorted_items):
            if len(keep_items) >= target_count or current_bytes >= target_bytes:
                break
            keep_items.append((key, value))
            current_bytes += len(value[1])

        self._gif_base64_cache.clear()
        self._gif_base64_cache.update(keep_items)
        logger.debug(
            f"_gif_base64_cache 淘汰完成，当前 {len(self._gif_base64_cache)} 条，"
            f"总大小 {current_bytes / 1024 / 1024:.2f}MB"
        )

    def cleanup(self):
        """清理资源。"""
        self._image_cache.clear()
        self._gif_base64_cache.clear()
        logger.debug("ImageProcessorService 资源已清理")

    async def _file_to_base64(self, file_path: str) -> str:
        """将文件转换为base64编码。

        Args:
            file_path: 文件路径

        Returns:
            str: base64编码
        """

        def _sync_read_and_encode(fp: str) -> str:
            with open(fp, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        try:
            return await asyncio.to_thread(_sync_read_and_encode, file_path)
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
                    self._evict_gif_base64_cache()
                return b64

            if PILImage is None:
                return await self._file_to_base64(file_path)

            def _sync_convert_to_gif(fp: str) -> str:
                with PILImage.open(fp) as im:
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

                    return base64.b64encode(buf.getvalue()).decode("utf-8")

            b64 = await asyncio.to_thread(_sync_convert_to_gif, file_path)
            if b64:
                self._gif_base64_cache[cache_key] = (now, b64)
                self._evict_gif_base64_cache()
            return b64
        except Exception as e:
            logger.error(f"文件转换为GIF base64失败: {e}")
            return await self._file_to_base64(file_path)

    async def safe_remove_file(self, file_path: str) -> bool:
        """安全删除文件。

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否删除成功
        """
        try:
            if os.path.exists(file_path):
                await asyncio.to_thread(os.remove, file_path)
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
        return await self._resolve_vision_provider(event)

    async def _resolve_vision_provider(self, event=None) -> str | None:
        """统一的视觉模型 provider 解析逻辑。

        优先级：
        1. 插件配置的 vision_provider_id
        2. AstrBot 框架配置的 default_image_caption_provider_id（视觉描述模型）
        3. 都未配置时返回 None

        Args:
            event: 消息事件对象（可选）

        Returns:
            str | None: 提供商ID，未配置时返回 None
        """
        # 1. 优先使用插件配置的视觉模型
        if self.vision_provider_id:
            logger.debug(f"[视觉模型] 使用插件配置的提供商: {self.vision_provider_id}")
            return self.vision_provider_id
        else:
            logger.debug(
                "[视觉模型] 插件未配置 vision_provider_id，尝试使用框架全局配置"
            )

        # 2. 使用缓存的框架 VLM provider（避免每次都读配置）
        if self._cached_framework_vlm_id is not None:
            # 空字符串表示已查询过但没有配置
            return self._cached_framework_vlm_id or None

        # 3. 首次查询：从 AstrBot 框架配置获取 default_image_caption_provider_id
        framework_vlm_id = ""
        try:
            if hasattr(self.plugin, "context"):
                astrbot_config = self.plugin.context.get_config()
                provider_settings = astrbot_config.get("provider_settings", {})
                framework_vlm_id = str(
                    provider_settings.get("default_image_caption_provider_id", "") or ""
                )
        except Exception as e:
            logger.debug(f"读取框架视觉模型配置失败: {e}")

        # 缓存结果（update_config 时会重置为 None 以便重新查询）
        self._cached_framework_vlm_id = framework_vlm_id

        if framework_vlm_id:
            logger.info(f"使用框架全局图片描述模型: {framework_vlm_id}")
            return framework_vlm_id

        logger.warning(
            "未配置视觉模型，无法进行图片分类。"
            "请在插件配置中设置 vision_provider_id，"
            "或在 AstrBot 全局配置中设置 default_image_caption_provider_id。"
        )
        return None
