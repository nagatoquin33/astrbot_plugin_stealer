import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..util.safe_io import protect_file, commit_image_file, safe_remove_file, unique_image_path
from ..util.normalization import normalize_label_list
from .semantic_schema import MAX_DESC_CHARS, clip_chars
from ..db.index_manager import refresh_search_entry

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
    from PIL import ImageFont as PILImageFont

    try:
        LANCZOS = PILImage.Resampling.LANCZOS
    except AttributeError:
        LANCZOS = PILImage.LANCZOS
except Exception:
    PILImage = None
    PILImageDraw = None
    PILImageFont = None
    LANCZOS = None

try:
    import numpy as np
except Exception:
    np = None


class ImageProcessorService:
    """图片处理服务类，负责处理所有与图片相关的操作。"""

    # 分类 key 迁移映射，用于移动分类文件及更新索引记录。
    CATEGORY_MIGRATION_MAP = {
        "smirk": "troll",  # 坏笑 -> 发癫
    }

    # 分类结果常量
    CATEGORY_FILTERED = "过滤不通过"
    CATEGORY_NOT_EMOJI = "非表情包"

    # 缓存常量
    IMAGE_CACHE_MAX_SIZE = 500  # 最大缓存条目数
    CACHE_EXPIRE_TIME = 3600  # 缓存过期时间（秒）

    def __init__(self, plugin_instance):
        """初始化图片处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance
        self.plugin_config = plugin_instance.plugin_config

        self.raw_dir = self.plugin_config.raw_dir if self.plugin_config else None
        self.categories_dir = self.plugin_config.categories_dir if self.plugin_config else None

        # 图片分类结果缓存，key为图片哈希，value为分类结果元组
        self._image_cache: dict[str, dict] = {}
        self._cache_expire_time = self.CACHE_EXPIRE_TIME

        # 图片处理锁，防止并发去重竞态
        self._process_lock = asyncio.Lock()
        # 正在处理中的哈希集合，防止同一图片被并发重复处理
        self._processing_hashes: set[str] = set()

        # 配置参数（初始值从 plugin_config 读取，后续通过 update_config 更新）
        self.categories = list(self.plugin_config.categories or []) if self.plugin_config else []
        self.content_filtration = (
            bool(getattr(self.plugin_config, "content_filtration", False))
            if self.plugin_config
            else False
        )
        # 子服务（职责拆分）
        from .prompt_manager import PromptManager
        from .phash_dedup_service import PHashDedupService
        from .classification_parser import ClassificationParser
        from .vlm_call_service import VLMCallService

        self._prompt_manager = PromptManager(plugin_instance)
        self._phash_service = PHashDedupService(plugin_instance)
        self._classification_parser = ClassificationParser(plugin_instance)
        self._vlm_call_service = VLMCallService(plugin_instance)

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
                        await asyncio.to_thread(shutil.move, str(img_file), str(target_path))
                        migrated_files += 1
                    except Exception as e:
                        logger.error(f"迁移文件失败 {img_file} -> {target_path}: {e}")

            # 迁移索引数据
            old_index = old_dir / "index.json"
            if old_index.exists():
                try:

                    def _migrate_index(old_index_path, new_dir_path, old_cat, new_cat):
                        """同步迁移索引（在线程中执行）"""
                        with open(old_index_path, encoding="utf-8") as f:
                            old_data = json.load(f)
                        for item in old_data:
                            if isinstance(item, dict) and item.get("category") == old_cat:
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
        emoji_classification_with_filter_prompt=None,
    ):
        """更新图片处理器配置。

        Args:
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            vision_provider_id: 视觉模型提供者ID
            emoji_classification_prompt: 表情包分类提示词
            emoji_classification_with_filter_prompt: 带审核的表情包分析提示词
        """
        if categories is not None:
            self.categories = categories
        if content_filtration is not None:
            self.content_filtration = content_filtration
        if vision_provider_id is not None:
            self._vlm_call_service.vision_provider_id = str(vision_provider_id or "").strip()
        # 同步到子服务
        self._prompt_manager.update_config(
            categories=categories,
            emoji_classification_prompt=emoji_classification_prompt,
            emoji_classification_with_filter_prompt=emoji_classification_with_filter_prompt,
        )

    async def _store_and_index_image(
        self,
        file_path: str,
        is_temp: bool,
        category: str,
        hash_val: str,
        idx: dict[str, Any],
        extra_meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
        already_in_raw: bool = False,
        phash_val: str = "",
        source_url: str = "",
        original_name: str = "",
        add_method: str = "auto",
    ) -> tuple[bool, dict[str, Any] | None]:
        """Publish one asset and commit only its metadata, retaining the source on failure."""
        if not os.path.isfile(file_path):
            return False, None
        db = getattr(self.plugin, "db_service", None)
        if db is None:
            raise RuntimeError("database unavailable")
        async with db.file_lock:
            if category not in self.plugin_config.categories:
                raise ValueError("分类已被移除")
            cat_dir = self.plugin_config.ensure_category_dir(category)
            filename = os.path.basename(file_path)
            cat_path = str(unique_image_path(cat_dir, filename))
            async def commit_entry(_target):
                if category not in self.plugin_config.categories:
                    raise ValueError("分类已被移除")
                entry = await self._build_image_entry(
                    cat_path, hash_val, category, extra_meta, tags, desc, scenes,
                    phash_val, source_url, original_name, add_method,
                )
                await db.sync_index({cat_path: entry})
                return entry
            entry = await commit_image_file(file_path, cat_path, commit_entry, consume=is_temp or already_in_raw)
        idx[cat_path] = entry
        await refresh_search_entry(self.plugin, cat_path, entry)
        return True, {cat_path: entry}

    async def _build_image_entry(
        self, path, image_hash, category, extra_meta, tags, desc, scenes,
        phash, source_url, original_name, add_method,
    ) -> dict[str, Any]:
        entry = {
            "hash": image_hash, "category": category,
            "created_at": int(time.time()), "use_count": 0, "last_used_at": 0,
            "phash": phash or None, "tags": tags or [],
            "desc": clip_chars(desc, MAX_DESC_CHARS), "scenes": scenes or [],
        }
        for key, value in (extra_meta or {}).items():
            if key not in entry and key != "path":
                entry[key] = value
        entry.update(await asyncio.to_thread(self._probe_image_metadata, path))
        if source_url:
            entry["source_url"] = source_url
        if original_name:
            entry["original_name"] = original_name
        entry["add_method"] = add_method
        return entry

    async def _store_to_pending(
        self,
        file_path: str,
        is_temp: bool,
        category: str,
        hash_val: str,
        extra_meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
        already_in_raw: bool = False,
        phash_val: str = "",
        source_url: str = "",
        original_name: str = "",
        add_method: str = "auto",
        analysis_state: str = "ready",
    ) -> tuple[bool, dict[str, Any] | None]:
        """将偷取的图片落点到待审核池（pending 目录 + emoji_pending 表）。

        不写 emoji/tag/scene/embedding；不参与发送与检索。
        原子写入：文件落盘成功 → 写 DB；DB 失败 → 删除已落盘 pending 文件。

        Returns:
            (True, None) 成功落 pending（idx 为 None，上层跳过索引合并与 _save_index）；
            (False, None) 失败。
        """
        if not hash_val or not os.path.exists(file_path):
            logger.warning(f"pending: 源文件缺失或无哈希: {file_path}")
            return False, None

        # 去重：哈希已在正式库 / 待审核池 / 黑名单中则跳过
        db = self.plugin.db_service
        if db and hash_val:
            # 查正式库
            if db.get_emoji_by_hash(hash_val):
                logger.debug(f"pending: 哈希已存在于正式库，跳过: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await safe_remove_file(file_path)
                return False, None
            # 查待审核池
            if db.get_pending_by_hash(hash_val):
                logger.debug(f"pending: 哈希已在待审核池中，跳过: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await safe_remove_file(file_path)
                return False, None
            # 查黑名单
            if db.is_blacklisted(hash_val):
                logger.debug(f"pending: 哈希已在黑名单中，跳过: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await safe_remove_file(file_path)
                return False, None

        pending_dir = self.plugin_config.ensure_pending_dir()
        if not pending_dir:
            logger.warning("pending: 无法创建 pending 目录")
            return False, None

        base_path = Path(file_path)
        ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
        pending_filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
        pending_path = str(unique_image_path(pending_dir, pending_filename))
        try:
            async def commit_pending(_target):
                meta = await self._build_image_entry(
                    pending_path, hash_val, category, extra_meta, tags, desc, scenes,
                    phash_val, source_url, original_name, add_method,
                )
                meta["path"] = pending_path
                meta["analysis_state"] = analysis_state
                if analysis_state == 'unanalysed':
                    meta['_pending_capacity'] = int(getattr(self.plugin_config, 'steal_pool_capacity', 200))
                pending_id = await db.insert_pending(meta)
                if pending_id is None:
                    raise RuntimeError("pending metadata insertion failed")
                return pending_id
            pending_id = await commit_image_file(file_path, pending_path, commit_pending,
                                                 consume=is_temp or already_in_raw, lock=db.file_lock)
        except Exception as exc:
            logger.error(f"pending publication failed: {exc}")
            return False, None
        logger.info(f"pending: id={pending_id} cat={category} hash={hash_val[:16]}")
        return True, None

    async def process_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        is_temp: bool = False,
        idx: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        content_filtration: bool | None = None,
        is_platform_emoji: bool = False,
        extra_meta: dict[str, Any] | None = None,
        to_pending: bool = False,
        source_url: str = "",
        original_name: str = "",
        add_method: str = "auto",
    ) -> tuple[bool, dict[str, Any] | None]:
        """统一处理图片：存储、分类、过滤。

        Args:
            event: 消息事件
            file_path: 图片路径
            is_temp: 是否为临时文件
            idx: 索引字典
            categories: 分类列表
            content_filtration: 是否进行内容过滤
            is_platform_emoji: 是否为平台标记的表情包
            to_pending: 是否将原图及本地元数据保存到待审核池，等待人工通过后后台识图。

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

        raw_path: str | None = None  # 在异常处理中用于清理孤儿 raw 文件

        # 1. 并行计算 SHA256 和感知哈希（锁外）
        hash_val, phash_val = await asyncio.gather(
            self._compute_hash(file_path),
            self._phash_service.compute_phash(file_path),
        )

        if not hash_val:
            logger.warning(f"无法计算图片哈希: {file_path}")
            return False, None

        # 2. 去重检查 + 标记处理中（锁内）
        async with self._process_lock:
            # 检查是否已在处理中（防止并发重复）
            if hash_val in self._processing_hashes:
                logger.debug(f"图片正在处理中，跳过: {hash_val[:16]}")
                return False, None
            # 检查去重
            if await self._is_duplicate_or_blacklisted(
                hash_val, idx, file_path, is_temp, phash_val
            ):
                return False, None
            # 标记为处理中
            self._processing_hashes.add(hash_val)

        try:
            if to_pending:
                # Manual review needs only the original image and local metadata.
                metadata = await asyncio.to_thread(self._probe_image_metadata, file_path)
                if not metadata.get('format'):
                    return False, None
                async with self._process_lock:
                    return await self._store_to_pending(
                        file_path, is_temp, "", hash_val, extra_meta=extra_meta,
                        phash_val=phash_val, source_url=source_url,
                        original_name=original_name, add_method=add_method,
                        analysis_state="unanalysed",
                    )
            # 3. 缓存检查（锁外；缓存带 VLM model_sig，换模型即失效）
            model_sig = str(await self._vlm_call_service._resolve_vision_provider(event) or "")
            cached = self._get_valid_cache(hash_val, model_sig)
            if cached is not None:
                async with self._process_lock:
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
                        extra_meta=extra_meta,
                        from_cache=True,
                        phash_val=phash_val,
                        to_pending=to_pending,
                        source_url=source_url,
                        original_name=original_name,
                        add_method=add_method,
                        overlay_text=cached.get("overlay_text", ""),
                        emotions=cached.get("emotions", []),
                    )

            # 4. 存入 raw 目录（锁外）
            raw_path = await self._move_to_raw(file_path, hash_val, is_temp)

            # 5. VLM 分类（锁外，耗时操作）
            category, tags, desc, emotion, scenes, overlay_text, emotions = (
                await self.classify_image(
                    event=event,
                    file_path=raw_path,
                    categories=categories,
                    content_filtration=content_filtration,
                )
            )

            # 6. 缓存结果（锁外）
            self._put_image_cache(
                hash_val,
                category,
                tags,
                desc,
                emotion,
                scenes,
                model_sig,
                overlay_text=overlay_text,
                emotions=emotions,
            )

            # 7. 处理分类结果（锁内）
            async with self._process_lock:
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
                    extra_meta=extra_meta,
                    from_cache=False,
                    already_in_raw=True,
                    phash_val=phash_val,
                    to_pending=to_pending,
                    source_url=source_url,
                    original_name=original_name,
                    add_method=add_method,
                    overlay_text=overlay_text,
                    emotions=emotions,
                )

        except Exception as e:
            logger.error(f"处理图片失败 [{file_path}]: {e}")
            # 清理临时文件
            if is_temp:
                if os.path.exists(file_path):
                    await safe_remove_file(file_path)
            raise
        finally:
            # 8. 清理处理中标记（确保总是清理）
            self._processing_hashes.discard(hash_val)

    # ── process_image 的辅助方法 ──

    async def _is_duplicate_or_blacklisted(
        self,
        hash_val: str,
        idx: dict,
        file_path: str,
        is_temp: bool,
        phash_val: str = "",
    ) -> bool:
        """检查图片是否已存在于索引或黑名单中。

        双重去重策略：
        1. SHA256 精确匹配 - 字节完全相同的文件
        2. 感知哈希 (pHash) 视觉相似度 - 编码不同但视觉相同的图片
        """

        async def _cleanup_temp():
            if is_temp and os.path.exists(file_path):
                await safe_remove_file(file_path)

        db = getattr(self.plugin, "db_service", None)
        if db is not None:
            if db.hash_exists(hash_val) or db.is_blacklisted(hash_val):
                await _cleanup_temp()
                return True
            image_phash = phash_val or await self._phash_service.compute_phash(file_path)
            if image_phash:
                for existing_path, existing_phash in db.get_phash_map().items():
                    distance = self._phash_service.hamming_distance(image_phash, existing_phash)
                    if distance <= self._phash_service.PHASH_HAMMING_THRESHOLD:
                        # Older databases can contain first-frame hashes for animations.
                        if not os.path.isfile(existing_path):
                            continue
                        if not await self._phash_service.compute_phash(existing_path):
                            continue
                        logger.debug(f"[去重] 感知哈希命中: 距离={distance}")
                        await _cleanup_temp()
                        return True
        elif any(isinstance(meta, dict) and meta.get("hash") == hash_val for meta in idx.values()):
            await _cleanup_temp()
            return True

        return False

    def _get_valid_cache(self, hash_val: str, model_sig: str = "") -> dict | None:
        """获取有效（未过期且模型签名一致）的分类缓存，过期或换模型则清除。"""
        cached = self._image_cache.get(hash_val)
        if cached is None:
            return None
        if str(cached.get("model_sig", "") or "") != str(model_sig or ""):
            # VLM 模型已更换：旧分类结果不可复用
            self._image_cache.pop(hash_val, None)
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
        model_sig: str = "",
        overlay_text: str = "",
        emotions: list | None = None,
    ) -> None:
        """写入分类缓存并淘汰过期条目。"""
        self._image_cache[hash_val] = {
            "category": category,
            "tags": tags,
            "desc": desc,
            "emotion": emotion,
            "scenes": scenes,
            "overlay_text": overlay_text or "",
            "emotions": list(emotions or []),
            "model_sig": str(model_sig or ""),
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
        extra_meta: dict[str, Any] | None = None,
        from_cache: bool = False,
        already_in_raw: bool = False,
        phash_val: str = "",
        to_pending: bool = False,
        source_url: str = "",
        original_name: str = "",
        add_method: str = "auto",
        overlay_text: str = "",
        emotions: list | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """根据分类结果决定存储、跳过或清理。"""
        source = "缓存" if from_cache else "VLM"

        # 过滤不通过
        if category == self.CATEGORY_FILTERED or emotion == self.CATEGORY_FILTERED:
            logger.debug(f"图片过滤不通过（{source}）: {hash_val}")
            if is_temp and os.path.exists(file_path):
                await safe_remove_file(file_path)
            elif not from_cache and os.path.exists(file_path):
                await safe_remove_file(file_path)
            return False, None

        # 非表情包
        if category == self.CATEGORY_NOT_EMOJI or emotion == self.CATEGORY_NOT_EMOJI:
            logger.debug(f"非表情包（{source}）: {hash_val}")
            if is_temp and os.path.exists(file_path):
                await safe_remove_file(file_path)
            elif not from_cache and os.path.exists(file_path):
                await safe_remove_file(file_path)
            return False, None

        if not category or category == "other" or category not in self.categories:
            fallback = (
                self.plugin_config.closest_category(category)
                if self.plugin_config
                else "confused"
            )
            logger.info(f"分类无效（{source}）: {category!r}，归入 {fallback}")
            category = fallback

        semantic_meta = dict(extra_meta) if isinstance(extra_meta, dict) else {}
        if overlay_text:
            semantic_meta["overlay_text"] = overlay_text
        if emotions:
            semantic_meta["emotions"] = list(emotions)

        logger.debug(f"分类有效（{source}）: {category}")
        if to_pending:
            return await self._store_to_pending(
                file_path,
                is_temp,
                category,
                hash_val,
                extra_meta=semantic_meta,
                tags=tags,
                desc=desc,
                scenes=scenes,
                already_in_raw=already_in_raw,
                phash_val=phash_val,
                source_url=source_url,
                original_name=original_name,
                add_method=add_method,
            )
        return await self._store_and_index_image(
            file_path,
            is_temp,
            category,
            hash_val,
            idx,
            extra_meta=semantic_meta,
            tags=tags,
            desc=desc,
            scenes=scenes,
            already_in_raw=already_in_raw,
            phash_val=phash_val,
            source_url=source_url,
            original_name=original_name,
            add_method=add_method,
        )

    async def steal_image_direct(
        self,
        file_path: str,
        category: str,
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
        is_temp: bool = False,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """LLM 自行打标后直接入库，跳过 VLM 分类。

        执行哈希计算、去重检查和文件存储，将标注合并到索引后写入数据库。

        Returns:
            (成功与否, 结果消息)
        """
        base_path = Path(file_path)
        if not base_path.exists():
            return False, f"图片文件不存在: {file_path}"

        if not category:
            return False, "分类为空"
        if category == "other" or category not in self.categories:
            return False, f"分类 '{category}' 不在可用分类列表中"

        normalized_tags = normalize_label_list(tags or [], allow_duplicates=True)
        normalized_desc = str(desc or "").strip()
        normalized_scenes = normalize_label_list(scenes or [], allow_duplicates=True)

        # 描述缺失时使用分类的中文显示名。
        if not normalized_desc:
            cat_info = self.plugin_config.category_info or {}
            cat_meta = cat_info.get(category, {}) if isinstance(cat_info, dict) else {}
            cn_name = str(cat_meta.get("name", "") or "").strip()
            cat_display = cn_name or category
            normalized_desc = f"来自群聊的{cat_display}表情包"

        # 1. 加载完整索引（避免覆盖已有数据）
        full_idx = {}

        # 2. 计算哈希
        hash_val = await self._compute_hash(file_path)
        if not hash_val:
            return False, "无法计算图片哈希"

        # 3. 去重检查（使用完整索引）
        async with self._process_lock:
            if hash_val in self._processing_hashes:
                return False, "图片正在处理中，跳过"
            if await self._is_duplicate_or_blacklisted(hash_val, full_idx, file_path, is_temp):
                return False, "图片已存在或相似度过高"
            self._processing_hashes.add(hash_val)

        try:
            # 4. 移到 raw 目录
            raw_path = await self._move_to_raw(file_path, hash_val, is_temp)

            # 5. 直接存储到分类目录并更新完整索引
            success, merged_idx = await self._store_and_index_image(
                file_path=raw_path,
                is_temp=False,
                category=category,
                hash_val=hash_val,
                idx=full_idx,
                extra_meta=extra_meta,
                tags=normalized_tags,
                desc=normalized_desc,
                scenes=normalized_scenes,
                already_in_raw=True,
                add_method="llm",
            )

            if success and merged_idx is not None:
                tag_hint = f"，标签: {', '.join(normalized_tags)}" if normalized_tags else ""
                scene_hint = f"，场景: {', '.join(normalized_scenes)}" if normalized_scenes else ""
                return True, f"偷取成功！已入库到分类 '{category}'{tag_hint}{scene_hint}"

            return False, "存储图片失败"

        except Exception as e:
            logger.error(f"LLM 直接入库失败 [{file_path}]: {e}")
            return False, f"入库失败: {e}"
        finally:
            self._processing_hashes.discard(hash_val)

    async def classify_image(
        self,
        event: AstrMessageEvent | None,
        file_path: str,
        categories=None,
        content_filtration=None,
    ) -> tuple[str, list[str], str, str, list[str], str, list[str]]:
        """使用视觉模型对图片进行分类并返回详细信息。

        Args:
            event: 消息事件
            file_path: 图片绝对路径
            categories: 分类列表（可选，默认使用 self.categories）
            content_filtration: 是否进行内容过滤（可选，默认使用 self.content_filtration）

        Returns:
            tuple: (category, tags, desc, emotion, scenes, overlay_text, emotions)
        """
        # 调用视觉模型前验证输入文件。
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"分类图片时文件不存在: {file_path}")

        try:
            # 确定是否进行内容过滤
            should_filter = (
                content_filtration if content_filtration is not None else self.content_filtration
            )

            prompt_categories = categories if isinstance(categories, list) else None
            if prompt_categories is None:
                prompt_categories = self.categories or []
            if not prompt_categories:
                raise ValueError("未配置可用分类，无法进行图片分类")
            prompt = self._prompt_manager.build_classification_prompt(
                use_filter=should_filter, categories=prompt_categories
            )

            # 调用视觉模型
            with protect_file(file_path):
                response = await self._vlm_call_service._call_vision_model(
                    event, file_path, prompt,
                    validate_response=self._classification_parser.validate_response,
                )

            # 解析JSON响应
            return self._classification_parser._parse_classification_response(response, file_path)

        except (FileNotFoundError, ValueError):
            # 配置错误 / 文件不存在，直接抛出不吞异常
            raise
        except Exception as e:
            logger.error(f"图片分类失败 [{file_path}]: {e}")
            return "", [], "", "", [], "", []

    @staticmethod
    def _probe_image_metadata(file_path: str) -> dict[str, Any]:
        """探测图片宽高、格式和字节数。

        失败时静默返回空 dict（不阻塞入库）。
        """
        meta: dict[str, Any] = {}
        try:
            if PILImage is not None and file_path and os.path.exists(file_path):
                with PILImage.open(file_path) as img:
                    meta["width"], meta["height"] = img.size
                    fmt = str(img.format or "").lower()
                    meta["format"] = fmt if fmt else None
        except Exception:
            pass
        try:
            if file_path and os.path.exists(file_path):
                meta["bytes"] = os.path.getsize(file_path)
        except Exception:
            pass
        return meta

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
                while chunk := f.read(65536):
                    hasher.update(chunk)
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
        if len(self._image_cache) <= self.IMAGE_CACHE_MAX_SIZE:
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

    def cleanup(self):
        """清理资源。"""
        self._image_cache.clear()
        logger.debug("ImageProcessorService 资源已清理")
