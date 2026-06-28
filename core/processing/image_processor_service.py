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

        # 提示词配置：正常运行时由 prompts.json 加载并通过 update_config 注入，
        # 以下仅为 prompts.json 缺失时的最小化 fallback
        _FALLBACK_PROMPT = (
            "分析表情包：从 `{emotion_list}` 中选择情绪分类。"
            '返回JSON格式：{"category": "分类名", "tags": ["标签1", "标签2"], '
            '"description": "画面描述", "scenes": ["场景1", "场景2"]}'
        )
        _FALLBACK_FILTER_PROMPT = (
            '审核图片是否含不当内容，不当则返回{"approved": false, "reason": "审核不通过"}。'
            "否则从 `{emotion_list}` 中选择情绪分类。"
            '返回JSON格式：{"approved": true, "category": "分类名", "tags": ["标签1"], '
            '"description": "画面描述", "scenes": ["场景1"]}'
        )

        self.emoji_classification_prompt = getattr(
            plugin_instance, "EMOJI_CLASSIFICATION_PROMPT", _FALLBACK_PROMPT
        )
        self.emoji_classification_with_filter_prompt = getattr(
            plugin_instance,
            "EMOJI_CLASSIFICATION_WITH_FILTER_PROMPT",
            _FALLBACK_FILTER_PROMPT,
        )

        # 配置参数（初始值从 plugin_config 读取，后续通过 update_config 更新）
        self.categories = list(self.plugin_config.categories or []) if self.plugin_config else []
        self.content_filtration = (
            bool(getattr(self.plugin_config, "content_filtration", False))
            if self.plugin_config
            else False
        )
        self.vision_provider_id = (
            str(self.plugin_config.vision_provider_id or "") if self.plugin_config else ""
        )
        # 框架 VLM provider 缓存，None 表示未查询过
        self._cached_framework_vlm_id: str | None = None

        # 子服务（职责拆分）
        from .prompt_manager import PromptManager
        from .phash_dedup_service import PHashDedupService
        from .image_render_service import ImageRenderService
        from .classification_parser import ClassificationParser
        from .vlm_call_service import VLMCallService

        self._prompt_manager = PromptManager(plugin_instance)
        self._phash_service = PHashDedupService(plugin_instance)
        self._render_service = ImageRenderService(plugin_instance)
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
            self.vision_provider_id = vision_provider_id
            # 插件 provider 变更时，重置框架缓存以便重新解析
            self._cached_framework_vlm_id = None
        if emoji_classification_prompt is not None:
            self.emoji_classification_prompt = emoji_classification_prompt
        if emoji_classification_with_filter_prompt is not None:
            self.emoji_classification_with_filter_prompt = emoji_classification_with_filter_prompt
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
            "use_count": 0,
            "last_used_at": 0,
        }
        if phash_val:
            entry["phash"] = phash_val
        if tags:
            entry["tags"] = tags
        if desc:
            entry["desc"] = desc
        if scenes:
            entry["scenes"] = scenes
        if extra_meta and isinstance(extra_meta, dict):
            # Avoid overriding core fields that stealer relies on.
            for k, v in extra_meta.items():
                if k in {"hash", "category", "created_at", "use_count", "last_used_at"}:
                    continue
                entry[k] = v
        idx[cat_path] = entry

        # 入库后写入嵌入向量（失败不阻塞）
        try:
            smart_service = getattr(
                getattr(self.plugin, "emoji_selector", None),
                "_smart_select_service", None
            )
            if smart_service and smart_service._embedding_service:
                await smart_service._embedding_service.insert_emoji(cat_path, entry)
                smart_service._invalidate_embedding_index()
        except Exception as e:
            logger.debug(f"嵌入写入失败（不阻塞入库）: {e}")

        return True, idx

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
                    await self.plugin._safe_remove_file(file_path)
                return False, None
            # 查待审核池
            if db.get_pending_by_hash(hash_val):
                logger.debug(f"pending: 哈希已在待审核池中，跳过: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await self.plugin._safe_remove_file(file_path)
                return False, None
            # 查黑名单
            if hasattr(db, "blacklisted_hashes") and hash_val in db.blacklisted_hashes():
                logger.debug(f"pending: 哈希已在黑名单中，跳过: {hash_val}")
                if is_temp and os.path.exists(file_path):
                    await self.plugin._safe_remove_file(file_path)
                return False, None

        pending_dir = self.plugin_config.ensure_pending_dir()
        if not pending_dir:
            logger.warning("pending: 无法创建 pending 目录")
            return False, None

        base_path = Path(file_path)
        ext = base_path.suffix.lower() if base_path.suffix else ".jpg"
        pending_filename = f"{int(time.time())}_{hash_val[:8]}{ext}"
        pending_path = str(pending_dir / pending_filename)

        try:
            # 源文件统一 move 到 pending：temp（cache 命中/on_message）或 raw（主分支）均是中间文件
            if os.path.abspath(file_path) != os.path.abspath(pending_path):
                await asyncio.to_thread(shutil.move, file_path, pending_path)
        except Exception as e:
            logger.error(f"pending: 移动文件到 pending 失败: {file_path} -> {pending_path}, {e}")
            return False, None

        if not os.path.exists(pending_path):
            logger.warning(f"pending: 落盘后文件不存在: {pending_path}")
            return False, None

        meta: dict[str, Any] = {
            "path": pending_path,
            "hash": hash_val,
            "phash": phash_val or None,
            "category": category,
            "desc": desc or None,
            "tags": tags or [],
            "scenes": scenes or [],
        }
        if extra_meta and isinstance(extra_meta, dict):
            for k, v in extra_meta.items():
                if k in {"path", "hash", "phash", "category", "desc", "tags", "scenes"}:
                    continue
                meta[k] = v
            if "scope_mode" not in extra_meta:
                meta.setdefault("scope_mode", "public")

        try:
            pending_id = await self.plugin.db_service.insert_pending(meta)
        except Exception as e:
            logger.error(f"pending: 写 emoji_pending 失败，回滚删除 pending 文件: {e}")
            await self.plugin._safe_remove_file(pending_path)
            return False, None

        if pending_id is None:
            # 同 path 已在 pending（UNIQUE 冲突）：删除本次落盘的多余文件，保持池内唯一
            logger.debug(f"pending: 已存在同路径记录，移除重复文件: {pending_path}")
            await self.plugin._safe_remove_file(pending_path)
            return False, None

        logger.info(
            f"pending: 已存入待审核池 id={pending_id} cat={category} hash={hash_val[:16]}"
        )
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
            to_pending: 落点为待审核池（on_message 自动偷取专用）。
                为 True 时分类有效后写入 pending 目录与 emoji_pending 表，
                不写 emoji/tag/scene/embedding；force_capture/工具/WebUI 不传此参数。

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
            # 3. 缓存检查（锁外）
            cached = self._get_valid_cache(hash_val)
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
                    )

            # 4. 存入 raw 目录（锁外）
            raw_path = await self._move_to_raw(file_path, hash_val, is_temp)

            # 5. VLM 分类（锁外，耗时操作）
            category, tags, desc, emotion, scenes = await self.classify_image(
                event=event,
                file_path=raw_path,
                categories=categories,
                content_filtration=content_filtration,
            )

            # 6. 缓存结果（锁外）
            self._put_image_cache(hash_val, category, tags, desc, emotion, scenes)

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
                )

        except Exception as e:
            logger.error(f"处理图片失败 [{file_path}]: {e}")
            # 清理临时文件
            if is_temp and os.path.exists(file_path):
                await self.plugin._safe_remove_file(file_path)
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
                await self.plugin._safe_remove_file(file_path)

        # 持久化索引
        if hasattr(self.plugin, "cache_service"):
            db_service = getattr(self.plugin, "db_service", None)

            # 1) SHA256 精确匹配 — SQL 替代全量索引
            if db_service and db_service.hash_exists(hash_val):
                logger.debug(f"[去重] SHA256 精确匹配命中: {hash_val[:16]}...")
                await _cleanup_temp()
                return True

            # 2) 感知哈希视觉相似度匹配 — 只查 pHash 映射
            _phash = phash_val or await self._phash_service.compute_phash(file_path)
            if _phash and db_service:
                phash_map = db_service.get_phash_map()
                for entry_path, existing_phash in phash_map.items():
                    distance = self._phash_service.hamming_distance(_phash, existing_phash)
                    if distance <= self._phash_service.PHASH_HAMMING_THRESHOLD:
                        logger.info(
                            f"[去重] 感知哈希相似度匹配命中: "
                            f"距离={distance}, 阈值={self._phash_service.PHASH_HAMMING_THRESHOLD}, "
                            f"已有={entry_path}"
                        )
                        await _cleanup_temp()
                        return True

            # 3) 黑名单检查
            db = getattr(self.plugin, "db_service", None)
            if db is not None and hasattr(db, "blacklisted_hashes"):
                blacklist = db.blacklisted_hashes()
            else:
                blacklist = self.plugin.cache_service.get_cache("blacklist_cache")
            if blacklist and hash_val in blacklist:
                logger.debug(f"[去重] 图片在黑名单中: {hash_val[:16]}...")
                await _cleanup_temp()
                return True
        else:
            # 无 cache_service 时回退到传入的 idx
            db_service = getattr(self.plugin, "db_service", None)
            if db_service and db_service.hash_exists(hash_val):
                logger.debug(f"[去重] SHA256 匹配命中 (DB): {hash_val[:16]}...")
                await _cleanup_temp()
                return True

            for v in idx.values():
                if isinstance(v, dict) and v.get("hash") == hash_val:
                    logger.debug(f"[去重] SHA256 匹配命中 (idx): {hash_val[:16]}...")
                    await _cleanup_temp()
                    return True

            if phash_val and db_service:
                phash_map = db_service.get_phash_map()
                for entry_path, existing_phash in phash_map.items():
                    distance = self._phash_service.hamming_distance(phash_val, existing_phash)
                    if distance <= self._phash_service.PHASH_HAMMING_THRESHOLD:
                        logger.info(
                            f"[去重] 感知哈希匹配命中 (DB): 距离={distance}, 已有={entry_path}"
                        )
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
        extra_meta: dict[str, Any] | None = None,
        from_cache: bool = False,
        already_in_raw: bool = False,
        phash_val: str = "",
        to_pending: bool = False,
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
            if to_pending:
                return await self._store_to_pending(
                    file_path,
                    is_temp,
                    category,
                    hash_val,
                    extra_meta=extra_meta,
                    tags=tags,
                    desc=desc,
                    scenes=scenes,
                    already_in_raw=already_in_raw,
                    phash_val=phash_val,
                )
            return await self._store_and_index_image(
                file_path,
                is_temp,
                category,
                hash_val,
                idx,
                extra_meta=extra_meta,
                tags=tags,
                desc=desc,
                scenes=scenes,
                already_in_raw=already_in_raw,
                phash_val=phash_val,
            )

        # 无效分类
        logger.warning(f"分类无效（{source}）: {category!r}，清理文件")
        if os.path.exists(file_path):
            await self.plugin._safe_remove_file(file_path)
        return False, None

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

        仍然执行哈希计算、去重检查、文件存储和索引更新。
        从数据库加载完整索引，合并后写回，不会覆盖已有数据。

        Returns:
            (成功与否, 结果消息)
        """
        base_path = Path(file_path)
        if not base_path.exists():
            return False, f"图片文件不存在: {file_path}"

        if not category or category not in self.categories:
            return False, f"分类 '{category}' 不在可用分类列表中"

        normalized_tags = [str(t).strip() for t in (tags or []) if t and str(t).strip()]
        normalized_desc = str(desc or "").strip()
        normalized_scenes = [str(s).strip() for s in (scenes or []) if s and str(s).strip()]

        # 兜底描述：用分类信息中的中文名，而不是干巴巴的"LLM手动入库"
        if not normalized_desc:
            cat_info = self.plugin_config.category_info or {}
            cat_meta = cat_info.get(category, {}) if isinstance(cat_info, dict) else {}
            cn_name = str(cat_meta.get("name", "") or "").strip()
            cat_display = cn_name or category
            normalized_desc = f"来自群聊的{cat_display}表情包"

        # 1. 加载完整索引（避免覆盖已有数据）
        full_idx = await self.plugin._load_index()

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
            )

            if success and merged_idx is not None:
                await self.plugin._save_index(merged_idx)
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
    ) -> tuple[str, list[str], str, str, list[str]]:
        """使用视觉模型对图片进行分类并返回详细信息。

        Args:
            event: 消息事件
            file_path: 图片绝对路径
            categories: 分类列表（可选，默认使用 self.categories）
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
            response = await self._call_vision_model(event, file_path, prompt)

            # 解析JSON响应
            return self._parse_classification_response(response, file_path)

        except (FileNotFoundError, ValueError):
            # 配置错误 / 文件不存在，直接抛出不吞异常
            raise
        except Exception as e:
            logger.error(f"图片分类失败 [{file_path}]: {e}")
            return "", [], "", "", []

    def _normalize_category(self, raw: str) -> str:
        """将 VLM 返回的分类文本规范化为有效分类名。

        无法识别或输出 unknown 时返回空字符串，由调用方决定如何处理。
        支持处理带前缀的格式（如 "审核通过：surprised"），提取冒号后的内容。
        """
        # 清理输入
        raw = self._sanitize_model_scalar(raw).lower()

        # unknown 或空值视为分类失败
        if not raw or raw == "unknown":
            logger.debug(f"[分类规范化] 分类失败: {raw!r}")
            return ""

        # 处理带前缀的格式，如 "审核通过：surprised" -> "surprised"
        # 也处理 "审核不通过：过滤不通过" 这种情况
        if "：" in raw or ":" in raw:
            # 统一使用英文冒号处理
            normalized_raw = raw.replace("：", ":")
            # 提取冒号后的部分
            if ":" in normalized_raw:
                _, _, category_part = normalized_raw.partition(":")
                raw = category_part.strip()

        if self.plugin_config:
            try:
                normalized = self.plugin_config.normalize_category_strict(raw)
                if normalized and normalized in self.categories:
                    return normalized
            except Exception as e:
                logger.debug(f"[分类规范化] 异常: {e}")

        logger.warning(f"无法识别情绪分类: {raw!r}，跳过该图片")
        return ""

    # ===== 门面委托：ClassificationParser =====

    def _parse_classification_response(self, response: str, file_path: str):
        """Parse the classification payload returned by the VLM（已迁移到 ClassificationParser）。"""
        return self._classification_parser._parse_classification_response(response, file_path)

    def _sanitize_model_scalar(self, value):
        """Normalize single-value model outputs（已迁移到 ClassificationParser）。"""
        return self._classification_parser._sanitize_model_scalar(value)

    def _extract_json_payload(self, response: str):
        """Extract JSON payload from VLM response（已迁移到 ClassificationParser）。"""
        return self._classification_parser._extract_json_payload(response)

    def _try_parse_json_candidate(self, text: str):
        """Try to parse JSON candidate（已迁移到 ClassificationParser）。"""
        return self._classification_parser._try_parse_json_candidate(text)

    def _parse_legacy_format(self, response: str):
        """Parse legacy format（已迁移到 ClassificationParser）。"""
        return self._classification_parser._parse_legacy_format(response)

    # ===== 门面委托：VLMCallService =====

    async def _call_vision_model(self, event, img_path: str, prompt: str):
        """Call vision model（已迁移到 VLMCallService）。"""
        return await self._vlm_call_service._call_vision_model(event, img_path, prompt)

    async def _prepare_image_for_vlm(self, img_path: str):
        """Prepare image for VLM（已迁移到 VLMCallService）。"""
        return await self._vlm_call_service._prepare_image_for_vlm(img_path)

    async def _do_vlm_call(self, provider_id: str, prompt: str, file_url: str):
        """Execute VLM call（已迁移到 VLMCallService）。"""
        return await self._vlm_call_service._do_vlm_call(provider_id, prompt, file_url)

    async def _llm_generate_with_image_compat(self, provider_id: str, prompt: str, file_url: str):
        """Compat wrapper for llm_generate（已迁移到 VLMCallService）。"""
        return await self._vlm_call_service._llm_generate_with_image_compat(
            provider_id, prompt, file_url
        )

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
        self._render_service.cleanup()
        logger.debug("ImageProcessorService 资源已清理")

    async def _file_to_base64(self, file_path: str) -> str:
        """将文件转换为 base64 编码（门面方法，委托给 ImageRenderService）。"""
        return await self._render_service.file_to_base64(file_path)

    async def _file_to_gif_base64(self, file_path: str) -> str:
        """将文件转换为 GIF 格式的 base64 编码（门面方法，委托给 ImageRenderService）。"""
        return await self._render_service.file_to_gif_base64(file_path)

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
        except FileNotFoundError:
            # 文件可能被其他进程删除，属于正常情况
            logger.debug(f"文件已被删除: {file_path}")
            return True
        except PermissionError as e:
            logger.warning(f"删除文件权限不足: {file_path}, 错误: {e}")
            return False
        except Exception as e:
            logger.error(f"删除文件失败: {file_path}, 错误: {e}")
            return False

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
            logger.debug("[视觉模型] 插件未配置 vision_provider_id，尝试使用框架全局配置")

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

    # ── 渲染门面方法（委托给 ImageRenderService）────────────────

    async def render_emoji_list_page_file(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """使用 AstrBot 内置 html-to-pic 渲染列表，返回本地图片文件路径（门面方法）。"""
        return await self._render_service.render_emoji_list_page_file(
            items=items,
            page=page,
            total_pages=total_pages,
            total_filtered=total_filtered,
            total_all=total_all,
            category=category,
            per_page=per_page,
        )

    async def render_emoji_list_page_url(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """使用 AstrBot 内置 html-to-pic 渲染列表，返回可公网访问的图片 URL（门面方法）。"""
        return await self._render_service.render_emoji_list_page_url(
            items=items,
            page=page,
            total_pages=total_pages,
            total_filtered=total_filtered,
            total_all=total_all,
            category=category,
            per_page=per_page,
        )

    async def render_emoji_list_page_base64(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """把表情包列表渲染成一张 PNG，并返回 base64（门面方法）。"""
        return await self._render_service.render_emoji_list_page_base64(
            items=items,
            page=page,
            total_pages=total_pages,
            total_filtered=total_filtered,
            total_all=total_all,
            category=category,
            per_page=per_page,
        )
