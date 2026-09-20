"""索引管理器：负责索引加载、持久化、重建和迁移。"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from astrbot.api import logger




def invalidate_search(plugin) -> None:
    """统一失效所有检索缓存，避免各入口分别维护索引状态。"""
    try:
        selector = getattr(plugin, "meme_selector", None)
        if selector is not None:
            selector._invalidate_bm25_index()
            smart = getattr(selector, "_smart_select_service", None)
            if smart is not None:
                smart._invalidate_embedding_index()
    except Exception as exc:
        logger.warning(f"检索缓存失效失败: {exc}")


async def refresh_search_entry(plugin, path: str, entry=None, *, previous_path: str | None = None) -> None:
    """同步新增、更新、移动或删除后的向量；模型故障不回滚图库写入。"""
    invalidate_search(plugin)
    try:
        smart = getattr(getattr(plugin, "meme_selector", None), "_smart_select_service", None)
        embedding = getattr(smart, "_embedding_service", None)
        if embedding is None:
            return
        db = getattr(plugin, "db_service", None)
        if entry is None and db is not None:
            entry = db.get_emoji(path)
        if previous_path and previous_path != path:
            await embedding.delete_by_path(previous_path)
        await embedding.delete_by_path(path)
        if entry:
            await embedding.insert_emoji(path, entry)
    except Exception as exc:
        logger.warning(f"检索索引同步失败 [{path}]: {exc}")


async def delete_index_paths(plugin, paths: list[str]) -> int:
    """显式删除选定记录，并通过同一入口移除向量。"""
    db = getattr(plugin, "db_service", None)
    if db is None or not hasattr(db, "delete_paths"):
        return 0
    removed = await db.delete_paths(paths)
    for path in paths:
        await refresh_search_entry(plugin, path)
    return removed


class IndexManager:
    """管理表情包索引的生命周期（加载、保存、重建、迁移）。"""

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance
        self.db_service = getattr(plugin_instance, "db_service", None)
        self.base_dir = getattr(plugin_instance, "base_dir", None)
        self.categories_dir = getattr(plugin_instance, "categories_dir", None)
        self.cache_dir = getattr(plugin_instance, "cache_dir", None)
        self._migration_done: bool = False

    async def load_index(self) -> dict[str, Any]:
        """从数据库加载索引；DB 为空时尝试从旧 JSON 一次性迁移。

        Returns:
            dict[str, Any]: 索引数据（兼容旧接口的字典格式）
        """
        try:
            idx: dict[str, Any] = {}
            if self.db_service is None:
                return idx

            db_count = self.db_service.count_total()

            if db_count > 0:
                logger.debug(f"[DB] 从数据库加载 {db_count} 条索引")
                return self.db_service.get_index_cache_readonly()

            if self._migration_done:
                return idx

            # 旧实例迁移：cache/index_cache.json 存在时一次性导入 DB 后删除
            old_json_path = self.cache_dir / "index_cache.json" if self.cache_dir else None
            if old_json_path and old_json_path.exists():
                migrated = await self.db_service.migrate_from_json(old_json_path)
                if migrated > 0:
                    self._migration_done = True
                    logger.info(f"[DB] 迁移了 {migrated} 条旧记录到数据库")
                    return self.db_service.get_index_cache_readonly()

            # 兜底：从更老的 JSON 位置（base_dir/index.json 等）合并
            if self.base_dir:
                legacy_data = await self.migrate_legacy_data(self.base_dir)
                if legacy_data:
                    await self.db_service.save_index(legacy_data)
                    self._migration_done = True
                    logger.info("[DB] 迁移旧数据到数据库完成")
                    return self.db_service.get_index_cache_readonly()

            self._migration_done = True
            return idx

        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def save_index(self, idx: dict[str, Any]) -> None:
        """增量提交后刷新有语义变化的条目，未出现的路径保持不变。"""
        if self.db_service is None:
            return
        changed = await self.db_service.sync_index(idx)
        invalidate_search(self.plugin)
        for path in changed or []:
            await refresh_search_entry(self.plugin, path)

    def _iter_legacy_index_paths(self, base_dir) -> list[Path]:
        """收集所有可能的旧版索引 JSON 路径。"""
        base_dir_path = Path(base_dir) if base_dir else None
        roots: list[Path] = [self.cache_dir / "index_cache.json"]

        if base_dir_path:
            roots.extend(
                [
                    base_dir_path / "index.json",
                    base_dir_path / "image_index.json",
                    base_dir_path / "cache" / "index.json",
                    base_dir_path / "cache" / "index_cache.json",
                ]
            )

        candidates: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            for path in (
                root,
                root.with_suffix(root.suffix + ".migrated"),
                root.with_suffix(root.suffix + ".backup"),
            ):
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
        return candidates

    @staticmethod
    def _legacy_record_score(info: dict[str, Any]) -> int:
        """为旧记录打分，优先保留 metadata 更完整的版本。"""
        score = 0
        if str(info.get("desc", "") or "").strip():
            score += 3
        if info.get("tags"):
            score += 3
        if info.get("scenes") or info.get("scene"):
            score += 2
        if str(info.get("hash", "") or "").strip():
            score += 2
        for key in (
            "source",
            "origin_target",
            "scope_mode",
            "qq_emoji_id",
            "qq_emoji_package_id",
            "origin_url",
            "qq_key",
            "phash",
        ):
            if info.get(key):
                score += 1
        return score

    async def load_legacy_index_data(
        self, base_dir
    ) -> tuple[dict[str, Any], list[Path]]:
        """加载旧版 JSON 索引，并按路径合并出信息最完整的记录。"""

        def load_old_file(path: Path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        merged_data: dict[str, Any] = {}
        loaded_paths: list[Path] = []

        for old_path in self._iter_legacy_index_paths(base_dir):
            if not old_path.exists():
                continue

            try:
                old_data = await asyncio.to_thread(load_old_file, old_path)
            except Exception as e:
                logger.warning(f"Failed to load legacy index file {old_path}: {e}")
                continue

            if not isinstance(old_data, dict) or not old_data:
                continue

            loaded_paths.append(old_path)
            logger.info(f"Loaded {len(old_data)} legacy records from {old_path}")

            for record_path, record in old_data.items():
                if not isinstance(record_path, str) or not isinstance(record, dict):
                    continue

                # 相同 path 存在多份旧记录时，优先保留 metadata 更丰富的版本。
                existing = merged_data.get(record_path)
                if existing is None or self._legacy_record_score(
                    record
                ) > self._legacy_record_score(existing):
                    merged_data[record_path] = dict(record)

        return merged_data, loaded_paths

    async def migrate_legacy_data(self, base_dir) -> dict[str, Any]:
        """加载旧版 JSON 索引（一次性 DB 迁移辅助），返回合并后的数据。

        调用方负责将返回结果写入数据库。
        """
        try:
            import shutil

            logger.info("Starting legacy index migration scan")
            migrated_data, loaded_paths = await self.load_legacy_index_data(base_dir)
            if not migrated_data:
                logger.info("No legacy index JSON files found")
                return {}

            for old_path in loaded_paths:
                if old_path.suffix.endswith("backup") or old_path.suffix.endswith(
                    "migrated"
                ):
                    continue
                # 仅对原始 JSON 做一次 .backup，避免反复覆盖已有备份文件。
                backup_path = old_path.with_suffix(old_path.suffix + ".backup")
                try:
                    await asyncio.to_thread(shutil.copy2, old_path, backup_path)
                    if backup_path.exists():
                        logger.info(f"Backed up legacy index file to {backup_path}")
                except Exception as backup_err:
                    logger.warning(
                        f"Failed to back up legacy index file: {backup_err}"
                    )

            logger.info(
                f"Loaded {len(migrated_data)} legacy records (caller persists to DB)"
            )
            return migrated_data
        except Exception as e:
            logger.error(f"Legacy data migration failed: {e}", exc_info=True)
            return {}

    async def rebuild_index_from_files(self) -> dict[str, Any]:
        """从 `categories` 目录重建最小索引。"""
        try:
            rebuilt_index: dict[str, Any] = {}
            categories_dir_path = Path(self.categories_dir)
            if not categories_dir_path.exists():
                return rebuilt_index

            for category_dir in categories_dir_path.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name
                logger.info(f"Rebuilding index for category '{category_name}'")

                for img_file in category_dir.iterdir():
                    if not img_file.is_file():
                        continue
                    if img_file.suffix.lower() not in {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".webp",
                    }:
                        continue

                    # 重建时直接使用分类目录里的实际文件路径作为索引路径。
                    path_str = str(img_file)
                    try:
                        file_hash = hashlib.sha256(img_file.read_bytes()).hexdigest()
                    except Exception as e:
                        logger.debug(f"Failed to compute file hash for {img_file}: {e}")
                        file_hash = ""

                    rebuilt_index[path_str] = {
                        "hash": file_hash,
                        "category": category_name,
                        "created_at": int(img_file.stat().st_mtime),
                    }

            logger.info(f"Rebuilt {len(rebuilt_index)} index records from files")
            return rebuilt_index
        except Exception as e:
            logger.error(f"Failed to rebuild index from files: {e}", exc_info=True)
            return {}

    async def migrate_blacklist(self) -> None:
        """将旧的 blacklist_cache.json 迁移到数据库 blacklist 表。

        幂等：DB 已有同样 hash 时跳过。旧 JSON 保留作备份，运行期只读写数据库。
        """
        try:
            db = self.db_service
            if db is None or not hasattr(db, "add_blacklist_batch"):
                return
            legacy_path = self.cache_dir / "blacklist_cache.json"
            if not legacy_path.is_file():
                return
            cached = json.loads(await asyncio.to_thread(legacy_path.read_text, encoding="utf-8"))
            if not isinstance(cached, dict) or not cached:
                return
            hashes: dict[str, int] = {}
            for h, ts in cached.items():
                try:
                    hashes[str(h)] = int(ts) if ts else int(time.time())
                except Exception:
                    hashes[str(h)] = 0
            imported = await db.add_blacklist_batch(hashes)
            if imported > 0:
                logger.info(f"[DB] 黑名单从缓存迁移完成，新增 {imported} 条")
        except Exception as e:
            logger.warning(f"[DB] 黑名单迁移失败: {e}")
