import asyncio
import copy
import hashlib
import inspect
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable

from astrbot.api import logger

IndexCache = dict[str, Any]
IndexUpdater = Callable[[IndexCache], Any | Awaitable[Any]]


class CacheService:
    """缓存服务，负责管理内存缓存与可选的 JSON 持久化。"""

    _CACHE_MAX_SIZE = 100

    def __init__(self, cache_dir: str | Path | None = None):
        if not cache_dir:
            from astrbot.api.star import StarTools

            cache_dir = (
                Path(StarTools.get_data_dir("astrbot_plugin_stealer")).resolve()
                / "cache"
            )

        self._cache_dir = Path(cache_dir)
        self._ensure_cache_dir()

        self._caches: dict[str, OrderedDict[str, Any]] = {
            "image_cache": OrderedDict(),
            "text_cache": OrderedDict(),
            "index_cache": OrderedDict(),
            "bm25_cache": OrderedDict(),
            "desc_cache": OrderedDict(),
            "blacklist_cache": OrderedDict(),
        }
        self._no_persist_caches: set[str] = {"index_cache"}
        self._lock = threading.RLock()

        self._load_caches()

    def _load_caches(self) -> None:
        """加载已持久化的缓存文件。"""
        for cache_name in self._caches:
            cache_file = self._get_cache_file(cache_name)
            if not cache_file.exists():
                continue

            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached_data = json.load(f)
                if isinstance(cached_data, dict):
                    self._caches[cache_name] = OrderedDict(cached_data)
                    logger.info(
                        f"[load_caches] loaded {len(cached_data)} items for {cache_name}"
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Skipped invalid cache JSON {cache_file}: {e}")
                self._caches[cache_name] = OrderedDict()
            except OSError as e:
                logger.error(f"Failed to read cache file {cache_file}: {e}")
            except Exception as e:
                logger.error(f"Failed to load cache file {cache_file}: {e}")

    def _ensure_cache_dir(self) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create cache directory {self._cache_dir}: {e}")
            raise Exception(f"Unable to create cache directory: {e}") from e

    def _get_cache_file(self, cache_name: str) -> Path:
        return self._cache_dir / f"{cache_name}.json"

    def _clean_cache(self, cache: OrderedDict[str, Any]) -> None:
        """按 LRU 语义裁剪缓存大小。"""
        while len(cache) > self._CACHE_MAX_SIZE:
            cache.popitem(last=False)

    def _get_sync(self, cache_name: str, key: str) -> Any | None:
        """同步读取缓存，并将命中的条目移动到队尾。"""
        with self._lock:
            if cache_name in self._caches and key in self._caches[cache_name]:
                self._caches[cache_name].move_to_end(key)
                return self._caches[cache_name][key]
        return None

    def _set_sync(self, cache_name: str, key: str, value: Any) -> None:
        """同步写入缓存。"""
        with self._lock:
            if cache_name not in self._caches:
                return
            self._caches[cache_name][key] = value
            self._clean_cache(self._caches[cache_name])

    def _delete_sync(self, cache_name: str, key: str) -> bool:
        """同步删除缓存键。"""
        with self._lock:
            if cache_name in self._caches and key in self._caches[cache_name]:
                del self._caches[cache_name][key]
                return True
        return False

    def _clear_sync(self, cache_name: str | None = None) -> None:
        """同步清空指定缓存，或清空全部缓存。"""
        with self._lock:
            if cache_name:
                if cache_name in self._caches:
                    self._caches[cache_name].clear()
                return

            for name in self._caches:
                self._caches[name].clear()

    def _get_cache_copy_sync(self, cache_name: str) -> dict[str, Any]:
        """返回缓存副本，避免调用方直接修改内部状态。"""
        with self._lock:
            return dict(self._caches.get(cache_name, OrderedDict()))

    def _set_cache_sync(self, cache_name: str, cache_data: dict[str, Any]) -> None:
        """整体替换指定缓存。"""
        with self._lock:
            if cache_name not in self._caches:
                return
            self._caches[cache_name].clear()
            self._caches[cache_name].update(cache_data)
            self._clean_cache(self._caches[cache_name])

    def _save_cache_sync(self, cache_name: str) -> None:
        """将指定缓存写回 JSON 文件。"""
        if cache_name not in self._caches or cache_name in self._no_persist_caches:
            return

        try:
            self._ensure_cache_dir()
            cache_file = self._get_cache_file(cache_name)
            with self._lock:
                data_snapshot = copy.deepcopy(self._caches[cache_name])
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data_snapshot, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved cache file {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache file {cache_name}: {e}", exc_info=True)

    async def get(self, cache_name: str, key: str) -> Any | None:
        return await asyncio.to_thread(self._get_sync, cache_name, key)

    async def set(
        self, cache_name: str, key: str, value: Any, persist: bool = False
    ) -> None:
        await asyncio.to_thread(self._set_sync, cache_name, key, value)
        if persist:
            await asyncio.to_thread(self._save_cache_sync, cache_name)

    async def delete(self, cache_name: str, key: str, persist: bool = False) -> None:
        await asyncio.to_thread(self._delete_sync, cache_name, key)
        if persist:
            await asyncio.to_thread(self._save_cache_sync, cache_name)

    async def clear(self, cache_name: str | None = None, persist: bool = False) -> None:
        await asyncio.to_thread(self._clear_sync, cache_name)
        if persist:
            if cache_name:
                await asyncio.to_thread(self._save_cache_sync, cache_name)
            else:
                for name in self._caches:
                    await asyncio.to_thread(self._save_cache_sync, name)

    async def set_cache(
        self, cache_name: str, cache_data: dict[str, Any], persist: bool = True
    ) -> None:
        await asyncio.to_thread(self._set_cache_sync, cache_name, cache_data)
        if persist:
            await asyncio.to_thread(self._save_cache_sync, cache_name)

    def get_cache(self, cache_name: str) -> dict[str, Any]:
        return self._get_cache_copy_sync(cache_name)

    def get_index_cache(self) -> dict[str, Any]:
        return self._get_cache_copy_sync("index_cache")

    def get_index_cache_readonly(self) -> dict[str, Any]:
        return self._get_cache_copy_sync("index_cache")

    def get_cache_size(self, cache_name: str) -> int:
        with self._lock:
            return len(self._caches.get(cache_name, OrderedDict()))

    def compute_hash(self, data: str | bytes) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    async def load_index(self) -> dict[str, Any]:
        """读取索引缓存。"""
        return await asyncio.to_thread(self._get_cache_copy_sync, "index_cache")

    async def save_index(self, idx: IndexCache) -> None:
        """保存索引缓存。"""
        await self.set_cache("index_cache", idx, persist=True)

    async def update_index(
        self, updater: IndexUpdater, persist: bool = True
    ) -> IndexCache:
        """以原子方式更新索引缓存。支持同步和异步 updater。"""

        async def _update_async() -> IndexCache:
            with self._lock:
                snapshot = copy.deepcopy(
                    self._caches.get("index_cache", OrderedDict())
                )
                # updater 修改 current，失败时 snapshot 保持不变
                current = copy.deepcopy(snapshot)
                result = updater(current)
                # 支持异步 updater
                if inspect.isawaitable(result):
                    await result

                self._caches["index_cache"].clear()
                self._caches["index_cache"].update(current)
                self._clean_cache(self._caches["index_cache"])
                return current

        try:
            updated = await _update_async()
            if persist:
                await asyncio.to_thread(self._save_cache_sync, "index_cache")
            return updated
        except Exception as e:
            logger.error(f"Failed to update index cache: {e}", exc_info=True)
            return await self.load_index()

    async def update_config(self, max_cache_size: int | None = None) -> None:
        if max_cache_size is not None:
            self._CACHE_MAX_SIZE = max_cache_size

            def _clean_all():
                with self._lock:
                    for cache in self._caches.values():
                        self._clean_cache(cache)

            await asyncio.to_thread(_clean_all)
            await self.persist_all()

    async def persist_all(self) -> None:
        for cache_name in self._caches:
            if cache_name not in self._no_persist_caches:
                await asyncio.to_thread(self._save_cache_sync, cache_name)

    async def cleanup(self) -> None:
        await self.persist_all()

    def _iter_legacy_index_paths(self, base_dir) -> list[Path]:
        """收集所有可能的旧版索引 JSON 路径。"""
        base_dir_path = Path(base_dir) if base_dir else None
        roots: list[Path] = [self._cache_dir / "index_cache.json"]

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
        """将旧版 JSON 索引合并进当前缓存。"""
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

            current_index = self._get_cache_copy_sync("index_cache")
            if not current_index:
                # 数据库/缓存为空时，直接采用旧索引，保证新环境也能恢复。
                logger.info(
                    f"Cache is empty, using {len(migrated_data)} legacy records directly"
                )
                await self.save_index(migrated_data)
                return migrated_data

            current_hash_map = {}
            for path, meta in current_index.items():
                if isinstance(meta, dict) and meta.get("hash"):
                    current_hash_map[meta["hash"]] = path

            merged_count = 0
            for old_path, old_info in migrated_data.items():
                if not isinstance(old_info, dict):
                    continue

                target_path = None
                if old_path in current_index:
                    target_path = old_path
                elif old_info.get("hash") in current_hash_map:
                    target_path = current_hash_map[old_info["hash"]]

                if not target_path:
                    continue

                target_info = current_index[target_path]
                updated = False
                # 旧数据只用于补充缺失 metadata，不覆盖当前已有内容。
                if old_info.get("desc") and not target_info.get("desc"):
                    target_info["desc"] = old_info["desc"]
                    updated = True
                if old_info.get("tags") and not target_info.get("tags"):
                    target_info["tags"] = old_info["tags"]
                    updated = True
                if old_info.get("scenes") and not target_info.get("scenes"):
                    target_info["scenes"] = old_info["scenes"]
                    updated = True

                if updated:
                    merged_count += 1

            logger.info(
                f"Recovered metadata for {merged_count} records from legacy data"
            )
            await self.save_index(current_index)
            return current_index
        except Exception as e:
            logger.error(f"Legacy data migration failed: {e}", exc_info=True)
            return {}

    async def rebuild_index_from_files(self, base_dir, categories_dir) -> dict[str, Any]:
        """从 `categories` 目录重建最小索引。"""
        try:
            rebuilt_index: dict[str, Any] = {}
            categories_dir_path = Path(categories_dir)
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
                        file_hash = self.compute_hash(img_file.read_bytes())
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
