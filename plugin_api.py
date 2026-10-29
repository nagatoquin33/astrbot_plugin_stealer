import base64
import binascii
import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


from astrbot.api import logger

from .core.util.blacklist import add_blacklist_hash
from .core.db.index_manager import refresh_search_entry
from .core.processing.semantic_schema import SEARCH_METADATA_FIELDS
from .core.util.normalization import (
    canonicalize_path,
    normalize_character_key,
    normalize_label_list,
    normalize_scope_mode,
)
from .core.util.safe_io import storage_write, file_in_use, commit_image_file, safe_remove_file, unique_image_path, write_image_file

from .api.library import LibraryRoutes
from .api.preferences import PreferenceRoutes
from .api.external_sources import ExternalSourceRoutes
from .api.pending import PendingRoutes
from .api.image_mutations import ImageMutationRoutes
from .api.maintenance import MaintenanceRoutes
from .api.taxonomy import TaxonomyRoutes
from .api.reanalysis import ReanalysisRoutes

PLUGIN_NAME = "astrbot_plugin_stealer"


class PluginAPI(
    LibraryRoutes,
    PreferenceRoutes,
    ExternalSourceRoutes,
    PendingRoutes,
    ImageMutationRoutes,
    MaintenanceRoutes,
    TaxonomyRoutes,
    ReanalysisRoutes,
):
    """Backend API provider for plugin Pages."""

    ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    BATCH_TASK_TTL_SECONDS = 30 * 60
    DASHBOARD_PREFS_KEY = "dashboard_prefs"
    VALID_THEMES = frozenset({"auto", "dark", "light", "minecraft", "fallout"})

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.batch_upload_tasks: dict[str, dict] = {}

    # ── Registration ──────────────────────────────────────────

    def register(self, context) -> None:
        routes: list[tuple[str, str, list[str]]] = [
            ("/images", "handle_list_images", ["GET"]),
            ("/image-data", "handle_image_data", ["GET"]),
            ("/serve-image", "handle_serve_image", ["GET"]),
            ("/thumbnail", "handle_thumbnail", ["GET"]),
            ("/images/upload", "handle_upload_image", ["POST"]),
            ("/images/update", "handle_update_image", ["POST"]),
            ("/images/delete", "handle_delete_image", ["POST"]),
            ("/images/batch-delete", "handle_batch_delete", ["POST"]),
            ("/images/batch-move", "handle_batch_move", ["POST"]),
            ("/images/batch-scope", "handle_batch_scope", ["POST"]),
            ("/images/batch-favorite", "handle_batch_favorite", ["POST"]),
            ("/images/batch-upload", "handle_batch_upload", ["POST"]),
            ("/images/batch-upload-status", "handle_batch_upload_status", ["GET"]),
            ("/images/scope-repair", "handle_scope_repair", ["POST"]),
            ("/analyze", "handle_analyze_image", ["POST"]),
            ("/images/reanalyze", "handle_batch_reanalyze", ["POST"]),
            ("/images/reanalyze/status", "handle_batch_reanalyze_status", ["GET"]),
            ("/images/reanalyze/cancel", "handle_batch_reanalyze_cancel", ["POST"]),
            ("/images/reanalyze/result", "handle_batch_reanalyze_result", ["GET"]),
            ("/images/reanalyze/review", "handle_batch_reanalyze_review", ["POST"]),
            ("/images/reanalyze/retry", "handle_batch_reanalyze_retry", ["POST"]),
            ("/storage/scan", "handle_storage_scan", ["GET"]),
            ("/storage/cleanup", "handle_storage_cleanup", ["POST"]),
            ("/stats", "handle_get_stats", ["GET"]),
            ("/pending", "handle_list_pending", ["GET"]),
            ("/pending/approve", "handle_pending_approve", ["POST"]),
            ("/pending/reject", "handle_pending_reject", ["POST"]),
            ("/pending/update", "handle_pending_update", ["POST"]),
            ("/pending/stats", "handle_pending_stats", ["GET"]),
            ("/categories", "handle_categories", ["GET", "POST"]),
            ("/categories/delete", "handle_delete_category", ["POST"]),
            ("/characters", "handle_characters", ["GET", "POST"]),
            ("/characters/delete", "handle_delete_character", ["POST"]),
            ("/images/batch-character", "handle_batch_character", ["POST"]),
            ("/emotions", "handle_get_emotions", ["GET"]),
            ("/health", "handle_health_check", ["GET"]),
            ("/prefs", "handle_prefs", ["GET", "POST"]),
            ("/sources", "handle_sources", ["GET", "POST"]),
            ("/sources/inspect", "handle_source_inspect", ["POST"]),
            ("/sources/import", "handle_source_import", ["POST"]),
            ("/sources/sync", "handle_source_sync", ["POST"]),
            ("/sources/jobs", "handle_source_job", ["GET"]),
            ("/sources/jobs/cancel", "handle_source_job_cancel", ["POST"]),
            ("/sources/delete", "handle_source_delete", ["POST"]),
            ("/sources/upload", "handle_source_upload", ["POST"]),
        ]
        for route, handler_name, methods in routes:
            handler = getattr(self, handler_name)
            context.register_web_api(
                f"/{PLUGIN_NAME}{route}",
                handler,
                methods,
                f"Plugin Page: {handler_name}",
            )

    # ── Helpers ───────────────────────────────────────────────

    @property
    def _data_dir(self) -> Path:
        return self.plugin.base_dir

    @property
    def _db(self):
        return self.plugin.db_service

    @property
    def _cfg(self):
        return self.plugin.plugin_config

    @property
    def _sources(self):
        return getattr(self.plugin, "source_service", None)

    def _get_index(self) -> dict[str, Any]:
        """从数据库读取完整索引快照。"""
        return self._db.get_index_cache_readonly()

    @storage_write
    async def _move_image_file(self, old: str, new: str, category: str, updates=None) -> None:
        async def commit_move(_target):
            if category not in self._cfg.categories:
                raise ValueError("分类已被移除，请重新选择分类")
            if not await self._move_index_path(old, new, category, updates):
                raise RuntimeError("update image metadata failed")
        if category not in self._cfg.categories:
            raise ValueError("分类已被移除，请重新选择分类")
        await commit_image_file(old, new, commit_move)

    def _find_index_entry_by_hash(self, img_hash: str) -> tuple[str, dict[str, Any]] | None:
        return self._db.get_emoji_by_hash(img_hash)

    async def _add_blacklist_hash(self, image_hash: str) -> bool:
        return await add_blacklist_hash(self.plugin, image_hash)

    async def _update_index_path(self, path: str, updates: dict[str, Any]) -> bool:
        ok = await self._db.update_path(path, updates)
        if ok:
            if SEARCH_METADATA_FIELDS.intersection(updates):
                await refresh_search_entry(self.plugin, path)
        return ok

    async def _move_index_path(
        self,
        old_path: str,
        new_path: str,
        category: str,
        updates: dict[str, Any] | None = None,
    ) -> bool:
        ok = await self._db.move_path(old_path, new_path, category, updates or {})
        if ok:
            await refresh_search_entry(self.plugin, new_path, previous_path=old_path)
        return ok

    _unique_path = staticmethod(unique_image_path)

    def _get_category_keys(self) -> list[str]:
        cfg = getattr(self.plugin, "plugin_config", None)
        if cfg:
            raw = list(getattr(cfg, "categories", []) or [])
        else:
            raw = list(getattr(self.plugin, "categories", []) or [])
        seen: set[str] = set()
        keys: list[str] = []
        for item in raw:
            key = str(item or "").strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    @staticmethod
    def _normalize_character_key(value: str) -> str:
        return normalize_character_key(value)

    def _build_characters_list(self, counts: dict[str, int] | None = None) -> list[dict]:
        counts = counts or {}
        result: list[dict] = []
        known: set[str] = set()
        for item in self._cfg.get_character_info_list():
            key = item["key"]
            known.add(key)
            result.append(
                {
                    "key": key,
                    "name": item["name"],
                    "desc": item.get("desc", ""),
                    "count": int(counts.get(key, 0) or 0),
                }
            )
        for key, count in counts.items():
            if key and key not in known:
                result.append({"key": key, "name": key, "desc": "", "count": int(count)})
        result.sort(key=lambda x: (-int(x.get("count") or 0), x["key"]))
        return result


    def _file_base64(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            raw = f.read()
        ext = Path(file_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext, "image/png")
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    @staticmethod
    def _split_csv(values: Any) -> list[str]:
        return normalize_label_list(values, allow_duplicates=True, csv_only=True)

    @staticmethod
    def _split_scenes(scene_raw: Any) -> list[str]:
        return normalize_label_list(scene_raw)

    @staticmethod
    def _norm_scope(scope_mode: object) -> str:
        return normalize_scope_mode(scope_mode) or "public"

    def _is_allowed_ext(self, ext: str) -> bool:
        return str(ext or "").lower() in self.ALLOWED_IMAGE_EXTS

    @staticmethod
    def _decode_base64_payload(value: str) -> bytes | None:
        if not value:
            return None
        b64_data = value.split(",", 1)[1] if "," in value else value
        try:
            return base64.b64decode(b64_data.strip(), validate=True)
        except (binascii.Error, ValueError):
            return None

    @staticmethod
    def _task_now() -> float:
        return time.time()

    def _prune_batch_upload_tasks(self) -> int:
        now = self._task_now()
        expired = []
        for task_id, task in self.batch_upload_tasks.items():
            if task.get("status") == "processing":
                continue
            done_at = float(task.get("completed_at") or task.get("updated_at") or 0)
            if done_at and now - done_at > self.BATCH_TASK_TTL_SECONDS:
                expired.append(task_id)
        for task_id in expired:
            self.batch_upload_tasks.pop(task_id, None)
        return len(expired)

    def _is_under_data_dir(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._data_dir.resolve())
            return True
        except Exception:
            return False

    @staticmethod
    def _norm_path_key(path) -> str:
        """归一化路径用于比较：统一分隔符并忽略大小写（Windows）。
        避免存储路径与磁盘遍历路径的大小写/分隔符差异，导致有效表情被
        误判为过期索引或孤儿文件，进而错删索引或清空自定义类别。
        """
        return canonicalize_path(path)

    @staticmethod
    def _file_stat(path: Path) -> dict[str, Any]:
        try:
            stat = path.stat()
            return {"path": str(path), "size": int(stat.st_size), "mtime": int(stat.st_mtime)}
        except OSError:
            return {"path": str(path), "size": 0, "mtime": 0}

    def _iter_files(self, root: Path, *, allowed_exts_only: bool = False) -> list[Path]:
        if not root.exists() or not root.is_dir():
            return []
        files: list[Path] = []
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if allowed_exts_only and path.suffix.lower() not in self.ALLOWED_IMAGE_EXTS:
                    continue
                files.append(path)
            except OSError:
                continue
        return files

    def _build_storage_report(self, *, include_items: bool = False) -> dict[str, Any]:
        index = self._get_index()
        indexed_paths = {
            self._norm_path_key(path)
            for path in index.keys()
            if isinstance(path, str)
        }
        stale_index = []
        for path in index.keys():
            if not isinstance(path, str):
                continue
            # 用 resolve() 规范化后再判断文件是否存在，避免因路径前缀/分隔符
            # 差异把真实存在的文件误判为过期索引（否则会删了索引却留下文件）。
            try:
                file_exists = Path(path).resolve().is_file()
            except Exception:
                file_exists = Path(path).is_file()
            if not file_exists:
                stale_index.append(self._file_stat(Path(path)))

        category_files = self._iter_files(self._data_dir / "categories", allowed_exts_only=True)
        orphan_files = [
            self._file_stat(path)
            for path in category_files
            if self._norm_path_key(path) not in indexed_paths
        ]

        thumb_files = [self._file_stat(path) for path in self._iter_files(self._data_dir / "thumb_cache")]
        temp_files = [self._file_stat(path) for path in self._iter_files(self._data_dir / "temp")]
        raw_files = [self._file_stat(path) for path in self._iter_files(self._data_dir / "raw")]

        def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
            result = {
                "count": len(items),
                "bytes": sum(int(item.get("size", 0) or 0) for item in items),
                "samples": items[:20],
            }
            if include_items:
                result["items"] = items
            return result

        return {
            "success": True,
            "stale_index": summary(stale_index),
            "orphan_files": summary(orphan_files),
            "thumb_cache": summary(thumb_files),
            "temp_files": summary(temp_files),
            "raw_files": summary(raw_files),
        }

    async def _remove_report_files(self, items: list[dict[str, Any]]) -> int:
        removed = 0
        for item in items:
            path = Path(str(item.get("path", "")))
            if not path or not self._is_under_data_dir(path):
                continue
            try:
                if path.is_file() and not file_in_use(path):
                    ok = await safe_remove_file(str(path))
                    if ok:
                        removed += 1
            except Exception as e:
                logger.warning(f"cleanup file failed: {path}, {e}")
        return removed

    def _parse_upload_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        category = str(data.get("category", data.get("emotion", "")) or "").strip()
        tags = self._split_csv(data.get("tags", []))
        scenes = self._split_scenes(data.get("scenes", data.get("scene")))
        overlay_text = str(data.get("overlay_text", "") or "").strip()
        emotions = self._split_csv(data.get("emotions", []))
        scope_mode = self._norm_scope(data.get("scope_mode"))
        origin_target = str(data.get("origin_target", "") or "").strip()
        return {
            "category": category or "unknown",
            "tags": tags,
            "desc": str(data.get("desc", data.get("description", "")) or ""),
            "scenes": scenes,
            "overlay_text": overlay_text,
            "emotions": emotions,
            "scope_mode": scope_mode,
            "origin_target": origin_target,
            "character": self._normalize_character_key(str(data.get("character", "") or "")),
        }

    def _build_categories_list(self, counts: dict[str, int]) -> list[dict]:
        result: list[dict] = []
        if hasattr(self.plugin, "plugin_config"):
            for cat in self._cfg.get_category_info():
                key = cat["key"]
                result.append({"key": key, "name": cat["name"], "count": counts.get(key, 0)})
            known = {c["key"] for c in result}
            for cat_key, count in counts.items():
                if cat_key not in known:
                    result.append({"key": cat_key or '', "name": cat_key or '待识图', "count": count})
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def _count_favorites(self) -> int:
        return int(self._db.count_favorites())

    def _build_image_item(self, path_str: str, meta: dict) -> dict | None:
        try:
            Path(path_str)
            return {
                "hash": meta.get("hash", ""),
                "category": meta.get("category", "unknown"),
                "tags": meta.get("tags", []),
                "desc": meta.get("desc", ""),
                "scenes": self._split_scenes(meta.get("scenes", [])),
                "emotions": self._split_csv(meta.get("emotions", [])),
                "scope_mode": self._norm_scope(meta.get("scope_mode")),
                "origin_target": str(meta.get("origin_target", "") or ""),
                "created_at": meta.get("created_at", 0),
                "is_favorite": bool(meta.get("is_favorite", 0)),
                "use_count": meta.get("use_count", 0) or 0,
                "last_used_at": meta.get("last_used_at", 0) or 0,
                # WebUI 预览面板使用的图片元数据字段
                "width": meta.get("width"),
                "height": meta.get("height"),
                "format": meta.get("format"),
                "bytes": meta.get("bytes"),
                "add_method": meta.get("add_method"),
                "reviewed_at": meta.get("reviewed_at"),
                "source_url": meta.get("source_url"),
                "original_name": meta.get("original_name"),
                "overlay_text": str(meta.get("overlay_text", "") or ""),
                "character": str(meta.get("character", "") or ""),
                "source": str(meta.get("source", "") or ""),
                "retention_class": str(meta.get("retention_class", "native") or "native"),
            }
        except ValueError:
            return None

    @storage_write
    async def _persist_image(
        self,
        *,
        file_content: bytes,
        file_ext: str,
        category: str,
        file_hash: str | None = None,
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
        scope_mode: str = "public",
        origin_target: str = "",
        overlay_text: str = "",
        emotions: list[str] | None = None,
        character: str = "",
    ) -> dict:
        final_cat = str(category or "").strip() or "unknown"
        if final_cat not in self._cfg.categories:
            raise ValueError("分类已被移除，请重新选择分类")
        ts = int(datetime.now().timestamp())
        filename = f"{ts}_{uuid.uuid4().hex[:8]}{file_ext}"
        cat_dir = self._cfg.ensure_category_dir(final_cat)
        file_path = cat_dir / filename

        img_hash = file_hash or hashlib.sha256(file_content).hexdigest()
        data = {
            "hash": img_hash,
            "path": str(file_path),
            "category": final_cat,
            "tags": list(tags or []),
            "desc": str(desc or ""),
            "scenes": list(scenes or []),
            "overlay_text": str(overlay_text or ""),
            "emotions": list(emotions or []),
            "character": self._normalize_character_key(character),
            "scope_mode": self._norm_scope(scope_mode),
            "origin_target": str(origin_target or "").strip(),
            "created_at": ts,
        }
        db = self._db
        if db is None:
            raise RuntimeError("database unavailable")
        async def commit_upload(_target):
            if final_cat not in self._cfg.categories:
                raise ValueError("分类已被移除，请重新选择分类")
            if await db.insert_batch([data]) <= 0:
                raise RuntimeError("insert image metadata failed")
        await write_image_file(file_content, file_path, commit_upload)
        await refresh_search_entry(self.plugin, str(file_path), data)
        return {"hash": img_hash, "category": final_cat}
