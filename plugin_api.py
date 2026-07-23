import asyncio
import base64
import binascii
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from quart import jsonify, request

try:
    from quart import send_file
except ImportError:
    send_file = None  # type: ignore[assignment]

from astrbot.api import logger

from .core.util.safe_io import safe_remove_file

PLUGIN_NAME = "astrbot_plugin_stealer"


class PluginAPI:
    """Backend API provider for plugin Pages."""

    ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    BATCH_TASK_TTL_SECONDS = 30 * 60

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
            ("/emotions", "handle_get_emotions", ["GET"]),
            ("/health", "handle_health_check", ["GET"]),
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
    def _cache(self):
        return self.plugin.cache_service

    @property
    def _db(self):
        return getattr(self.plugin, "db_service", None)

    @property
    def _cfg(self):
        return self.plugin.plugin_config

    def _get_index(self) -> dict[str, Any]:
        """从数据库读取完整索引；DB 不存在时返回空 dict。"""
        db = self._db
        if db:
            return db.get_index_cache_readonly()
        return {}

    def _build_full_index_snapshot(self) -> dict[str, Any]:
        """获取完整索引快照。"""
        db = self._db
        if db:
            return db.get_index_cache_readonly()
        return {}

    def _find_index_entry_by_hash(self, img_hash: str) -> tuple[str, dict[str, Any]] | None:
        db = self._db
        if db and hasattr(db, "get_emoji_by_hash"):
            found = db.get_emoji_by_hash(img_hash)
            if found:
                return found
        for path, meta in self._build_full_index_snapshot().items():
            if isinstance(meta, dict) and meta.get("hash") == img_hash:
                return path, dict(meta)
        return None

    def _invalidate_bm25(self) -> None:
        """写入操作后使 BM25 索引失效，下次搜索时强制重建。
        防止批量导入/删除/更新后 BM25 仍使用旧语料，导致新表情无法被检索到。
        """
        try:
            selector = getattr(self.plugin, "meme_selector", None)
            if selector is not None:
                selector._invalidate_bm25_index()
        except Exception as e:
            logger.debug(f"[BM25] 失效索引失败: {e}")

    async def _add_blacklist_hash(self, image_hash: str) -> bool:
        if not image_hash:
            return False
        try:
            db = self._db
            if db and hasattr(db, "add_blacklist"):
                await db.add_blacklist(image_hash, int(time.time()))
                return True
            await self._cache.set(
                "blacklist_cache", image_hash, int(time.time()), persist=True
            )
            return True
        except Exception as e:
            logger.error(f"写入黑名单失败: {e}", exc_info=True)
            return False

    async def _delete_index_paths(self, paths: list[str]) -> None:
        """通过 db_service 删除索引条目。"""
        db = self._db
        if db is None or not hasattr(db, "delete_paths"):
            logger.warning("[PluginAPI] DB 不可用，跳过索引删除")
            return
        await db.delete_paths(paths)
        self._invalidate_bm25()

    async def _update_index_path(self, path: str, updates: dict[str, Any]) -> bool:
        db = self._db
        if db is None or not hasattr(db, "update_path"):
            return False
        ok = await db.update_path(path, updates)
        if ok:
            self._invalidate_bm25()
        return ok

    async def _move_index_path(
        self,
        old_path: str,
        new_path: str,
        category: str,
        updates: dict[str, Any] | None = None,
    ) -> bool:
        db = self._db
        if db is None or not hasattr(db, "move_path"):
            return False
        ok = await db.move_path(old_path, new_path, category, updates or {})
        if ok:
            self._invalidate_bm25()
        return ok

    @staticmethod
    def _unique_path(target_dir: Path, filename: str) -> Path:
        candidate = target_dir / filename
        if not candidate.exists():
            return candidate
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while True:
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

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
    def _split_csv(tags_raw: str) -> list[str]:
        return [t.strip() for t in str(tags_raw).split(",") if t.strip()]

    @staticmethod
    def _split_scenes(scene_raw: Any) -> list[str]:
        if scene_raw is None:
            return []
        if isinstance(scene_raw, list):
            raw_items = scene_raw
        else:
            raw_items = (
                str(scene_raw).replace("、", ",").replace("，", ",").replace("；", ",").split(",")
            )
        seen: set[str] = set()
        result: list[str] = []
        for item in raw_items:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _norm_scope(scope_mode: object) -> str:
        raw = str(scope_mode or "").strip().lower()
        if raw in {"public", "global", "all"}:
            return "public"
        if raw in {"local", "private", "scoped"}:
            return "local"
        return "public"

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
        try:
            return os.path.normcase(os.path.normpath(str(path)))
        except Exception:
            return os.path.normcase(str(path))

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
        index = self._build_full_index_snapshot()
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
                if path.is_file():
                    ok = await safe_remove_file(str(path))
                    if ok:
                        removed += 1
            except Exception as e:
                logger.warning(f"cleanup file failed: {path}, {e}")
        return removed

    def _parse_upload_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        category = str(data.get("category", data.get("emotion", "")) or "").strip()
        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            tags = self._split_csv(tags_raw)
        elif isinstance(tags_raw, list):
            tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
        else:
            tags = []
        scenes = self._split_scenes(data.get("scenes", data.get("scene")))
        scope_mode = self._norm_scope(data.get("scope_mode"))
        origin_target = str(data.get("origin_target", "") or "").strip()
        return {
            "category": category or "unknown",
            "tags": tags,
            "desc": str(data.get("desc", data.get("description", "")) or ""),
            "scenes": scenes,
            "scope_mode": scope_mode,
            "origin_target": origin_target,
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
                    result.append({"key": cat_key, "name": cat_key, "count": count})
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def _count_favorites(self) -> int:
        db = self._db
        if db and hasattr(db, "count_favorites"):
            return int(db.count_favorites())
        return sum(
            1
            for meta in self._get_index().values()
            if isinstance(meta, dict) and bool(meta.get("is_favorite", 0))
        )

    def _build_image_item(self, path_str: str, meta: dict) -> dict | None:
        try:
            Path(path_str)
            return {
                "hash": meta.get("hash", ""),
                "category": meta.get("category", "unknown"),
                "tags": meta.get("tags", []),
                "desc": meta.get("desc", ""),
                "scenes": self._split_scenes(meta.get("scenes", [])),
                "scope_mode": self._norm_scope(meta.get("scope_mode")),
                "origin_target": str(meta.get("origin_target", "") or ""),
                "created_at": meta.get("created_at", 0),
                "is_favorite": bool(meta.get("is_favorite", 0)),
                "use_count": meta.get("use_count", 0) or 0,
                "last_used_at": meta.get("last_used_at", 0) or 0,
            }
        except ValueError:
            return None

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
    ) -> dict:
        final_cat = str(category or "").strip() or "unknown"
        ts = int(datetime.now().timestamp())
        filename = f"{ts}_{uuid.uuid4().hex[:8]}{file_ext}"
        cat_dir = self._cfg.ensure_category_dir(final_cat)
        file_path = cat_dir / filename
        await asyncio.to_thread(file_path.write_bytes, file_content)

        img_hash = file_hash or self._cache.compute_hash(file_content)
        data = {
            "hash": img_hash,
            "path": str(file_path),
            "category": final_cat,
            "tags": list(tags or []),
            "desc": str(desc or ""),
            "scenes": list(scenes or []),
            "scope_mode": self._norm_scope(scope_mode),
            "origin_target": str(origin_target or "").strip(),
            "created_at": ts,
        }
        db = self._db
        if db and hasattr(db, "insert_batch"):
            inserted = await db.insert_batch([data])
            if inserted <= 0:
                try:
                    await safe_remove_file(str(file_path))
                finally:
                    raise RuntimeError("insert image metadata failed")
            self._invalidate_bm25()
        else:
            logger.warning("[PluginAPI] DB 不可用，无法插入图片元数据")
            raise RuntimeError("db_service unavailable for insert_batch")
        return {"hash": img_hash, "category": final_cat}

    # ── Image serving ─────────────────────────────────────────

    async def handle_serve_image(self):
        """直接服务图片文件（用于页面展示）。"""
        if send_file is None:
            return jsonify({"success": False, "error": "send_file 不可用"}), 500
        file_path = request.args.get("path", "")
        if not file_path or not os.path.isfile(file_path):
            return jsonify({"success": False, "error": "文件不存在"}), 404
        try:
            Path(file_path).resolve().relative_to(self._data_dir.resolve())
        except ValueError:
            return jsonify({"success": False, "error": "路径非法"}), 403
        return await send_file(file_path)

    async def handle_image_data(self):
        """返回图片的 base64 data URL。"""
        image_hash = request.args.get("hash", "").strip()
        if not image_hash:
            return jsonify({"success": False, "error": "缺少 hash"})
        for path_str, meta in self._get_index().items():
            if isinstance(meta, dict) and meta.get("hash") == image_hash:
                if os.path.isfile(path_str):
                    try:
                        data_url = self._file_base64(path_str)
                        return jsonify({"success": True, "hash": image_hash, "url": data_url})
                    except Exception as e:
                        logger.warning(f"读取图片失败: {e}")
                break
        return jsonify({"success": False, "error": "图片未找到"})

    async def _get_or_create_thumbnail(
        self, img_hash: str, file_path: str, max_size: int = 300
    ) -> str:
        """生成或返回缓存的缩略图路径。"""
        thumb_dir = self._data_dir / "thumb_cache"
        thumb_path = thumb_dir / f"{img_hash}_{max_size}.jpg"

        if thumb_path.exists():
            return str(thumb_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        thumb_dir.mkdir(parents=True, exist_ok=True)

        def _create() -> str:
            with Image.open(file_path) as img:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(thumb_path, "JPEG", quality=85)
            return str(thumb_path)

        return await asyncio.to_thread(_create)

    async def handle_thumbnail(self):
        """返回缩略图的 base64 data URL，用于列表展示。"""
        img_hash = request.args.get("hash", "").strip()
        max_size = request.args.get("size", 300, type=int)

        if not img_hash:
            return jsonify({"success": False, "error": "缺少 hash"}), 400

        file_path = None
        for p, m in self._get_index().items():
            if isinstance(m, dict) and m.get("hash") == img_hash:
                file_path = p
                break

        # 回退：正式库未命中时查待审核池（审核区缩略图走此路径）
        if not file_path:
            db = self._db
            if db and hasattr(db, "get_pending_by_hash"):
                try:
                    pending_row = db.get_pending_by_hash(img_hash)
                except Exception:
                    pending_row = None
                if pending_row:
                    file_path = pending_row.get("path")

        if not file_path or not os.path.isfile(file_path):
            return jsonify({"success": False, "error": "图片未找到"}), 404

        try:
            thumb_path = await self._get_or_create_thumbnail(
                img_hash, file_path, max_size
            )
            data_url = self._file_base64(thumb_path)
            return jsonify({"success": True, "hash": img_hash, "url": data_url})
        except Exception as e:
            logger.warning(f"生成缩略图失败: {e}")
            try:
                data_url = self._file_base64(file_path)
                return jsonify({"success": True, "hash": img_hash, "url": data_url})
            except Exception as e2:
                return jsonify({"success": False, "error": str(e2)}), 500

    # ── List / Stats / Health ─────────────────────────────────

    async def handle_list_images(self):
        """返回分页图片列表和分类统计。"""
        try:
            page = request.args.get("page", 1, type=int)
            page_size = request.args.get("size", 50, type=int)
            cat_filter = request.args.get("category", None)
            search = str(request.args.get("q", "")).lower()
            sort_order = request.args.get("sort", "newest")
            favorite_only = request.args.get("favorite_only", "false").lower() == "true"

            db = self._db
            get_paginated = getattr(db, "get_emojis_paginated", None) if db else None

            if db and callable(get_paginated) and db.count_total() > 0:
                raw, total, cat_counts = get_paginated(
                    page=page,
                    page_size=page_size,
                    category=cat_filter,
                    sort_order=sort_order,
                    search_query=search if search else None,
                    favorite_only=favorite_only,
                )
                images = [
                    item for item in (self._build_image_item(i["path"], i) for i in raw) if item
                ]
                cats = self._build_categories_list(cat_counts)
                return jsonify(
                    {
                        "success": True,
                        "total": total,
                        "page": page,
                        "size": page_size,
                        "images": images,
                        "categories": cats,
                        "favorite_count": self._count_favorites(),
                    }
                )

            index = self._get_index()
            images: list[dict] = []
            cat_counts: dict[str, int] = {}

            for path_str, meta in index.items():
                if not Path(path_str).exists():
                    continue
                item = self._build_image_item(path_str, meta)
                if not item:
                    continue
                if search and not (
                    any(search in str(t).lower() for t in item["tags"])
                    or search in item["desc"].lower()
                    or any(search in str(s).lower() for s in item.get("scenes", []))
                ):
                    continue
                cat = item["category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if cat_filter and item["category"] != cat_filter:
                    continue
                if favorite_only and not item.get("is_favorite"):
                    continue
                images.append(item)

            images.sort(
                key=lambda x: (int(x.get("created_at", 0) or 0), str(x.get("hash", ""))),
                reverse=(sort_order != "oldest"),
            )

            total = len(images)
            start = (page - 1) * page_size
            paged = images[start : start + page_size]
            cats = self._build_categories_list(cat_counts)

            return jsonify(
                {
                    "success": True,
                    "total": total,
                    "page": page,
                    "size": page_size,
                    "images": paged,
                    "categories": cats,
                    "favorite_count": self._count_favorites(),
                }
            )
        except Exception as e:
            logger.error(f"Error listing images: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_get_stats(self):
        try:
            index = self._get_index()
            today_start = (
                datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            )
            today_count = sum(
                1
                for m in index.values()
                if isinstance(m, dict) and m.get("created_at", 0) >= today_start
            )
            cat_count = len(self._cfg.categories) if hasattr(self.plugin, "plugin_config") else 0
            return jsonify(
                {
                    "success": True,
                    "stats": {"total": len(index), "categories": cat_count, "today": today_count},
                }
            )
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return jsonify({"success": False, "error": str(e)})

    async def handle_health_check(self):
        return jsonify({"success": True, "status": "ok", "service": "emoji-manager-webui"})

    # ── Pending (待审核池) ────────────────────────────────────

    def _build_pending_item(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """把 emoji_pending 行构造成前端 item（带 id 供审核定位）。"""
        try:
            path_str = str(row.get("path", "") or "")
            Path(path_str)
            return {
                "id": row.get("id"),
                "hash": str(row.get("hash", "") or ""),
                "category": str(row.get("category", "") or "unknown"),
                "tags": list(row.get("tags", []) or []),
                "desc": str(row.get("desc", "") or ""),
                "scenes": self._split_scenes(row.get("scenes", [])),
                "scope_mode": self._norm_scope(row.get("scope_mode")),
                "origin_target": str(row.get("origin_target", "") or ""),
                "source": str(row.get("source", "") or ""),
                "review_status": str(row.get("review_status", "pending") or "pending"),
                "created_at": int(row.get("created_at", 0) or 0),
            }
        except (ValueError, TypeError):
            return None

    async def handle_list_pending(self):
        """GET /pending —— 分页返回待审核列表（分类筛选/搜索/sort=newest）。"""
        try:
            db = self._db
            if not db or not hasattr(db, "get_pending_paginated"):
                return jsonify({"success": True, "images": [], "total": 0, "categories": {}})

            page = request.args.get("page", 1, type=int)
            page_size = request.args.get("size", 50, type=int)
            category = request.args.get("category", None)
            search = str(request.args.get("q", "")).strip().lower()

            raw, total, cat_counts = db.get_pending_paginated(
                page=max(1, page),
                page_size=max(1, min(page_size, 200)),
                category=category if category else None,
                search_query=search if search else None,
            )
            images = [item for item in (self._build_pending_item(r) for r in raw) if item]
            return jsonify(
                {
                    "success": True,
                    "images": images,
                    "total": total,
                    "categories": cat_counts,
                }
            )
        except Exception as e:
            logger.error(f"列出待审核失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_pending_stats(self):
        """GET /pending/stats —— 供审核区顶部进度条。"""
        try:
            db = self._db
            pending = db.count_pending() if db and hasattr(db, "count_pending") else 0
            capacity = int(getattr(self.plugin, "steal_pool_capacity", 200) or 200)
            return jsonify(
                {
                    "success": True,
                    "stats": {
                        "pending": pending,
                        "capacity": capacity,
                        "paused": pending >= capacity,
                    },
                }
            )
        except Exception as e:
            logger.error(f"待审核统计失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def _resolve_pending_ids(self, data: dict[str, Any]) -> list[int]:
        """支持 {id} 或 {ids:[...]} 两种入参，返回去重后的 int id 列表。"""
        raw = data.get("ids")
        if raw is None:
            single = data.get("id")
            raw = [single] if single is not None else []
        if not isinstance(raw, list):
            raw = []
        return list({int(i) for i in raw if i is not None})

    async def _approve_one(self, db, pending_id: int) -> tuple[bool, str]:
        """审核通过单条：pending 文件 → categories/，写 emoji+tag+scene，删 pending 行。

        原子性：文件 move 成功 → 写 emoji；emoji 写入失败 → 文件移回 pending，视为本次失败。
        embedding 在阶段 4 接入，此处不写向量。
        """
        row = db.get_pending(pending_id)
        if not row:
            return False, "pending not found"

        src_path = str(row.get("path", "") or "")
        if not src_path or not os.path.isfile(src_path):
            # 文件已丢失：清理孤儿 pending 行
            db.delete_pending(pending_id)
            return False, "pending file missing"

        category = str(row.get("category", "") or "").strip()
        if not category or category not in self._cfg.categories:
            # 分类无效：保留 pending 行让用户改，或可选拒绝。这里返回失败不删。
            return False, f"invalid category: {category!r}"

        cat_dir = self._cfg.ensure_category_dir(category)
        cat_path = str(cat_dir / os.path.basename(src_path))

        moved = False
        try:
            if os.path.abspath(src_path) != os.path.abspath(cat_path):
                await asyncio.to_thread(shutil.move, src_path, cat_path)
            moved = True

            emoji_entry: dict[str, Any] = {
                "path": cat_path,
                "hash": str(row.get("hash", "") or ""),
                "phash": row.get("phash") or "",
                "category": category,
                "desc": str(row.get("desc", "") or ""),
                "source": str(row.get("source", "") or ""),
                "origin_target": str(row.get("origin_target", "") or ""),
                "scope_mode": str(row.get("scope_mode", "public") or "public"),
                "created_at": int(time.time()),
                "use_count": 0,
                "last_used_at": 0,
                "tags": list(row.get("tags", []) or []),
                "scenes": list(row.get("scenes", []) or []),
            }
            inserted = await db.insert_batch([emoji_entry])
            if not inserted:
                raise RuntimeError("insert emoji returned 0")
            db.delete_pending(pending_id)

            # 审核通过后写入嵌入向量（仅在开启嵌入检索时，失败不阻塞）
            if getattr(self.plugin, "enable_embedding_search", True):
                try:
                    smart_service = getattr(getattr(self.plugin, "meme_selector", None), "_smart_select_service", None)
                    if smart_service and smart_service._embedding_service:
                        await smart_service._embedding_service.insert_emoji(cat_path, emoji_entry)
                        smart_service._invalidate_embedding_index()
                except Exception as embed_err:
                    logger.debug(f"审核通过后嵌入写入失败（不阻塞）: {embed_err}")

            return True, ""
        except Exception as e:
            # 回滚：把文件移回 pending 路径，保留 pending 行
            if moved and os.path.isfile(cat_path):
                try:
                    await asyncio.to_thread(shutil.move, cat_path, src_path)
                except Exception as rb:
                    logger.warning(f"审核回滚移动文件失败: {rb}")
            logger.error(f"审核通过失败 id={pending_id}: {e}", exc_info=True)
            return False, str(e)

    async def handle_pending_approve(self):
        """POST /pending/approve —— 批量通过 {id} 或 {ids:[]}。"""
        try:
            data = await request.get_json() or {}
            ids = await self._resolve_pending_ids(data)
            if not ids:
                return jsonify({"success": False, "error": "缺少 id/ids"})

            db = self._db
            if not db:
                return jsonify({"success": False, "error": "db 不可用"})

            approved = 0
            errors: list[str] = []
            for pending_id in ids:
                ok, msg = await self._approve_one(db, pending_id)
                if ok:
                    approved += 1
                elif msg and msg not in ("pending not found",):
                    errors.append(f"id={pending_id}: {msg}")

            # 审核通过后刷新 BM25 缓存索引
            try:
                selector = getattr(self.plugin, "meme_selector", None)
                if selector and hasattr(selector, "_invalidate_bm25_index"):
                    selector._invalidate_bm25_index()
            except Exception as e:
                logger.warning(f"审核后刷新缓存失败: {e}")

            return jsonify(
                {
                    "success": approved > 0,
                    "approved": approved,
                    "errors": errors,
                }
            )
        except Exception as e:
            logger.error(f"审核通过失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_pending_reject(self):
        """POST /pending/reject —— 批量拒绝 {id} 或 {ids:[...], blacklist?:bool}。"""
        try:
            data = await request.get_json() or {}
            ids = await self._resolve_pending_ids(data)
            if not ids:
                return jsonify({"success": False, "error": "缺少 id/ids"})
            blacklist = bool(data.get("blacklist", False))

            db = self._db
            if not db:
                return jsonify({"success": False, "error": "db 不可用"})

            removed_rows = db.delete_pending_batch(ids)
            deleted = 0
            blacklisted = 0
            for r in removed_rows:
                p = str(r.get("path", "") or "")
                h = str(r.get("hash", "") or "")
                if p:
                    try:
                        if await safe_remove_file(p):
                            deleted += 1
                    except Exception as e:
                        logger.warning(f"拒绝时删除文件失败 {p}: {e}")
                if blacklist and h:
                    try:
                        await db.add_blacklist(h)
                        blacklisted += 1
                    except Exception as e:
                        logger.warning(f"拉黑失败 {h}: {e}")
            return jsonify(
                {"success": True, "deleted": deleted, "blacklisted": blacklisted}
            )
        except Exception as e:
            logger.error(f"审核拒绝失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_pending_update(self):
        """POST /pending/update —— 修改一条 pending 的元数据（分类/描述/标签/场景/作用域）。"""
        try:
            data = await request.get_json() or {}
            try:
                pending_id = int(data.get("id") or 0)
            except (TypeError, ValueError):
                pending_id = 0
            if pending_id <= 0:
                return jsonify({"success": False, "error": "缺少 id"})

            db = self._db
            if not db or not hasattr(db, "update_pending"):
                return jsonify({"success": False, "error": "db 不可用"})

            # 字段白名单 + 类型归一化（与 update_pending 内部白名单一致）
            fields: dict[str, Any] = {}
            if "category" in data:
                category = str(data.get("category") or "").strip()
                if category and category in (self._cfg.categories or []):
                    fields["category"] = category
                elif category:
                    return jsonify(
                        {"success": False, "error": f"分类无效: {category!r}"}
                    )
            if "desc" in data:
                fields["desc"] = str(data.get("desc") or "").strip()
            if "scope_mode" in data:
                fields["scope_mode"] = str(data.get("scope_mode") or "public").strip()
            if "tags" in data:
                tags_raw = data.get("tags")
                if isinstance(tags_raw, list):
                    fields["tags"] = [
                        str(t).strip() for t in tags_raw if str(t or "").strip()
                    ]
                else:
                    fields["tags"] = [
                        t.strip()
                        for t in str(tags_raw or "").split(",")
                        if t.strip()
                    ]
            if "scenes" in data:
                scenes_raw = data.get("scenes")
                if isinstance(scenes_raw, list):
                    fields["scenes"] = [
                        str(s).strip() for s in scenes_raw if str(s or "").strip()
                    ]
                else:
                    fields["scenes"] = [
                        s.strip()
                        for s in str(scenes_raw or "").split(",")
                        if s.strip()
                    ]

            if not fields:
                return jsonify({"success": False, "error": "没有可更新字段"})

            updated = await db.update_pending(pending_id, fields)
            if not updated:
                return jsonify(
                    {"success": False, "error": "pending 不存在"}
                )

            return jsonify({"success": True, "item": self._build_pending_item(updated)})
        except Exception as e:
            logger.error(f"待审核更新失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    # ── Upload / Update / Delete ──────────────────────────────

    async def handle_upload_image(self):
        try:
            files = await request.files
            form = await request.form
            file_content = None
            filename = "upload.png"
            metadata_source: dict[str, Any] = {}

            if "file" in files:
                f = files["file"]
                file_content = f.read()
                filename = f.filename or "upload.png"
                metadata_source = dict(form)
            else:
                data = await request.get_json() or {}
                b64 = data.get("base64", "")
                if not b64:
                    return jsonify({"success": False, "error": "没有上传文件"})
                file_content = self._decode_base64_payload(b64)
                if file_content is None:
                    return jsonify({"success": False, "error": "图片数据无效"})
                filename = data.get("filename", "upload.png")
                metadata_source = data

            ext = Path(filename).suffix.lower()
            if not self._is_allowed_ext(ext):
                return jsonify({"success": False, "error": f"不支持的文件类型: {ext}"})
            if not file_content:
                return jsonify({"success": False, "error": "文件内容为空"})

            metadata = self._parse_upload_metadata(metadata_source)
            image = await self._persist_image(
                file_content=file_content,
                file_ext=ext,
                category=metadata["category"],
                tags=metadata["tags"],
                desc=metadata["desc"],
                scenes=metadata["scenes"],
                scope_mode=metadata["scope_mode"],
                origin_target=metadata["origin_target"],
            )
            if not self._db:
                logger.warning("[PluginAPI] DB 不可用，无法上传图片")
                return jsonify({"success": False, "error": "db_service unavailable"}), 503
            return jsonify({"success": True, "image": image, "hash": image["hash"]})
        except Exception as e:
            logger.error(f"上传图片失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_update_image(self):
        try:
            data = await request.get_json() or {}
            img_hash = data.get("hash")
            if not img_hash:
                return jsonify({"success": False, "error": "缺少 hash"})

            new_cat = data.get("category")
            new_tags = data.get("tags")
            new_desc = data.get("desc")
            new_scenes = data.get("scenes", data.get("scene"))
            new_scope = self._norm_scope(data.get("scope_mode"))
            new_favorite = data.get("is_favorite")
            found = self._find_index_entry_by_hash(str(img_hash))
            if not found:
                return jsonify({"success": False, "error": "Image not found"})
            target, meta = found

            updates: dict[str, Any] = {}
            if new_tags is not None:
                updates["tags"] = self._split_csv(new_tags) if isinstance(new_tags, str) else new_tags
            if new_desc is not None:
                updates["desc"] = new_desc
            if new_scenes is not None:
                updates["scenes"] = self._split_scenes(new_scenes)
            if new_scope:
                if new_scope == "local" and not str(meta.get("origin_target", "")).strip():
                    return jsonify({"success": False, "error": "Origin target missing"})
                updates["scope_mode"] = new_scope
            if new_favorite is not None:
                updates["is_favorite"] = 1 if new_favorite else 0

            if new_cat and new_cat != meta.get("category"):
                old_path = Path(target)
                if not old_path.exists():
                    return jsonify({"success": False, "error": "Source file not found"})
                target_dir = self._cfg.ensure_category_dir(new_cat)
                new_path = self._unique_path(target_dir, old_path.name)
                await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
                moved = False
                try:
                    moved = await self._move_index_path(target, str(new_path), new_cat, updates)
                finally:
                    if not moved and new_path.exists() and not old_path.exists():
                        try:
                            await asyncio.to_thread(shutil.move, str(new_path), str(old_path))
                        except Exception as rollback_error:
                            logger.error(
                                f"rollback moved image failed: {new_path} -> {old_path}, {rollback_error}"
                            )
                if not moved:
                    return jsonify({"success": False, "error": "Update index failed"})
            elif updates:
                if not await self._update_index_path(target, updates):
                    return jsonify({"success": False, "error": "Update index failed"})
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"更新图片失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_delete_image(self):
        try:
            data = await request.get_json() or {}
            img_hash = (data.get("hash", "") or "").strip()
            if not img_hash:
                return jsonify({"success": False, "error": "缺少 hash"})
            blacklist = data.get("blacklist", False)

            index = self._build_full_index_snapshot()
            removed: list[str] = []
            for p, m in index.items():
                if isinstance(m, dict) and m.get("hash") == img_hash:
                    removed.append(p)

            if removed:
                if blacklist and not await self._add_blacklist_hash(img_hash):
                    return jsonify({"success": False, "error": "write blacklist failed"})
                deleted_paths: list[str] = []
                for target in removed:
                    try:
                        if await safe_remove_file(target):
                            deleted_paths.append(target)
                    except Exception as e:
                        logger.warning(f"删除文件失败: {e}")
                if not deleted_paths:
                    return jsonify({"success": False, "error": "delete file failed"})
                await self._delete_index_paths(deleted_paths)
                if hasattr(self.plugin, "image_processor_service"):
                    self.plugin.image_processor_service.invalidate_cache(img_hash)
                return jsonify({"success": True, "count": len(deleted_paths)})
            return jsonify({"success": False, "error": "图片未找到"})
        except Exception as e:
            logger.error(f"删除图片失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    # ── Batch operations ──────────────────────────────────────

    async def handle_batch_delete(self):
        try:
            data = await request.get_json() or {}
            hashes = set(data.get("hashes", []))
            if not hashes:
                return jsonify({"success": True, "count": 0})
            index = self._build_full_index_snapshot()
            removed_paths = [
                p for p, m in index.items() if isinstance(m, dict) and m.get("hash") in hashes
            ]
            deleted_paths: list[str] = []
            for p in removed_paths:
                try:
                    if await safe_remove_file(p):
                        deleted_paths.append(p)
                except Exception as e:
                    logger.warning(f"删除文件失败 {p}: {e}")
            if deleted_paths:
                await self._delete_index_paths(deleted_paths)
            return jsonify({"success": True, "count": len(deleted_paths)})
        except Exception as e:
            logger.error(f"批量删除失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_batch_move(self):
        try:
            data = await request.get_json() or {}
            hashes = set(data.get("hashes", []))
            target_cat = data.get("category")
            if not hashes or not target_cat:
                return jsonify({"success": False, "error": "缺少参数"})
            moved_count = 0

            target_dir = self._cfg.ensure_category_dir(target_cat)
            index = self._build_full_index_snapshot()
            for p, m in list(index.items()):
                if not isinstance(m, dict) or m.get("hash") not in hashes:
                    continue
                if m.get("category") == target_cat:
                    continue
                old = Path(p)
                if not old.exists():
                    continue
                new = self._unique_path(target_dir, old.name)
                await asyncio.to_thread(shutil.move, str(old), str(new))
                moved = False
                try:
                    moved = await self._move_index_path(p, str(new), target_cat)
                finally:
                    if not moved and new.exists() and not old.exists():
                        try:
                            await asyncio.to_thread(shutil.move, str(new), str(old))
                        except Exception as rollback_error:
                            logger.error(
                                f"rollback batch move failed: {new} -> {old}, {rollback_error}"
                            )
                if moved:
                    moved_count += 1
            return jsonify({"success": True, "count": moved_count})
        except Exception as e:
            logger.error(f"批量移动失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_batch_scope(self):
        try:
            data = await request.get_json() or {}
            hashes = set(data.get("hashes", []))
            scope = self._norm_scope(data.get("scope_mode"))
            if not hashes or not scope:
                return jsonify({"success": False, "error": "缺少参数"})
            updated = 0
            skipped = 0

            index = self._build_full_index_snapshot()
            for p, m in index.items():
                if not isinstance(m, dict) or m.get("hash") not in hashes:
                    continue
                if scope == "local" and not str(m.get("origin_target", "")).strip():
                    skipped += 1
                    continue
                if await self._update_index_path(p, {"scope_mode": scope}):
                    updated += 1
            return jsonify({"success": True, "count": updated, "skipped": skipped})
        except Exception as e:
            logger.error(f"批量作用域更新失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_batch_favorite(self):
        try:
            data = await request.get_json() or {}
            hashes = set(data.get("hashes", []))
            favorite = str(data.get("favorite", "true")).lower() != "false"
            if not hashes:
                return jsonify({"success": True, "count": 0})
            updated = 0

            index = self._build_full_index_snapshot()
            for p, m in index.items():
                if not isinstance(m, dict) or m.get("hash") not in hashes:
                    continue
                if await self._update_index_path(p, {"is_favorite": 1 if favorite else 0}):
                    updated += 1
            return jsonify({"success": True, "count": updated})
        except Exception as e:
            logger.error(f"批量收藏失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_batch_upload(self):
        try:
            self._prune_batch_upload_tasks()
            files_data = []
            try:
                data = await request.get_json()
            except Exception:
                data = None
            if data and "_files" in data:
                for fi in data.get("_files", []):
                    b64 = fi.get("base64", "")
                    content = self._decode_base64_payload(b64)
                    ext = Path(fi.get("name", "upload.png")).suffix.lower()
                    if not self._is_allowed_ext(ext):
                        continue
                    if content:
                        files_data.append(
                            {
                                "filename": fi.get("name", "upload.png"),
                                "content": content,
                                "hash": self._cache.compute_hash(content),
                                "ext": ext,
                            }
                        )
                category = str(data.get("category", "")).strip()
                auto_analyze = str(data.get("auto_analyze", "false")).lower() == "true"
            else:
                files = await request.files
                form = await request.form
                category = form.get("category", "").strip()
                auto_analyze = form.get("auto_analyze", "false").lower() == "true"
                for field_name in files:
                    f = files[field_name]
                    ext = Path(f.filename or "upload.png").suffix.lower()
                    if not self._is_allowed_ext(ext):
                        continue
                    content = f.read()
                    if content:
                        files_data.append(
                            {
                                "filename": f.filename or "upload.png",
                                "content": content,
                                "hash": self._cache.compute_hash(content),
                                "ext": ext,
                            }
                        )

            if not files_data:
                return jsonify({"success": False, "error": "没有上传有效的图片文件"})

            fallback = category or (
                self._get_category_keys()[0] if self._get_category_keys() else None
            )
            if not fallback:
                return jsonify({"success": False, "error": "未配置任何分类"})

            task_id = str(uuid.uuid4())
            now = self._task_now()
            self.batch_upload_tasks[task_id] = {
                "status": "processing",
                "total": len(files_data),
                "processed": 0,
                "success": 0,
                "failed": 0,
                "results": [],
                "created_at": now,
                "updated_at": now,
            }
            asyncio.create_task(
                self._process_batch(task_id, files_data, category, auto_analyze, fallback)
            )
            return jsonify({"success": True, "task_id": task_id, "total": len(files_data)})
        except Exception as e:
            logger.error(f"批量上传失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def _process_batch(
        self, task_id: str, files_data: list[dict], category: str, auto_analyze: bool, fallback: str
    ) -> None:
        try:
            task = self.batch_upload_tasks.get(task_id)
            if not task:
                return
            for fd in files_data:
                tmp: Path | None = None
                try:
                    tags, desc, scenes = [], "", []
                    final_cat = category or fallback
                    if auto_analyze:
                        try:
                            img_hash = fd["hash"]
                            tmp = self._data_dir / "temp" / f"{img_hash}{fd['ext']}"
                            tmp.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(lambda: tmp.write_bytes(fd["content"]))
                            proc = self.plugin.image_processor_service
                            if proc:
                                rc, rt, rd, _, rs = await proc.classify_image(
                                    event=None,
                                    file_path=str(tmp),
                                    categories=list(self._cfg.categories or []),
                                    content_filtration=False,
                                )
                                if rc and rc != getattr(proc, "CATEGORY_FILTERED", None):
                                    final_cat = rc
                                    tags = rt or []
                                    desc = rd or ""
                                    scenes = rs or []
                        except Exception as e:
                            logger.warning(f"自动分析失败: {e}")
                        finally:
                            if tmp is not None:
                                await asyncio.to_thread(lambda: tmp.unlink() if tmp.exists() else None)

                    img = await self._persist_image(
                        file_content=fd["content"],
                        file_ext=fd["ext"],
                        category=final_cat,
                        file_hash=fd["hash"],
                        tags=tags,
                        desc=desc,
                        scenes=scenes,
                    )
                    task["results"].append(
                        {"hash": img["hash"], "category": img["category"], "success": True}
                    )
                    task["success"] += 1
                except Exception as e:
                    logger.error(f"处理文件 {fd['filename']} 失败: {e}")
                    task["results"].append(
                        {"filename": fd["filename"], "success": False, "error": str(e)}
                    )
                    task["failed"] += 1
                task["processed"] += 1
                task["updated_at"] = self._task_now()
            if not self._db:
                task["status"] = "failed"
                task["error"] = "db_service unavailable"
                return
            task["status"] = "completed"
            task["completed_at"] = self._task_now()
            task["updated_at"] = task["completed_at"]
        except Exception as e:
            logger.error(f"批量上传任务 {task_id} 失败: {e}")
            if task_id in self.batch_upload_tasks:
                self.batch_upload_tasks[task_id]["status"] = "failed"
                self.batch_upload_tasks[task_id]["error"] = str(e)
                self.batch_upload_tasks[task_id]["completed_at"] = self._task_now()
                self.batch_upload_tasks[task_id]["updated_at"] = self.batch_upload_tasks[task_id]["completed_at"]

    async def handle_batch_upload_status(self):
        self._prune_batch_upload_tasks()
        task_id = request.args.get("task_id", "").strip()
        if not task_id:
            return jsonify({"success": False, "error": "无效的任务ID"})
        task = self.batch_upload_tasks.get(task_id)
        if not task:
            return jsonify({"success": False, "error": "任务不存在或已过期"})
        return jsonify(
            {
                "success": True,
                "task_id": task_id,
                "status": task["status"],
                "total": task["total"],
                "processed": task["processed"],
                "success_count": task["success"],
                "failed_count": task["failed"],
                "error": task.get("error", ""),
                "results": task.get("results", []),
            }
        )

    # ── Repair / Storage maintenance ─────────────────────────

    async def handle_scope_repair(self):
        try:
            data = await request.get_json() or {}
            origin_target = str(data.get("origin_target", "") or "").strip()
            if not origin_target:
                return jsonify({"success": False, "error": "缺少 origin_target"})

            hashes_raw = data.get("hashes", [])
            hashes = {str(h).strip() for h in hashes_raw if str(h).strip()}
            scope_mode = self._norm_scope(data.get("scope_mode", "local"))
            only_missing = str(data.get("only_missing", "true")).lower() != "false"

            updated = 0
            skipped = 0
            index = self._build_full_index_snapshot()
            for path, meta in index.items():
                if not isinstance(meta, dict):
                    continue
                img_hash = str(meta.get("hash", "") or "")
                if hashes and img_hash not in hashes:
                    continue
                if only_missing and str(meta.get("origin_target", "") or "").strip():
                    skipped += 1
                    continue
                updates = {"origin_target": origin_target}
                if scope_mode:
                    updates["scope_mode"] = scope_mode
                if await self._update_index_path(path, updates):
                    updated += 1

            return jsonify({"success": True, "count": updated, "skipped": skipped})
        except Exception as e:
            logger.error(f"作用域来源修复失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_storage_scan(self):
        try:
            return jsonify(self._build_storage_report())
        except Exception as e:
            logger.error(f"存储扫描失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_storage_cleanup(self):
        try:
            data = await request.get_json() or {}
            report = self._build_storage_report(include_items=True)
            sections_raw = data.get("sections")
            if isinstance(sections_raw, list):
                sections = {str(section) for section in sections_raw}
            else:
                sections = {
                    key
                    for key in ("stale_index", "orphan_files", "thumb_cache", "temp_files", "raw_files")
                    if str(data.get(key, "false")).lower() == "true"
                }

            if not sections:
                strategy = str(
                    data.get(
                        "strategy",
                        getattr(self.plugin, "storage_cleanup_strategy", "balanced"),
                    )
                    or "balanced"
                )
                if strategy == "conservative":
                    sections = {"stale_index", "temp_files"}
                elif strategy == "aggressive":
                    sections = {
                        "stale_index",
                        "orphan_files",
                        "thumb_cache",
                        "temp_files",
                        "raw_files",
                    }
                else:
                    sections = {"stale_index", "orphan_files", "thumb_cache", "temp_files"}

            removed: dict[str, int] = {}
            if "stale_index" in sections:
                index = self._build_full_index_snapshot()
                stale_paths: list[str] = []
                for path in index.keys():
                    if not isinstance(path, str):
                        continue
                    # stale_index 语义：物理文件已丢失但索引条目仍存在。
                    # 只删索引条目，绝不删文件——文件已经不在了。
                    try:
                        file_exists = Path(path).resolve().is_file()
                    except Exception:
                        file_exists = Path(path).is_file()
                    if not file_exists:
                        stale_paths.append(path)
                if stale_paths:
                    await self._delete_index_paths(stale_paths)
                removed["stale_index"] = len(stale_paths)

            for key in ("orphan_files", "thumb_cache", "temp_files", "raw_files"):
                if key not in sections:
                    continue
                removed[key] = await self._remove_report_files(
                    report.get(key, {}).get("items", [])
                )

            return jsonify(
                {
                    "success": True,
                    "removed": removed,
                    "report": self._build_storage_report(),
                }
            )
        except Exception as e:
            logger.error(f"存储清理失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    # ── VLM Analyze ───────────────────────────────────────────

    async def handle_analyze_image(self):
        try:
            proc = getattr(self.plugin, "image_processor_service", None)
            if not proc:
                return jsonify({"success": False, "error": "图片处理服务不可用"})

            data = await request.get_json() or {}
            img_hash = (data.get("hash", "") or "").strip()
            img_base64 = (data.get("base64", "") or "").strip()
            tmp_file_to_cleanup = None

            file_path = None

            # 优先通过 hash 从索引查找文件路径
            if img_hash:
                index = self._get_index()
                for p, m in index.items():
                    if isinstance(m, dict) and m.get("hash") == img_hash:
                        file_path = p
                        break
                if not file_path or not os.path.isfile(file_path):
                    file_path = None

            # hash 查不到或未提供 hash 时，回退到 base64 方式
            if not file_path and img_base64:
                import tempfile

                file_content = self._decode_base64_payload(img_base64)
                if file_content is None:
                    return jsonify({"success": False, "error": "图片数据无效"})
                ext = ".png"
                if img_base64.startswith("data:image/jpeg") or img_base64.startswith("data:image/jpg"):
                    ext = ".jpg"
                elif img_base64.startswith("data:image/gif"):
                    ext = ".gif"
                elif img_base64.startswith("data:image/webp"):
                    ext = ".webp"

                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.write(file_content)
                tmp.close()
                file_path = tmp.name
                tmp_file_to_cleanup = file_path

            if not file_path:
                return jsonify({"success": False, "error": "缺少 hash 或 base64 图片数据"})

            cat, tags, desc, _, scenes = await proc.classify_image(
                event=None,
                file_path=file_path,
                categories=list(self._cfg.categories or []),
                content_filtration=False,
            )
            if cat == getattr(proc, "CATEGORY_FILTERED", None):
                return jsonify({"success": False, "error": "图片内容审核不通过"})
            if not cat:
                return jsonify({"success": False, "error": "无法识别图片分类"})

            return jsonify(
                {
                    "success": True,
                    "category": cat,
                    "tags": tags,
                    "description": desc,
                    "scenes": scenes or [],
                }
            )
        except Exception as e:
            logger.error(f"VLM分析失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": f"分析失败: {e}"})
        finally:
            if tmp_file_to_cleanup:
                try:
                    os.unlink(tmp_file_to_cleanup)
                except Exception:
                    pass

    # ── Categories ────────────────────────────────────────────

    async def handle_categories(self):
        if request.method == "POST":
            return await self._categories_update()
        return await self._categories_list()

    async def _categories_list(self):
        try:
            cats = {key: 0 for key in self._get_category_keys()}
            for meta in self._get_index().values():
                if isinstance(meta, dict):
                    c = str(meta.get("category", "unknown"))
                    cats[c] = cats.get(c, 0) + 1
            return jsonify({"success": True, "categories": cats})
        except Exception as e:
            logger.error(f"获取分类失败: {e}")
            return jsonify({"success": False, "error": str(e)})

    async def _categories_update(self):
        try:
            data = await request.get_json() or {}
            items = data.get("categories", [])
            if not isinstance(items, list) or not items:
                return jsonify({"success": False, "error": "分类列表无效"})

            keys: list[str] = []
            info: dict[str, dict] = {}
            seen: set[str] = set()
            for item in items:
                if isinstance(item, dict) and item.get("key"):
                    key = str(item["key"]).strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    keys.append(key)
                    name = str(item.get("name", "")).strip()
                    desc = str(item.get("desc", "")).strip()
                    if name or desc:
                        info[key] = {"name": name, "desc": desc}
                elif isinstance(item, str):
                    key = item.strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    keys.append(key)

            if not keys:
                return jsonify({"success": False, "error": "分类列表无效"})

            self.plugin.update_config({"categories": keys})

            cur_info = dict(getattr(self._cfg, "category_info", {}) or {})
            self._cfg.category_info = {k: cur_info.get(k, {}) for k in keys}
            self._cfg.category_info.update(info)
            self._cfg.ensure_category_dirs(keys)
            self._cfg.save_category_info()
            return jsonify({"success": True, "categories": keys})
        except Exception as e:
            logger.error(f"更新分类失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_delete_category(self):
        try:
            data = await request.get_json() or {}
            key = str(data.get("key", "")).strip()
            if not key:
                return jsonify({"success": False, "error": "分类Key无效"})

            cur_cats = list(self._cfg.categories or [])
            if key not in cur_cats:
                return jsonify({"success": False, "error": "分类不存在"})
            if len(cur_cats) <= 1:
                return jsonify({"success": False, "error": "至少需要保留1个分类"})

            updated = [c for c in cur_cats if c != key]
            deleted = 0

            index = self._build_full_index_snapshot()
            deleted_paths: list[str] = []
            for p, m in list(index.items()):
                if not isinstance(m, dict) or m.get("category") != key:
                    continue
                old = Path(p)
                try:
                    if not old.exists() or await safe_remove_file(str(old)):
                        deleted_paths.append(p)
                        deleted += 1
                        h = m.get("hash")
                        if h and hasattr(self.plugin, "image_processor_service"):
                            self.plugin.image_processor_service.invalidate_cache(h)
                except Exception as ex:
                    logger.warning(f"删除分类文件失败: {old}, {ex}")

            if deleted_paths:
                await self._delete_index_paths(deleted_paths)

            cat_dir = self._data_dir / "categories" / key
            try:
                if cat_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, cat_dir, True)
            except Exception as e:
                logger.warning(f"删除分类目录失败: {cat_dir}, {e}")

            if key in getattr(self._cfg, "category_info", {}):
                del self._cfg.category_info[key]
                self._cfg.save_category_info()

            self.plugin.update_config({"categories": updated})

            return jsonify(
                {"success": True, "deleted": key, "categories": updated, "deleted_files": deleted}
            )
        except Exception as e:
            logger.error(f"删除分类失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_get_emotions(self):
        try:
            info = self._cfg.get_category_info()
            return jsonify({"success": True, "emotions": info})
        except Exception as e:
            logger.error(f"获取情绪分类失败: {e}")
            return jsonify({"success": False, "error": str(e)})
