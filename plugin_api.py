import asyncio
import base64
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

PLUGIN_NAME = "astrbot_plugin_stealer"


class PluginAPI:
    """Backend API provider for plugin Pages."""

    ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

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
            ("/images/batch-upload", "handle_batch_upload", ["POST"]),
            ("/images/batch-upload-status", "handle_batch_upload_status", ["GET"]),
            ("/analyze", "handle_analyze_image", ["POST"]),
            ("/stats", "handle_get_stats", ["GET"]),
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
        db = self._db
        if db:
            idx = db.get_index_cache_readonly()
            if idx:
                return idx
        return self._cache.get_index_cache_readonly()

    async def _sync_index(self) -> bool:
        db = self._db
        if not db:
            return True
        try:
            idx = self._cache.get_index_cache_readonly()
            if hasattr(db, "sync_index"):
                await db.sync_index(idx)
                return True
            if hasattr(db, "save_index"):
                await db.save_index(idx)
                return True
        except Exception as e:
            logger.error(f"同步索引失败: {e}")
        return False

    async def _update_index(self, updater) -> dict:
        current = dict(self._get_index())
        result = updater(current)
        if hasattr(result, "__await__"):
            await result
        await self._cache.set_cache("index_cache", current, persist=False)
        await self._sync_index()
        return current

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
            "created_at": ts,
        }
        await self._cache.update_index(lambda cur: cur.__setitem__(str(file_path), data))
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

            db = self._db
            get_paginated = getattr(db, "get_emojis_paginated", None) if db else None

            if db and callable(get_paginated) and db.count_total() > 0:
                raw, total, cat_counts = get_paginated(
                    page=page,
                    page_size=page_size,
                    category=cat_filter,
                    sort_order=sort_order,
                    search_query=search if search else None,
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

    # ── Upload / Update / Delete ──────────────────────────────

    async def handle_upload_image(self):
        try:
            files = await request.files
            if "file" not in files:
                return jsonify({"success": False, "error": "没有上传文件"})
            f = files["file"]
            ext = Path(f.filename or "upload.png").suffix.lower()
            if not self._is_allowed_ext(ext):
                return jsonify({"success": False, "error": f"不支持的文件类型: {ext}"})
            content = f.read()
            image = await self._persist_image(
                file_content=content, file_ext=ext, category="unknown"
            )
            await self._sync_index()
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
            updated = {"ok": False, "error": ""}

            async def updater(current: dict):
                target = None
                meta = None
                for p, m in current.items():
                    if isinstance(m, dict) and m.get("hash") == img_hash:
                        target, meta = p, m
                        break
                if not target or not meta:
                    updated["error"] = "Image not found"
                    return
                if new_tags is not None:
                    meta["tags"] = (
                        self._split_csv(new_tags) if isinstance(new_tags, str) else new_tags
                    )
                if new_desc is not None:
                    meta["desc"] = new_desc
                if new_scenes is not None:
                    meta["scenes"] = self._split_scenes(new_scenes)
                if new_scope:
                    if new_scope == "local" and not str(meta.get("origin_target", "")).strip():
                        updated["error"] = "Origin target missing"
                        return
                    meta["scope_mode"] = new_scope
                if new_cat and new_cat != meta.get("category"):
                    old_path = Path(target)
                    if not old_path.exists():
                        updated["error"] = "Source file not found"
                        return
                    target_dir = self._cfg.ensure_category_dir(new_cat)
                    new_path = target_dir / old_path.name
                    await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
                    del current[target]
                    meta["path"] = str(new_path)
                    meta["category"] = new_cat
                    current[str(new_path)] = meta
                else:
                    current[target] = meta
                updated["ok"] = True

            await self._update_index(updater)
            await self._sync_index()
            if not updated["ok"]:
                return jsonify({"success": False, "error": updated["error"] or "Update failed"})
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

            removed: list[str] = []

            async def remover(current: dict):
                for p, m in list(current.items()):
                    if isinstance(m, dict) and m.get("hash") == img_hash:
                        removed.append(p)
                        del current[p]
                        return

            await self._update_index(remover)
            await self._sync_index()

            if removed:
                target = removed[0]
                try:
                    await self.plugin._safe_remove_file(target)
                except Exception as e:
                    logger.warning(f"删除文件失败: {e}")
                if blacklist:
                    await self._cache.set(
                        "blacklist_cache", img_hash, int(time.time()), persist=True
                    )
                if hasattr(self.plugin, "image_processor_service"):
                    self.plugin.image_processor_service.invalidate_cache(img_hash)
                return jsonify({"success": True})
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
            removed_paths: list[str] = []

            async def updater(current: dict):
                for p, m in list(current.items()):
                    if isinstance(m, dict) and m.get("hash") in hashes:
                        removed_paths.append(p)
                        del current[p]

            await self._update_index(updater)
            await self._sync_index()
            deleted = 0
            for p in removed_paths:
                try:
                    await self.plugin._safe_remove_file(p)
                    deleted += 1
                except Exception as e:
                    logger.warning(f"删除文件失败 {p}: {e}")
            return jsonify({"success": True, "count": deleted})
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

            async def updater(current: dict):
                nonlocal moved_count
                target_dir = self._cfg.ensure_category_dir(target_cat)
                for p, m in list(current.items()):
                    if not isinstance(m, dict) or m.get("hash") not in hashes:
                        continue
                    if m.get("category") == target_cat:
                        continue
                    old = Path(p)
                    if not old.exists():
                        continue
                    new = target_dir / old.name
                    await asyncio.to_thread(shutil.move, str(old), str(new))
                    del current[p]
                    m["path"] = str(new)
                    m["category"] = target_cat
                    current[str(new)] = m
                    moved_count += 1

            await self._update_index(updater)
            await self._sync_index()
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

            async def updater(current: dict):
                nonlocal updated, skipped
                for _, m in current.items():
                    if not isinstance(m, dict) or m.get("hash") not in hashes:
                        continue
                    if scope == "local" and not str(m.get("origin_target", "")).strip():
                        skipped += 1
                        continue
                    m["scope_mode"] = scope
                    updated += 1

            await self._update_index(updater)
            await self._sync_index()
            return jsonify({"success": True, "count": updated, "skipped": skipped})
        except Exception as e:
            logger.error(f"批量作用域更新失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def handle_batch_upload(self):
        try:
            files_data = []
            try:
                data = await request.get_json()
            except Exception:
                data = None
            if data and "_files" in data:
                for fi in data.get("_files", []):
                    b64 = fi.get("base64", "")
                    if "," in b64:
                        b64 = b64.split(",")[1]
                    content = base64.b64decode(b64)
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
            self.batch_upload_tasks[task_id] = {
                "status": "processing",
                "total": len(files_data),
                "processed": 0,
                "success": 0,
                "failed": 0,
                "results": [],
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
                            await asyncio.to_thread(lambda: tmp.unlink() if tmp.exists() else None)
                        except Exception as e:
                            logger.warning(f"自动分析失败: {e}")

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
            await self._sync_index()
            task["status"] = "completed"
        except Exception as e:
            logger.error(f"批量上传任务 {task_id} 失败: {e}")
            if task_id in self.batch_upload_tasks:
                self.batch_upload_tasks[task_id]["status"] = "failed"
                self.batch_upload_tasks[task_id]["error"] = str(e)

    async def handle_batch_upload_status(self):
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
                import base64
                import tempfile

                b64_data = img_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                file_content = base64.b64decode(b64_data)
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

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": keys})
            else:
                self._cfg.categories = keys
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = keys

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

            async def updater(current: dict):
                nonlocal deleted
                for p, m in list(current.items()):
                    if isinstance(m, dict) and m.get("category") == key:
                        old = Path(p)
                        if old.exists():
                            try:
                                await self.plugin._safe_remove_file(str(old))
                                deleted += 1
                                h = m.get("hash")
                                if h and hasattr(self.plugin, "image_processor_service"):
                                    self.plugin.image_processor_service.invalidate_cache(h)
                            except Exception as ex:
                                logger.warning(f"删除分类文件失败: {old}, {ex}")
                        del current[p]

            await self._update_index(updater)
            await self._sync_index()

            cat_dir = self._data_dir / "categories" / key
            try:
                if cat_dir.exists():
                    await asyncio.to_thread(shutil.rmtree, cat_dir, True)
            except Exception as e:
                logger.warning(f"删除分类目录失败: {cat_dir}, {e}")

            if key in getattr(self._cfg, "category_info", {}):
                del self._cfg.category_info[key]
                self._cfg.save_category_info()

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": updated})
            else:
                self._cfg.categories = updated
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = updated

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
