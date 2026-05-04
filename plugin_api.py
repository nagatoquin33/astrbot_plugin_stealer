import asyncio
import base64
import hashlib
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from quart import request, jsonify

from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_stealer"


class PluginAPI:
    """Backend API provider for plugin Pages.

    Registers web APIs through context.register_web_api() so the Bridge SDK
    can relay calls to them. Handlers use Quart globals (request, jsonify).
    """

    ALLOWED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.batch_upload_tasks: dict[str, dict] = {}

    # ── Registration ──────────────────────────────────────────

    def register(self, context) -> None:
        routes: list[tuple[str, str, list[str]]] = [
            ("/images", "handle_list_images", ["GET"]),
            ("/image-data", "handle_image_data", ["GET"]),
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

    # ── Response helpers ───────────────────────────────────────

    @staticmethod
    def ok(data: dict | None = None, **kwargs) -> dict:
        body: dict = {"success": True}
        if data:
            body.update(data)
        if kwargs:
            body.update(kwargs)
        return body

    @staticmethod
    def err(msg: str) -> dict:
        return {"success": False, "error": msg}

    # ── Utility ────────────────────────────────────────────────

    @property
    def data_dir(self) -> Path:
        return self.plugin.base_dir

    @property
    def cache_service(self):
        return self.plugin.cache_service

    @property
    def db_service(self):
        return getattr(self.plugin, "db_service", None)

    @property
    def plugin_config(self):
        return self.plugin.plugin_config

    @staticmethod
    def split_csv_tags(tags_raw: str) -> list[str]:
        return [t.strip() for t in str(tags_raw).split(",") if t.strip()]

    @staticmethod
    def split_scene_terms(scene_raw: Any) -> list[str]:
        if scene_raw is None:
            return []
        if isinstance(scene_raw, list):
            raw_items = scene_raw
        else:
            raw_items = (
                str(scene_raw)
                .replace("、", ",")
                .replace("，", ",")
                .replace("；", ",")
                .split(",")
            )
        seen: set[str] = set()
        scenes: list[str] = []
        for item in raw_items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            scenes.append(text)
        return scenes

    @staticmethod
    def normalize_scope_mode(scope_mode: object) -> str:
        raw = str(scope_mode or "").strip().lower()
        if raw in {"public", "global", "all"}:
            return "public"
        if raw in {"local", "private", "scoped"}:
            return "local"
        return ""

    def get_configured_category_keys(self) -> list[str]:
        plugin_config = getattr(self.plugin, "plugin_config", None)
        raw_categories = []
        if plugin_config is not None:
            raw_categories = list(getattr(plugin_config, "categories", []) or [])
        elif hasattr(self.plugin, "categories"):
            raw_categories = list(getattr(self.plugin, "categories", []) or [])
        category_keys: list[str] = []
        seen: set[str] = set()
        for item in raw_categories:
            key = str(item or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            category_keys.append(key)
        return category_keys

    def get_runtime_index_snapshot(self) -> dict[str, Any]:
        cache_idx = self.cache_service.get_index_cache_readonly()
        if cache_idx:
            return cache_idx
        db_service = self.db_service
        if db_service and getattr(db_service, "count_total", None):
            try:
                if db_service.count_total() > 0:
                    get_index = getattr(db_service, "get_index_cache_readonly", None)
                    if callable(get_index):
                        return get_index()
            except Exception:
                pass
        return {}

    async def sync_index_to_db(self, *, raise_on_error: bool = False) -> bool:
        db_service = self.db_service
        if not db_service:
            return True
        try:
            idx = self.cache_service.get_index_cache_readonly()
            sync_index = getattr(db_service, "sync_index", None)
            if callable(sync_index):
                await sync_index(idx)
                return True
            save_index = getattr(db_service, "save_index", None)
            if callable(save_index):
                await save_index(idx)
                return True
        except Exception as e:
            logger.error(f"同步索引到数据库失败: {e}", exc_info=True)
            if raise_on_error:
                raise
            return False

    async def update_runtime_index(self, updater, *, raise_on_sync_error: bool = False) -> dict[str, Any]:
        current = dict(self.get_runtime_index_snapshot())
        result = updater(current)
        if hasattr(result, "__await__"):
            await result
        await self.cache_service.set_cache("index_cache", current, persist=False)
        await self.sync_index_to_db(raise_on_error=raise_on_sync_error)
        return current

    def build_image_url(self, file_path: Path) -> str:
        return f"/images/{file_path.relative_to(self.data_dir).as_posix()}"

    def is_allowed_image_ext(self, file_ext: str) -> bool:
        return str(file_ext or "").lower() in self.ALLOWED_IMAGE_EXTS

    async def persist_uploaded_image(
        self,
        *,
        file_content: bytes,
        file_ext: str,
        category: str,
        file_hash: str | None = None,
        tags: list[str] | None = None,
        desc: str = "",
        scenes: list[str] | None = None,
    ) -> dict[str, Any]:
        final_category = str(category or "").strip() or "unknown"
        timestamp = int(datetime.now().timestamp())
        unique_filename = f"{timestamp}_{uuid.uuid4().hex[:8]}{file_ext}"
        category_dir = self.plugin_config.ensure_category_dir(final_category)
        file_path = category_dir / unique_filename
        await asyncio.to_thread(file_path.write_bytes, file_content)

        image_hash = file_hash or self.cache_service.compute_hash(file_content)
        image_data = {
            "hash": image_hash,
            "path": str(file_path),
            "category": final_category,
            "tags": list(tags or []),
            "desc": str(desc or ""),
            "scenes": list(scenes or []),
            "created_at": timestamp,
        }
        await self.cache_service.update_index(
            lambda current: current.__setitem__(str(file_path), image_data)
        )
        return {
            "hash": image_hash,
            "url": self.build_image_url(file_path),
            "category": final_category,
            "tags": image_data["tags"],
            "desc": image_data["desc"],
            "scenes": image_data["scenes"],
            "created_at": timestamp,
        }

    async def collect_removed_paths_by_hashes(self, hashes: set[str], *, raise_on_sync_error: bool = False) -> list[str]:
        removed_paths: list[str] = []
        await self.update_runtime_index(
            lambda current: self._remove_by_hashes(current, hashes, removed_paths),
            raise_on_sync_error=raise_on_sync_error,
        )
        return removed_paths

    @staticmethod
    def _remove_by_hashes(current: dict, hashes: set[str], removed_paths: list[str]) -> None:
        for path_str, meta in list(current.items()):
            if isinstance(meta, dict) and meta.get("hash") in hashes:
                removed_paths.append(path_str)
                del current[path_str]

    def collect_items_by_hashes(self, index_map: dict, hashes: set[str]) -> list[tuple[str, dict]]:
        items: list[tuple[str, dict]] = []
        for path_str, meta in index_map.items():
            if isinstance(meta, dict) and meta.get("hash") in hashes:
                items.append((path_str, meta))
        return items

    async def move_items_to_category(
        self, index_map: dict, items: list[tuple[str, dict]], target_category: str
    ) -> tuple[int, list[str]]:
        import shutil

        moved_count = 0
        moved_hashes: list[str] = []
        target_dir = self.plugin_config.ensure_category_dir(target_category)

        for old_path_str, meta in items:
            try:
                old_path = Path(old_path_str)
                if not old_path.exists():
                    continue
                new_path = target_dir / old_path.name
                await asyncio.to_thread(shutil.move, str(old_path), str(new_path))
                if old_path_str in index_map:
                    del index_map[old_path_str]
                meta["path"] = str(new_path)
                meta["category"] = target_category
                index_map[str(new_path)] = meta
                moved_count += 1
                if meta.get("hash"):
                    moved_hashes.append(meta["hash"])
            except Exception as e:
                logger.warning(f"Failed to move {old_path_str}: {e}")
        return moved_count, moved_hashes

    def invalidate_hashes(self, hashes: list[str]) -> None:
        if not hashes or not hasattr(self.plugin, "image_processor_service"):
            return
        for image_hash in hashes:
            self.plugin.image_processor_service.invalidate_cache(image_hash)

    async def delete_paths_best_effort(self, paths: list[str], warn_template: str) -> int:
        deleted_count = 0
        for path_str in paths:
            try:
                await self.plugin._safe_remove_file(path_str)
                deleted_count += 1
            except Exception as e:
                logger.warning(warn_template.format(path=path_str, error=e))
        return deleted_count

    def build_categories_list(self, category_counts: dict[str, int]) -> list[dict]:
        categories_list: list[dict] = []
        if hasattr(self.plugin, "plugin_config"):
            base_categories = self.plugin_config.get_category_info()
            for cat_info in base_categories:
                key = cat_info["key"]
                categories_list.append({
                    "key": key,
                    "name": cat_info["name"],
                    "count": category_counts.get(key, 0),
                })
            known_keys = {c["key"] for c in categories_list}
            for cat_key, count in category_counts.items():
                if cat_key not in known_keys:
                    categories_list.append({"key": cat_key, "name": cat_key, "count": count})
        categories_list.sort(key=lambda x: x["count"], reverse=True)
        return categories_list

    @staticmethod
    def _read_file_base64(file_path: str) -> str:
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

    # ── Handlers ───────────────────────────────────────────────

    async def handle_health_check(self) -> dict:
        return self.ok({"status": "ok", "service": "emoji-manager-webui"})

    async def handle_get_stats(self) -> dict:
        try:
            index = self.get_runtime_index_snapshot()
            today_start = (
                datetime.now()
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .timestamp()
            )
            today_count = 0
            categories_count = 0
            if hasattr(self.plugin, "plugin_config"):
                categories_count = len(self.plugin_config.categories)

            db_service = self.db_service
            total_count = len(index)
            get_stats = getattr(db_service, "get_stats", None) if db_service else None
            count_created_since = getattr(db_service, "count_created_since", None) if db_service else None
            db_today_loaded = False
            if db_service and callable(get_stats):
                try:
                    db_stats = get_stats()
                    total_count = int(db_stats.get("total_emojis", total_count) or 0)
                    if callable(count_created_since):
                        today_count = int(count_created_since(today_start) or 0)
                        db_today_loaded = True
                except Exception as db_error:
                    logger.warning(f"Falling back to cache stats: {db_error}")
            if not db_today_loaded:
                for meta in index.values():
                    if isinstance(meta, dict) and meta.get("created_at", 0) >= today_start:
                        today_count += 1

            return self.ok({
                "stats": {
                    "total": total_count,
                    "categories": categories_count,
                    "today": today_count,
                }
            })
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return self.err(str(e))

    async def handle_list_images(self) -> dict:
        try:
            page = request.args.get("page", 1, type=int)
            page_size = request.args.get("size", 50, type=int)
            category_filter = request.args.get("category", None)
            search_query = str(request.args.get("q", "")).lower()
            sort_order = request.args.get("sort", "newest")

            db_service = self.db_service
            get_emojis_paginated = getattr(db_service, "get_emojis_paginated", None) if db_service else None

            if db_service and callable(get_emojis_paginated) and db_service.count_total() > 0:
                raw_images, total, category_counts = get_emojis_paginated(
                    page=page,
                    page_size=page_size,
                    category=category_filter,
                    sort_order=sort_order,
                    search_query=search_query if search_query else None,
                )
                images: list[dict[str, Any]] = []
                for item in raw_images:
                    try:
                        abs_path = Path(item["path"])
                        rel_path = abs_path.relative_to(self.data_dir)
                        image_item = {
                            "hash": item.get("hash", ""),
                            "thumb_url": "",
                            "url": "",
                            "category": item.get("category", "unknown"),
                            "tags": item.get("tags", []),
                            "desc": item.get("desc", ""),
                            "scenes": self.split_scene_terms(item.get("scenes", [])),
                            "scope_mode": self.normalize_scope_mode(item.get("scope_mode", "public")) or "public",
                            "origin_target": str(item.get("origin_target", "") or ""),
                            "created_at": item.get("created_at", 0),
                        }
                        images.append(image_item)
                    except ValueError:
                        continue
                categories_list = self.build_categories_list(category_counts)
                return self.ok({
                    "total": total,
                    "page": page,
                    "size": page_size,
                    "images": images,
                    "categories": categories_list,
                })

            index = self.get_runtime_index_snapshot()
            images: list[dict[str, Any]] = []
            category_counts: dict[str, int] = {}

            for path_str, meta in index.items():
                try:
                    abs_path = Path(path_str)
                    rel_path = abs_path.relative_to(self.data_dir)
                    if not abs_path.exists():
                        continue
                    item = {
                        "hash": meta.get("hash", ""),
                        "thumb_url": "",
                        "url": "",
                        "category": meta.get("category", "unknown"),
                        "tags": meta.get("tags", []),
                        "desc": meta.get("desc", ""),
                        "scenes": self.split_scene_terms(meta.get("scenes", [])),
                        "scope_mode": self.normalize_scope_mode(meta.get("scope_mode", "public")) or "public",
                        "origin_target": str(meta.get("origin_target", "") or ""),
                        "created_at": meta.get("created_at", 0),
                    }
                    if search_query and not (
                        any(search_query in str(t).lower() for t in item["tags"])
                        or search_query in item["desc"].lower()
                        or any(search_query in str(scene).lower() for scene in item.get("scenes", []))
                    ):
                        continue
                    cat = item["category"]
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    if category_filter and item["category"] != category_filter:
                        continue
                    images.append(item)
                except ValueError:
                    continue

            if sort_order == "oldest":
                images.sort(key=lambda x: (int(x.get("created_at", 0) or 0), str(x.get("hash", "") or "")))
            else:
                images.sort(key=lambda x: (int(x.get("created_at", 0) or 0), str(x.get("hash", "") or "")), reverse=True)

            total = len(images)
            start = (page - 1) * page_size
            end = start + page_size
            paged_images = images[start:end]
            categories_list = self.build_categories_list(category_counts)

            return self.ok({
                "total": total,
                "page": page,
                "size": page_size,
                "images": paged_images,
                "categories": categories_list,
            })
        except Exception as e:
            logger.error(f"Error listing images: {e}")
            return self.err(str(e))

    async def handle_image_data(self) -> dict:
        """Return base64 data URI for a single image."""
        image_hash = request.args.get("hash", "").strip()
        if not image_hash:
            return self.err("缺少 hash 参数")

        index = self.get_runtime_index_snapshot()
        for path_str, meta in index.items():
            if isinstance(meta, dict) and meta.get("hash") == image_hash:
                try:
                    if os.path.exists(path_str):
                        data_url = self._read_file_base64(path_str)
                        return self.ok({"hash": image_hash, "url": data_url})
                except Exception as e:
                    logger.warning(f"读取图片失败: {e}")
                break
        return self.err("图片未找到")

    async def handle_upload_image(self) -> dict:
        try:
            files = await request.files
            if "file" not in files:
                return self.err("没有上传文件")

            uploaded_file = files["file"]
            filename = uploaded_file.filename or "upload.png"
            file_ext = Path(filename).suffix.lower()
            if not self.is_allowed_image_ext(file_ext):
                return self.err(f"不支持的文件类型，允许: {', '.join(self.ALLOWED_IMAGE_EXTS)}")

            file_content = uploaded_file.read()
            image = await self.persist_uploaded_image(
                file_content=file_content,
                file_ext=file_ext,
                category="unknown",
            )
            await self.sync_index_to_db(raise_on_error=True)
            return self.ok({"image": image, "hash": image["hash"]})
        except Exception as e:
            logger.error(f"上传图片失败: {e}", exc_info=True)
            return self.err(str(e))

    async def handle_update_image(self) -> dict:
        try:
            data = await request.get_json()
            image_hash = data.get("hash") if data else None
            if not image_hash:
                return self.err("缺少 hash")

            new_category = data.get("category") if data else None
            new_tags = data.get("tags") if data else None
            new_desc = data.get("desc") if data else None
            new_scenes = data.get("scenes", data.get("scene")) if data else None
            new_scope_mode = self.normalize_scope_mode(data.get("scope_mode") if data else None)

            updated = {"ok": False, "error": ""}

            async def updater(current: dict):
                target_path = None
                meta = None
                for path_str, m in current.items():
                    if isinstance(m, dict) and m.get("hash") == image_hash:
                        target_path = path_str
                        meta = m
                        break
                if not target_path or not meta:
                    updated["error"] = "Image not found"
                    return
                if new_tags is not None:
                    meta["tags"] = self.split_csv_tags(new_tags) if isinstance(new_tags, str) else new_tags
                if new_desc is not None:
                    meta["desc"] = new_desc
                if new_scenes is not None:
                    meta["scenes"] = self.split_scene_terms(new_scenes)
                if new_scope_mode:
                    if new_scope_mode == "local" and not str(meta.get("origin_target", "") or "").strip():
                        updated["error"] = "Origin target missing"
                        return
                    meta["scope_mode"] = new_scope_mode
                if new_category and new_category != meta.get("category"):
                    old_path = Path(target_path)
                    if not old_path.exists():
                        updated["error"] = "Source file not found"
                        return
                    moved, _ = await self.move_items_to_category(current, [(target_path, meta)], new_category)
                    if moved <= 0:
                        updated["error"] = "Move failed"
                        return
                else:
                    current[target_path] = meta
                updated["ok"] = True

            await self.update_runtime_index(updater, raise_on_sync_error=True)
            if not updated["ok"]:
                return self.err(updated["error"] or "Update failed")
            return self.ok()
        except Exception as e:
            logger.error(f"Failed to update image: {e}")
            return self.err(str(e))

    async def handle_delete_image(self) -> dict:
        try:
            data = await request.get_json()
            image_hash = (data or {}).get("hash", "").strip()
            if not image_hash:
                return self.err("缺少 hash")
            add_to_blacklist = (data or {}).get("blacklist", False)

            removed_paths = await self.collect_removed_paths_by_hashes({image_hash}, raise_on_sync_error=True)
            target_path = removed_paths[0] if removed_paths else None

            if target_path:
                try:
                    await self.plugin._safe_remove_file(target_path)
                    if add_to_blacklist:
                        await self.cache_service.set("blacklist_cache", image_hash, int(time.time()), persist=True)
                    self.invalidate_hashes([image_hash])
                    return self.ok()
                except Exception as e:
                    logger.error(f"Failed to delete image {target_path}: {e}")
                    return self.err(str(e))
            return self.err("Image not found")
        except Exception as e:
            logger.error(f"Delete image failed: {e}")
            return self.err(str(e))

    async def handle_batch_delete(self) -> dict:
        try:
            data = await request.get_json()
            hashes = (data or {}).get("hashes", [])
            if not hashes:
                return self.ok(count=0)
            hash_set = set(hashes)
            removed_paths = await self.collect_removed_paths_by_hashes(hash_set, raise_on_sync_error=True)
            deleted_count = await self.delete_paths_best_effort(removed_paths, "Failed to delete {path}: {error}")
            return self.ok(count=deleted_count)
        except Exception as e:
            logger.error(f"Batch delete failed: {e}")
            return self.err(str(e))

    async def handle_batch_move(self) -> dict:
        try:
            data = await request.get_json()
            hashes = (data or {}).get("hashes", [])
            target_category = (data or {}).get("category")
            if not hashes or not target_category:
                return self.err("Missing hashes or category")
            hash_set = set(hashes)
            moved_count = 0
            moved_hashes: list[str] = []

            async def updater(current: dict):
                nonlocal moved_count, moved_hashes
                items = self.collect_items_by_hashes(current, hash_set)
                items = [(p, m) for p, m in items if m.get("category") != target_category]
                moved_count, moved_hashes = await self.move_items_to_category(current, items, target_category)

            await self.update_runtime_index(updater, raise_on_sync_error=True)
            if moved_count > 0:
                self.invalidate_hashes(moved_hashes)
            return self.ok(count=moved_count)
        except Exception as e:
            logger.error(f"Batch move failed: {e}")
            return self.err(str(e))

    async def handle_batch_scope(self) -> dict:
        try:
            data = await request.get_json()
            hashes = (data or {}).get("hashes", [])
            scope_mode = self.normalize_scope_mode((data or {}).get("scope_mode"))
            if not hashes or not scope_mode:
                return self.err("Missing hashes or scope_mode")
            result = {"updated": 0, "skipped": 0}
            hash_set = set(hashes)

            async def updater(current: dict):
                for _, meta in self.collect_items_by_hashes(current, hash_set):
                    if not isinstance(meta, dict):
                        continue
                    if scope_mode == "local" and not str(meta.get("origin_target", "") or "").strip():
                        result["skipped"] += 1
                        continue
                    meta["scope_mode"] = scope_mode
                    result["updated"] += 1

            await self.update_runtime_index(updater, raise_on_sync_error=True)
            return self.ok(count=result["updated"], skipped=result["skipped"])
        except Exception as e:
            logger.error(f"Batch scope update failed: {e}")
            return self.err(str(e))

    async def handle_batch_upload(self) -> dict:
        try:
            files = await request.files
            form = await request.form
            category = form.get("category", "").strip()
            auto_analyze = form.get("auto_analyze", "false").strip().lower() == "true"

            files_data = []
            for field_name in files:
                file_obj = files[field_name]
                filename = file_obj.filename or "upload.png"
                file_ext = Path(filename).suffix.lower()
                if not self.is_allowed_image_ext(file_ext):
                    continue
                content = file_obj.read()
                if content:
                    files_data.append({
                        "filename": filename,
                        "content": content,
                        "hash": self.cache_service.compute_hash(content),
                        "ext": file_ext,
                    })

            if not files_data:
                return self.err("没有上传有效的图片文件")
            if not category and not auto_analyze:
                return self.err("请选择情绪分类或启用自动识别")

            fallback_category = category or self.get_configured_category_keys()[0] if self.get_configured_category_keys() else None
            if not fallback_category:
                return self.err("未配置任何情绪分类")

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
                self._process_batch_upload(task_id, files_data, category, auto_analyze, fallback_category)
            )
            return self.ok({"task_id": task_id, "total": len(files_data)})
        except Exception as e:
            logger.error(f"批量上传失败: {e}", exc_info=True)
            return self.err(str(e))

    async def _process_batch_upload(
        self, task_id: str, files_data: list[dict], category: str, auto_analyze: bool, fallback: str
    ) -> None:
        try:
            task_info = self.batch_upload_tasks.get(task_id)
            if not task_info:
                return
            for file_data in files_data:
                try:
                    tags = []
                    desc = ""
                    scenes = []
                    final_category = category or fallback
                    configured_categories = self.get_configured_category_keys()

                    if auto_analyze:
                        try:
                            img_hash = file_data["hash"]
                            temp_path = self.data_dir / "temp" / f"{img_hash}{file_data['ext']}"
                            temp_path.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(lambda: temp_path.write_bytes(file_data["content"]))
                            image_processor = self.plugin.image_processor_service
                            result_category, result_tags, result_desc, _, result_scenes = (
                                await image_processor.classify_image(
                                    event=None,
                                    file_path=str(temp_path),
                                    categories=list(self.plugin.plugin_config.categories or []),
                                    content_filtration=False,
                                )
                            )
                            if (
                                result_category
                                and result_category != image_processor.CATEGORY_FILTERED
                                and (not configured_categories or result_category in configured_categories)
                            ):
                                final_category = result_category
                                tags = result_tags or []
                                desc = result_desc or ""
                                scenes = result_scenes or []
                            await asyncio.to_thread(lambda: temp_path.unlink() if temp_path.exists() else None)
                        except Exception as e:
                            logger.warning(f"自动分析失败: {e}")

                    image = await self.persist_uploaded_image(
                        file_content=file_data["content"],
                        file_ext=file_data["ext"],
                        category=final_category,
                        file_hash=file_data["hash"],
                        tags=tags,
                        desc=desc,
                        scenes=scenes,
                    )
                    task_info["results"].append({"hash": image["hash"], "url": image["url"], "category": image["category"], "success": True})
                    task_info["success"] += 1
                except Exception as e:
                    logger.error(f"处理文件 {file_data['filename']} 失败: {e}")
                    task_info["results"].append({"filename": file_data["filename"], "success": False, "error": str(e)})
                    task_info["failed"] += 1
                task_info["processed"] += 1

            await self.sync_index_to_db(raise_on_error=True)
            task_info["status"] = "completed"
        except Exception as e:
            logger.error(f"批量上传任务 {task_id} 失败: {e}")
            if task_id in self.batch_upload_tasks:
                self.batch_upload_tasks[task_id]["status"] = "failed"
                self.batch_upload_tasks[task_id]["error"] = str(e)

    async def handle_batch_upload_status(self) -> dict:
        task_id = request.args.get("task_id", "").strip()
        if not task_id:
            return self.err("无效的任务ID")
        task_info = self.batch_upload_tasks.get(task_id)
        if not task_info:
            return self.err("任务不存在或已过期")
        return self.ok({
            "task_id": task_id,
            "status": task_info["status"],
            "total": task_info["total"],
            "processed": task_info["processed"],
            "success_count": task_info["success"],
            "failed_count": task_info["failed"],
            "error": task_info.get("error", ""),
            "results": task_info.get("results", []),
        })

    async def handle_analyze_image(self) -> dict:
        try:
            if not hasattr(self.plugin, "image_processor_service"):
                return self.err("图片处理服务不可用")

            data = await request.get_json()
            image_hash = (data or {}).get("hash", "").strip()
            if not image_hash:
                return self.err("缺少 hash")

            # Find the file path from index
            index = self.get_runtime_index_snapshot()
            file_path = None
            for path_str, meta in index.items():
                if isinstance(meta, dict) and meta.get("hash") == image_hash:
                    file_path = path_str
                    break

            if not file_path or not os.path.exists(file_path):
                return self.err("图片文件不存在")

            image_processor = self.plugin.image_processor_service
            category, tags, desc, _, scenes = await image_processor.classify_image(
                event=None,
                file_path=file_path,
                categories=list(self.plugin.plugin_config.categories or []),
                content_filtration=False,
            )

            if category == image_processor.CATEGORY_FILTERED:
                return self.err("图片内容审核不通过")
            if not category:
                return self.err("无法识别图片分类")

            return self.ok({
                "category": category,
                "tags": tags,
                "description": desc,
                "scenes": scenes or [],
            })
        except Exception as e:
            logger.error(f"VLM分析失败: {e}", exc_info=True)
            return self.err(f"分析失败: {str(e)}")

    async def handle_categories(self) -> dict:
        if request.method == "POST":
            return await self._handle_update_categories()
        return await self._handle_get_categories()

    async def _handle_get_categories(self) -> dict:
        try:
            categories = {key: 0 for key in self.get_configured_category_keys()}
            db_service = self.db_service
            get_stats = getattr(db_service, "get_stats", None) if db_service else None
            if db_service and callable(get_stats):
                try:
                    db_stats = get_stats()
                    for key, count in (db_stats.get("categories", {}) or {}).items():
                        categories[str(key)] = int(count or 0)
                    return self.ok({"categories": categories})
                except Exception as db_error:
                    logger.warning(f"Falling back to runtime categories: {db_error}")
            index = self.get_runtime_index_snapshot()
            for meta in index.values():
                if isinstance(meta, dict):
                    cat = str(meta.get("category", "unknown") or "unknown")
                    categories[cat] = categories.get(cat, 0) + 1
            return self.ok({"categories": categories})
        except Exception as e:
            logger.error(f"获取分类列表失败: {e}")
            return self.err(str(e))

    async def _handle_update_categories(self) -> dict:
        try:
            data = await request.get_json()
            new_categories_data = (data or {}).get("categories", [])
            if not isinstance(new_categories_data, list) or not new_categories_data:
                return self.err("分类列表无效")

            category_keys: list[str] = []
            category_info: dict[str, dict[str, str]] = {}
            seen: set[str] = set()
            for item in new_categories_data:
                if isinstance(item, dict) and item.get("key"):
                    key = str(item["key"]).strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    category_keys.append(key)
                    name = str(item.get("name", "") or "").strip()
                    desc = str(item.get("desc", "") or "").strip()
                    if name or desc:
                        category_info[key] = {"name": name, "desc": desc}
                elif isinstance(item, str):
                    key = item.strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    category_keys.append(key)
            if not category_keys:
                return self.err("分类列表无效")

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": category_keys})
            else:
                self.plugin.plugin_config.categories = category_keys
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = category_keys

            current_info = dict(getattr(self.plugin_config, "category_info", {}) or {})
            self.plugin_config.category_info = {key: current_info.get(key, {}) for key in category_keys}
            self.plugin_config.category_info.update(category_info)
            self.plugin_config.ensure_category_dirs(category_keys)
            self.plugin_config.save_category_info()
            return self.ok(categories=category_keys)
        except Exception as e:
            logger.error(f"更新分类列表失败: {e}", exc_info=True)
            return self.err(str(e))

    async def handle_delete_category(self) -> dict:
        try:
            data = await request.get_json()
            category_key = str((data or {}).get("key", "")).strip()
            if not category_key:
                return self.err("分类Key无效")
            if not hasattr(self.plugin, "plugin_config") or not self.plugin.plugin_config:
                return self.err("配置服务不可用")

            current_categories = list(self.plugin.plugin_config.categories or [])
            if category_key not in current_categories:
                return self.err("分类不存在")
            if len(current_categories) <= 1:
                return self.err("至少需要保留1个分类")

            updated_categories = [c for c in current_categories if c != category_key]

            # Remove from runtime index
            import shutil

            deleted_file_count = 0
            index_copy = self.get_runtime_index_snapshot()
            for path_str, meta in list(index_copy.items()):
                if isinstance(meta, dict) and meta.get("category") == category_key:
                    old_path = Path(path_str)
                    if old_path.exists():
                        try:
                            await self.plugin._safe_remove_file(str(old_path))
                            deleted_file_count += 1
                            image_hash = meta.get("hash")
                            if image_hash and hasattr(self.plugin, "image_processor_service"):
                                try:
                                    self.plugin.image_processor_service.invalidate_cache(image_hash)
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.warning(f"删除分类文件失败: {old_path}, {e}")

            async def updater(current: dict):
                for path_str, meta in list(current.items()):
                    if isinstance(meta, dict) and meta.get("category") == category_key:
                        del current[path_str]

            await self.update_runtime_index(updater, raise_on_sync_error=True)

            # Remove category dir
            category_dir = Path(self.data_dir) / "categories" / category_key
            try:
                if category_dir.exists() and category_dir.is_dir():
                    orphan_files = [p for p in category_dir.rglob("*") if p.is_file()]
                    deleted_file_count += len(orphan_files)
                    await asyncio.to_thread(shutil.rmtree, category_dir, True)
            except Exception as e:
                logger.warning(f"删除空分类目录失败: {category_dir}, {e}")

            if category_key in getattr(self.plugin_config, "category_info", {}):
                del self.plugin_config.category_info[category_key]
                self.plugin_config.save_category_info()

            if hasattr(self.plugin, "_update_config_from_dict"):
                self.plugin._update_config_from_dict({"categories": updated_categories})
            else:
                self.plugin.plugin_config.categories = updated_categories
                if hasattr(self.plugin, "categories"):
                    self.plugin.categories = updated_categories

            return self.ok({"deleted": category_key, "categories": updated_categories, "deleted_files": deleted_file_count})
        except Exception as e:
            logger.error(f"删除分类失败: {e}", exc_info=True)
            return self.err(str(e))

    async def handle_get_emotions(self) -> dict:
        try:
            category_info = self.plugin_config.get_category_info()
            return self.ok({"emotions": category_info})
        except Exception as e:
            logger.error(f"获取情绪分类失败: {e}")
            return self.err(str(e))
