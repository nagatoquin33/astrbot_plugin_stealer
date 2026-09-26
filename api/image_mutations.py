"""ImageMutationRoutes methods for PluginAPI."""

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any

from quart import jsonify, request

from astrbot.api import logger

from ..core.db.index_manager import delete_index_paths
from ..core.util.safe_io import safe_remove_file

class ImageMutationRoutes:
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
                overlay_text=metadata.get("overlay_text", ""),
                emotions=metadata.get("emotions") or [],
                character=metadata.get("character", ""),
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
            new_overlay = data.get("overlay_text")
            new_emotions = data.get("emotions")
            new_character = data.get("character") if "character" in data else None
            new_scope = self._norm_scope(data["scope_mode"]) if "scope_mode" in data else None
            new_favorite = data.get("is_favorite")
            found = self._find_index_entry_by_hash(str(img_hash))
            if not found:
                return jsonify({"success": False, "error": "Image not found"})
            target, meta = found

            updates: dict[str, Any] = {}
            if new_tags is not None:
                updates["tags"] = self._split_csv(new_tags)
            if new_desc is not None:
                updates["desc"] = new_desc
            if new_scenes is not None:
                updates["scenes"] = self._split_scenes(new_scenes)
            if new_overlay is not None:
                updates["overlay_text"] = str(new_overlay or "").strip()
            if new_emotions is not None:
                updates["emotions"] = self._split_csv(new_emotions)
            if new_character is not None:
                updates["character"] = self._normalize_character_key(str(new_character or ""))
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

            index = self._get_index()
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
                await delete_index_paths(self.plugin, deleted_paths)
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
            index = self._get_index()
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
                await delete_index_paths(self.plugin, deleted_paths)
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
            index = self._get_index()
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

    async def handle_batch_character(self):
        try:
            data = await request.get_json() or {}
            hashes = set(data.get("hashes", []))
            character = self._normalize_character_key(str(data.get("character", "") or ""))
            if not hashes:
                return jsonify({"success": False, "error": "缺少 hashes"})
            if character and character not in set(self._cfg.get_characters()):
                return jsonify({"success": False, "error": f"角色无效: {character}"})
            updated = 0
            index = self._get_index()
            for path, meta in index.items():
                if not isinstance(meta, dict) or meta.get("hash") not in hashes:
                    continue
                if str(meta.get("character", "") or "") == character:
                    continue
                if await self._update_index_path(path, {"character": character}):
                    updated += 1
            return jsonify({"success": True, "count": updated})
        except Exception as e:
            logger.error(f"批量分配角色失败: {e}", exc_info=True)
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

            index = self._get_index()
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

            index = self._get_index()
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
                                "hash": hashlib.sha256(content).hexdigest(),
                                "ext": ext,
                            }
                        )
                category = str(data.get("category", "")).strip()
                auto_analyze = str(data.get("auto_analyze", "false")).lower() == "true"
                character = self._normalize_character_key(str(data.get("character", "") or ""))
            else:
                files = await request.files
                form = await request.form
                category = form.get("category", "").strip()
                auto_analyze = form.get("auto_analyze", "false").lower() == "true"
                character = self._normalize_character_key(str(form.get("character", "") or ""))
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
                                "hash": hashlib.sha256(content).hexdigest(),
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
                self._process_batch(task_id, files_data, category, auto_analyze, fallback, character)
            )
            return jsonify({"success": True, "task_id": task_id, "total": len(files_data)})
        except Exception as e:
            logger.error(f"批量上传失败: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)})

    async def _process_batch(
        self, task_id: str, files_data: list[dict], category: str, auto_analyze: bool, fallback: str, character: str = ""
    ) -> None:
        try:
            task = self.batch_upload_tasks.get(task_id)
            if not task:
                return
            for fd in files_data:
                tmp: Path | None = None
                try:
                    tags, desc, scenes = [], "", []
                    overlay_text, emotions = "", []
                    final_cat = category or fallback
                    if auto_analyze:
                        try:
                            img_hash = fd["hash"]
                            tmp = self._data_dir / "temp" / f"{img_hash}{fd['ext']}"
                            tmp.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(lambda: tmp.write_bytes(fd["content"]))
                            proc = self.plugin.image_processor_service
                            if proc:
                                classified = await proc.classify_image(
                                    event=None,
                                    file_path=str(tmp),
                                    categories=list(self._cfg.categories or []),
                                    content_filtration=False,
                                )
                                rc, rt, rd, _, rs, overlay_text, emotions = classified
                                if rc and rc != getattr(proc, "CATEGORY_FILTERED", None):
                                    final_cat = rc
                                    tags = rt or []
                                    desc = rd or ""
                                    scenes = rs or []
                                    overlay_text = overlay_text or ""
                                    emotions = emotions or []
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
                        overlay_text=overlay_text,
                        emotions=emotions,
                        character=character,
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
