"""PendingRoutes methods for PluginAPI."""

import asyncio
import os
import shutil
import time
from typing import Any

from quart import jsonify, request

from astrbot.api import logger

from ..core.util.blacklist import add_blacklist_hash
from ..core.db.index_manager import refresh_search_entry
from ..core.util.safe_io import safe_remove_file

class PendingRoutes:
    # ── Pending (待审核池) ────────────────────────────────────

    def _build_pending_item(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """待审核记录与图库使用同一套展示字段。"""
        item = self._build_image_item(str(row.get("path") or ""), row)
        if item is not None:
            item.update(id=row.get("id"), review_status=str(row.get("review_status") or "pending"))
        return item

    async def handle_list_pending(self):
        """GET /pending —— 分页返回待审核列表（分类筛选/搜索/sort=newest）。"""
        try:
            db = self._db
            if not db or not hasattr(db, "get_pending_paginated"):
                return jsonify(
                    {
                        "success": True,
                        "images": [],
                        "total": 0,
                        "category_total": 0,
                        "categories": {},
                    }
                )

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
                    "category_total": sum(int(count) for count in cat_counts.values()),
                    "categories": self._build_categories_list(cat_counts),
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
        known = set(self._cfg.get_vlm_categories() if hasattr(self._cfg, "get_vlm_categories") else (self._cfg.categories or []))
        if not category or category == "other" or category not in known:
            category = self._cfg.closest_category(category) if hasattr(self._cfg, "closest_category") else "confused"

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
                "reviewed_at": int(time.time()),
                "tags": list(row.get("tags", []) or []),
                "scenes": list(row.get("scenes", []) or []),
                "overlay_text": str(row.get("overlay_text", "") or ""),
                "emotions": list(row.get("emotions", []) or []),
                "character": str(row.get("character", "") or ""),
                # v5：从 pending 继承元数据（宽高/格式/字节/来源/入库方式）
                "source_url": row.get("source_url"),
                "original_name": row.get("original_name"),
                "width": row.get("width"),
                "height": row.get("height"),
                "format": row.get("format"),
                "bytes": row.get("bytes"),
                "add_method": row.get("add_method"),
                "retention_class": str(row.get("retention_class", "native") or "native"),
            }
            inserted = await db.insert_batch([emoji_entry])
            if not inserted:
                raise RuntimeError("insert emoji returned 0")
            if hasattr(db, "promote_source_pending_path"):
                await db.promote_source_pending_path(src_path, cat_path)
            db.delete_pending(pending_id)

            await refresh_search_entry(self.plugin, cat_path, emoji_entry)

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
                    if await add_blacklist_hash(self.plugin, h):
                        blacklisted += 1
                    else:
                        logger.warning(f"拉黑失败 {h}")
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
            if "character" in data:
                character = self._normalize_character_key(str(data.get("character") or ""))
                if character and character not in set(self._cfg.get_characters()):
                    return jsonify({"success": False, "error": f"角色无效: {character}"})
                fields["character"] = character

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
