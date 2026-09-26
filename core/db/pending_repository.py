"""SQLite pending repository operations."""

import asyncio
import sqlite3
import time
from typing import Any


from ..util.normalization import normalize_scope_mode


class PendingRepository:
    # ── 待审核池 (emoji_pending) CRUD ──

    @staticmethod
    def _split_multi(text: Any) -> list[str]:
        if not text:
            return []
        return [s.strip() for s in str(text).split(",") if s.strip()]

    def count_pending(self) -> int:
        """统计待审核池数量（O(1) COUNT），用于偷取护栏与审核区进度条。"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM emoji_pending").fetchone()
            return int(row["cnt"] if row else 0)

    async def insert_pending(self, meta: dict[str, Any]) -> int | None:
        """插入一条待审核记录。

        Args:
            meta: 至少含 path/hash；可选 phash/category/desc/source/origin_target/
                  scope_mode/tags/scenes/created_at。

        Returns:
            新插入行的 id；path 冲突(UNIQUE)返回 None。
        """
        if not meta or not meta.get("path"):
            return None
        async with self._write_lock:
            return await asyncio.to_thread(self._insert_pending_sync, meta)

    def _insert_pending_sync(self, meta: dict[str, Any]) -> int | None:
        created_at = int(meta.get("created_at") or int(time.time()))
        with self._get_connection() as conn:
            try:
                cur = conn.execute(
                    f"""
                    INSERT INTO emoji_pending
                    ({", ".join(self._PENDING_INSERT_COLUMNS)})
                    VALUES ({", ".join("?" * len(self._PENDING_INSERT_COLUMNS))})
                    """,
                    (
                        meta.get("path"),
                        meta.get("hash", ""),
                        meta.get("phash"),
                        meta.get("category"),
                        meta.get("desc"),
                        meta.get("source"),
                        meta.get("origin_target"),
                        meta.get("scope_mode", "public"),
                        "pending",
                        created_at,
                        "",  # tags_text 已废弃（v5 改关联表）
                        "",  # scenes_text 已废弃
                        meta.get("source_url"),
                        meta.get("original_name"),
                        meta.get("width"),
                        meta.get("height"),
                        meta.get("format"),
                        meta.get("bytes"),
                        meta.get("add_method"),
                        meta.get("overlay_text"),
                        self._emotions_json_from_meta(meta),
                        str(meta.get("character") or "").strip(),
                        str(meta.get("retention_class") or "native").strip() or "native",
                    ),
                )
                path = str(meta.get("path") or "")
                # 标签/场景写入关联表（与正式表 emoji_tag/emoji_scene 同构）
                for tag in self._normalize_multi_value(meta.get("tags")):
                    conn.execute(
                        "INSERT OR IGNORE INTO emoji_pending_tag (path, tag) VALUES (?, ?)",
                        (path, tag),
                    )
                for scene in self._normalize_multi_value(meta.get("scenes")):
                    conn.execute(
                        "INSERT OR IGNORE INTO emoji_pending_scene (path, scene) VALUES (?, ?)",
                        (path, scene),
                    )
                return int(cur.lastrowid) if cur.lastrowid else None
            except sqlite3.IntegrityError:
                # 仅作 pending 内去重：同路径已存在说明该图已在池中
                return None

    def get_pending_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        category: str | None = None,
        search_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        """分页获取待审核列表，支持分类筛选与文本搜索。固定按 created_at 降序。"""
        where_clauses: list[str] = []
        params: list[Any] = []

        if category:
            where_clauses.append("p.category = ?")
            params.append(category)

        if search_query:
            search_pattern = f"%{search_query}%"
            where_clauses.append(
                "(p.desc LIKE ? OR p.category LIKE ? OR p.hash LIKE ?"
                " OR p.origin_target LIKE ? OR p.source LIKE ? OR p.path LIKE ?"
                " OR p.overlay_text LIKE ?"
                " OR EXISTS("
                "SELECT 1 FROM emoji_pending_tag t WHERE t.path = p.path AND t.tag LIKE ?"
                ") OR EXISTS("
                "SELECT 1 FROM emoji_pending_scene s WHERE s.path = p.path AND s.scene LIKE ?"
                "))"
            )
            params.extend([search_pattern] * 9)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._get_connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) as cnt FROM emoji_pending p {where_sql}", params
            ).fetchone()["cnt"]

            category_where_clauses = [
                clause for clause in where_clauses if clause != "p.category = ?"
            ]
            category_params = params[1:] if category else params
            category_where_sql = (
                "WHERE " + " AND ".join(category_where_clauses)
                if category_where_clauses
                else ""
            )
            cat_rows = conn.execute(
                f"SELECT p.category, COUNT(*) as cnt FROM emoji_pending p {category_where_sql} "
                "GROUP BY p.category",
                category_params,
            ).fetchall()
            category_counts = {r["category"]: r["cnt"] for r in cat_rows}

            offset = max(0, (page - 1)) * page_size
            rows = conn.execute(
                f"""
                SELECT p.id, p.path, p.hash, p.phash, p.category, p.desc,
                       p.source, p.origin_target, p.scope_mode, p.review_status,
                       p.created_at, p.tags_text, p.scenes_text,
                       p.source_url, p.original_name, p.width, p.height,
                       p.format, p.bytes, p.add_method,
                       p.overlay_text, p.emotions_json, p.character, p.retention_class
                FROM emoji_pending p {where_sql}
                ORDER BY p.created_at DESC, p.id DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()

            items: list[dict[str, Any]] = []
            if rows:
                paths = [r["path"] for r in rows]
                tags_map = self._load_related_map(
                    conn, table="emoji_pending_tag", value_column="tag", paths=paths
                )
                scenes_map = self._load_related_map(
                    conn, table="emoji_pending_scene", value_column="scene", paths=paths
                )
                for row in rows:
                    item = dict(row)
                    tags = tags_map.get(row["path"], [])
                    scenes = scenes_map.get(row["path"], [])
                    # 兼容旧数据：关联表为空时回退到旧逗号列
                    if not tags and item.get("tags_text"):
                        tags = self._split_multi(item.pop("tags_text", ""))
                    if not scenes and item.get("scenes_text"):
                        scenes = self._split_multi(item.pop("scenes_text", ""))
                    item.pop("tags_text", None)
                    item.pop("scenes_text", None)
                    item["tags"] = tags
                    item["scenes"] = scenes
                    items.append(self._hydrate_entry(item))
            return items, total, category_counts

    def get_pending(self, pending_id: int) -> dict[str, Any] | None:
        """获取单条待审核记录（含拆分后的 tags/scenes，来自关联表）。"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM emoji_pending WHERE id = ?", (pending_id,)
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            path = str(item.get("path") or "")
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM emoji_pending_tag WHERE path = ? ORDER BY rowid", (path,)
                ).fetchall()
            ]
            scenes = [
                r["scene"]
                for r in conn.execute(
                    "SELECT scene FROM emoji_pending_scene WHERE path = ? ORDER BY rowid", (path,)
                ).fetchall()
            ]
            # 兼容旧数据：关联表为空时回退到旧逗号列
            if not tags:
                tags = self._split_multi(item.pop("tags_text", ""))
            if not scenes:
                scenes = self._split_multi(item.pop("scenes_text", ""))
            item.pop("tags_text", None)
            item.pop("scenes_text", None)
            item["tags"] = tags
            item["scenes"] = scenes
            return self._hydrate_entry(item)

    def get_pending_by_hash(self, hash_val: str) -> dict[str, Any] | None:
        """按内容哈希查一条待审核记录（用于缩略图回退）。返回含 path 字段。"""
        if not hash_val:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, path, hash, phash, category, character "
                "FROM emoji_pending WHERE hash = ? LIMIT 1",
                (hash_val,),
            ).fetchone()
            return dict(row) if row else None

    async def update_pending(
        self,
        pending_id: int,
        fields: dict[str, Any],
        *,
        allowed_fields: tuple[str, ...] = (
            "category",
            "desc",
            "tags",
            "scenes",
            "scope_mode",
            "phash",
            "overlay_text",
            "emotions",
            "emotions_json",
            "character",
        ),
    ) -> dict[str, Any] | None:
        """更新一条待审核记录。仅允许白名单字段，避免改写 path/hash/source/origin_target。

        Args:
            pending_id: 待审核行 id
            fields: 待更新字段；tags/scenes 会 join 为多值列，其它原样写入
            allowed_fields: 白名单，防止调用方误改 path/hash

        Returns:
            更新后的完整行（dict），不存在或字段非白名单导致无写入时返回 None。
        """
        if not pending_id or not isinstance(fields, dict) or not fields:
            return None

        # 字段白名单过滤
        clean_fields: dict[str, Any] = {}
        for key, value in fields.items():
            if key in allowed_fields:
                clean_fields[key] = value

        if not clean_fields:
            return None

        # tags/scenes 拆分出来走关联表（不再写 tags_text/scenes_text 列）
        new_tags = clean_fields.pop("tags", None)
        new_scenes = clean_fields.pop("scenes", None)
        if "emotions" in clean_fields:
            clean_fields["emotions_json"] = self._dump_emotions_json(
                clean_fields.pop("emotions")
            )
        elif "emotions_json" in clean_fields:
            clean_fields["emotions_json"] = self._dump_emotions_json(
                clean_fields.get("emotions_json")
            )

        # scope_mode 兜底
        if "scope_mode" in clean_fields:
            clean_fields["scope_mode"] = normalize_scope_mode(clean_fields["scope_mode"])

        # category 不能为空
        if "category" in clean_fields:
            cat = str(clean_fields["category"] or "").strip()
            if not cat:
                return None
            clean_fields["category"] = cat

        # desc 兜底字符串
        if "desc" in clean_fields:
            clean_fields["desc"] = str(clean_fields["desc"] or "").strip() or None

        async with self._write_lock:
            return await asyncio.to_thread(
                self._update_pending_sync, pending_id, clean_fields, new_tags, new_scenes
            )

    def _update_pending_sync(
        self,
        pending_id: int,
        clean_fields: dict[str, Any],
        new_tags: Any = None,
        new_scenes: Any = None,
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            if clean_fields:
                set_clause = ", ".join(f"{col} = ?" for col in clean_fields.keys())
                params: list[Any] = list(clean_fields.values()) + [pending_id]
                conn.execute(
                    f"UPDATE emoji_pending SET {set_clause} WHERE id = ?",
                    params,
                )
            # 注：SQLite 对值未变的 no-op UPDATE 会返回 rowcount=0，
            # 不能据此判断"行不存在"。统一回查一次行存在性。
            row = conn.execute(
                "SELECT * FROM emoji_pending WHERE id = ?", (pending_id,)
            ).fetchone()
            if not row:
                return None

            path = str(row["path"] or "")
            # 标签/场景写关联表（全量替换，与正式表 update 语义一致）
            if new_tags is not None:
                conn.execute("DELETE FROM emoji_pending_tag WHERE path = ?", (path,))
                for tag in self._normalize_multi_value(new_tags):
                    conn.execute(
                        "INSERT OR IGNORE INTO emoji_pending_tag (path, tag) VALUES (?, ?)",
                        (path, tag),
                    )
            if new_scenes is not None:
                conn.execute("DELETE FROM emoji_pending_scene WHERE path = ?", (path,))
                for scene in self._normalize_multi_value(new_scenes):
                    conn.execute(
                        "INSERT OR IGNORE INTO emoji_pending_scene (path, scene) VALUES (?, ?)",
                        (path, scene),
                    )

            item = dict(row)
            tags = [
                r["tag"]
                for r in conn.execute(
                    "SELECT tag FROM emoji_pending_tag WHERE path = ? ORDER BY rowid", (path,)
                ).fetchall()
            ]
            scenes = [
                r["scene"]
                for r in conn.execute(
                    "SELECT scene FROM emoji_pending_scene WHERE path = ? ORDER BY rowid", (path,)
                ).fetchall()
            ]
            # 兼容旧数据：关联表为空时回退到旧逗号列
            if not tags:
                tags = self._split_multi(item.pop("tags_text", ""))
            if not scenes:
                scenes = self._split_multi(item.pop("scenes_text", ""))
            item.pop("tags_text", None)
            item.pop("scenes_text", None)
            item["tags"] = tags
            item["scenes"] = scenes
            return self._hydrate_entry(item)

    def delete_pending(self, pending_id: int) -> dict[str, Any] | None:
        """删除单条待审核记录，返回被删行的 path/hash（供删除磁盘文件用）；不存在返回 None。"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT path, hash FROM emoji_pending WHERE id = ?", (pending_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM emoji_pending WHERE id = ?", (pending_id,))
            return {"path": row["path"], "hash": row["hash"]}

    def delete_pending_batch(self, ids: list[int]) -> list[dict[str, Any]]:
        """批量删除待审核记录，返回每条被删行的 {path, hash}。"""
        clean_ids = [i for i in ids if isinstance(i, int)]
        if not clean_ids:
            return []
        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(clean_ids))
            rows = conn.execute(
                f"SELECT id, path, hash FROM emoji_pending WHERE id IN ({placeholders})",
                clean_ids,
            ).fetchall()
            conn.execute(
                f"DELETE FROM emoji_pending WHERE id IN ({placeholders})", clean_ids
            )
            return [{"path": r["path"], "hash": r["hash"]} for r in rows]
