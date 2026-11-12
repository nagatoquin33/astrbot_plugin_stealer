"""SQLite pending repository operations."""

import asyncio
import json
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
            conn.execute("BEGIN IMMEDIATE")
            try:
                if meta.get('hash') and conn.execute('SELECT 1 FROM emoji_pending WHERE hash=?', (meta['hash'],)).fetchone():
                    return None
                capacity = meta.get('_pending_capacity')
                if capacity and conn.execute('SELECT COUNT(*) FROM emoji_pending').fetchone()[0] >= capacity:
                    return None
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
                        "",  # 标签从关联表读取，tags_text 列写入空值。
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
                        str(meta.get("analysis_state") or "ready"),
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
                conn.execute("COMMIT")
                return int(cur.lastrowid) if cur.lastrowid else None
            except sqlite3.IntegrityError:
                conn.execute("ROLLBACK")
                # 仅作 pending 内去重：同路径已存在说明该图已在池中
                return None

    async def approve_pending(self, pending_id: int, path: str, category: str) -> bool:
        """Promote metadata and source links in one transaction."""
        async with self._write_lock:
            return await asyncio.to_thread(self._approve_pending_sync, pending_id, path, category)

    def _approve_pending_sync(self, pending_id: int, path: str, category: str) -> bool:
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM emoji_pending WHERE id = ?", (pending_id,)).fetchone()
            if row is None:
                return False
            if conn.execute("SELECT 1 FROM emoji WHERE path = ?", (path,)).fetchone():
                raise sqlite3.IntegrityError("target metadata already exists")
            if row['hash'] and conn.execute("SELECT 1 FROM emoji WHERE hash = ?", (row['hash'],)).fetchone():
                raise sqlite3.IntegrityError('image already exists in library')
            meta = dict(row)
            old_path = meta["path"]
            now = int(time.time())
            meta.update(category=category, reviewed_at=now)
            conn.execute(self._INSERT_EMOJI_SQL, self._emoji_insert_values(path, meta, now=now))
            for table, column in (("emoji_tag", "tag"), ("emoji_scene", "scene")):
                conn.execute(
                    f"INSERT INTO {table}(path, {column}) SELECT ?, {column} "
                    f"FROM {table.replace('emoji_', 'emoji_pending_')} WHERE path = ?",
                    (path, old_path),
                )
            self._promote_source_pending_in_connection(conn, old_path, path)
            conn.execute("DELETE FROM emoji_pending WHERE id = ?", (pending_id,))
            conn.execute("COMMIT")
            return True

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
                       p.overlay_text, p.emotions_json, p.character, p.retention_class,
                       p.analysis_state, p.analysis_error
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

    def get_pending_phash_map(self) -> dict[str, str]:
        """返回待审核池的 path→phash 映射，用于视觉去重。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, phash FROM emoji_pending WHERE phash IS NOT NULL AND phash != ''"
            ).fetchall()
            return {r["path"]: r["phash"] for r in rows}

    def list_pending_missing_phash(self, limit: int = 500) -> list[dict[str, Any]]:
        """列出缺少感知哈希的待审核记录，供启动回填。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, path FROM emoji_pending "
                "WHERE (phash IS NULL OR phash = '') AND path != '' "
                "ORDER BY id LIMIT ?",
                (max(0, int(limit)),),
            ).fetchall()
            return [dict(row) for row in rows]

    async def set_pending_phash(self, path: str, phash: str) -> bool:
        """只更新待审核记录的感知哈希，不触碰审核状态和人工字段。"""
        if not path or not phash:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._set_pending_phash_sync, path, phash)

    def _set_pending_phash_sync(self, path: str, phash: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE emoji_pending SET phash = ? "
                "WHERE path = ? AND (phash IS NULL OR phash = '')",
                (phash, path),
            )
            return bool(cur.rowcount)

    async def queue_pending_analysis(self, pending_id: int) -> bool:
        async with self._write_lock:
            return await asyncio.to_thread(self._queue_pending_analysis_sync, pending_id)

    def _queue_pending_analysis_sync(self, pending_id: int) -> bool:
        with self._get_connection() as conn:
            return bool(conn.execute(
                "UPDATE emoji_pending SET review_status='queued', analysis_error='' "
                "WHERE id=? AND review_status IN ('pending','failed')", (pending_id,),
            ).rowcount)

    def recover_pending_analysis(self) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE emoji_pending SET review_status='queued' WHERE review_status='processing'")

    async def claim_pending_analysis(self) -> dict[str, Any] | None:
        async with self._write_lock:
            task = asyncio.create_task(asyncio.to_thread(self._claim_pending_analysis_sync))
            try:
                pending_id = await asyncio.shield(task)
            except asyncio.CancelledError:
                pending_id = await task
                if pending_id is not None:
                    await asyncio.to_thread(self._finish_pending_analysis_sync, pending_id, '')
                raise
        return self.get_pending(pending_id) if pending_id is not None else None

    def _claim_pending_analysis_sync(self) -> int | None:
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT id FROM emoji_pending WHERE review_status='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE emoji_pending SET review_status='processing' WHERE id=?", (row['id'],))
            conn.execute("COMMIT")
            return row['id']

    async def finish_pending_analysis(self, pending_id: int, error: str = "") -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._finish_pending_analysis_sync, pending_id, error)

    def _finish_pending_analysis_sync(self, pending_id: int, error: str) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE emoji_pending SET review_status=?, analysis_error=? "
                         "WHERE id=? AND review_status='processing'",
                         ('failed' if error else 'queued', error[:500], pending_id))

    def pending_analysis_stats(self) -> dict[str, int]:
        with self._get_connection() as conn:
            return {row['review_status']: row['count'] for row in conn.execute(
                "SELECT review_status, COUNT(*) AS count FROM emoji_pending GROUP BY review_status")}

    async def update_pending(
        self,
        pending_id: int,
        fields: dict[str, Any],
        *,
        from_analysis: bool = False,
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

        # tags/scenes 分别写入标签和场景关联表。
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
                self._update_pending_sync, pending_id, clean_fields, new_tags, new_scenes, from_analysis
            )

    def _update_pending_sync(
        self,
        pending_id: int,
        clean_fields: dict[str, Any],
        new_tags: Any = None,
        new_scenes: Any = None,
        from_analysis: bool = False,
    ) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute("SELECT * FROM emoji_pending WHERE id=?", (pending_id,)).fetchone()
            if previous is None or (not from_analysis and previous['review_status'] in ('queued', 'processing')):
                return None
            if not from_analysis:
                manual = set(json.loads(previous['manual_fields']))
                manual.update(key for key, value in clean_fields.items()
                              if (previous[key] or '') != (value or ''))
                if new_tags is not None:
                    old_tags = [row['tag'] for row in conn.execute('SELECT tag FROM emoji_pending_tag WHERE path=?', (previous['path'],))]
                    if self._normalize_multi_value(new_tags) != old_tags:
                        manual.add('tags')
                if new_scenes is not None:
                    old_scenes = [row['scene'] for row in conn.execute('SELECT scene FROM emoji_pending_scene WHERE path=?', (previous['path'],))]
                    if self._normalize_multi_value(new_scenes) != old_scenes:
                        manual.add('scenes')
                conn.execute("UPDATE emoji_pending SET manual_fields=? WHERE id=?", (json.dumps(sorted(manual)), pending_id))
            else:
                conn.execute("UPDATE emoji_pending SET analysis_state='ready' WHERE id=?", (pending_id,))
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

            conn.execute('COMMIT')

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
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute(
                "SELECT path, hash FROM emoji_pending WHERE id = ? AND review_status != 'processing'", (pending_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute("DELETE FROM emoji_pending WHERE id = ?", (pending_id,))
            conn.execute('COMMIT')
            return {"path": row["path"], "hash": row["hash"]}

    def delete_pending_batch(self, ids: list[int]) -> list[dict[str, Any]]:
        """批量删除待审核记录，返回每条被删行的 {path, hash}。"""
        clean_ids = [i for i in ids if isinstance(i, int)]
        if not clean_ids:
            return []
        with self._get_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            placeholders = ",".join("?" * len(clean_ids))
            rows = conn.execute(
                f"SELECT id, path, hash FROM emoji_pending WHERE id IN ({placeholders}) AND review_status != 'processing'",
                clean_ids,
            ).fetchall()
            conn.execute(
                f"DELETE FROM emoji_pending WHERE id IN ({placeholders}) AND review_status != 'processing'", clean_ids
            )
            conn.execute('COMMIT')
            return [{"path": r["path"], "hash": r["hash"]} for r in rows]
