"""SQLite 数据库服务，用于存储表情包索引。

替代原有的 JSON 文件存储，提供：
- 增量更新（单行 UPDATE，而非全量重写）
- 索引查询（快速搜索）
- 事务支持（并发安全）
- 低内存占用（无需全量驻留）
"""

import asyncio
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from astrbot.api import logger

from ..util.normalization import canonicalize_path, normalize_label_list
from ..processing.semantic_schema import SEARCH_METADATA_FIELDS


from .schema import DatabaseSchema
from .source_repository import SourceRepository
from .library_queries import LibraryQueries
from .pending_repository import PendingRepository
from .embedding_repository import EmbeddingRepository


class DatabaseService(
    DatabaseSchema,
    SourceRepository,
    LibraryQueries,
    PendingRepository,
    EmbeddingRepository,
):
    """SQLite 数据库服务，管理表情包索引存储。"""

    _RELATED_FETCH_CHUNK_SIZE = 400

    # 表结构版本，用于迁移检测
    # Keep the historical schema version stable for existing integrations.
    # External-source tables are tracked independently through
    # ``external_schema_version`` so v2 databases migrate without changing the
    # value older tooling expects.
    SCHEMA_VERSION = 6
    EXTERNAL_SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path | None = None):
        """初始化数据库服务。

        Args:
            db_path: 数据库文件路径，默认为插件数据目录下的 emoji.db
        """
        if db_path is None:
            from astrbot.api.star import StarTools

            db_path = Path(StarTools.get_data_dir("astrbot_plugin_stealer")) / "emoji.db"

        self._db_path = Path(db_path)
        self._ensure_db_dir()

        # 初始化数据库表结构
        self._init_schema()

        # 异步锁，保护并发写入
        self._write_lock = asyncio.Lock()

    def _ensure_db_dir(self) -> None:
        """确保数据库目录存在。"""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建数据库目录失败: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """获取数据库连接的上下文管理器。

        使用 WAL 模式支持并发读写。
        """
        conn = sqlite3.connect(
            self._db_path,
            timeout=30.0,
            isolation_level=None,  # 自动提交模式，配合 WAL
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        conn.execute("PRAGMA foreign_keys=ON")  # 启用外键约束，支持 CASCADE
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _dump_emotions_json(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.startswith("["):
                return text
            items = normalize_label_list(text, allow_duplicates=True, csv_only=True)
            return json.dumps(items, ensure_ascii=False) if items else None
        if isinstance(value, list):
            items = normalize_label_list(value, allow_duplicates=True)
            return json.dumps(items, ensure_ascii=False) if items else None
        return None

    @staticmethod
    def _load_emotions_json(value: Any) -> list[str]:
        if isinstance(value, list):
            return normalize_label_list(value, allow_duplicates=True)
        text = str(value or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return normalize_label_list(text, allow_duplicates=True, csv_only=True)
        if isinstance(parsed, list):
            return normalize_label_list(parsed, allow_duplicates=True)
        return []

    def _emotions_json_from_meta(self, meta: dict[str, Any]) -> str | None:
        if meta.get("emotions_json"):
            return self._dump_emotions_json(meta.get("emotions_json"))
        if "emotions" in meta:
            return self._dump_emotions_json(meta.get("emotions"))
        return None

    def _hydrate_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry["emotions"] = self._load_emotions_json(entry.get("emotions_json"))
        if not entry.get("overlay_text"):
            entry["overlay_text"] = ""
        entry["character"] = str(entry.get("character") or "").strip()
        return entry

    def get_character_counts(self, *, exclude_favorites: bool = False) -> dict[str, int]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(character, '') as character, COUNT(*) as cnt
                FROM emoji WHERE (? = 0 OR COALESCE(is_favorite, 0) = 0)
                GROUP BY COALESCE(character, '')
                """, (int(exclude_favorites),)
            ).fetchall()
            return {str(r["character"] or ""): int(r["cnt"]) for r in rows}

    def clear_character(self, character: str) -> int:
        key = str(character or "").strip()
        if not key:
            return 0
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE emoji SET character = '' WHERE character = ?", (key,)
            )
            conn.execute(
                "UPDATE emoji_pending SET character = '' WHERE character = ?", (key,)
            )
            return int(cur.rowcount or 0)

    def _row_get(self, row: Any, column: str, default: Any = None) -> Any:
        try:
            keys = row.keys() if hasattr(row, "keys") else []
            if column in keys:
                value = row[column]
                return default if value is None else value
        except Exception:
            return default
        return default

    def _emoji_insert_values(
        self,
        path: str,
        meta: dict[str, Any],
        *,
        now: int | None = None,
        row: Any = None,
        category_override: str | None = None,
    ) -> tuple[Any, ...]:
        created_at = int(now if now is not None else time.time())
        values: list[Any] = []
        for col in self._EMOJI_INSERT_COLUMNS:
            if col == "path":
                values.append(path)
                continue
            if col == "category" and category_override is not None:
                values.append(category_override)
                continue
            if col == "emotions_json":
                dumped = self._emotions_json_from_meta(meta)
                if dumped is None and row is not None:
                    dumped = self._row_get(row, "emotions_json")
                values.append(dumped)
                continue
            if col in meta and meta[col] is not None:
                value = meta[col]
                if col == "is_favorite":
                    value = self._coerce_int_flag(value)
                values.append(value)
                continue
            if row is not None:
                fallback = self._row_get(row, col)
                if fallback is not None:
                    values.append(fallback)
                    continue
            if col == "hash":
                values.append("")
            elif col == "category":
                values.append("unknown")
            elif col == "scope_mode":
                values.append("public")
            elif col == "created_at":
                values.append(created_at)
            elif col == "retention_class":
                values.append("native")
            elif col in {"use_count", "last_used_at", "is_favorite"}:
                values.append(0)
            else:
                values.append(None)
        return tuple(values)

    def clear_all_embeddings(self) -> None:
        """清空文本向量，用于嵌入语料格式升级后重建。"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_embedding")

    def get_meta_value(self, key: str) -> str | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row and row["value"] is not None else None

    def set_meta_value(self, key: str, value: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value),
            )

    @staticmethod
    def _normalize_multi_value(values: Any) -> list[str]:
        if isinstance(values, list):
            return normalize_label_list(values, allow_duplicates=True)
        if values is None:
            return []
        text = str(values).strip()
        return [text] if text else []

    @staticmethod
    def _coerce_int_flag(value: Any) -> int:
        if isinstance(value, str):
            return 1 if value.strip().lower() in {"1", "true", "yes", "on"} else 0
        return 1 if bool(value) else 0

    def _chunk_paths(self, paths: list[str]):
        chunk_size = max(1, int(self._RELATED_FETCH_CHUNK_SIZE))
        for start in range(0, len(paths), chunk_size):
            yield paths[start : start + chunk_size]

    def _load_related_map(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        value_column: str,
        paths: list[str],
    ) -> dict[str, list[str]]:
        related_map: dict[str, list[str]] = {path: [] for path in paths}
        if not paths:
            return related_map

        for chunk in self._chunk_paths(paths):
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"""
                    SELECT path, {value_column} FROM {table}
                    WHERE path IN ({placeholders})
                    ORDER BY rowid
                """,
                chunk,
            ).fetchall()
            for row in rows:
                related_map[row["path"]].append(row[value_column])
        return related_map

    @staticmethod
    def _insert_related_values_sync(
        conn: sqlite3.Connection,
        *,
        table: str,
        value_column: str,
        path: str,
        values: list[str],
    ) -> None:
        for value in values:
            if value:
                conn.execute(
                    f"INSERT OR IGNORE INTO {table} (path, {value_column}) VALUES (?, ?)",
                    (path, value),
                )

    def _replace_related_values_sync(
        self,
        conn: sqlite3.Connection,
        *,
        table: str,
        value_column: str,
        path: str,
        current_values: list[str],
        desired_values: list[str],
    ) -> None:
        desired = [value for value in desired_values if value]
        if desired == (current_values or []):
            return
        conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))
        self._insert_related_values_sync(
            conn,
            table=table,
            value_column=value_column,
            path=path,
            values=desired,
        )

    def _sync_existing_emoji_sync(
        self,
        conn: sqlite3.Connection,
        *,
        path: str,
        current: dict[str, Any],
        desired: dict[str, Any],
    ) -> bool:
        search_metadata_changed = any(
            field in desired
            and (desired[field] or "") != (current.get(field) or "")
            for field in SEARCH_METADATA_FIELDS
        )

        changed_fields = {
            field: desired.get(field)
            for field in self._EMOJI_SCALAR_COLUMNS
            if field not in {"use_count", "last_used_at"}
            and field in desired
            and desired.get(field) != current.get(field)
        }
        if changed_fields:
            clauses = ", ".join(f"{field} = ?" for field in changed_fields)
            values = list(changed_fields.values()) + [path]
            conn.execute(
                f"UPDATE emoji SET {clauses} WHERE path = ?",
                values,
            )

        if "tags" in desired:
            self._replace_related_values_sync(
                conn,
                table="emoji_tag",
                value_column="tag",
                path=path,
                current_values=current.get("tags") or [],
                desired_values=desired.get("tags") or [],
            )
        if "scenes" in desired:
            self._replace_related_values_sync(
                conn,
                table="emoji_scene",
                value_column="scene",
                path=path,
                current_values=current.get("scenes") or [],
                desired_values=desired.get("scenes") or [],
            )
        return search_metadata_changed

    def _build_search_signature_from_index(self, idx: dict[str, dict[str, Any]]) -> str:
        if not idx:
            return "empty"

        hasher = hashlib.sha256()
        for path in sorted(idx.keys()):
            data = idx.get(path)
            if not isinstance(data, dict):
                continue

            category = str(data.get("category", "") or "")
            desc = str(data.get("desc", "") or "")
            overlay = str(data.get("overlay_text", "") or "")
            tags = self._normalize_multi_value(data.get("tags", []))
            scenes = self._normalize_multi_value(data.get("scenes", []))
            emotions = self._normalize_multi_value(data.get("emotions", []))
            payload = "\x1f".join(
                [
                    path,
                    category,
                    desc,
                    overlay,
                    "\x1e".join(tags),
                    "\x1e".join(scenes),
                    "\x1e".join(emotions),
                ]
            )
            hasher.update(payload.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    # ── 基础 CRUD 操作 ──

    def get_emoji(self, path: str) -> dict[str, Any] | None:
        """获取单个表情包的完整信息。"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM emoji WHERE path = ?", (path,)).fetchone()
            if not row:
                return None

            result = dict(row)

            # 获取标签和场景（按插入顺序）
            tags = conn.execute(
                "SELECT tag FROM emoji_tag WHERE path = ? ORDER BY rowid", (path,)
            ).fetchall()
            result["tags"] = [r["tag"] for r in tags]

            scenes = conn.execute(
                "SELECT scene FROM emoji_scene WHERE path = ? ORDER BY rowid", (path,)
            ).fetchall()
            result["scenes"] = [r["scene"] for r in scenes]
            return self._hydrate_entry(result)

    def get_emoji_by_hash(self, hash_val: str) -> tuple[str, dict[str, Any]] | None:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM emoji
                WHERE hash = ?
                ORDER BY created_at ASC, path ASC
                LIMIT 1
                """,
                (hash_val,),
            ).fetchone()
            if not row:
                return None

            path = row["path"]
            result = dict(row)
            tags = conn.execute(
                "SELECT tag FROM emoji_tag WHERE path = ? ORDER BY rowid", (path,)
            ).fetchall()
            result["tags"] = [r["tag"] for r in tags]
            scenes = conn.execute(
                "SELECT scene FROM emoji_scene WHERE path = ? ORDER BY rowid", (path,)
            ).fetchall()
            result["scenes"] = [r["scene"] for r in scenes]
            return path, self._hydrate_entry(result)

    def get_all_paths(self) -> list[str]:
        """获取所有表情包路径。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji").fetchall()
            return [r["path"] for r in rows]

    def get_paths_sorted_newest(self) -> list[str]:
        """按列表指令的稳定顺序返回路径，无需加载全量元数据。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path FROM emoji ORDER BY created_at DESC, path DESC"
            ).fetchall()
            return [row["path"] for row in rows]

    def hash_exists(self, hash_val: str) -> bool:
        """O(1) 哈希查重，不走全量索引加载。"""
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM emoji WHERE hash = ? LIMIT 1", (hash_val,)).fetchone()
            return row is not None

    def get_phash_map(self) -> dict[str, str]:
        """只返回 path→phash 映射，用于感知哈希去重（轻量替代全量索引）。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, phash FROM emoji WHERE phash IS NOT NULL AND phash != ''"
            ).fetchall()
            return {r["path"]: r["phash"] for r in rows}

    def count_total(self) -> int:
        """统计表情包总数。"""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM emoji").fetchone()
            return result["cnt"] if result else 0

    def blacklisted_hashes(self) -> set[str]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT hash FROM blacklist").fetchall()
            return {r["hash"] for r in rows} if rows else set()

    async def add_blacklist(self, hash_val: str, ts: int | None = None) -> None:
        if not hash_val:
            return
        ts_val = ts if ts is not None else int(time.time())
        async with self._write_lock:
            await asyncio.to_thread(self._add_blacklist_sync, hash_val, ts_val)

    def _add_blacklist_sync(self, hash_val: str, ts: int) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blacklist (hash, created_at) VALUES (?, ?)",
                (hash_val, ts),
            )

    async def remove_blacklist(self, hash_val: str) -> bool:
        if not hash_val:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._remove_blacklist_sync, hash_val)

    def _remove_blacklist_sync(self, hash_val: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM blacklist WHERE hash = ?", (hash_val,))
            return cur.rowcount > 0

    async def add_blacklist_batch(self, hashes: dict[str, int]) -> int:
        if not hashes:
            return 0
        async with self._write_lock:
            return await asyncio.to_thread(self._add_blacklist_batch_sync, hashes)

    def _add_blacklist_batch_sync(self, hashes: dict[str, int]) -> int:
        inserted = 0
        with self._get_connection() as conn:
            for hash_val, ts in hashes.items():
                cur = conn.execute(
                    "INSERT OR IGNORE INTO blacklist (hash, created_at) VALUES (?, ?)",
                    (hash_val, int(ts)),
                )
                inserted += cur.rowcount
        return inserted

    def count_favorites(self) -> int:
        """统计收藏表情包总数。"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji WHERE is_favorite = 1"
            ).fetchone()
            return result["cnt"] if result else 0

    def increment_usage_sync(self, path: str) -> bool:
        """增加已入库图片的使用次数，并报告是否命中记录。"""
        if not path:
            return False
        now = int(time.time())
        with self._get_connection() as conn:
            updated = conn.execute(
                "UPDATE emoji SET use_count = use_count + 1, last_used_at = ? WHERE path = ?",
                (now, path),
            )
            if updated.rowcount:
                return True

            # 历史记录在 Windows 上保留反斜杠和大小写；比较键仅用于查找。
            wanted = canonicalize_path(path)
            matches = [
                row["path"]
                for row in conn.execute("SELECT path FROM emoji")
                if canonicalize_path(row["path"]) == wanted
            ]
            if len(matches) != 1:
                return False
            updated = conn.execute(
                "UPDATE emoji SET use_count = use_count + 1, last_used_at = ? WHERE path = ?",
                (now, matches[0]),
            )
            return updated.rowcount == 1

    async def delete_paths(self, paths: list[str]) -> int:
        clean_paths = [p for p in paths if isinstance(p, str) and p]
        if not clean_paths:
            return 0
        async with self._write_lock:
            return await asyncio.to_thread(self._delete_paths_sync, clean_paths)

    def _delete_paths_sync(self, paths: list[str]) -> int:
        deleted = 0
        with self._get_connection() as conn:
            transaction_started = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                transaction_started = True
                for path in paths:
                    cur = conn.execute("DELETE FROM emoji WHERE path = ?", (path,))
                    deleted += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
                conn.execute("COMMIT")
            except Exception:
                if transaction_started and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
        return deleted

    async def update_path(self, path: str, updates: dict[str, Any]) -> bool:
        if not path or not updates:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(self._update_path_sync, path, updates)

    def _update_path_sync(self, path: str, updates: dict[str, Any]) -> bool:
        scalar_fields = self._EMOJI_SCALAR_COLUMNS
        scalar_updates = {
            key: self._coerce_int_flag(value) if key == "is_favorite" else value
            for key, value in updates.items()
            if key in scalar_fields
        }
        if "emotions" in updates and "emotions_json" not in scalar_updates:
            scalar_updates["emotions_json"] = self._dump_emotions_json(updates.get("emotions"))

        with self._get_connection() as conn:
            transaction_started = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                transaction_started = True
                row = conn.execute("SELECT 1 FROM emoji WHERE path = ?", (path,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    return False

                if scalar_updates:
                    clauses = ", ".join(f"{field} = ?" for field in scalar_updates)
                    values = list(scalar_updates.values()) + [path]
                    conn.execute(f"UPDATE emoji SET {clauses} WHERE path = ?", values)

                if "tags" in updates:
                    tags = self._normalize_multi_value(updates.get("tags"))
                    conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                    for tag in tags:
                        conn.execute(
                            "INSERT OR IGNORE INTO emoji_tag (path, tag) VALUES (?, ?)",
                            (path, tag),
                        )

                if "scenes" in updates:
                    scenes = self._normalize_multi_value(updates.get("scenes"))
                    conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))
                    for scene in scenes:
                        conn.execute(
                            "INSERT OR IGNORE INTO emoji_scene (path, scene) VALUES (?, ?)",
                            (path, scene),
                        )

                conn.execute("COMMIT")
                return True
            except Exception:
                if transaction_started and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    async def move_path(
        self,
        old_path: str,
        new_path: str,
        category: str,
        updates: dict[str, Any] | None = None,
    ) -> bool:
        if not old_path or not new_path or old_path == new_path:
            return False
        async with self._write_lock:
            return await asyncio.to_thread(
                self._move_path_sync, old_path, new_path, category, updates or {}
            )

    def _move_path_sync(
        self, old_path: str, new_path: str, category: str, updates: dict[str, Any]
    ) -> bool:
        with self._get_connection() as conn:
            transaction_started = False
            try:
                conn.execute("BEGIN IMMEDIATE")
                transaction_started = True
                row = conn.execute("SELECT * FROM emoji WHERE path = ?", (old_path,)).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    return False

                existing = conn.execute("SELECT 1 FROM emoji WHERE path = ?", (new_path,)).fetchone()
                if existing:
                    conn.execute("ROLLBACK")
                    return False

                scalar = {
                    key: self._coerce_int_flag(value) if key == "is_favorite" else value
                    for key, value in updates.items()
                    if key in self._EMOJI_SCALAR_COLUMNS
                }

                conn.execute(
                    self._INSERT_EMOJI_SQL,
                    self._emoji_insert_values(
                        new_path,
                        scalar,
                        row=row,
                        category_override=category,
                    ),
                )
                if "tags" in updates:
                    for tag in self._normalize_multi_value(updates.get("tags")):
                        conn.execute(
                            "INSERT OR IGNORE INTO emoji_tag (path, tag) VALUES (?, ?)",
                            (new_path, tag),
                        )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO emoji_tag (path, tag)
                        SELECT ?, tag FROM emoji_tag WHERE path = ?
                        """,
                        (new_path, old_path),
                    )

                if "scenes" in updates:
                    for scene in self._normalize_multi_value(updates.get("scenes")):
                        conn.execute(
                            "INSERT OR IGNORE INTO emoji_scene (path, scene) VALUES (?, ?)",
                            (new_path, scene),
                        )
                else:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO emoji_scene (path, scene)
                        SELECT ?, scene FROM emoji_scene WHERE path = ?
                        """,
                        (new_path, old_path),
                    )
                # Preserve external-source provenance when a user moves an
                # imported image between categories.
                conn.execute(
                    "UPDATE meme_source_item SET path = ? WHERE path = ?",
                    (new_path, old_path),
                )
                conn.execute("DELETE FROM emoji WHERE path = ?", (old_path,))
                conn.execute("COMMIT")
                return True
            except Exception:
                if transaction_started and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    # ── 批量操作 ──

    async def insert_batch(self, emojis: list[dict[str, Any]]) -> int:
        """批量插入表情包记录。

        Args:
            emojis: 表情包数据列表，每个元素包含 path, hash, category 等字段

        Returns:
            int: 成功插入的数量
        """
        if not emojis:
            return 0

        async with self._write_lock:
            try:
                count = await asyncio.to_thread(self._insert_batch_sync, emojis)
                return count
            except Exception as e:
                logger.error(f"[DB] 批量插入失败: {e}")
                return 0

    def _insert_batch_sync(self, emojis: list[dict[str, Any]]) -> int:
        """同步批量插入。"""
        now = int(time.time())
        count = 0

        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")

            try:
                for emoji in emojis:
                    path = emoji.get("path")
                    if not path:
                        continue

                    conn.execute(
                        self._INSERT_EMOJI_SQL,
                        self._emoji_insert_values(path, emoji, now=now),
                    )

                    # 删除旧标签/场景
                    conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                    conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))

                    # 插入标签
                    for tag in emoji.get("tags") or []:
                        if tag:
                            conn.execute(
                                "INSERT OR IGNORE INTO emoji_tag (path, tag) VALUES (?, ?)", (path, tag)
                            )

                    # 插入场景
                    for scene in emoji.get("scenes") or []:
                        if scene:
                            conn.execute(
                                "INSERT OR IGNORE INTO emoji_scene (path, scene) VALUES (?, ?)", (path, scene)
                            )

                    count += 1

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"[DB] 批量插入事务回滚: {e}")
                raise

        return count

    # ── 兼容旧接口 ──

    def get_index_cache_readonly(self) -> dict[str, Any]:
        """获取完整索引（兼容旧接口，用于迁移过渡）。

        注意：此方法返回全量数据，仅用于兼容过渡，
        新代码应使用具体的搜索方法。
        """
        result: dict[str, Any] = {}

        with self._get_connection() as conn:
            # 单次查询获取所有表情包
            rows = conn.execute("SELECT * FROM emoji").fetchall()
            if not rows:
                return result

            paths = [r["path"] for r in rows]
            tags_map = self._load_related_map(
                conn, table="emoji_tag", value_column="tag", paths=paths
            )
            scenes_map = self._load_related_map(
                conn, table="emoji_scene", value_column="scene", paths=paths
            )

            # 构建结果
            for row in rows:
                path = row["path"]
                entry = dict(row)
                entry["tags"] = tags_map.get(path, [])
                entry["scenes"] = scenes_map.get(path, [])
                result[path] = self._hydrate_entry(entry)

        return result

    async def save_index(self, idx: dict[str, Any]) -> None:
        """保存索引（兼容旧接口，用于迁移过渡）。

        注意：此方法会清空数据库并重新插入，仅用于兼容过渡，
        新代码应使用 insert_emoji / insert_batch。
        """
        emojis = []
        for path, meta in idx.items():
            if isinstance(meta, dict):
                emoji = {"path": path, **meta}
                emojis.append(emoji)

        await self.clear_all()
        await self.insert_batch(emojis)

    async def sync_index(self, idx: dict[str, Any]) -> list[str]:
        """增量同步索引到数据库（仅插入/更新，不删除）。

        删除已移交 MaintenanceService 通过孤儿扫描处理，
        避免并发 on_message 时误删其他消息刚入库的条目。
        """
        async with self._write_lock:
            return await asyncio.to_thread(self._sync_index_sync, idx)

    def _sync_index_sync(self, idx: dict[str, Any]) -> list[str]:
        desired_index = {
            path: dict(meta)
            for path, meta in idx.items()
            if isinstance(path, str) and isinstance(meta, dict)
        }

        for meta in desired_index.values():
            if "emotions" in meta:
                meta["emotions_json"] = self._dump_emotions_json(meta.pop("emotions"))
            for field in ("tags", "scenes"):
                if field in meta:
                    meta[field] = self._normalize_multi_value(meta[field])

        with self._get_connection() as conn:
            transaction_started = False

            try:
                conn.execute("BEGIN IMMEDIATE")
                transaction_started = True
                current_rows = conn.execute("SELECT * FROM emoji").fetchall()
                current_index: dict[str, dict[str, Any]] = {
                    row["path"]: dict(row) for row in current_rows
                }
                current_paths = list(current_index.keys())

                if current_paths:
                    tags_map = self._load_related_map(
                        conn,
                        table="emoji_tag",
                        value_column="tag",
                        paths=current_paths,
                    )
                    scenes_map = self._load_related_map(
                        conn,
                        table="emoji_scene",
                        value_column="scene",
                        paths=current_paths,
                    )

                    for path in current_paths:
                        current_index[path]["tags"] = tags_map.get(path, [])
                        current_index[path]["scenes"] = scenes_map.get(path, [])

                desired_paths = set(desired_index.keys())
                existing_paths = set(current_index.keys())
                changed_paths = set(desired_paths - existing_paths)

                for path in desired_paths - existing_paths:
                    meta = desired_index[path]
                    now = int(time.time())
                    conn.execute(
                        self._INSERT_EMOJI_SQL,
                        self._emoji_insert_values(path, meta, now=now),
                    )
                    self._insert_related_values_sync(
                        conn,
                        table="emoji_tag",
                        value_column="tag",
                        path=path,
                        values=meta.get("tags") or [],
                    )
                    self._insert_related_values_sync(
                        conn,
                        table="emoji_scene",
                        value_column="scene",
                        path=path,
                        values=meta.get("scenes") or [],
                    )

                for path in desired_paths & existing_paths:
                    meta = desired_index[path]
                    current = current_index[path]
                    if self._sync_existing_emoji_sync(
                        conn,
                        path=path,
                        current=current,
                        desired=meta,
                    ):
                        changed_paths.add(path)

                conn.execute("COMMIT")
                return sorted(changed_paths)
            except Exception:
                if transaction_started and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise

    async def clear_all(self) -> None:
        """清空所有数据。"""
        async with self._write_lock:
            try:
                await asyncio.to_thread(self._clear_all_sync)
            except Exception as e:
                logger.error(f"[DB] 清空数据失败: {e}")

    def _clear_all_sync(self) -> None:
        """同步清空所有数据。"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_tag")
            conn.execute("DELETE FROM emoji_scene")
            conn.execute("DELETE FROM emoji")

    # ── 迁移工具 ──

    async def migrate_from_json(self, json_path: Path) -> int:
        """从旧版 JSON 文件迁移数据。

        Args:
            json_path: 旧版 index_cache.json 文件路径

        Returns:
            int: 成功迁移的数量
        """
        if not json_path.exists():
            logger.info(f"[DB] 无需迁移，JSON 文件不存在: {json_path}")
            return 0

        try:
            with open(json_path, encoding="utf-8") as f:
                old_data = json.load(f)

            if not isinstance(old_data, dict) or not old_data:
                logger.info(f"[DB] JSON 文件无有效数据: {json_path}")
                return 0

            # 转换格式
            emojis = []
            for path, meta in old_data.items():
                if isinstance(meta, dict):
                    emoji = {"path": path, **meta}
                    emojis.append(emoji)

            count = await self.insert_batch(emojis)
            logger.info(f"[DB] 从 JSON 迁移了 {count} 条记录")

            # 备份旧文件
            backup_path = json_path.with_suffix(".json.migrated")
            json_path.rename(backup_path)
            logger.info(f"[DB] 旧 JSON 文件已备份到: {backup_path}")

            return count

        except Exception as e:
            logger.error(f"[DB] 迁移 JSON 失败: {e}", exc_info=True)
            return 0

    # ── 统计与调试 ──

    def get_stats(self) -> dict[str, Any]:
        """获取数据库统计信息。"""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) as cnt FROM emoji").fetchone()["cnt"]
            categories = conn.execute(
                "SELECT category, COUNT(*) as cnt FROM emoji GROUP BY category"
            ).fetchall()
            tags_count = conn.execute("SELECT COUNT(*) as cnt FROM emoji_tag").fetchone()["cnt"]
            scenes_count = conn.execute("SELECT COUNT(*) as cnt FROM emoji_scene").fetchone()["cnt"]
            pending_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji_pending"
            ).fetchone()["cnt"]
            external_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji WHERE retention_class = 'external'"
            ).fetchone()["cnt"]

            return {
                "total_emojis": total,
                "total_tags": tags_count,
                "total_scenes": scenes_count,
                "pending_count": pending_count,
                "external_count": external_count,
                "categories": {r["category"]: r["cnt"] for r in categories},
                "db_size_bytes": self._db_path.stat().st_size if self._db_path.exists() else 0,
            }

    def get_tag_stats(self, top_n: int = 15) -> dict[str, Any]:
        """标签/场景统计（供 /meme tag_stats 命令与 WebUI 使用）。

        Returns:
            dict: {
                top_tags: [{tag, count}]          按使用次数降序
                single_use_tags: [tag]            仅出现 1 次的标签（疑似噪声/拼写差异）
                zero_tag_count: int               无任何标签的表情数量
                total_emojis: int                 表情总数
                total_with_tags: int              有标签的表情数
                top_scenes: [{scene, count}]      场景统计（同标签口径）
            }
        """
        with self._get_connection() as conn:
            top_tags = conn.execute(
                "SELECT tag, COUNT(*) as cnt FROM emoji_tag "
                "GROUP BY tag ORDER BY cnt DESC, tag ASC LIMIT ?",
                (int(top_n),),
            ).fetchall()
            single = conn.execute(
                "SELECT tag FROM emoji_tag GROUP BY tag HAVING COUNT(*) = 1 ORDER BY tag LIMIT 20"
            ).fetchall()
            zero = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji e WHERE NOT EXISTS "
                "(SELECT 1 FROM emoji_tag t WHERE t.path = e.path)"
            ).fetchone()
            total = conn.execute("SELECT COUNT(*) as cnt FROM emoji").fetchone()
            top_scenes = conn.execute(
                "SELECT scene, COUNT(*) as cnt FROM emoji_scene "
                "GROUP BY scene ORDER BY cnt DESC, scene ASC LIMIT ?",
                (int(top_n),),
            ).fetchall()
            return {
                "top_tags": [{"tag": r["tag"], "count": r["cnt"]} for r in top_tags],
                "single_use_tags": [r["tag"] for r in single],
                "zero_tag_count": int(zero["cnt"] if zero else 0),
                "total_emojis": int(total["cnt"] if total else 0),
                "total_with_tags": int(total["cnt"] if total else 0)
                - int(zero["cnt"] if zero else 0),
                "top_scenes": [{"scene": r["scene"], "count": r["cnt"]} for r in top_scenes],
            }

    def count_created_since(self, created_at: int | float) -> int:
        """Count emojis whose created_at is at or after the given timestamp."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji WHERE created_at >= ?",
                (int(created_at),),
            ).fetchone()
            return int(row["cnt"] if row else 0)

    def get_corpus_signature(self) -> str:
        """获取语料库签名，用于 BM25 索引变更检测。"""
        return self._build_search_signature_from_index(self.get_index_cache_readonly())
