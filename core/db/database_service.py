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


class DatabaseService:
    """SQLite 数据库服务，管理表情包索引存储。"""

    _RELATED_FETCH_CHUNK_SIZE = 400

    # 表结构版本，用于迁移检测
    SCHEMA_VERSION = 4

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

    def _init_schema(self) -> None:
        """初始化数据库表结构。"""
        with self._get_connection() as conn:
            # 创建元数据表（存储版本等信息）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # 检查 schema 版本
            result = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            current_version = int(result["value"] if result else 0)

            if current_version < self.SCHEMA_VERSION:
                logger.info(f"[DB] 升级数据库 schema: {current_version} -> {self.SCHEMA_VERSION}")
                self._create_tables(conn)

                # 版本 2 迁移：添加 is_favorite 字段
                if current_version < 2:
                    try:
                        conn.execute("ALTER TABLE emoji ADD COLUMN is_favorite INTEGER DEFAULT 0")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_favorite ON emoji(is_favorite)")
                        logger.info("[DB] 迁移完成: 添加 is_favorite 字段")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e).lower():
                            logger.info("[DB] is_favorite 字段已存在，跳过")
                        else:
                            raise

                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),),
                )

                # v3: blacklist table created by _create_tables
                if current_version < 3:
                    logger.info("[DB] migration: blacklist table ready")

                # v4: 待审核池 emoji_pending / 嵌入向量 emoji_embedding 表
                # （均由 _create_tables 用 IF NOT EXISTS 创建，此处仅记录）
                if current_version < 4:
                    logger.info("[DB] migration: emoji_pending / emoji_embedding tables ready")

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """创建所有数据表。"""
        # 主表：表情包元数据
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                phash TEXT,
                category TEXT NOT NULL,
                desc TEXT,
                source TEXT,
                origin_target TEXT,
                scope_mode TEXT DEFAULT 'public',
                created_at INTEGER DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                last_used_at INTEGER DEFAULT 0,
                is_favorite INTEGER DEFAULT 0
            )
        """)

        # 标签表：一对多关系
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_tag (
                path TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (path, tag),
                FOREIGN KEY (path) REFERENCES emoji(path) ON DELETE CASCADE
            )
        """)

        # 场景表：一对多关系
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_scene (
                path TEXT NOT NULL,
                scene TEXT NOT NULL,
                PRIMARY KEY (path, scene),
                FOREIGN KEY (path) REFERENCES emoji(path) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS blacklist (
                hash TEXT PRIMARY KEY,
                created_at INTEGER DEFAULT 0
            )
        """)

        # 待审核池：on_message 自动偷取先进 pending，人工审核通过后入库
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                hash TEXT NOT NULL,
                phash TEXT,
                category TEXT,
                desc TEXT,
                source TEXT,
                origin_target TEXT,
                scope_mode TEXT DEFAULT 'public',
                review_status TEXT DEFAULT 'pending',
                created_at INTEGER DEFAULT 0,
                tags_text TEXT,
                scenes_text TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_created ON emoji_pending(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_category ON emoji_pending(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_hash ON emoji_pending(hash)")

        # 嵌入向量：审核通过入库时计算，检索阶段优先用向量召回，缺失则降级 BM25
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_embedding (
                path TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model_sig TEXT NOT NULL,
                updated_at INTEGER DEFAULT 0,
                FOREIGN KEY (path) REFERENCES emoji(path) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_model ON emoji_embedding(model_sig)")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_category ON emoji(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_hash ON emoji(hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_last_used ON emoji(last_used_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_tag ON emoji_tag(tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scene_scene ON emoji_scene(scene)")

    @staticmethod
    def _normalize_multi_value(values: Any) -> list[str]:
        if isinstance(values, list):
            return [str(v).strip() for v in values if str(v).strip()]
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
            tags = self._normalize_multi_value(data.get("tags", []))
            scenes = self._normalize_multi_value(data.get("scenes", []))
            payload = "\x1f".join([path, category, desc, "\x1e".join(tags), "\x1e".join(scenes)])
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

            return result

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
            return path, result

    def get_all_paths(self) -> list[str]:
        """获取所有表情包路径。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji").fetchall()
            return [r["path"] for r in rows]

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

    def increment_usage_sync(self, path: str) -> None:
        """同步增加使用次数。"""
        now = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE emoji SET use_count = use_count + 1, last_used_at = ? WHERE path = ?",
                (now, path),
            )

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
        scalar_fields = {
            "hash",
            "phash",
            "category",
            "desc",
            "source",
            "origin_target",
            "scope_mode",
            "created_at",
            "use_count",
            "last_used_at",
            "is_favorite",
        }
        scalar_updates = {
            key: self._coerce_int_flag(value) if key == "is_favorite" else value
            for key, value in updates.items()
            if key in scalar_fields
        }

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
                    if key
                    in {
                        "hash",
                        "phash",
                        "desc",
                        "source",
                        "origin_target",
                        "scope_mode",
                        "created_at",
                        "use_count",
                        "last_used_at",
                        "is_favorite",
                    }
                }

                conn.execute(
                    """
                    INSERT INTO emoji
                    (path, hash, phash, category, desc, source, origin_target,
                     scope_mode, created_at, use_count, last_used_at, is_favorite)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_path,
                        scalar.get("hash", row["hash"]),
                        scalar.get("phash", row["phash"]),
                        category,
                        scalar.get("desc", row["desc"]),
                        scalar.get("source", row["source"]),
                        scalar.get("origin_target", row["origin_target"]),
                        scalar.get("scope_mode", row["scope_mode"]),
                        scalar.get("created_at", row["created_at"]),
                        scalar.get("use_count", row["use_count"]),
                        scalar.get("last_used_at", row["last_used_at"]),
                        scalar.get("is_favorite", row["is_favorite"]),
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

                    # 插入主记录
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO emoji
                        (path, hash, phash, category, desc, source, origin_target,
                         scope_mode, created_at, use_count, last_used_at, is_favorite)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            path,
                            emoji.get("hash", ""),
                            emoji.get("phash"),
                            emoji.get("category", "unknown"),
                            emoji.get("desc"),
                            emoji.get("source"),
                            emoji.get("origin_target"),
                            emoji.get("scope_mode", "public"),
                            emoji.get("created_at", now),
                            emoji.get("use_count", 0),
                            emoji.get("last_used_at", 0),
                            int(bool(emoji.get("is_favorite", 0))),
                        ),
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
                result[path] = entry

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

    async def sync_index(self, idx: dict[str, Any]) -> None:
        """Synchronize the database to match a full index snapshot."""
        async with self._write_lock:
            await asyncio.to_thread(self._sync_index_sync, idx)

    def _sync_index_sync(self, idx: dict[str, Any]) -> None:
        desired_index = {
            path: meta
            for path, meta in idx.items()
            if isinstance(path, str) and isinstance(meta, dict)
        }

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

                for stale_path in existing_paths - desired_paths:
                    conn.execute("DELETE FROM emoji WHERE path = ?", (stale_path,))

                for path in desired_paths - existing_paths:
                    meta = desired_index[path]
                    now = int(time.time())
                    conn.execute(
                        """
                            INSERT INTO emoji
                            (path, hash, phash, category, desc, source, origin_target,
                             scope_mode, created_at, use_count, last_used_at, is_favorite)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            path,
                            meta.get("hash", ""),
                            meta.get("phash"),
                            meta.get("category", "unknown"),
                            meta.get("desc"),
                            meta.get("source"),
                            meta.get("origin_target"),
                            meta.get("scope_mode", "public"),
                            meta.get("created_at", now),
                            meta.get("use_count", 0),
                            meta.get("last_used_at", 0),
                            int(bool(meta.get("is_favorite", 0))),
                        ),
                    )

                    for tag in meta.get("tags") or []:
                        if tag:
                            conn.execute(
                                "INSERT OR IGNORE INTO emoji_tag (path, tag) VALUES (?, ?)",
                                (path, tag),
                            )

                    for scene in meta.get("scenes") or []:
                        if scene:
                            conn.execute(
                                "INSERT OR IGNORE INTO emoji_scene (path, scene) VALUES (?, ?)",
                                (path, scene),
                            )

                scalar_fields = (
                    "hash",
                    "phash",
                    "category",
                    "desc",
                    "source",
                    "origin_target",
                    "scope_mode",
                    "created_at",
                    "use_count",
                    "last_used_at",
                    "is_favorite",
                )

                for path in desired_paths & existing_paths:
                    meta = desired_index[path]
                    current = current_index[path]

                    changed_fields: dict[str, Any] = {}
                    for field in scalar_fields:
                        if field not in meta:
                            continue
                        if meta.get(field) != current.get(field):
                            changed_fields[field] = meta.get(field)

                    if changed_fields:
                        clauses = ", ".join(f"{field} = ?" for field in changed_fields)
                        values = list(changed_fields.values()) + [path]
                        conn.execute(
                            f"UPDATE emoji SET {clauses} WHERE path = ?",
                            values,
                        )

                    if "tags" in meta:
                        desired_tags = [tag for tag in (meta.get("tags") or []) if tag]
                        if desired_tags != (current.get("tags") or []):
                            conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                            for tag in desired_tags:
                                conn.execute(
                                    "INSERT OR IGNORE INTO emoji_tag (path, tag) VALUES (?, ?)",
                                    (path, tag),
                                )

                    if "scenes" in meta:
                        desired_scenes = [scene for scene in (meta.get("scenes") or []) if scene]
                        if desired_scenes != (current.get("scenes") or []):
                            conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))
                            for scene in desired_scenes:
                                conn.execute(
                                    "INSERT OR IGNORE INTO emoji_scene (path, scene) VALUES (?, ?)",
                                    (path, scene),
                                )

                conn.execute("COMMIT")
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

            return {
                "total_emojis": total,
                "total_tags": tags_count,
                "total_scenes": scenes_count,
                "categories": {r["category"]: r["cnt"] for r in categories},
                "db_size_bytes": self._db_path.stat().st_size if self._db_path.exists() else 0,
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

    # ── 分页查询 ──

    # 排序字段白名单（防止SQL注入）
    _VALID_ORDER_FIELDS = {
        "newest": "e.created_at DESC, e.path DESC",
        "oldest": "e.created_at ASC, e.path ASC",
        "most_used": "e.use_count DESC, e.last_used_at DESC, e.path ASC",
        "last_used": "e.last_used_at DESC, e.use_count DESC, e.path ASC",
    }

    def get_emojis_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        category: str | None = None,
        sort_order: str = "newest",
        search_query: str | None = None,
        scope_target: str | None = None,
        favorite_only: bool = False,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        """分页获取表情包列表，支持过滤、搜索和排序。

        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            category: 分类过滤（可选）
            sort_order: 排序方式 - "newest"(最新), "oldest"(最旧), "most_used"(最常用)
            search_query: 搜索关键词（匹配标签、描述、场景）
            scope_target: scope 过滤目标（可选）
            favorite_only: 仅显示收藏的表情包

        Returns:
            tuple: (图片列表, 总数, 分类统计)
        """
        # 白名单验证排序字段（防止SQL注入）
        order_sql = self._VALID_ORDER_FIELDS.get(sort_order, "e.created_at DESC")

        with self._get_connection() as conn:
            # 构建基础查询条件
            where_clauses: list[str] = []
            params: list[Any] = []
            category_count_where_clauses: list[str] = []
            category_count_params: list[Any] = []

            # 分类过滤
            if category:
                where_clauses.append("e.category = ?")
                params.append(category)

            # 收藏过滤
            if favorite_only:
                where_clauses.append("e.is_favorite = 1")
                category_count_where_clauses.append("e.is_favorite = 1")

            # scope 过滤
            if scope_target:
                where_clauses.append("(e.scope_mode = 'public' OR e.origin_target = ?)")
                params.append(scope_target)
                category_count_where_clauses.append(
                    "(e.scope_mode = 'public' OR e.origin_target = ?)"
                )
                category_count_params.append(scope_target)

            # 搜索过滤（描述/标签/场景/分类/hash/来源/文件名/路径）
            if search_query:
                search_pattern = f"%{search_query}%"
                search_clause = (
                    "(e.desc LIKE ? OR e.category LIKE ? OR e.hash LIKE ?"
                    " OR e.origin_target LIKE ? OR e.source LIKE ? OR e.path LIKE ?"
                    " OR EXISTS("
                    "SELECT 1 FROM emoji_tag t WHERE t.path = e.path AND t.tag LIKE ?"
                    ") OR EXISTS("
                    "SELECT 1 FROM emoji_scene s WHERE s.path = e.path AND s.scene LIKE ?"
                    "))"
                )
                where_clauses.append(search_clause)
                params.extend([search_pattern] * 8)
                category_count_where_clauses.append(search_clause)
                category_count_params.extend([search_pattern] * 8)

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            category_count_where_sql = ""
            if category_count_where_clauses:
                category_count_where_sql = "WHERE " + " AND ".join(category_count_where_clauses)

            # 计算总数
            count_sql = f"SELECT COUNT(*) as cnt FROM emoji e {where_sql}"
            total = conn.execute(count_sql, params).fetchone()["cnt"]

            # 分类统计（用于侧边栏显示）
            cat_count_sql = f"""
                SELECT e.category, COUNT(*) as cnt
                FROM emoji e {category_count_where_sql}
                GROUP BY e.category
            """
            cat_rows = conn.execute(cat_count_sql, category_count_params).fetchall()
            category_counts = {r["category"]: r["cnt"] for r in cat_rows}

            # 分页查询
            offset = (page - 1) * page_size
            limit = page_size

            data_sql = f"""
                SELECT e.path, e.hash, e.category, e.desc, e.scope_mode,
                       e.origin_target, e.created_at, e.use_count, e.last_used_at,
                       e.is_favorite
                FROM emoji e {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(data_sql, params + [limit, offset]).fetchall()

            if not rows:
                return [], total, category_counts

            # 批量获取标签和场景（解决N+1问题）
            paths = [r["path"] for r in rows]
            tags_map = self._load_related_map(
                conn, table="emoji_tag", value_column="tag", paths=paths
            )
            scenes_map = self._load_related_map(
                conn, table="emoji_scene", value_column="scene", paths=paths
            )

            # 构建结果列表
            images: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["tags"] = tags_map.get(row["path"], [])
                item["scenes"] = scenes_map.get(row["path"], [])
                images.append(item)

            return images, total, category_counts

    # ── 待审核池 (emoji_pending) CRUD ──

    @staticmethod
    def _join_multi(values: Any) -> str:
        return ",".join(str(v).strip() for v in DatabaseService._normalize_multi_value(values))

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
                    """
                    INSERT INTO emoji_pending
                    (path, hash, phash, category, desc, source, origin_target,
                     scope_mode, review_status, created_at, tags_text, scenes_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
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
                        created_at,
                        self._join_multi(meta.get("tags")),
                        self._join_multi(meta.get("scenes")),
                    ),
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
                "(p.desc LIKE ? OR p.tags_text LIKE ? OR p.scenes_text LIKE ?"
                " OR p.category LIKE ? OR p.hash LIKE ?"
                " OR p.origin_target LIKE ? OR p.source LIKE ? OR p.path LIKE ?)"
            )
            params.extend([search_pattern] * 8)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._get_connection() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) as cnt FROM emoji_pending p {where_sql}", params
            ).fetchone()["cnt"]

            cat_rows = conn.execute(
                f"SELECT p.category, COUNT(*) as cnt FROM emoji_pending p {where_sql} "
                "GROUP BY p.category",
                params,
            ).fetchall()
            category_counts = {r["category"]: r["cnt"] for r in cat_rows}

            offset = max(0, (page - 1)) * page_size
            rows = conn.execute(
                f"""
                SELECT p.id, p.path, p.hash, p.phash, p.category, p.desc,
                       p.source, p.origin_target, p.scope_mode, p.review_status,
                       p.created_at, p.tags_text, p.scenes_text
                FROM emoji_pending p {where_sql}
                ORDER BY p.created_at DESC, p.id DESC
                LIMIT ? OFFSET ?
                """,
                params + [page_size, offset],
            ).fetchall()

            items: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["tags"] = self._split_multi(item.pop("tags_text", ""))
                item["scenes"] = self._split_multi(item.pop("scenes_text", ""))
                items.append(item)
            return items, total, category_counts

    def get_pending(self, pending_id: int) -> dict[str, Any] | None:
        """获取单条待审核记录（含拆分后的 tags/scenes）。"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM emoji_pending WHERE id = ?", (pending_id,)
            ).fetchone()
            if not row:
                return None
            item = dict(row)
            item["tags"] = self._split_multi(item.pop("tags_text", ""))
            item["scenes"] = self._split_multi(item.pop("scenes_text", ""))
            return item

    def get_pending_by_hash(self, hash_val: str) -> dict[str, Any] | None:
        """按内容哈希查一条待审核记录（用于缩略图回退）。返回含 path 字段。"""
        if not hash_val:
            return None
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT path, hash, phash, category FROM emoji_pending WHERE hash = ? LIMIT 1",
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

        # tags/scenes 需要 join_multi
        if "tags" in clean_fields:
            clean_fields["tags_text"] = self._join_multi(clean_fields.pop("tags"))
        if "scenes" in clean_fields:
            clean_fields["scenes_text"] = self._join_multi(clean_fields.pop("scenes"))

        # scope_mode 兜底
        if "scope_mode" in clean_fields:
            sm = str(clean_fields["scope_mode"] or "").strip().lower()
            clean_fields["scope_mode"] = sm if sm in ("public", "local") else "public"

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
                self._update_pending_sync, pending_id, clean_fields
            )

    def _update_pending_sync(
        self, pending_id: int, clean_fields: dict[str, Any]
    ) -> dict[str, Any] | None:
        set_clause = ", ".join(f"{col} = ?" for col in clean_fields.keys())
        params: list[Any] = list(clean_fields.values()) + [pending_id]
        with self._get_connection() as conn:
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
            item = dict(row)
            item["tags"] = self._split_multi(item.pop("tags_text", ""))
            item["scenes"] = self._split_multi(item.pop("scenes_text", ""))
            return item

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

    # ── 嵌入向量 (emoji_embedding) CRUD ──

    def upsert_embedding(
        self,
        path: str,
        vector_blob: bytes,
        dim: int,
        model_sig: str,
    ) -> None:
        """写入或更新某 path 的向量。"""
        if not path or not vector_blob:
            return
        now = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO emoji_embedding (path, vector, dim, model_sig, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    vector = excluded.vector,
                    dim = excluded.dim,
                    model_sig = excluded.model_sig,
                    updated_at = excluded.updated_at
                """,
                (path, sqlite3.Binary(vector_blob), int(dim), model_sig, now),
            )

    def delete_embedding(self, path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emoji_embedding WHERE path = ?", (path,))

    def load_embeddings_by_sig(self, model_sig: str) -> list[dict[str, Any]]:
        """加载某 model_sig 的所有向量行，用于构建内存索引矩阵。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path, vector, dim FROM emoji_embedding WHERE model_sig = ?",
                (model_sig,),
            ).fetchall()
            return [
                {
                    "path": r["path"],
                    "vector": bytes(r["vector"]),
                    "dim": int(r["dim"]),
                }
                for r in rows
            ]

    def count_embeddings_by_sig(self, model_sig: str) -> int:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji_embedding WHERE model_sig = ?",
                (model_sig,),
            ).fetchone()
            return int(row["cnt"] if row else 0)

    def get_all_embedding_paths(self) -> list[str]:
        """所有已存向量的 path（用于对比 emoji 表，检测缺失/陈旧向量）。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji_embedding").fetchall()
            return [r["path"] for r in rows]
