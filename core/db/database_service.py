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

from ..util.normalization import normalize_label_list, normalize_scope_mode


class DatabaseService:
    """SQLite 数据库服务，管理表情包索引存储。"""

    _RELATED_FETCH_CHUNK_SIZE = 400

    # 表结构版本，用于迁移检测
    SCHEMA_VERSION = 6

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

                # v5: emoji/emoji_pending 元数据列 + pending 标签/场景关联表
                # （关联表由 _create_tables 用 IF NOT EXISTS 创建）
                if current_version < 5:
                    self._migrate_v5(conn)

                if current_version < 6:
                    self._migrate_v6(conn)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """创建所有数据表。"""
        # 主表：表情包元数据（v5 起含 reviewed_at 与图片元数据列）
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
                is_favorite INTEGER DEFAULT 0,
                reviewed_at INTEGER,
                source_url TEXT,
                original_name TEXT,
                width INTEGER,
                height INTEGER,
                format TEXT,
                bytes INTEGER,
                add_method TEXT,
                overlay_text TEXT,
                emotions_json TEXT,
                character TEXT
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
        # （v5 起含图片元数据列；tags_text/scenes_text 为废弃列，实际标签存关联表）
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
                scenes_text TEXT,
                source_url TEXT,
                original_name TEXT,
                width INTEGER,
                height INTEGER,
                format TEXT,
                bytes INTEGER,
                add_method TEXT,
                overlay_text TEXT,
                emotions_json TEXT,
                character TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_created ON emoji_pending(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_category ON emoji_pending(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_hash ON emoji_pending(hash)")

        # 待审核池标签/场景关联表（v5，与 emoji_tag/emoji_scene 同构，
        # 取代 emoji_pending 的 tags_text/scenes_text 逗号拼接列）
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_pending_tag (
                path TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (path, tag),
                FOREIGN KEY (path) REFERENCES emoji_pending(path) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS emoji_pending_scene (
                path TEXT NOT NULL,
                scene TEXT NOT NULL,
                PRIMARY KEY (path, scene),
                FOREIGN KEY (path) REFERENCES emoji_pending(path) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_tag_tag ON emoji_pending_tag(tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_scene_scene ON emoji_pending_scene(scene)")

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
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_character ON emoji(character)")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_hash ON emoji(hash)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_last_used ON emoji(last_used_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_tag ON emoji_tag(tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scene_scene ON emoji_scene(scene)")

    # ── v5 迁移：元数据列 + pending 标签/场景关联表 ──
    # 元数据列（emoji 与 emoji_pending 同构），供 WebUI 展示与统计
    _META_COLUMNS: tuple[tuple[str, str], ...] = (
        ("source_url", "TEXT"),      # 图片来源 URL（尽力获取，可能为空）
        ("original_name", "TEXT"),   # 原始文件名
        ("width", "INTEGER"),        # 图片宽度（像素）
        ("height", "INTEGER"),       # 图片高度（像素）
        ("format", "TEXT"),          # 图片格式 png/jpg/gif/webp
        ("bytes", "INTEGER"),        # 文件字节数
        ("add_method", "TEXT"),      # 入库方式 auto/manual/llm/api
    )
    _SEMANTIC_COLUMNS: tuple[tuple[str, str], ...] = (
        ("overlay_text", "TEXT"),
        ("emotions_json", "TEXT"),
        ("character", "TEXT"),
    )
    # 元数据标量字段名（供 INSERT/UPDATE 透传）
    _EMOJI_SCALAR_COLUMNS: frozenset[str] = frozenset(
        {
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
            "reviewed_at",
            *(col for col, _ in _META_COLUMNS),
            *(col for col, _ in _SEMANTIC_COLUMNS),
        }
    )
    # INSERT 语句用列清单（顺序与 _INSERT_EMOJI_SQL 的 VALUES 占位对应）
    _EMOJI_INSERT_COLUMNS: tuple[str, ...] = (
        "path",
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
        "reviewed_at",
        *(col for col, _ in _META_COLUMNS),
        *(col for col, _ in _SEMANTIC_COLUMNS),
    )
    _INSERT_EMOJI_SQL: str = (
        "INSERT OR REPLACE INTO emoji ("
        + ", ".join(_EMOJI_INSERT_COLUMNS)
        + ") VALUES ("
        + ", ".join("?" * len(_EMOJI_INSERT_COLUMNS))
        + ")"
    )
    # 待审核池 INSERT 列清单（含元数据列，不含 reviewed_at）
    _PENDING_INSERT_COLUMNS: tuple[str, ...] = (
        "path",
        "hash",
        "phash",
        "category",
        "desc",
        "source",
        "origin_target",
        "scope_mode",
        "review_status",
        "created_at",
        "tags_text",
        "scenes_text",
        *(col for col, _ in _META_COLUMNS),
        *(col for col, _ in _SEMANTIC_COLUMNS),
    )

    def _migrate_v5(self, conn: sqlite3.Connection) -> None:
        """v4 -> v5：为 emoji/emoji_pending 添加元数据列，并把旧 tags_text/scenes_text 拆入关联表。"""
        for col, ddl in self._META_COLUMNS:
            for table in ("emoji", "emoji_pending"):
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
        try:
            conn.execute("ALTER TABLE emoji ADD COLUMN reviewed_at INTEGER")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

        # 拆旧逗号拼接列到关联表（幂等：重复行用 INSERT OR IGNORE 跳过）
        rows = conn.execute(
            "SELECT path, tags_text, scenes_text FROM emoji_pending"
        ).fetchall()
        migrated_tags = 0
        migrated_scenes = 0
        for row in rows:
            path = row["path"]
            for tag in self._split_multi(row["tags_text"]):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO emoji_pending_tag (path, tag) VALUES (?, ?)",
                    (path, tag),
                )
                migrated_tags += cur.rowcount
            for scene in self._split_multi(row["scenes_text"]):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO emoji_pending_scene (path, scene) VALUES (?, ?)",
                    (path, scene),
                )
                migrated_scenes += cur.rowcount

        # 清空旧列（新代码不再写入；读取时以关联表为准）
        conn.execute("UPDATE emoji_pending SET tags_text = '', scenes_text = ''")
        logger.info(
            f"[DB] 迁移完成 (v5): 元数据列就绪，pending 标签/场景拆分 "
            f"{migrated_tags}/{migrated_scenes} 条"
        )

    def _migrate_v6(self, conn: sqlite3.Connection) -> None:
        """v5 -> v6：overlay_text / emotions_json / character。"""
        for col, ddl in self._SEMANTIC_COLUMNS:
            for table in ("emoji", "emoji_pending"):
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_character ON emoji(character)")
        except sqlite3.OperationalError:
            pass
        logger.info("[DB] 迁移完成 (v6): overlay_text / emotions_json / character 就绪")

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

    def get_character_counts(self) -> dict[str, int]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(character, '') as character, COUNT(*) as cnt
                FROM emoji GROUP BY COALESCE(character, '')
                """
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

    async def sync_index(self, idx: dict[str, Any]) -> None:
        """增量同步索引到数据库（仅插入/更新，不删除）。

        删除已移交 MaintenanceService 通过孤儿扫描处理，
        避免并发 on_message 时误删其他消息刚入库的条目。
        """
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

                for path in desired_paths - existing_paths:
                    meta = desired_index[path]
                    now = int(time.time())
                    conn.execute(
                        self._INSERT_EMOJI_SQL,
                        self._emoji_insert_values(path, meta, now=now),
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

                scalar_fields = tuple(self._EMOJI_SCALAR_COLUMNS)

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
            pending_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji_pending"
            ).fetchone()["cnt"]

            return {
                "total_emojis": total,
                "total_tags": tags_count,
                "total_scenes": scenes_count,
                "pending_count": pending_count,
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
        character: str | None = None,
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

            if character == "__none__":
                where_clauses.append("(e.character IS NULL OR e.character = '')")
            elif character:
                where_clauses.append("e.character = ?")
                params.append(character)

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
                    " OR e.overlay_text LIKE ? OR e.character LIKE ?"
                    " OR EXISTS("
                    "SELECT 1 FROM emoji_tag t WHERE t.path = e.path AND t.tag LIKE ?"
                    ") OR EXISTS("
                    "SELECT 1 FROM emoji_scene s WHERE s.path = e.path AND s.scene LIKE ?"
                    "))"
                )
                where_clauses.append(search_clause)
                params.extend([search_pattern] * 10)
                category_count_where_clauses.append(search_clause)
                category_count_params.extend([search_pattern] * 10)

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
                       e.is_favorite, e.reviewed_at,
                       e.source_url, e.original_name, e.width, e.height,
                       e.format, e.bytes, e.add_method,
                       e.overlay_text, e.emotions_json, e.character
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
                images.append(self._hydrate_entry(item))

            return images, total, category_counts

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
                       p.overlay_text, p.emotions_json, p.character
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
