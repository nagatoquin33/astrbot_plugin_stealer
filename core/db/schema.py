"""SQLite schema creation, migrations, and column contracts."""

import sqlite3

from astrbot.api import logger



class DatabaseSchema:
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

            # v3 external-source schema is additive and intentionally runs for
            # every database, including databases already at schema version 6.
            self._ensure_external_schema(conn)

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
                character TEXT,
                retention_class TEXT DEFAULT 'native'
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
                character TEXT,
                retention_class TEXT DEFAULT 'native'
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
            "retention_class",
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
        "retention_class",
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
        "retention_class",
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

    def _ensure_external_schema(self, conn: sqlite3.Connection) -> None:
        """Create the additive v3 source registry and retention metadata.

        This method is idempotent and also repairs databases created by older
        versions where ``_create_tables`` was never called after v6.
        """

        for table in ("emoji", "emoji_pending"):
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN retention_class TEXT DEFAULT 'native'"
                )
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emoji_retention ON emoji(retention_class)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pending_retention ON emoji_pending(retention_class)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meme_source (
                source_id TEXT PRIMARY KEY,
                source_type TEXT NOT NULL,
                name TEXT NOT NULL,
                endpoint TEXT,
                config_json TEXT,
                enabled INTEGER DEFAULT 1,
                status TEXT DEFAULT 'idle',
                last_error TEXT,
                item_count INTEGER DEFAULT 0,
                last_sync_at INTEGER,
                created_at INTEGER DEFAULT 0,
                updated_at INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meme_source_item (
                source_id TEXT NOT NULL,
                external_id TEXT NOT NULL,
                path TEXT,
                source_category TEXT,
                source_url TEXT,
                license TEXT,
                attribution TEXT,
                remote_hash TEXT,
                metadata_json TEXT,
                first_seen_at INTEGER DEFAULT 0,
                last_seen_at INTEGER DEFAULT 0,
                stale INTEGER DEFAULT 0,
                PRIMARY KEY (source_id, external_id),
                FOREIGN KEY (source_id) REFERENCES meme_source(source_id) ON DELETE CASCADE,
                FOREIGN KEY (path) REFERENCES emoji(path) ON DELETE SET NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_item_path ON meme_source_item(path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source_item_source ON meme_source_item(source_id)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('external_schema_version', ?)",
            (str(self.EXTERNAL_SCHEMA_VERSION),),
        )
