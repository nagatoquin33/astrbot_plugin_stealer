"""SQLite schema creation, migrations, and column contracts."""

import sqlite3

from astrbot.api import logger

from .storage_paths import legacy_image_path



class DatabaseSchema:
    def _init_schema(self) -> None:
        """初始化数据库表结构。"""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
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

                # 记录由 _create_tables 创建的黑名单表已就绪。
                if current_version < 3:
                    logger.info("[DB] migration: blacklist table ready")

                # 记录由 _create_tables 创建的待审核池和嵌入向量表已就绪。
                if current_version < 4:
                    logger.info("[DB] migration: emoji_pending / emoji_embedding tables ready")

                # 补齐图片元数据列，并迁移待审核记录的标签和场景。
                if current_version < 5:
                    self._migrate_v5(conn)

                if current_version < 6:
                    self._migrate_v6(conn)

                if current_version < 8:
                    for column, ddl in (("analysis_state", "TEXT NOT NULL DEFAULT 'ready'"),
                                        ("analysis_error", "TEXT NOT NULL DEFAULT ''"),
                                        ("manual_fields", "TEXT NOT NULL DEFAULT '[]'")):
                        columns = {row[1] for row in conn.execute("PRAGMA table_info(emoji_pending)")}
                        if column not in columns:
                            conn.execute(f"ALTER TABLE emoji_pending ADD COLUMN {column} {ddl}")

            # 为每个数据库补齐外部源表结构和检索修订信息。
            self._ensure_external_schema(conn)
            if self._data_root is not None and (current_version < 9 or conn.execute(
                "SELECT 1 FROM meta WHERE key = 'image_path_migration_pending'"
            ).fetchone()):
                self._migrate_image_paths(conn)
            self._ensure_search_revision_schema(conn)
            conn.execute("COMMIT")

    def _migrate_image_paths(self, conn: sqlite3.Connection) -> None:
        """改写存储路径和关联键，保留已有语义、向量及队列状态。"""
        root = self._data_root
        if root is None:
            return
        changes = {}
        missing = set()
        for row in conn.execute("SELECT path FROM emoji UNION SELECT path FROM emoji_pending"):
            old = row["path"]
            new = legacy_image_path(old, root)
            if new == old:
                continue
            if new != self._store_path(old) and not (root / new).is_file():
                missing.add(old)
                continue
            changes[old] = new

        # 旧库的刷新队列还可能含已删除图片的路径。
        has_queue = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'search_refresh'").fetchone()
        if has_queue:
            for row in conn.execute("SELECT path FROM search_refresh"):
                old = row["path"]
                new = legacy_image_path(old, root)
                if old != new and old not in missing:
                    changes.setdefault(old, new)

        if missing:
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('image_path_migration_pending', '1')")
            logger.warning(f"[DB] 有 {len(missing)} 张旧库图片未在新目录找到，保留记录并暂停启动清理")
        else:
            conn.execute("DELETE FROM meta WHERE key = 'image_path_migration_pending'")
        if not changes:
            return

        conn.execute("PRAGMA defer_foreign_keys = ON")
        conn.execute("CREATE TEMP TABLE image_path_map (old TEXT PRIMARY KEY, new TEXT NOT NULL)")
        conn.executemany("INSERT INTO image_path_map VALUES (?, ?)", changes.items())
        # 这里只换路径，不重新计算已有向量或重置失败重试时间。
        for table in ("emoji", "emoji_tag", "emoji_scene"):
            for operation in ("insert", "update", "delete"):
                conn.execute(f"DROP TRIGGER IF EXISTS revision_{table}_{operation}")
                conn.execute(f"DROP TRIGGER IF EXISTS corpus_{table}_{operation}")
        tables = (
            "emoji", "emoji_tag", "emoji_scene", "emoji_pending", "emoji_pending_tag",
            "emoji_pending_scene", "emoji_embedding", "meme_source_item",
        )
        if has_queue:
            tables += ("search_refresh",)
        for table in tables:
            conn.execute(f"UPDATE {table} SET path = (SELECT new FROM image_path_map WHERE old = path) "
                         "WHERE path IN (SELECT old FROM image_path_map)")
        conn.execute("""
            UPDATE meme_source_item
            SET metadata_json = json_set(metadata_json, '$.pending_path', mapping.new)
            FROM image_path_map AS mapping
            WHERE CASE WHEN json_valid(metadata_json)
                  THEN json_extract(metadata_json, '$.pending_path') END = mapping.old
        """)
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 "
                     "WHERE key IN ('library_revision', 'corpus_revision')")
        logger.info(f"[DB] 已将 {len(changes)} 个图库路径迁移为相对路径")

    def _ensure_search_revision_schema(self, conn: sqlite3.Connection) -> None:
        """创建图库修订信息和可合并同路径任务的持久化向量刷新队列。"""
        for key in ("library_revision", "corpus_revision"):
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, '0')", (key,))
        conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('corpus_id', hex(randomblob(16)))")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_refresh (
                path TEXT PRIMARY KEY,
                generation INTEGER NOT NULL DEFAULT 1,
                attempts INTEGER NOT NULL DEFAULT 0,
                retry_at INTEGER NOT NULL DEFAULT 0,
                last_error TEXT
            )
        """)
        for table in ("emoji", "emoji_tag", "emoji_scene"):
            for operation in ("INSERT", "UPDATE", "DELETE"):
                conn.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS revision_{table}_{operation.lower()}
                    AFTER {operation} ON {table}
                    BEGIN
                        UPDATE meta SET value = CAST(value AS INTEGER) + 1
                        WHERE key = 'library_revision';
                    END
                """)
                when = ""
                if table == "emoji" and operation == "UPDATE":
                    fields = ("path", "category", "desc", "overlay_text", "emotions_json", "character")
                    when = "WHEN " + " OR ".join(f"OLD.{field} IS NOT NEW.{field}" for field in fields)
                path = "OLD.path" if operation == "DELETE" else "NEW.path"
                conn.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS corpus_{table}_{operation.lower()}
                    AFTER {operation} ON {table} {when}
                    BEGIN
                        UPDATE meta SET value = CAST(value AS INTEGER) + 1
                        WHERE key = 'corpus_revision';
                        INSERT INTO search_refresh(path) VALUES ({path})
                        ON CONFLICT(path) DO UPDATE SET
                            generation = generation + 1, attempts = 0,
                            retry_at = 0, last_error = NULL;
                    END
                """)

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        """创建所有数据表。"""
        # 表情包主表：存储分类、审核时间和图片元数据。
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
        # 图片元数据存于本表，标签和场景存于关联表。
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

        # 待审核池标签和场景关联表，与正式图库关联表结构一致。
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
        "analysis_state",
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

        # 清空已迁入关联表的逗号分隔字段。
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
        """幂等创建外部源注册表，并补齐图库和待审核池的来源与保留策略字段。
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
