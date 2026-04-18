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
import os
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
    SCHEMA_VERSION = 1

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

        # 内存缓存（可选，用于热点数据加速）
        self._category_cache: dict[str, list[str]] = {}
        self._cache_valid = False

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
            result = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            current_version = int(result["value"] if result else 0)

            if current_version < self.SCHEMA_VERSION:
                logger.info(f"[DB] 升级数据库 schema: {current_version} -> {self.SCHEMA_VERSION}")
                self._create_tables(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),)
                )

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
                last_used_at INTEGER DEFAULT 0
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

        # 创建索引加速搜索
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

    def _build_search_signature_from_index(
        self, idx: dict[str, dict[str, Any]]
    ) -> str:
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
            payload = "\x1f".join(
                [path, category, desc, "\x1e".join(tags), "\x1e".join(scenes)]
            )
            hasher.update(payload.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    # ── 基础 CRUD 操作 ──

    def get_emoji(self, path: str) -> dict[str, Any] | None:
        """获取单个表情包的完整信息。"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM emoji WHERE path = ?", (path,)
            ).fetchone()
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

    def get_all_paths(self) -> list[str]:
        """获取所有表情包路径。"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT path FROM emoji").fetchall()
            return [r["path"] for r in rows]

    def get_paths_by_category(self, category: str) -> list[str]:
        """获取指定分类的所有表情包路径。"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT path FROM emoji WHERE category = ?", (category,)
            ).fetchall()
            return [r["path"] for r in rows]

    def count_by_category(self, category: str) -> int:
        """统计指定分类的表情包数量。"""
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) as cnt FROM emoji WHERE category = ?", (category,)
            ).fetchone()
            return result["cnt"] if result else 0

    def count_total(self) -> int:
        """统计表情包总数。"""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM emoji").fetchone()
            return result["cnt"] if result else 0

    # ── 插入和更新操作 ──

    async def insert_emoji(
        self,
        path: str,
        hash_val: str,
        category: str,
        tags: list[str] | None = None,
        scenes: list[str] | None = None,
        desc: str | None = None,
        phash: str | None = None,
        source: str | None = None,
        origin_target: str | None = None,
        scope_mode: str = "public",
    ) -> bool:
        """插入新表情包记录。

        Args:
            path: 文件路径
            hash_val: SHA256 哈希
            category: 分类
            tags: 标签列表
            scenes: 场景列表
            desc: 描述
            phash: 感知哈希
            source: 来源
            origin_target: 来源目标
            scope_mode: scope 模式

        Returns:
            bool: 是否成功插入
        """
        async with self._write_lock:
            try:
                now = int(time.time())
                await asyncio.to_thread(
                    self._insert_emoji_sync,
                    path, hash_val, category, tags or [], scenes or [],
                    desc, phash, source, origin_target, scope_mode, now
                )
                self._cache_valid = False
                return True
            except Exception as e:
                logger.error(f"[DB] 插入表情包失败: {e}")
                return False

    def _insert_emoji_sync(
        self,
        path: str,
        hash_val: str,
        category: str,
        tags: list[str],
        scenes: list[str],
        desc: str | None,
        phash: str | None,
        source: str | None,
        origin_target: str | None,
        scope_mode: str,
        created_at: int,
    ) -> None:
        """同步插入表情包。"""
        with self._get_connection() as conn:
            # 使用事务确保一致性
            conn.execute("BEGIN IMMEDIATE")

            try:
                # 插入主记录
                conn.execute("""
                    INSERT OR REPLACE INTO emoji
                    (path, hash, phash, category, desc, source, origin_target,
                     scope_mode, created_at, use_count, last_used_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
                """, (path, hash_val, phash, category, desc, source,
                      origin_target, scope_mode, created_at))

                # 删除旧标签/场景（如果是 REPLACE）
                conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))

                # 插入标签
                for tag in tags:
                    if tag:
                        conn.execute(
                            "INSERT INTO emoji_tag (path, tag) VALUES (?, ?)",
                            (path, tag)
                        )

                # 插入场景
                for scene in scenes:
                    if scene:
                        conn.execute(
                            "INSERT INTO emoji_scene (path, scene) VALUES (?, ?)",
                            (path, scene)
                        )

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"[DB] 插入事务回滚: {e}")
                raise

    async def update_emoji(
        self,
        path: str,
        **updates: Any,
    ) -> bool:
        """更新表情包记录（增量更新）。

        支持更新的字段：category, desc, tags, scenes, use_count, last_used_at
        """
        if not updates:
            return True

        async with self._write_lock:
            try:
                await asyncio.to_thread(self._update_emoji_sync, path, updates)
                self._cache_valid = False
                return True
            except Exception as e:
                logger.error(f"[DB] 更新表情包失败: {e}")
                return False

    # 允许更新的字段白名单（防止SQL注入）
    _VALID_UPDATE_FIELDS = frozenset({
        "category", "desc", "use_count", "last_used_at",
        "hash", "phash", "source", "origin_target", "scope_mode"
    })

    def _update_emoji_sync(self, path: str, updates: dict[str, Any]) -> None:
        """同步更新表情包。"""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")

            try:
                # 白名单过滤字段名（防止SQL注入）
                main_updates = {
                    k: v for k, v in updates.items()
                    if k in self._VALID_UPDATE_FIELDS
                }

                if main_updates:
                    # 使用白名单验证的字段名构建SQL（安全）
                    clauses = ", ".join(f"{k} = ?" for k in main_updates.keys())
                    values = list(main_updates.values()) + [path]
                    conn.execute(
                        f"UPDATE emoji SET {clauses} WHERE path = ?",
                        values
                    )

                # 更新标签（完整替换）
                if "tags" in updates:
                    conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                    for tag in updates["tags"] or []:
                        if tag:
                            conn.execute(
                                "INSERT INTO emoji_tag (path, tag) VALUES (?, ?)",
                                (path, tag)
                            )

                # 更新场景（完整替换）
                if "scenes" in updates:
                    conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))
                    for scene in updates["scenes"] or []:
                        if scene:
                            conn.execute(
                                "INSERT INTO emoji_scene (path, scene) VALUES (?, ?)",
                                (path, scene)
                            )

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"[DB] 更新事务回滚: {e}")
                raise

    async def increment_usage(self, path: str) -> bool:
        """增加使用次数并更新最后使用时间。"""
        async with self._write_lock:
            try:
                await asyncio.to_thread(self.increment_usage_sync, path)
                return True
            except Exception as e:
                logger.error(f"[DB] 增加使用次数失败: {e}")
                return False

    def increment_usage_sync(self, path: str) -> None:
        """同步增加使用次数。"""
        now = int(time.time())
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE emoji SET use_count = use_count + 1, last_used_at = ? WHERE path = ?",
                (now, path)
            )

    async def delete_emoji(self, path: str) -> bool:
        """删除表情包记录。"""
        async with self._write_lock:
            try:
                await asyncio.to_thread(self._delete_emoji_sync, path)
                self._cache_valid = False
                return True
            except Exception as e:
                logger.error(f"[DB] 删除表情包失败: {e}")
                return False

    def _delete_emoji_sync(self, path: str) -> None:
        """同步删除表情包。"""
        with self._get_connection() as conn:
            # CASCADE 会自动删除关联的标签和场景
            conn.execute("DELETE FROM emoji WHERE path = ?", (path,))

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
                self._cache_valid = False
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
                    conn.execute("""
                        INSERT OR REPLACE INTO emoji
                        (path, hash, phash, category, desc, source, origin_target,
                         scope_mode, created_at, use_count, last_used_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
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
                    ))

                    # 删除旧标签/场景
                    conn.execute("DELETE FROM emoji_tag WHERE path = ?", (path,))
                    conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))

                    # 插入标签
                    for tag in emoji.get("tags") or []:
                        if tag:
                            conn.execute(
                                "INSERT INTO emoji_tag (path, tag) VALUES (?, ?)",
                                (path, tag)
                            )

                    # 插入场景
                    for scene in emoji.get("scenes") or []:
                        if scene:
                            conn.execute(
                                "INSERT INTO emoji_scene (path, scene) VALUES (?, ?)",
                                (path, scene)
                            )

                    count += 1

                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"[DB] 批量插入事务回滚: {e}")
                raise

        return count

    # ── 搜索操作 ──

    def search_by_category(
        self,
        category: str,
        exclude_recent: list[str] | None = None,
        limit: int = 10,
    ) -> list[str]:
        """按分类搜索表情包路径。

        Args:
            category: 分类名
            exclude_recent: 要排除的路径列表（最近使用的）
            limit: 返回数量限制

        Returns:
            list[str]: 匹配的表情包路径列表
        """
        with self._get_connection() as conn:
            if exclude_recent:
                # 使用 NOT IN 排除最近使用的
                placeholders = ", ".join("?" for _ in exclude_recent)
                query = f"""
                    SELECT path FROM emoji
                    WHERE category = ? AND path NOT IN ({placeholders})
                    ORDER BY use_count DESC, last_used_at ASC
                    LIMIT ?
                """
                params = [category] + exclude_recent + [limit]
            else:
                query = """
                    SELECT path FROM emoji
                    WHERE category = ?
                    ORDER BY use_count DESC, last_used_at ASC
                    LIMIT ?
                """
                params = [category, limit]

            rows = conn.execute(query, params).fetchall()
            return [r["path"] for r in rows]

    def search_by_tag(
        self,
        tag: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """按标签搜索表情包。

        Args:
            tag: 标签关键词
            category: 可选的分类过滤
            limit: 返回数量限制

        Returns:
            list[str]: 匹配的表情包路径列表
        """
        with self._get_connection() as conn:
            if category:
                query = """
                    SELECT DISTINCT e.path FROM emoji e
                    JOIN emoji_tag t ON e.path = t.path
                    WHERE t.tag LIKE ? AND e.category = ?
                    ORDER BY e.use_count DESC
                    LIMIT ?
                """
                params = [f"%{tag}%", category, limit]
            else:
                query = """
                    SELECT DISTINCT e.path FROM emoji e
                    JOIN emoji_tag t ON e.path = t.path
                    WHERE t.tag LIKE ?
                    ORDER BY e.use_count DESC
                    LIMIT ?
                """
                params = [f"%{tag}%", limit]

            rows = conn.execute(query, params).fetchall()
            return [r["path"] for r in rows]

    def search_by_desc(
        self,
        keyword: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """按描述关键词搜索。

        Args:
            keyword: 描述关键词
            category: 可选的分类过滤
            limit: 返回数量限制

        Returns:
            list[str]: 匹配的表情包路径列表
        """
        with self._get_connection() as conn:
            if category:
                query = """
                    SELECT path FROM emoji
                    WHERE desc LIKE ? AND category = ?
                    ORDER BY use_count DESC
                    LIMIT ?
                """
                params = [f"%{keyword}%", category, limit]
            else:
                query = """
                    SELECT path FROM emoji
                    WHERE desc LIKE ?
                    ORDER BY use_count DESC
                    LIMIT ?
                """
                params = [f"%{keyword}%", limit]

            rows = conn.execute(query, params).fetchall()
            return [r["path"] for r in rows]

    def search_comprehensive(
        self,
        query: str,
        categories: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """综合搜索：分类、标签、描述匹配。

        Args:
            query: 搜索关键词
            categories: 候选分类列表
            exclude_paths: 要排除的路径
            limit: 返回数量限制

        Returns:
            list[tuple[str, float]]: (路径, 匹配得分) 列表
        """
        results: dict[str, float] = {}

        with self._get_connection() as conn:
            # 1. 分类精确匹配（得分最高）
            if categories:
                for cat in categories:
                    rows = conn.execute(
                        "SELECT path FROM emoji WHERE category = ?",
                        (cat,)
                    ).fetchall()
                    for r in rows:
                        results[r["path"]] = max(results.get(r["path"], 0), 0.8)

            # 2. 标签匹配
            rows = conn.execute(
                "SELECT DISTINCT e.path FROM emoji e "
                "JOIN emoji_tag t ON e.path = t.path "
                "WHERE t.tag LIKE ?",
                (f"%{query}%",)
            ).fetchall()
            for r in rows:
                results[r["path"]] = max(results.get(r["path"], 0), 0.6)

            # 3. 描述匹配
            rows = conn.execute(
                "SELECT path FROM emoji WHERE desc LIKE ?",
                (f"%{query}%",)
            ).fetchall()
            for r in rows:
                results[r["path"]] = max(results.get(r["path"], 0), 0.4)

            # 4. 场景匹配
            rows = conn.execute(
                "SELECT DISTINCT e.path FROM emoji e "
                "JOIN emoji_scene s ON e.path = s.path "
                "WHERE s.scene LIKE ?",
                (f"%{query}%",)
            ).fetchall()
            for r in rows:
                results[r["path"]] = max(results.get(r["path"], 0), 0.5)

        # 排除指定路径
        if exclude_paths:
            for p in exclude_paths:
                results.pop(p, None)

        # 按得分排序并限制数量
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]

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
            self._cache_valid = False

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
                             scope_mode, created_at, use_count, last_used_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        ),
                    )

                    for tag in meta.get("tags") or []:
                        if tag:
                            conn.execute(
                                "INSERT INTO emoji_tag (path, tag) VALUES (?, ?)",
                                (path, tag),
                            )

                    for scene in meta.get("scenes") or []:
                        if scene:
                            conn.execute(
                                "INSERT INTO emoji_scene (path, scene) VALUES (?, ?)",
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
                                    "INSERT INTO emoji_tag (path, tag) VALUES (?, ?)",
                                    (path, tag),
                                )

                    if "scenes" in meta:
                        desired_scenes = [
                            scene for scene in (meta.get("scenes") or []) if scene
                        ]
                        if desired_scenes != (current.get("scenes") or []):
                            conn.execute("DELETE FROM emoji_scene WHERE path = ?", (path,))
                            for scene in desired_scenes:
                                conn.execute(
                                    "INSERT INTO emoji_scene (path, scene) VALUES (?, ?)",
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
                self._cache_valid = False
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

    def vacuum(self) -> None:
        """执行 VACUUM 优化数据库空间。"""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("[DB] VACUUM 完成")

    # ── 分页查询 ──

    # 排序字段白名单（防止SQL注入）
    _VALID_ORDER_FIELDS = {
        "newest": "e.created_at DESC, e.path DESC",
        "oldest": "e.created_at ASC, e.path ASC",
        "most_used": "e.use_count DESC, e.last_used_at DESC, e.path ASC",
    }

    def get_emojis_paginated(
        self,
        page: int = 1,
        page_size: int = 50,
        category: str | None = None,
        sort_order: str = "newest",
        search_query: str | None = None,
        scope_target: str | None = None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
        """分页获取表情包列表，支持过滤、搜索和排序。

        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            category: 分类过滤（可选）
            sort_order: 排序方式 - "newest"(最新), "oldest"(最旧), "most_used"(最常用)
            search_query: 搜索关键词（匹配标签、描述、场景）
            scope_target: scope 过滤目标（可选）

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

            # scope 过滤
            if scope_target:
                where_clauses.append(
                    "(e.scope_mode = 'public' OR e.origin_target = ?)"
                )
                params.append(scope_target)
                category_count_where_clauses.append(
                    "(e.scope_mode = 'public' OR e.origin_target = ?)"
                )
                category_count_params.append(scope_target)

            # 搜索过滤（标签、描述、场景）
            if search_query:
                search_pattern = f"%{search_query}%"
                search_clause = (
                    "(e.desc LIKE ? OR EXISTS("
                    "SELECT 1 FROM emoji_tag t WHERE t.path = e.path AND t.tag LIKE ?"
                    ") OR EXISTS("
                    "SELECT 1 FROM emoji_scene s WHERE s.path = e.path AND s.scene LIKE ?"
                    "))"
                )
                where_clauses.append(search_clause)
                params.extend([search_pattern, search_pattern, search_pattern])
                category_count_where_clauses.append(search_clause)
                category_count_params.extend(
                    [search_pattern, search_pattern, search_pattern]
                )

            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)

            category_count_where_sql = ""
            if category_count_where_clauses:
                category_count_where_sql = (
                    "WHERE " + " AND ".join(category_count_where_clauses)
                )

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
                       e.origin_target, e.created_at, e.use_count
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
