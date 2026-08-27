"""数据库服务单元测试。

测试 DatabaseService 的核心功能：
- 基础 CRUD 操作
- 批量插入和迁移
- 搜索功能
- 并发安全
"""

import asyncio
import json
import tempfile
import pytest
import sys
import types
from pathlib import Path

# 安装 astrbot stubs
def _install_stubs():
    package_name = Path(__file__).resolve().parents[1].name

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    api_module = types.ModuleType("astrbot.api")
    api_module.logger = logger
    api_module.AstrBotConfig = object

    sys.modules["astrbot.api"] = api_module

    # StarTools stub
    star_tools = types.SimpleNamespace(
        get_data_dir=lambda name: str(Path(tempfile.gettempdir()) / "astrbot_test" / name)
    )
    star_module = types.ModuleType("astrbot.api.star")
    star_module.StarTools = star_tools
    star_module.Context = object
    star_module.Star = object

    sys.modules["astrbot.api.star"] = star_module

    return package_name

PACKAGE_NAME = _install_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db.database_service import DatabaseService


class TestDatabaseInit:
    """测试数据库初始化。"""

    def test_init_creates_database_file(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        DatabaseService(db_path)
        assert db_path.exists()

    def test_init_creates_tables(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = DatabaseService(db_path)

        with db._get_connection() as conn:
            # 检查主表存在
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='emoji'"
            ).fetchone()
            assert result is not None

            # 检查标签表存在
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='emoji_tag'"
            ).fetchone()
            assert result is not None

            # 检查场景表存在
            result = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='emoji_scene'"
            ).fetchone()
            assert result is not None

    def test_schema_version_recorded(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = DatabaseService(db_path)

        with db._get_connection() as conn:
            result = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            assert result is not None
            assert int(result["value"]) == DatabaseService.SCHEMA_VERSION

    def test_wal_mode_enabled(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = DatabaseService(db_path)

        with db._get_connection() as conn:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result["journal_mode"].lower() == "wal"


class TestBasicCRUD:
    """测试基础增删改查操作。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        return DatabaseService(db_path)






    def test_get_all_paths(self, db: DatabaseService):
        asyncio.run(db.insert_batch([
            {"path": "/test/a.gif", "hash": "hash_a", "category": "happy"},
            {"path": "/test/b.gif", "hash": "hash_b", "category": "sad"},
            {"path": "/test/c.gif", "hash": "hash_c", "category": "angry"},
        ]))

        paths = db.get_all_paths()
        assert len(paths) == 3
        assert "/test/a.gif" in paths
        assert "/test/b.gif" in paths
        assert "/test/c.gif" in paths



    def test_count_total(self, db: DatabaseService):
        asyncio.run(db.insert_batch([
            {"path": "/test/a.gif", "hash": "hash_a", "category": "happy"},
            {"path": "/test/b.gif", "hash": "hash_b", "category": "sad"},
        ]))

        assert db.count_total() == 2

    def test_count_favorites(self, db: DatabaseService):
        asyncio.run(db.insert_batch([
            {"path": "/test/a.gif", "hash": "hash_a", "category": "happy", "is_favorite": 1},
            {"path": "/test/b.gif", "hash": "hash_b", "category": "sad"},
            {"path": "/test/c.gif", "hash": "hash_c", "category": "happy", "is_favorite": True},
        ]))

        assert db.count_favorites() == 2


class TestBatchOperations:
    """测试批量操作。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        return DatabaseService(db_path)

    def test_insert_batch(self, db: DatabaseService):
        async def run():
            emojis = [
                {
                    "path": "/test/a.gif",
                    "hash": "hash_a",
                    "category": "happy",
                    "tags": ["开心"],
                    "desc": "描述A",
                },
                {
                    "path": "/test/b.gif",
                    "hash": "hash_b",
                    "category": "sad",
                    "tags": ["难过"],
                    "scenes": ["伤心"],
                },
                {
                    "path": "/test/c.gif",
                    "hash": "hash_c",
                    "category": "angry",
                },
            ]

            count = await db.insert_batch(emojis)
            assert count == 3

            assert db.count_total() == 3
            assert db.get_emoji("/test/a.gif")["tags"] == ["开心"]
            assert db.get_emoji("/test/b.gif")["scenes"] == ["伤心"]

        asyncio.run(run())

    def test_insert_batch_replaces_existing(self, db: DatabaseService):
        async def run():
            # 先插入一条
            await db.insert_batch([{"path": "/test/a.gif", "hash": "hash_old", "category": "happy", "desc": "旧描述"}])

            # 批量插入包含相同路径的新数据
            emojis = [
                {
                    "path": "/test/a.gif",
                    "hash": "hash_new",
                    "category": "sad",
                    "desc": "新描述",
                },
            ]

            count = await db.insert_batch(emojis)
            assert count == 1

            emoji = db.get_emoji("/test/a.gif")
            assert emoji["hash"] == "hash_new"
            assert emoji["category"] == "sad"
            assert emoji["desc"] == "新描述"

        asyncio.run(run())


class TestSearchOperations:
    """测试搜索操作。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = DatabaseService(db_path)
        # 预填充测试数据
        asyncio.run(db.insert_batch([
            {"path": "/test/happy1.gif", "hash": "h1", "category": "happy", "tags": ["开心", "大笑"], "desc": "开心表情"},
            {"path": "/test/happy2.gif", "hash": "h2", "category": "happy", "tags": ["微笑"], "desc": "微笑表情"},
            {"path": "/test/sad1.gif", "hash": "s1", "category": "sad", "tags": ["难过"], "desc": "难过表情"},
            {"path": "/test/angry1.gif", "hash": "a1", "category": "angry", "tags": ["生气"], "desc": "生气表情"},
        ]))
        return db


class TestPendingUpdate:
    """issue #87: 待审核区信息修改（DB 层）。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "pending.db"
        svc = DatabaseService(db_path)
        return svc

    def _insert(self, db: DatabaseService, **overrides):
        meta = {
            "path": "/data/pending/123_abcdef.jpg",
            "hash": "abcdef0123456789",
            "phash": "phash01",
            "category": "happy",
            "desc": "原始描述",
            "tags": ["原标签"],
            "scenes": ["原场景"],
            "scope_mode": "public",
            "source": "auto",
            "origin_target": "group:1",
        }
        meta.update(overrides)
        return asyncio.run(db.insert_pending(meta))

    def test_update_pending_changes_category_desc_tags_scenes(self, db: DatabaseService):
        pid = self._insert(db)
        assert pid is not None
        updated = asyncio.run(
            db.update_pending(
                pid,
                {
                    "category": "troll",
                    "desc": "新描述",
                    "tags": ["搞笑", "梗图"],
                    "scenes": ["搞笑", "吐槽"],
                    "scope_mode": "local",
                },
            )
        )
        assert updated is not None
        assert updated["category"] == "troll"
        assert updated["desc"] == "新描述"
        assert set(updated["tags"]) == {"搞笑", "梗图"}
        assert set(updated["scenes"]) == {"搞笑", "吐槽"}
        assert updated["scope_mode"] == "local"

    def test_update_pending_rejects_path_and_hash(self, db: DatabaseService):
        """path/hash/source/origin_target 不在白名单，必须被忽略（不抛错也不写入）。"""
        pid = self._insert(db)
        updated = asyncio.run(
            db.update_pending(
                pid,
                {
                    "category": "happy",
                    "path": "/evil/override.jpg",  # 应被忽略
                    "hash": "deadbeef",  # 应被忽略
                    "source": "manual",  # 应被忽略
                },
            )
        )
        assert updated is not None
        assert updated["path"] == "/data/pending/123_abcdef.jpg"
        assert updated["hash"] == "abcdef0123456789"
        assert updated["source"] == "auto"

    def test_update_pending_unknown_id_returns_none(self, db: DatabaseService):
        result = asyncio.run(db.update_pending(99999, {"category": "happy"}))
        assert result is None

    def test_update_pending_empty_fields_returns_none(self, db: DatabaseService):
        pid = self._insert(db)
        # 空字段或全部为白名单外字段都返回 None
        assert asyncio.run(db.update_pending(pid, {})) is None
        assert asyncio.run(db.update_pending(pid, {"path": "/x"})) is None

    def test_update_pending_noop_returns_row_not_none(self, db: DatabaseService):
        """SQLite 的 no-op UPDATE（值未变）会返回 rowcount=0，不能据此判定行不存在。"""
        pid = self._insert(db, category="happy", desc="hi", tags=["t"])
        # 提交与现有完全相同的值
        same = asyncio.run(
            db.update_pending(pid, {"category": "happy", "desc": "hi", "tags": ["t"]})
        )
        assert same is not None
        assert same["id"] == pid
        assert same["category"] == "happy"
        assert same["desc"] == "hi"








class TestLegacyCompatibility:
    """测试旧接口兼容性。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        return DatabaseService(db_path)

    def test_get_index_cache_readonly(self, db: DatabaseService):
        asyncio.run(db.insert_batch([
            {"path": "/test/a.gif", "hash": "h1", "category": "happy", "tags": ["开心"]},
            {"path": "/test/b.gif", "hash": "h2", "category": "sad", "scenes": ["伤心"]},
        ]))

        idx = db.get_index_cache_readonly()
        assert isinstance(idx, dict)
        assert len(idx) == 2
        assert "/test/a.gif" in idx
        assert idx["/test/a.gif"]["tags"] == ["开心"]
        assert idx["/test/b.gif"]["scenes"] == ["伤心"]

    def test_save_index(self, db: DatabaseService):
        async def run():
            idx = {
                "/test/a.gif": {"hash": "h1", "category": "happy", "tags": ["开心"], "desc": "描述A"},
                "/test/b.gif": {"hash": "h2", "category": "sad", "scenes": ["伤心"]},
            }

            await db.save_index(idx)
            assert db.count_total() == 2

            emoji = db.get_emoji("/test/a.gif")
            assert emoji["category"] == "happy"
            assert emoji["tags"] == ["开心"]

        asyncio.run(run())

    def test_sync_index_updates_existing_metadata(self, db: DatabaseService):
        async def run():
            await db.insert_batch([{
                "path": "/test/a.gif",
                "hash": "h1",
                "category": "happy",
                "tags": ["旧标签"],
                "scenes": ["旧场景"],
                "desc": "旧描述",
                "source": "qq_store",
                "origin_target": "group:1",
                "scope_mode": "public",
            }])

            await db.sync_index(
                {
                    "/test/a.gif": {
                        "hash": "h1",
                        "category": "happy",
                        "tags": ["新标签1", "新标签2"],
                        "scenes": ["新场景"],
                        "desc": "新描述",
                        "source": "manual",
                        "origin_target": "group:2",
                        "scope_mode": "local",
                    }
                }
            )

            emoji = db.get_emoji("/test/a.gif")
            assert emoji["desc"] == "新描述"
            assert emoji["tags"] == ["新标签1", "新标签2"]
            assert emoji["scenes"] == ["新场景"]
            assert emoji["source"] == "manual"
            assert emoji["origin_target"] == "group:2"
            assert emoji["scope_mode"] == "local"

        asyncio.run(run())

    def test_sync_index_inserts_new_and_updates_existing(self, db: DatabaseService):
        """sync_index 增量同步：插入新条目、更新已有条目，不删除未出现在索引中的条目。"""
        async def run():
            await db.insert_batch(
                [
                    {
                        "path": "/test/a.gif",
                        "hash": "h1",
                        "category": "happy",
                        "desc": "A",
                    },
                    {
                        "path": "/test/b.gif",
                        "hash": "h2",
                        "category": "sad",
                        "desc": "B",
                        "tags": ["旧标签"],
                    },
                ]
            )

            await db.sync_index(
                {
                    "/test/b.gif": {
                        "hash": "h2",
                        "category": "sad",
                        "desc": "B2",
                        "tags": ["新标签"],
                    },
                    "/test/c.gif": {
                        "hash": "h3",
                        "category": "angry",
                        "desc": "C",
                    },
                }
            )

            # /test/a.gif 不在传入索引中，但不应被删除
            assert db.get_emoji("/test/a.gif")["desc"] == "A"
            # /test/b.gif 应被更新
            assert db.get_emoji("/test/b.gif")["desc"] == "B2"
            assert db.get_emoji("/test/b.gif")["tags"] == ["新标签"]
            # /test/c.gif 应被插入
            assert db.get_emoji("/test/c.gif")["category"] == "angry"
            # 总共 3 条（a 保留，b 更新，c 新增）
            assert db.count_total() == 3

        asyncio.run(run())

    def test_sync_index_chunks_related_queries_for_large_indexes(
        self, db: DatabaseService, monkeypatch
    ):
        async def run():
            await db.insert_batch(
                [
                    {
                        "path": f"/test/{i}.gif",
                        "hash": f"h{i}",
                        "category": "happy",
                        "tags": [f"tag-{i}"],
                        "scenes": [f"scene-{i}"],
                        "desc": f"desc-{i}",
                    }
                    for i in range(5)
                ]
            )

            monkeypatch.setattr(db, "_RELATED_FETCH_CHUNK_SIZE", 2)

            await db.sync_index(
                {
                    f"/test/{i}.gif": {
                        "hash": f"h{i}",
                        "category": "happy",
                        "tags": [f"tag-{i}-new"],
                        "scenes": [f"scene-{i}-new"],
                        "desc": f"desc-{i}-new",
                    }
                    for i in range(5)
                }
            )

            idx = db.get_index_cache_readonly()
            assert len(idx) == 5
            assert idx["/test/4.gif"]["tags"] == ["tag-4-new"]
            assert idx["/test/4.gif"]["scenes"] == ["scene-4-new"]
            assert idx["/test/4.gif"]["desc"] == "desc-4-new"

        asyncio.run(run())

    def test_clear_all(self, db: DatabaseService):
        async def run():
            await db.insert_batch([
                {"path": "/test/a.gif", "hash": "h1", "category": "happy"},
                {"path": "/test/b.gif", "hash": "h2", "category": "sad"},
            ])

            await db.clear_all()
            assert db.count_total() == 0

        asyncio.run(run())


class TestMigration:
    """测试从旧版 JSON 迁移。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        return DatabaseService(db_path)

    def test_migrate_from_json(self, db: DatabaseService, tmp_path: Path):
        async def run():
            # 创建旧版 JSON 文件
            json_path = tmp_path / "index_cache.json"
            old_data = {
                "/test/a.gif": {
                    "hash": "h1",
                    "category": "happy",
                    "tags": ["开心"],
                    "desc": "描述A",
                    "use_count": 5,
                },
                "/test/b.gif": {
                    "hash": "h2",
                    "category": "sad",
                    "scenes": ["伤心"],
                    "use_count": 3,
                },
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(old_data, f)

            count = await db.migrate_from_json(json_path)
            assert count == 2

            # 验证数据正确迁移
            emoji_a = db.get_emoji("/test/a.gif")
            assert emoji_a["category"] == "happy"
            assert emoji_a["tags"] == ["开心"]
            assert emoji_a["use_count"] == 5

            # 验证旧文件被备份
            backup_path = tmp_path / "index_cache.json.migrated"
            assert backup_path.exists()
            assert not json_path.exists()

        asyncio.run(run())

    def test_migrate_from_nonexistent_json(self, db: DatabaseService, tmp_path: Path):
        async def run():
            json_path = tmp_path / "nonexistent.json"
            count = await db.migrate_from_json(json_path)
            assert count == 0

        asyncio.run(run())


class TestConcurrency:
    """测试并发安全性。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        return DatabaseService(db_path)




class TestStats:
    """测试统计功能。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = DatabaseService(db_path)
        asyncio.run(db.insert_batch([
            {"path": "/test/a.gif", "hash": "h1", "category": "happy", "tags": ["开心", "大笑"]},
            {"path": "/test/b.gif", "hash": "h2", "category": "happy", "tags": ["微笑"]},
            {"path": "/test/c.gif", "hash": "h3", "category": "sad", "scenes": ["伤心"]},
        ]))
        return db

    def test_get_stats(self, db: DatabaseService):
        stats = db.get_stats()
        assert stats["total_emojis"] == 3
        assert stats["total_tags"] == 3  # 开心, 大笑, 微笑
        assert stats["total_scenes"] == 1  # 伤心
        assert stats["categories"]["happy"] == 2
        assert stats["categories"]["sad"] == 1
        assert stats["db_size_bytes"] > 0

    def test_get_emojis_paginated_keeps_all_category_counts_when_filtered(
        self, db: DatabaseService
    ):
        images, total, category_counts = db.get_emojis_paginated(
            page=1,
            page_size=50,
            category="happy",
        )

        assert total == 2
        assert len(images) == 2
        assert category_counts["happy"] == 2
        assert category_counts["sad"] == 1

    def test_get_emojis_paginated_includes_favorite_and_usage_fields(
        self, db: DatabaseService
    ):
        asyncio.run(db.insert_batch([
            {
                "path": "/test/favorite.gif",
                "hash": "fav1",
                "category": "happy",
                "is_favorite": 1,
                "use_count": 7,
                "last_used_at": 1700000000,
            }
        ]))

        images, total, _ = db.get_emojis_paginated(
            page=1,
            page_size=50,
            favorite_only=True,
        )

        assert total == 1
        assert images[0]["hash"] == "fav1"
        assert images[0]["is_favorite"] == 1
        assert images[0]["use_count"] == 7
        assert images[0]["last_used_at"] == 1700000000

    def test_character_is_independent_of_category(self, db: DatabaseService):
        asyncio.run(db.insert_batch([
            {
                "path": "/test/neuro_happy.gif",
                "hash": "n1",
                "category": "happy",
                "character": "neurosama",
                "desc": "棕发小女孩唱歌",
            },
            {
                "path": "/test/plain_happy.gif",
                "hash": "n2",
                "category": "happy",
                "character": "",
                "desc": "普通开心",
            },
        ]))
        neuro, total, _ = db.get_emojis_paginated(page=1, page_size=50, character="neurosama")
        assert total == 1
        assert neuro[0]["hash"] == "n1"
        assert neuro[0]["character"] == "neurosama"
        assert neuro[0]["category"] == "happy"

        none_assigned, none_total, _ = db.get_emojis_paginated(
            page=1, page_size=50, character="__none__", category="happy"
        )
        assert none_total >= 1
        assert all(not (item.get("character") or "") for item in none_assigned)

        counts = db.get_character_counts()
        assert counts.get("neurosama") == 1

    def test_get_corpus_signature_changes_when_searchable_metadata_changes(
        self, db: DatabaseService
    ):
        before = db.get_corpus_signature()

        asyncio.run(
            db.sync_index({
                "/test/a.gif": {
                    "hash": "h1",
                    "category": "happy",
                    "desc": "完全不同的描述",
                    "tags": ["替换标签1", "替换标签2"],
                }
            })
        )

        after = db.get_corpus_signature()
        assert before != after


class TestPendingPool:
    """待审核池分页 / 哈希回退（PR #90 修复）。"""

    @pytest.fixture
    def db(self, tmp_path: Path):
        return DatabaseService(tmp_path / "pending_pool.db")

    def _insert(self, db: DatabaseService, **overrides):
        meta = {
            "path": "/data/pending/a.jpg",
            "hash": "hash_a",
            "category": "happy",
            "desc": "测试描述",
        }
        meta.update(overrides)
        return asyncio.run(db.insert_pending(meta))

    def test_get_pending_paginated_keeps_all_category_counts_when_filtered(
        self, db: DatabaseService
    ):
        """PR #90：按分类筛选时，分类计数不应再被同一分类条件锁死。"""
        self._insert(db, path="/data/pending/a.jpg", hash="ha", category="happy")
        self._insert(db, path="/data/pending/b.jpg", hash="hb", category="happy")
        self._insert(db, path="/data/pending/c.jpg", hash="hc", category="sad")

        items, total, counts = db.get_pending_paginated(
            page=1, page_size=50, category="happy"
        )

        assert total == 2
        assert len(items) == 2
        assert counts["happy"] == 2
        assert counts["sad"] == 1

    def test_get_pending_paginated_search_keeps_all_category_counts(
        self, db: DatabaseService
    ):
        self._insert(db, path="/data/pending/a.jpg", hash="ha", category="happy", desc="猫咪")
        self._insert(db, path="/data/pending/b.jpg", hash="hb", category="happy", desc="狗狗")
        self._insert(db, path="/data/pending/c.jpg", hash="hc", category="sad", desc="猫咪")

        items, total, counts = db.get_pending_paginated(
            page=1, page_size=50, search_query="猫咪"
        )

        assert total == 2
        assert counts["happy"] == 1
        assert counts["sad"] == 1

    def test_get_pending_paginated_category_plus_search(self, db: DatabaseService):
        self._insert(db, path="/data/pending/a.jpg", hash="ha", category="happy", desc="猫咪")
        self._insert(db, path="/data/pending/b.jpg", hash="hb", category="happy", desc="狗狗")
        self._insert(db, path="/data/pending/c.jpg", hash="hc", category="sad", desc="猫咪")

        items, total, counts = db.get_pending_paginated(
            page=1, page_size=50, category="happy", search_query="猫咪"
        )

        assert total == 1
        assert len(items) == 1
        assert items[0]["hash"] == "ha"
        assert counts["happy"] == 1

    def test_get_pending_paginated_splits_tags_and_scenes(self, db: DatabaseService):
        self._insert(
            db,
            path="/data/pending/a.jpg",
            hash="ha",
            category="happy",
            tags=["搞笑", "梗图"],
            scenes=["群聊"],
        )
        items, total, _ = db.get_pending_paginated(page=1, page_size=50)
        assert items[0]["tags"] == ["搞笑", "梗图"]
        assert items[0]["scenes"] == ["群聊"]

    def test_get_pending_by_hash_returns_path(self, db: DatabaseService):
        self._insert(db, path="/data/pending/a.jpg", hash="hash123", category="happy")
        row = db.get_pending_by_hash("hash123")
        assert row is not None
        assert row["path"] == "/data/pending/a.jpg"
        assert row["category"] == "happy"
        assert row["hash"] == "hash123"

    def test_get_pending_by_hash_unknown_returns_none(self, db: DatabaseService):
        assert db.get_pending_by_hash("nope") is None

    def test_get_pending_by_hash_empty_returns_none(self, db: DatabaseService):
        assert db.get_pending_by_hash("") is None

    def test_count_pending(self, db: DatabaseService):
        self._insert(db, path="/data/pending/a.jpg", hash="ha")
        self._insert(db, path="/data/pending/b.jpg", hash="hb")
        assert db.count_pending() == 2

    def test_insert_pending_duplicate_path_returns_none(self, db: DatabaseService):
        pid1 = self._insert(db, path="/data/pending/a.jpg", hash="ha")
        assert pid1 is not None
        pid2 = self._insert(db, path="/data/pending/a.jpg", hash="ha")
        assert pid2 is None