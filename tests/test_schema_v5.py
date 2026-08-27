"""v5 schema 回归：v4→v5 迁移、pending 关联表、元数据列、标签规范化、缓存 model_sig。"""

import json
import os
import sqlite3
import tempfile

import pytest

from core.db.database_service import DatabaseService


def _make_v4_db(db_path: str) -> None:
    """构造一个 v4 旧库（含 tags_text 逗号列数据）。"""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema_version','4');
        CREATE TABLE emoji (path TEXT PRIMARY KEY, hash TEXT, phash TEXT, category TEXT, desc TEXT,
          source TEXT, origin_target TEXT, scope_mode TEXT DEFAULT 'public',
          created_at INTEGER DEFAULT 0, use_count INTEGER DEFAULT 0,
          last_used_at INTEGER DEFAULT 0, is_favorite INTEGER DEFAULT 0);
        CREATE TABLE emoji_tag (path TEXT, tag TEXT, PRIMARY KEY(path,tag));
        CREATE TABLE emoji_scene (path TEXT, scene TEXT, PRIMARY KEY(path,scene));
        CREATE TABLE blacklist (hash TEXT PRIMARY KEY, created_at INTEGER DEFAULT 0);
        CREATE TABLE emoji_pending (id INTEGER PRIMARY KEY AUTOINCREMENT,
          path TEXT NOT NULL UNIQUE, hash TEXT, phash TEXT, category TEXT, desc TEXT,
          source TEXT, origin_target TEXT, scope_mode TEXT DEFAULT 'public',
          review_status TEXT DEFAULT 'pending', created_at INTEGER DEFAULT 0,
          tags_text TEXT, scenes_text TEXT);
        CREATE TABLE emoji_embedding (path TEXT PRIMARY KEY, vector BLOB,
          dim INTEGER, model_sig TEXT, updated_at INTEGER DEFAULT 0);
        """
    )
    # 旧 pending 数据
    conn.execute(
        "INSERT INTO emoji_pending (path,hash,category,tags_text,scenes_text) VALUES (?,?,?,?,?)",
        ("P1.png", "h1", "happy", "开心,猫,开心,大笑", "群聊,吹水"),
    )
    conn.execute(
        "INSERT INTO emoji (path,hash,category) VALUES (?,?,?)", ("old.png", "h0", "sad")
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_v5_migration_and_metadata():
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "emoji.db")
    _make_v4_db(db_path)

    db = DatabaseService(db_path)

    # 迁移后版本号
    with sqlite3.connect(db_path) as conn:
        ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert ver == "6"

    # 旧 pending tags 拆入关联表（主键 (path,tag) 天然去重，与正式表 emoji_tag 语义一致）
    p = db.get_pending(1)
    assert p["tags"] == ["开心", "猫", "大笑"], p["tags"]
    assert p["scenes"] == ["群聊", "吹水"], p["scenes"]
    assert "tags_text" not in p

    # 新 pending 插入：关联表 + 元数据
    pid = await db.insert_pending(
        {
            "path": "P2.png",
            "hash": "h2",
            "category": "angry",
            "tags": ["生气", "摔桌"],
            "scenes": ["被惹毛"],
            "source_url": "https://x/y.png",
            "original_name": "y.png",
            "width": 100,
            "height": 200,
            "format": "png",
            "bytes": 1234,
            "add_method": "auto",
        }
    )
    row = db.get_pending(pid)
    assert row["tags"] == ["生气", "摔桌"], row["tags"]
    assert row["source_url"] == "https://x/y.png"
    assert row["width"] == 100 and row["height"] == 200
    assert row["add_method"] == "auto"

    # update_pending 全量替换标签（走关联表）
    upd = await db.update_pending(pid, {"tags": ["恼火", "气炸"], "scenes": ["吵架"]})
    assert upd["tags"] == ["恼火", "气炸"], upd["tags"]
    assert upd["scenes"] == ["吵架"], upd["scenes"]

    # pending 分页查询含关联表标签
    items, total, _ = db.get_pending_paginated(page=1, page_size=10)
    by_path = {i["path"]: i for i in items}
    assert by_path["P2.png"]["tags"] == ["恼火", "气炸"]
    assert by_path["P1.png"]["tags"] == ["开心", "猫", "大笑"]

    # 正式库插入带元数据
    n = await db.insert_batch(
        [
            {
                "path": "new.png",
                "hash": "h3",
                "category": "happy",
                "tags": ["开心"],
                "scenes": ["庆祝"],
                "width": 640,
                "height": 480,
                "format": "gif",
                "bytes": 999,
                "add_method": "llm",
                "reviewed_at": 1234567890,
            }
        ]
    )
    assert n == 1
    e = db.get_emoji("new.png")
    assert e["width"] == 640 and e["format"] == "gif"
    assert e["reviewed_at"] == 1234567890 and e["add_method"] == "llm"

    # 分页查询返回元数据
    imgs, _, _ = db.get_emojis_paginated(page=1, page_size=10)
    new_img = next(i for i in imgs if i["path"] == "new.png")
    assert new_img["width"] == 640 and new_img["format"] == "gif"
    assert new_img["add_method"] == "llm" and new_img["reviewed_at"] == 1234567890

    # tag_stats
    st = db.get_tag_stats(top_n=5)
    assert st["total_emojis"] == 2
    assert st["top_tags"][0]["tag"] == "开心"
    assert st["zero_tag_count"] == 1  # old.png 无标签
    assert st["total_with_tags"] == 1
    print("tag_stats:", json.dumps(st, ensure_ascii=False))

    # 删除 pending → 关联表级联删除
    db.delete_pending(pid)
    with sqlite3.connect(db_path) as conn:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM emoji_pending_tag WHERE path='P2.png'"
        ).fetchone()[0]
    assert cnt == 0


def test_fresh_db_creates_full_schema():
    """全新数据库：CREATE 直接建全列（含元数据），无需走 ALTER 迁移路径。"""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "fresh.db")
    DatabaseService(db_path)

    # 版本直接到当前 schema
    with sqlite3.connect(db_path) as conn:
        ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert ver == "6"

    # 全列存在（fresh 库 CREATE 含元数据列，_migrate_v5 的 ALTER 被 duplicate 容错跳过）
    with sqlite3.connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(emoji)").fetchall()]
    for col in (
        "reviewed_at",
        "source_url",
        "original_name",
        "width",
        "height",
        "format",
        "bytes",
        "add_method",
        "overlay_text",
        "emotions_json",
        "character",
    ):
        assert col in cols, f"emoji 缺少列 {col}"

    with sqlite3.connect(db_path) as conn:
        pcols = [r[1] for r in conn.execute("PRAGMA table_info(emoji_pending)").fetchall()]
    for col in (
        "source_url",
        "original_name",
        "width",
        "height",
        "format",
        "bytes",
        "add_method",
        "overlay_text",
        "emotions_json",
        "character",
    ):
        assert col in pcols, f"emoji_pending 缺少列 {col}"


def test_parser_label_normalization():
    """B2：标签/场景规范化（去重、截断到上限）。"""
    from core.processing.classification_parser import MAX_SCENES, MAX_TAGS, ClassificationParser

    n = ClassificationParser.normalize_label_list
    # 列表 + 去重 + 截断
    assert n(["震惊", "瞪眼", "震惊", "卧槽", "无语", "傻眼"], MAX_TAGS) == [
        "震惊",
        "瞪眼",
        "卧槽",
    ]
    # 字符串（中文逗号/顿号/分号分隔）+ 去重 + 截断
    assert n("开心，猫、猫；大笑", MAX_TAGS) == ["开心", "猫", "大笑"]
    # 超过上限截断
    assert n(["a", "b", "c", "d", "e", "f", "g"], MAX_TAGS) == ["a", "b", "c"]
    assert n(["s1", "s2", "s3", "s4"], MAX_SCENES) == ["s1", "s2"]
    # 空/非列表
    assert n(None, MAX_TAGS) == []
    assert n(123, MAX_TAGS) == []
    assert n([], MAX_TAGS) == []


def test_classification_cache_model_sig():
    """B3：分类缓存带 model_sig，换模型即失效。"""
    import types

    from core.processing.image_processor_service import ImageProcessorService

    fake_plugin = types.SimpleNamespace(
        plugin_config=types.SimpleNamespace(
            raw_dir=None, categories_dir=None, categories=[],
            content_filtration=False, vision_provider_id="",
        )
    )
    svc = ImageProcessorService(fake_plugin)
    svc._put_image_cache("hash1", "happy", ["开心"], "desc", "happy", ["场景"], "model-A")
    # 相同模型命中
    assert svc._get_valid_cache("hash1", "model-A") is not None
    # 换模型失效
    assert svc._get_valid_cache("hash1", "model-B") is None
    # 未配置模型（空签名）时不命中带签名的缓存
    assert svc._get_valid_cache("hash1", "") is None
    # 无签名缓存与空签名互认
    svc._put_image_cache("hash2", "sad", [], "d", "sad", [], "")
    assert svc._get_valid_cache("hash2", "") is not None
