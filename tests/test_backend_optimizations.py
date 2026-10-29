"""Regression cases from the architecture audit, using real SQLite and image bytes."""
import asyncio
import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image, ImageDraw

from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.events.meme_sender_engine import MemeSenderEngine
from core.processing.classification_parser import ClassificationParser
from core.processing.image_processor_service import ImageProcessorService
from core.processing.phash_dedup_service import PHashDedupService
from core.search.meme_selector import MemeSelector
from core.search.meme_search_engine import MemeSearchEngine
from core.search.meme_smart_select_service import MemeSmartSelectService


def smart_service(index):
    search = SimpleNamespace(get_index=lambda: index, _bm25_dirty=False,
                             _bm25_index=SimpleNamespace(get_top_k=lambda *a, **k: []))
    scope = SimpleNamespace(_is_entry_allowed_for_event=lambda data, event: not data.get("blocked"))
    strategy = SimpleNamespace(_calculate_recent_penalty=lambda *a: 0)
    return MemeSmartSelectService(None, search, strategy, scope)


def test_overlay_does_not_replace_reply_scenes():
    scenes = ["你能帮帮我吗", "这个任务太难了"]
    assert ClassificationParser(None).sanitize_scenes(scenes, "我不干了") == scenes


@pytest.mark.asyncio
async def test_distinct_animations_and_legacy_first_frame_hashes(tmp_path):
    first = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(first).rectangle((10, 10, 40, 40), fill="black")
    png = tmp_path / "first.png"
    first.save(png)
    files = [tmp_path / "a.gif", tmp_path / "b.gif"]
    for path, color in zip(files, ["red", "blue"]):
        first.save(path, save_all=True, append_images=[Image.new("RGB", (64, 64), color)], duration=100, loop=0)
    assert files[0].read_bytes() != files[1].read_bytes()
    service = PHashDedupService()
    assert await service.compute_phash(str(files[1])) == ""
    legacy = await service.compute_phash(str(png))
    db = DatabaseService(tmp_path / "animation.db")
    await db.insert_batch([{
        "path": str(files[0]), "hash": hashlib.sha256(files[0].read_bytes()).hexdigest(),
        "phash": legacy,
    }])
    processor = ImageProcessorService.__new__(ImageProcessorService)
    processor.plugin, processor._phash_service = SimpleNamespace(db_service=db), service
    image_hash = hashlib.sha256(files[1].read_bytes()).hexdigest()
    assert not await processor._is_duplicate_or_blacklisted(image_hash, {}, str(files[1]), False)
    assert not await processor._is_duplicate_or_blacklisted("static", {}, str(png), False, legacy)
    db.hash_exists = lambda _: True
    assert await processor._is_duplicate_or_blacklisted(image_hash, {}, str(files[1]), False)


@pytest.mark.asyncio
async def test_preferences_do_not_admit_unrelated_candidates(tmp_path):
    path = str(tmp_path / "a.png")
    tmp_path.joinpath("a.png").touch()
    index = {path: {"category": "happy", "desc": "galaxy", "is_favorite": True, "use_count": 999}}
    service = smart_service(index)
    assert await service._rank_emoji_candidates("happy", "quartz", deterministic=True) == []
    index[path]["overlay_text"] = "quartz"
    assert (await service._rank_emoji_candidates("happy", "quartz", deterministic=True))[0][0] == path


@pytest.mark.asyncio
async def test_smart_abstention_does_not_fall_back_to_random():
    selector = MemeSelector.__new__(MemeSelector)
    selector.plugin = SimpleNamespace(plugin_config=SimpleNamespace(smart_meme_selection=True))
    selector._selection_lock = asyncio.Lock()
    selector.normalize_category = lambda value: value
    selector._selection_strategy = SimpleNamespace(_get_candidate_categories=lambda _: ["happy"])
    selector._smart_select_service = SimpleNamespace(_select_emoji_smart_impl=AsyncMock(return_value=None))
    selector._select_emoji_random_impl = Mock(side_effect=AssertionError("random fallback"))
    assert await selector.select_emoji("happy", "谢谢") is None


@pytest.mark.asyncio
async def test_recall_fusion_keeps_embedding_and_bm25_channels():
    index = {f"p{i}": {"category": "happy"} for i in range(50)}
    service = smart_service(index)
    service._overlay_recall_paths = lambda *a: [f"p{i}" for i in range(16)]
    service._character_recall_paths = lambda *a: [f"p{i}" for i in range(16, 32)]
    service._scene_recall_paths = lambda *a: [f"p{i}" for i in range(32, 48)]
    service._is_embedding_ready = lambda: True
    service._embedding_service = SimpleNamespace(search=AsyncMock(return_value=[("p48", 0.99)]))
    service._search_engine._bm25_index = SimpleNamespace(get_top_k=lambda *a, **k: [(0, 10)])
    service._search_engine._bm25_doc_paths = ["p49"]
    paths, _ = await service._recall_candidate_paths(index, "query", None, set())
    assert len(paths) == 48
    assert {"p48", "p49"} <= set(paths)


@pytest.mark.parametrize("text,skip", [
    ("你不能再可爱了", False), ("安全感满满", False), ("失败乃成功之母", False),
    ("抱歉，请求执行失败", True), ("连接失败，请稍后重试", True),
    ("无法访问该文件", True), ("/meme status", True),
])
def test_contextual_intent_gate(text, skip):
    engine = MemeSenderEngine(SimpleNamespace())
    assert engine.should_skip_auto_emoji_by_gate(text) is skip


@pytest.mark.asyncio
async def test_snapshot_cache_isolated_and_invalidated_by_other_connection(tmp_path, monkeypatch):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy", "tags": ["original"]}])
    load = Mock(wraps=db._read_index)
    monkeypatch.setattr(db, "_read_index", load)
    snapshot = db.get_index_cache_readonly()
    snapshot["a"]["tags"].append("mutated")
    snapshot["a"]["is_favorite"] = 1
    assert db.get_index_cache_readonly()["a"]["tags"] == ["original"]
    assert load.call_count == 1
    signature = db.get_corpus_signature()
    with sqlite3.connect(db._db_path) as conn:
        conn.execute("UPDATE emoji SET is_favorite=1, use_count=8 WHERE path='a'")
    assert db.get_index_cache_readonly()["a"]["use_count"] == 8
    assert db.get_corpus_signature() == signature
    await db.update_path("a", {"desc": "changed"})
    assert db.get_index_cache_readonly()["a"]["desc"] == "changed"
    assert db.get_corpus_signature() != signature


@pytest.mark.asyncio
async def test_partial_sync_reads_only_submitted_paths_and_noop_keeps_revision(tmp_path, monkeypatch):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": str(i), "hash": str(i), "category": "happy", "tags": ["t"]} for i in range(1000)])
    statements = []
    original = db._get_connection
    @contextmanager
    def traced():
        with original() as connection:
            connection.set_trace_callback(statements.append)
            yield connection
    monkeypatch.setattr(db, "_get_connection", traced)
    await db.sync_index({"5": {"desc": "changed"}})
    assert not any(sql == "SELECT * FROM emoji" for sql in statements)
    assert db.count_total() == 1000
    signature = db.get_corpus_signature()
    await db.update_path("5", {"desc": "changed", "tags": ["t"]})
    assert db.get_corpus_signature() == signature


@pytest.mark.asyncio
async def test_refresh_failure_survives_restart_and_newer_generation(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy"}])
    embedding = SimpleNamespace(_embedding_enabled=lambda: True, delete_by_path=AsyncMock(),
                                insert_emoji=AsyncMock(return_value=False))
    plugin = SimpleNamespace(db_service=db, meme_selector=SimpleNamespace(
        _smart_select_service=SimpleNamespace(_embedding_service=embedding)))
    manager = IndexManager(plugin)
    assert await manager.drain_search_refresh() == 1
    restarted = DatabaseService(db._db_path)
    with restarted._get_connection() as conn:
        failed = dict(conn.execute("SELECT * FROM search_refresh").fetchone())
    assert failed["attempts"] == 1 and failed["last_error"]
    await restarted.update_path("a", {"desc": "new"})
    restarted.finish_search_refresh("a", failed["generation"])
    assert restarted.get_search_refresh_jobs()[0]["generation"] > failed["generation"]
    plugin.db_service = restarted
    manager = IndexManager(plugin)
    embedding.insert_emoji.return_value = True
    assert await manager.drain_search_refresh() == 1
    assert restarted.get_search_refresh_jobs() == []


@pytest.mark.asyncio
async def test_bm25_concurrent_update_keeps_old_build_dirty(tmp_path, monkeypatch):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy", "desc": "galaxy"}])
    engine = MemeSearchEngine(SimpleNamespace(db_service=db), None, None)
    entered, release = threading.Event(), threading.Event()
    prepare = engine._prepare_bm25
    def blocked(index):
        entered.set()
        assert release.wait(5)
        return prepare(index)
    monkeypatch.setattr(engine, "_prepare_bm25", blocked)
    build = asyncio.create_task(engine._build_bm25_index())
    assert await asyncio.to_thread(entered.wait, 5)
    await db.update_path("a", {"desc": "quartz"})
    engine._invalidate_bm25_index()
    release.set()
    await build
    assert engine._bm25_dirty
    assert engine._bm25_signature != db.get_corpus_signature()
    await engine._build_bm25_index()
    assert not engine._bm25_dirty
    assert engine._bm25_signature == db.get_corpus_signature()
    await db.delete_paths(["a"])
    await engine._build_bm25_index()
    assert engine._bm25_index is None and engine._bm25_doc_paths == []


@pytest.mark.asyncio
async def test_v6_migration_keeps_existing_library_and_vectors(tmp_path):
    path = tmp_path / "emoji.db"
    db = DatabaseService(path)
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy", "tags": ["t"],
                            "scenes": ["s"], "character": "fish", "is_favorite": 1,
                            "scope_mode": "local", "origin_target": "group:1", "use_count": 8}])
    db.upsert_embedding("a", b"vector", 1, "model")
    with db._get_connection() as conn:
        names = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")]
        for name in names:
            conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute("DROP TABLE search_refresh")
        conn.execute("DELETE FROM meta WHERE key IN ('library_revision', 'corpus_revision', 'corpus_id')")
        conn.execute("UPDATE meta SET value='6' WHERE key='schema_version'")
    migrated = DatabaseService(path)
    entry = migrated.get_emoji("a")
    assert entry["is_favorite"] == 1 and entry["use_count"] == 8
    assert entry["scope_mode"] == "local" and entry["origin_target"] == "group:1"
    assert entry["character"] == "fish" and entry["tags"] == ["t"] and entry["scenes"] == ["s"]
    assert migrated.load_embeddings_by_sig("model")[0]["vector"] == b"vector"
    assert migrated.get_search_refresh_jobs() == []
    await migrated.update_path("a", {"desc": "changed"})
    assert migrated.get_search_refresh_jobs()[0]["path"] == "a"


@pytest.mark.asyncio
async def test_status_and_selected_hash_queries_never_load_full_snapshot(tmp_path, monkeypatch):
    from core.commands.command_handler import CommandHandler

    db = DatabaseService(tmp_path / "performance.db")
    await db.insert_batch([
        {"path": "a", "hash": "same", "category": "happy", "tags": ["one"]},
        {"path": "b", "hash": "same", "category": "sad", "is_favorite": True},
        {"path": "other", "hash": "other", "category": "happy"},
    ])
    monkeypatch.setattr(db, "get_index_cache_readonly", lambda: pytest.fail("不应读取全库快照"))
    selected = db.get_entries_by_hashes(["same"])
    assert set(selected) == {"a", "b"} and selected["a"]["tags"] == ["one"]
    assert db.get_entries_by_hashes([]) == {}
    await db.add_blacklist("blocked")
    monkeypatch.setattr(db, "blacklisted_hashes", lambda: pytest.fail("不应加载整个黑名单"))
    assert db.is_blacklisted("blocked") and not db.is_blacklisted("other")
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(
        steal_meme=True, auto_send_meme=False, max_reg_num=100,
    ))
    messages = [message async for message in CommandHandler(plugin).status(
        SimpleNamespace(plain_result=lambda text: text)
    )]
    assert "表情：3" in messages[0] and "收藏：1" in messages[0]
    assert "happy 2" in messages[0] and len(messages[0].splitlines()) <= 4
