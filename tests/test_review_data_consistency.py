"""Data-root, cleanup and model-isolation regressions from the follow-up review."""
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from astrbot.api.star import StarTools

from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.events.background_steal_queue import BackgroundStealQueue, StealJob
from core.maintenance.service import MaintenanceService
from core.search.embedding_service import EmbeddingService
from core.sources.source_service import SourceService
from core.util.safe_io import commit_image_file


def test_missing_data_roots_use_framework_directory(tmp_path, monkeypatch):
    calls = []
    def data_dir(name):
        calls.append(name)
        return tmp_path
    monkeypatch.setattr(StarTools, "get_data_dir", staticmethod(data_dir))
    plugin = SimpleNamespace()
    assert Path(EmbeddingService(plugin)._resolve_data_dir()) == tmp_path
    assert SourceService(plugin).data_dir == tmp_path
    assert BackgroundStealQueue(plugin).staging_dir == tmp_path / "staging"
    assert calls == ["astrbot_plugin_stealer"] * 3


@pytest.mark.parametrize("kind", ["embedding", "source", "queue"])
def test_relative_data_root_is_rejected(kind):
    plugin = SimpleNamespace(base_dir="relative")
    with pytest.raises(ValueError, match="absolute"):
        if kind == "embedding":
            EmbeddingService(plugin)._resolve_data_dir()
        elif kind == "source":
            SourceService(plugin)
        else:
            BackgroundStealQueue(plugin)


@pytest.mark.asyncio
async def test_cleanup_preserves_equivalent_indexed_paths(tmp_path):
    categories, pending_dir = tmp_path / "categories", tmp_path / "pending"
    (categories / "happy").mkdir(parents=True)
    pending_dir.mkdir()
    image, waiting = categories / "happy" / "a.png", pending_dir / "p.png"
    image.write_bytes(b"image")
    waiting.write_bytes(b"pending")
    # A /./ segment is valid on Windows/Linux but differs from directory enumeration.
    stored = str(image.parent) + "/./" + image.name
    stored_pending = str(waiting.parent) + "/./" + waiting.name
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": stored, "hash": "h", "category": "happy", "is_favorite": 1}])
    pid = await db.insert_pending({"path": stored_pending, "hash": "p", "category": "happy"})
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(categories_dir=categories, pending_dir=pending_dir))
    await MaintenanceService(plugin)._cleanup_orphans()
    assert image.exists() and waiting.exists()
    assert db.get_emoji(stored)["is_favorite"] == 1 and db.get_pending(pid)


@pytest.mark.asyncio
async def test_pending_only_library_cleans_missing_pending_rows(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    pid = await db.insert_pending({"path": str(tmp_path / "missing.png"), "hash": "p", "tags": ["tag"]})
    assert db.count_total() == 0
    await MaintenanceService(SimpleNamespace(db_service=db))._cleanup_orphans()
    assert db.get_pending(pid) is None


@pytest.mark.asyncio
async def test_startup_cleanup_does_not_delete_an_inflight_publication(tmp_path):
    categories, pending = tmp_path / "categories", tmp_path / "pending"
    categories.mkdir()
    pending.mkdir()
    existing, source, target = categories / "existing.png", tmp_path / "raw.png", categories / "new.png"
    existing.write_bytes(b"existing")
    source.write_bytes(b"new")
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": str(existing), "hash": "existing", "category": "happy"}])
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(categories_dir=categories, pending_dir=pending))
    entered, release = asyncio.Event(), asyncio.Event()
    async def commit(path):
        entered.set()
        await release.wait()
        await db.sync_index({str(path): {"hash": "new", "category": "happy"}})
    publication = asyncio.create_task(commit_image_file(source, target, commit))
    try:
        await asyncio.wait_for(entered.wait(), timeout=3)
        await MaintenanceService(plugin)._cleanup_orphans()
        assert target.exists()
    finally:
        release.set()
        await publication
    assert db.get_emoji(str(target)) is not None and target.exists()


@pytest.mark.asyncio
async def test_startup_cleanup_removes_orphans_after_grace_period(tmp_path, monkeypatch):
    import core.maintenance.service as module
    categories, pending = tmp_path / "categories", tmp_path / "pending"
    categories.mkdir()
    pending.mkdir()
    existing, orphan = categories / "existing.png", categories / "orphan.png"
    existing.write_bytes(b"existing")
    orphan.write_bytes(b"orphan")
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": str(existing), "hash": "existing", "category": "happy"}])
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(categories_dir=categories, pending_dir=pending))
    later = module.time.time() + 600
    monkeypatch.setattr(module.time, "time", lambda: later)
    await MaintenanceService(plugin)._cleanup_orphans()
    assert existing.exists() and not orphan.exists()


@pytest.mark.asyncio
async def test_background_processing_preserves_edits_after_the_ingestion_commit(tmp_path):
    source = tmp_path / "staged.png"
    source.write_bytes(b"fixture")
    db = DatabaseService(tmp_path / "worker.db")
    await db.insert_batch([{"path": "existing.png", "hash": "old", "desc": "existing"}])
    delta = {"new.png": {"hash": "new", "category": "happy", "desc": "model result"}}

    async def process(*args, **kwargs):
        await db.insert_batch([{"path": path, **entry} for path, entry in delta.items()])
        await db.update_path("new.png", {"desc": "later user edit"})
        return True, delta

    plugin = SimpleNamespace(base_dir=tmp_path, db_service=db, _process_image=process)
    manager = IndexManager(plugin)
    plugin.index_manager = manager
    manager.load_index = AsyncMock(side_effect=AssertionError("full snapshot"))
    manager.save_index = AsyncMock(wraps=manager.save_index)
    queue = BackgroundStealQueue(plugin, worker_count=1)
    await queue.start()
    try:
        await queue._queue.put(StealJob(str(source), to_pending=False))
        await asyncio.wait_for(queue._queue.join(), timeout=3)
        manager.load_index.assert_not_awaited()
        manager.save_index.assert_not_awaited()
        assert db.get_emoji("new.png")["desc"] == "later user edit"
        assert db.get_emoji("existing.png")["desc"] == "existing"
    finally:
        await queue.stop()


def fallback_fixture(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    model = ["model-a"]
    provider = SimpleNamespace(provider_config={"id": "provider"}, get_model=lambda: model[0], get_dim=lambda: 2)
    async def embed(text):
        return [1.0, 0.0] if model[0] == "model-a" or text == "query" else [0.0, 1.0]
    provider.get_embedding = AsyncMock(side_effect=embed)
    service = EmbeddingService(SimpleNamespace(db_service=db, enable_embedding_search=True,
        plugin_config=SimpleNamespace(enable_embedding_search=True, embedding_provider_id="", category_info={}, character_info={})))
    service._get_provider = lambda: provider
    service._init_faiss = lambda: False
    return service, db, provider, model



@pytest.mark.asyncio
async def test_same_dimension_model_change_does_not_reuse_old_vectors(tmp_path):
    service, db, _, model = fallback_fixture(tmp_path)
    await db.insert_batch([{"path": "a", "hash": "a", "category": "happy", "desc": "picture"}])
    assert await service._fallback_insert("a", "picture")
    assert (await service._fallback_search("query", 5))[0][0] == "a"
    model[0] = "model-b"
    assert await service._fallback_search("query", 5) == []
    assert db.get_all_embedding_paths() == ["a"]  # Old row remains until actual backfill.
    assert await service.backfill_existing() == 1
    assert await service._fallback_search("query", 5) == []
    assert (await service._fallback_search("picture", 5))[0][0] == "a"


@pytest.mark.asyncio
async def test_legacy_mixed_dimension_and_nonfinite_vectors_are_not_loaded(tmp_path):
    service, db, provider, _ = fallback_fixture(tmp_path)
    signature = service._fallback_signature(provider)
    await db.insert_batch([{"path": path, "hash": path, "category": "happy"}
                           for path in ("legacy", "bad", "nan", "good")])
    for path, values, sig in [("legacy", [1, 0], "fallback"), ("bad", [1, 0, 1], signature),
                              ("nan", [float("nan"), 0], signature), ("good", [1, 0], signature)]:
        array = np.asarray(values, dtype=np.float32)
        db.upsert_embedding(path, array.tobytes(), len(values), sig)
    service._load_fallback_matrix()
    assert service._fallback_paths == ["good"]
    assert len(db.get_all_embedding_paths()) == 4


@pytest.mark.asyncio
async def test_unavailable_provider_does_not_consume_vector_refresh_jobs(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    await db.insert_batch([{"path": "a", "hash": "a", "category": "happy"}])
    db.delete_embedding = Mock(side_effect=AssertionError("must preserve vectors"))
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(enable_embedding_search=True, embedding_provider_id=""),
                             context=SimpleNamespace(get_all_embedding_providers=lambda: []))
    service = EmbeddingService(plugin)
    plugin.meme_selector = SimpleNamespace(embedding_service=service)
    assert await IndexManager(plugin).drain_search_refresh() == 0
    assert db.get_search_refresh_jobs()[0]["attempts"] == 0


def test_provider_registration_recovers_after_a_negative_probe():
    providers = []
    inst_map = {}
    service = EmbeddingService(SimpleNamespace(plugin_config=SimpleNamespace(enable_embedding_search=True, embedding_provider_id=""),
        context=SimpleNamespace(get_all_embedding_providers=lambda: providers,
                                provider_manager=SimpleNamespace(inst_map=inst_map))))
    assert service._get_provider() is None
    service._faiss_available = False
    provider = SimpleNamespace(provider_config={"id": "new"}, get_dim=lambda: 2)
    providers.append(provider)
    inst_map["new"] = provider
    assert service._get_provider() is provider
    assert service._faiss_available is None


@pytest.mark.asyncio
async def test_fallback_rejects_model_change_during_embedding_request(tmp_path):
    service, db, provider, model = fallback_fixture(tmp_path)
    await db.insert_batch([{"path": "a", "hash": "a", "category": "happy"}])
    async def changed(_text):
        model[0] = "model-b"
        return [1.0, 0.0]
    provider.get_embedding.side_effect = changed
    assert not await service._fallback_insert("a", "picture")
    assert db.get_all_embedding_paths() == []


@pytest.mark.asyncio
async def test_moved_v8_library_keeps_related_rows_and_uses_new_root(tmp_path):
    old = tmp_path / "old" / "astrbot_plugin_stealer"
    restored = tmp_path / "restored" / old.name
    final = tmp_path / "second" / old.name
    image = "categories/happy/emoji.png"
    waiting = "pending/review.png"
    for relative in (image, waiting):
        path = old / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    cache = Path("cache") / "emoji.db"
    db = DatabaseService(old / cache)
    db.set_meta_value("schema_version", "8")
    # 用另一系统的旧路径验证数据库迁移不依赖宿主路径语法。
    foreign = r"/srv/astrbot/data/plugin_data/astrbot_plugin_stealer"
    image_old, waiting_old = f"{foreign}/{image}", f"{foreign}/{waiting}"
    await db.insert_batch([{"path": image_old, "hash": "main", "category": "happy",
                            "is_favorite": 1, "tags": ["tag"], "scenes": ["scene"]}])
    pending_id = await db.insert_pending({"path": waiting_old, "hash": "waiting", "tags": ["review"]})
    db.upsert_embedding(image_old, b"vector", 1, "test")
    generation = db.get_search_refresh_generations([image_old])[image_old]
    db.finish_search_refresh(image_old, generation, "retry")
    await db.upsert_source({"source_id": "source"})
    await db.link_source_item({"source_id": "source", "external_id": "approved", "path": image_old})
    await db.link_source_item({"source_id": "source", "external_id": "pending",
                               "metadata": {"pending_path": waiting_old}})

    restored.parent.mkdir()
    old.rename(restored)
    db = DatabaseService(restored / cache, data_root=restored)
    with db._get_connection() as conn:
        assert conn.execute("SELECT path FROM emoji").fetchone()[0] == image
        assert conn.execute("SELECT path FROM emoji_pending").fetchone()[0] == waiting
        assert tuple(conn.execute("SELECT path, generation, attempts FROM search_refresh").fetchone()) == (
            image, generation, 1
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []

    final.parent.mkdir()
    restored.rename(final)
    db = DatabaseService(final / cache, data_root=final)
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(
        categories_dir=final / "categories", pending_dir=final / "pending",
    ))
    await MaintenanceService(plugin).run_startup_cleanup()
    path = str(final / image)
    assert (final / image).is_file() and (final / waiting).is_file()
    assert db.get_all_paths() == [path]
    entry = db.get_entries([path])[path]
    assert entry["is_favorite"] == 1 and entry["tags"] == ["tag"] and entry["scenes"] == ["scene"]
    assert db.load_embeddings_by_sig("test")[0]["path"] == path
    assert db.get_source_item("source", "approved")["path"] == path
    assert db.get_pending(pending_id)["path"] == str(final / waiting)

    new_image = str(final / "categories" / "happy" / "review.png")
    assert await db.approve_pending(pending_id, new_image, "happy")
    assert db.get_emoji(new_image)["tags"] == ["review"]
    assert db.get_source_item("source", "pending")["path"] == new_image
    assert await db.update_path(path, {"desc": "edit"})
    assert db.increment_usage_sync(path)
    assert db.get_emoji(path)["desc"] == "edit"


@pytest.mark.asyncio
async def test_incomplete_v8_restore_retries_before_cleaning(tmp_path):
    root = tmp_path / "astrbot_plugin_stealer"
    categories, pending = root / "categories", root / "pending"
    categories.mkdir(parents=True)
    pending.mkdir()
    database = root / "cache" / "emoji.db"
    db = DatabaseService(database)
    db.set_meta_value("schema_version", "8")
    old_path = "/backup/astrbot_plugin_stealer/categories/happy/missing.png"
    await db.insert_batch([{"path": old_path, "hash": "old", "category": "happy"}])

    fresh = categories / "happy" / "fresh.png"
    fresh.parent.mkdir()
    fresh.write_bytes(b"new")
    await db.insert_batch([{"path": str(fresh), "hash": "fresh", "category": "happy"}])
    db = DatabaseService(database, data_root=root)
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(
        categories_dir=categories, pending_dir=pending,
    ))
    await MaintenanceService(plugin).run_startup_cleanup()
    assert db.get_emoji(old_path) and fresh.is_file()

    restored = fresh.with_name("missing.png")
    restored.write_bytes(b"restored")
    db = DatabaseService(database, data_root=root)
    assert db.get_meta_value("image_path_migration_pending") is None
    assert db.get_emoji(str(restored))["hash"] == "old"
