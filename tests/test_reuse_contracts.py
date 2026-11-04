"""Regression coverage for existing library/framework integration points."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rank_bm25 import BM25Plus

from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.search.embedding_service import EmbeddingService
from core.search.text_similarity import BM25
from task_scheduler import TaskScheduler


def test_bm25_library_adapter_returns_true_top_k():
    model = BM25([["term"] * count for count in [1, 2, 3, 8, 7]])
    assert isinstance(model._model, BM25Plus)
    scores = model.get_scores(["TERM"])
    expected = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:2]
    assert [index for index, _ in model.get_top_k(["term"], 2)] == expected == [3, 4]
    assert model.get_top_k(["absent"], 2) == []
    assert model.get_top_k(["term"], 0) == []


def embedding_fixture(tmp_path):
    db = DatabaseService(tmp_path / "emoji.db")
    plugin = SimpleNamespace(db_service=db, enable_embedding_search=True, plugin_config=SimpleNamespace(
        enable_embedding_search=True, embedding_provider_id="", category_info={}, character_info={}))
    service = EmbeddingService(plugin)
    sdk = SimpleNamespace(initialize=AsyncMock(), insert_batch=AsyncMock(return_value=[1]), delete_documents=AsyncMock(),
                          document_storage=SimpleNamespace(get_documents=AsyncMock(return_value=[])))
    service._faiss_db = sdk
    service._init_faiss = lambda: True
    service._ensure_provider_ready = AsyncMock()
    return service, db, sdk, plugin


@pytest.mark.asyncio
async def test_path_delete_reuses_framework_filter_and_propagates_failure(tmp_path):
    service, db, sdk, _ = embedding_fixture(tmp_path)
    db.delete_embedding = Mock(side_effect=AssertionError("wrong backend"))
    assert await service.delete_by_path("a")
    sdk.delete_documents.assert_awaited_once_with(metadata_filters={"path": "a"})
    sdk.delete_documents.side_effect = RuntimeError("vector deletion failed")
    with pytest.raises(RuntimeError):
        await service.delete_by_path("a")


@pytest.mark.asyncio
async def test_active_faiss_failure_does_not_write_a_different_backend(tmp_path):
    service, _, sdk, _ = embedding_fixture(tmp_path)
    service._fallback_insert = AsyncMock(side_effect=AssertionError("backend switch"))
    assert await service.insert_emoji("a", {"category": "happy", "desc": "hello"})
    assert sdk.insert_batch.await_args.kwargs["metadatas"] == [{"path": "a", "category": "happy"}]
    sdk.insert_batch.side_effect = RuntimeError("provider failed")
    assert not await service.insert_emoji("a", {"category": "happy", "desc": "hello"})
    sdk.insert_batch.side_effect = None
    sdk.insert_batch.return_value = []
    assert not await service.insert_emoji("a", {"desc": "hello"})
    service._fallback_insert.assert_not_awaited()


@pytest.mark.asyncio
async def test_late_faiss_client_initializes_once_before_concurrent_operations(tmp_path):
    service, _, sdk, _ = embedding_fixture(tmp_path)
    await asyncio.gather(service.delete_by_path("a"), service.insert_emoji("b", {"desc": "hello"}))
    sdk.initialize.assert_awaited_once()
    sdk.delete_documents.assert_awaited_once()
    sdk.insert_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_reuses_sdk_batch_and_preserves_concurrent_update(tmp_path):
    service, db, sdk, _ = embedding_fixture(tmp_path)
    await db.insert_batch([{"path": str(i), "hash": str(i), "category": "happy", "desc": "old"} for i in range(5)])
    async def insert(**kwargs):
        if kwargs["metadatas"][0]["path"] == "0":
            await db.update_path("0", {"desc": "changed during request"})
        return list(range(len(kwargs["contents"])))
    sdk.insert_batch.side_effect = insert
    assert await service.backfill_existing(batch_size=2) == 5
    assert sdk.insert_batch.await_count == 3
    assert all(call.kwargs["tasks_limit"] == 1 for call in sdk.insert_batch.await_args_list)
    jobs = db.get_search_refresh_jobs()
    assert len(jobs) == 1 and jobs[0]["path"] == "0"
    assert db.get_emoji("0")["desc"] == "changed during request"


@pytest.mark.asyncio
async def test_failed_backfill_cleans_partial_documents_and_queues_retry(tmp_path):
    service, db, sdk, _ = embedding_fixture(tmp_path)
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy"}])
    sdk.insert_batch.side_effect = RuntimeError("partial batch failure")
    assert await service.backfill_existing() == 0
    sdk.delete_documents.assert_awaited_once_with(metadata_filters={"path": "a"})
    assert db.get_search_refresh_jobs()[0]["path"] == "a"
    assert db.get_emoji("a") is not None


@pytest.mark.asyncio
async def test_vector_delete_failure_is_retained_in_durable_queue(tmp_path):
    service, db, sdk, plugin = embedding_fixture(tmp_path)
    await db.insert_batch([{"path": "a", "hash": "h", "category": "happy"}])
    sdk.delete_documents.side_effect = RuntimeError("delete failed")
    plugin.meme_selector = SimpleNamespace(_smart_select_service=SimpleNamespace(_embedding_service=service))
    assert await IndexManager(plugin).drain_search_refresh() == 1
    with db._get_connection() as conn:
        job = conn.execute("SELECT * FROM search_refresh WHERE path='a'").fetchone()
    assert job["attempts"] == 1 and "delete failed" in job["last_error"]
    sdk.insert_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_worker_reuses_existing_task_scheduler():
    scheduler = TaskScheduler()
    manager = IndexManager(SimpleNamespace(task_scheduler=scheduler))
    manager.start_search_worker()
    first = manager._search_task
    assert scheduler._tasks["stealer_search_refresh"] is first
    manager.start_search_worker()
    assert manager._search_task is first
    await asyncio.sleep(0)
    await manager.stop_search_worker()
    assert "stealer_search_refresh" not in scheduler._tasks
    assert not manager.search_worker_running()
