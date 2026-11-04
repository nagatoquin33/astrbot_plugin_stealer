import asyncio
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from astrbot_plugin_stealer.core.config import config as config_module
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.core.processing.image_processor_service import ImageProcessorService
from astrbot_plugin_stealer.core.processing.pending_analysis_worker import PendingAnalysisWorker
from astrbot_plugin_stealer.plugin_api import PluginAPI
from astrbot_plugin_stealer.task_scheduler import TaskScheduler


@pytest.fixture
def review(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, 'StarTools', SimpleNamespace(get_data_dir=lambda _: tmp_path))
    cfg = config_module.PluginConfig({'vision_provider_id': 'vision', 'model_request_interval': 0})
    db = DatabaseService(tmp_path / 'emoji.db')
    plugin = SimpleNamespace(plugin_config=cfg, db_service=db, task_scheduler=TaskScheduler(),
                             context=SimpleNamespace(llm_generate=AsyncMock()))
    processor = ImageProcessorService(plugin)
    plugin.image_processor_service = processor
    worker = PendingAnalysisWorker(plugin)
    plugin.pending_analysis_worker = worker
    return SimpleNamespace(plugin=plugin, db=db, processor=processor, worker=worker, api=PluginAPI(plugin), root=tmp_path)


async def collect(review, name='image', color='red'):
    path = review.root / f'{name}.png'
    Image.new('RGB', (32, 32), color).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ok, index = await review.processor.process_image(None, str(path), is_temp=True, to_pending=True,
        extra_meta={'source': 'auto', 'scope_mode': 'local', 'origin_target': 'group:1'})
    assert ok and index is None
    return review.db.get_pending_by_hash(digest)['id']


@pytest.mark.asyncio
async def test_collection_and_rejection_never_resolve_or_call_vision(review):
    review.processor._vlm_call_service._resolve_vision_provider = AsyncMock(side_effect=AssertionError('model probe'))
    pid = await collect(review)
    row = review.db.get_pending(pid)
    assert row['analysis_state'] == 'unanalysed' and row['category'] == ''
    assert row['width'] == 32 and row['scope_mode'] == 'local'
    assert review.db.count_total() == 0
    review.plugin.context.llm_generate.assert_not_awaited()
    assert review.db.delete_pending(pid)


@pytest.mark.asyncio
async def test_approval_returns_before_slow_vision_and_is_idempotent(review):
    pid = await collect(review)
    started, release = asyncio.Event(), asyncio.Event()
    async def classify(*args, **kwargs):
        started.set()
        await release.wait()
        return 'happy', ['hello'], 'description', 'happy', ['greeting'], 'text', ['happy']
    review.processor.classify_image = AsyncMock(side_effect=classify)
    assert await asyncio.wait_for(review.api._approve_one(review.db, pid), .5) == (True, 'queued')
    assert await review.api._approve_one(review.db, pid) == (False, 'already queued')
    task = asyncio.create_task(review.worker.process_one())
    await asyncio.wait_for(started.wait(), 1)
    assert review.db.get_pending(pid)['review_status'] == 'processing'
    assert review.db.delete_pending_batch([pid]) == []
    assert await review.db.update_pending(pid, {'desc': 'race'}) is None
    assert review.db.count_total() == 0
    release.set()
    await task
    assert review.db.get_pending(pid) is None
    entries = review.db.get_index_cache_readonly()
    path, entry = next(iter(entries.items()))
    assert Path(path).is_file() and entry['tags'] == ['hello']
    assert entry['scope_mode'] == 'local' and entry['origin_target'] == 'group:1'


@pytest.mark.asyncio
async def test_failure_retains_original_and_retry_succeeds_with_manual_fields(review):
    pid = await collect(review)
    await review.db.update_pending(pid, {'desc': 'manual description', 'tags': ['manual']})
    review.processor.classify_image = AsyncMock(return_value=('', [], '', '', [], '', []))
    await review.api._approve_one(review.db, pid)
    await review.worker.process_one()
    failed = review.db.get_pending(pid)
    assert failed['review_status'] == 'failed' and Path(failed['path']).exists()
    assert failed['analysis_error'] and review.db.count_total() == 0
    review.processor.classify_image.return_value = ('happy', ['model'], 'model description', 'happy', ['scene'], 'text', ['happy'])
    assert await review.api._approve_one(review.db, pid) == (True, 'queued')
    await review.worker.process_one()
    entry = next(iter(review.db.get_index_cache_readonly().values()))
    assert entry['desc'] == 'manual description' and entry['tags'] == ['manual']
    assert entry['scenes'] == ['scene'] and entry['overlay_text'] == 'text'


@pytest.mark.asyncio
async def test_restart_recovers_claim_and_cancellation_keeps_job(review):
    pid = await collect(review)
    await review.api._approve_one(review.db, pid)
    assert (await review.db.claim_pending_analysis())['id'] == pid
    reopened = DatabaseService(review.db._db_path)
    reopened.recover_pending_analysis()
    assert reopened.get_pending(pid)['review_status'] == 'queued'
    started = asyncio.Event()
    async def classify(*args):
        started.set()
        await asyncio.Event().wait()
    review.processor.classify_image = AsyncMock(side_effect=classify)
    task = asyncio.create_task(review.worker.process_one())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reopened.get_pending(pid)['review_status'] == 'queued'
    assert Path(reopened.get_pending(pid)['path']).is_file()


@pytest.mark.asyncio
async def test_legacy_pending_promotes_without_analysis(review):
    path = review.root / 'legacy.png'
    Image.new('RGB', (32, 32)).save(path)
    pid = await review.db.insert_pending({'path': str(path), 'hash': 'legacy', 'category': 'happy', 'tags': ['old']})
    review.processor.classify_image = AsyncMock(side_effect=AssertionError('legacy should not call model'))
    assert (await review.api._approve_one(review.db, pid))[0]
    assert review.db.get_pending(pid) is None
    assert review.db.count_total() == 1


@pytest.mark.asyncio
async def test_queue_worker_owned_by_existing_scheduler(review):
    await review.worker.start()
    assert 'pending_analysis' in review.plugin.task_scheduler._tasks
    await review.worker.stop()
    assert 'pending_analysis' not in review.plugin.task_scheduler._tasks


@pytest.mark.asyncio
async def test_approval_api_accepts_multiple_jobs_without_calling_model(review, monkeypatch):
    import astrbot_plugin_stealer.api.pending as module
    ids = [await collect(review, 'first'), await collect(review, 'second', 'blue')]
    monkeypatch.setattr(module, 'request', SimpleNamespace(get_json=AsyncMock(return_value={'ids': ids})))
    monkeypatch.setattr(module, 'jsonify', lambda result: result)
    response = await asyncio.wait_for(review.api.handle_pending_approve(), .5)
    assert response == {'success': True, 'approved': 2, 'queued': 2, 'errors': []}
    assert review.db.pending_analysis_stats() == {'queued': 2}
    review.plugin.context.llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_during_sqlite_claim_requeues_committed_job(review, monkeypatch):
    pid = await collect(review)
    await review.api._approve_one(review.db, pid)
    started, release = threading.Event(), threading.Event()
    original = review.db._claim_pending_analysis_sync
    def claim():
        result = original()
        started.set()
        release.wait(5)
        return result
    monkeypatch.setattr(review.db, '_claim_pending_analysis_sync', claim)
    task = asyncio.create_task(review.db.claim_pending_analysis())
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert review.db.get_pending(pid)['review_status'] == 'queued'


@pytest.mark.asyncio
async def test_pending_capacity_is_enforced_at_commit(review):
    review.plugin.plugin_config.steal_pool_capacity = 1
    await collect(review)
    path = review.root / 'overflow.png'
    Image.new('RGB', (32, 32), 'blue').save(path)
    assert (await review.processor.process_image(None, str(path), is_temp=True, to_pending=True))[0] is False
    assert review.db.count_pending() == 1 and path.exists()


@pytest.mark.asyncio
async def test_pending_analysis_discards_emotion_removed_during_model_call(review):
    pending_id = await collect(review)
    started, release = asyncio.Event(), asyncio.Event()
    async def classify(*args, **kwargs):
        started.set()
        await release.wait()
        return 'happy', [], 'description', 'happy', [], '', ['happy', 'sad']
    review.processor.classify_image = AsyncMock(side_effect=classify)
    assert (await review.api._approve_one(review.db, pending_id))[0]
    task = asyncio.create_task(review.worker.process_one())
    await started.wait()
    cfg = review.plugin.plugin_config
    assert cfg.update_config({'categories': [key for key in cfg.categories if key != 'sad']})
    release.set()
    await task
    entry = next(iter(review.db.get_index_cache_readonly().values()))
    assert entry['emotions'] == ['happy']
