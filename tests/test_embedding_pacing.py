import asyncio
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock

import pytest

from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.search.embedding_service import EmbeddingService
from core.search.embedding_provider_adapter import EmbeddingProviderAdapter
from core.util.model_requests import ModelRequestGate, model_request_context, retry_after


def service_fixture(tmp_path, dimension=99):
    db = DatabaseService(tmp_path / 'emoji.db')
    model = ['model-a']
    provider = SimpleNamespace(provider_config={'id': 'embedding'}, get_dim=lambda: dimension,
                               get_model=lambda: model[0], get_embedding=AsyncMock(return_value=[1., 0.]))
    service = EmbeddingService(SimpleNamespace(db_service=db, enable_embedding_search=True,
        plugin_config=SimpleNamespace(enable_embedding_search=True, embedding_provider_id='', model_request_interval=0, category_info={}, character_info={})))
    service._get_provider = lambda: provider
    service._init_faiss = lambda: False
    return service, db, provider, model


@pytest.mark.asyncio
async def test_actual_dimensions_override_incorrect_provider_config(tmp_path):
    service, db, provider, _ = service_fixture(tmp_path)
    await db.insert_batch([{'path': 'a', 'hash': 'a', 'category': 'happy'}])
    assert await service.insert_emoji('a', {'desc': 'hello'})
    assert service._provider_dim == 2 and provider.get_dim() == 99
    assert db.load_embeddings_by_sig(service._fallback_signature(provider))[0]['dim'] == 2
    assert (await service.search('hello'))[0][0] == 'a'


@pytest.mark.asyncio
async def test_faiss_receives_observed_dimension_and_isolated_paths(tmp_path, monkeypatch):
    service, _, provider, _ = service_fixture(tmp_path, dimension=0)
    service.plugin.base_dir = tmp_path
    created = []
    module = ModuleType('astrbot.core.db.vec_db.faiss_impl.vec_db')
    def fake(db_path, index_path, adapter):
        created.append((db_path, index_path, adapter.get_dim()))
        return SimpleNamespace()
    module.FaissVecDB = fake
    monkeypatch.setitem(sys.modules, 'faiss', ModuleType('faiss'))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    service._last_enable_embedding_search = True
    service._last_embedding_provider_id = ''
    await service._ensure_provider_ready()
    assert EmbeddingService._init_faiss(service)
    assert created[0][2] == 2 and 'emoji_faiss_' in created[0][0]
    assert EmbeddingProviderAdapter(service, provider).get_dim() == 2


@pytest.mark.asyncio
async def test_output_dimension_change_preserves_old_vectors_and_queues_rebuild(tmp_path):
    service, db, provider, _ = service_fixture(tmp_path, dimension=2)
    await db.insert_batch([{'path': 'a', 'hash': 'a', 'category': 'happy'}])
    assert await service._fallback_insert('a', 'hello')
    old_sig = service._fallback_signature(provider)
    provider.get_embedding.return_value = [1., 0., 0.]
    assert not await service._fallback_insert('a', 'hello')
    assert db.load_embeddings_by_sig(old_sig)
    assert service._effective_dim(provider) == 3
    assert db.get_search_refresh_jobs()[0]['path'] == 'a'
    assert await service._fallback_insert('a', 'hello')
    assert db.load_embeddings_by_sig(service._fallback_signature(provider))[0]['dim'] == 3


@pytest.mark.asyncio
async def test_shared_gate_serializes_batch_and_online_calls(tmp_path):
    service, _, provider, _ = service_fixture(tmp_path, dimension=2)
    active, maximum = 0, 0
    async def embed(_text):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(.01)
        active -= 1
        return [1., 0.]
    provider.get_embedding.side_effect = embed
    adapter = EmbeddingProviderAdapter(service, provider)
    await asyncio.gather(adapter.get_embeddings(['a', 'b', 'c']), adapter.get_embedding('query'))
    assert maximum == 1


@pytest.mark.asyncio
async def test_retry_after_cools_other_requests_without_sleeping_for_real(monkeypatch):
    import core.util.model_requests as module
    now = [0.]
    waits = []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    async def sleep(delay):
        waits.append(delay)
        now[0] += delay
    monkeypatch.setattr(module.asyncio, 'sleep', sleep)
    gate = ModelRequestGate(SimpleNamespace(plugin_config=SimpleNamespace(model_request_interval=5)))
    error = RuntimeError('429 limit')
    error.response = SimpleNamespace(headers={'Retry-After': '12'}, status_code=429)
    with pytest.raises(RuntimeError):
        await gate.call('same', AsyncMock(side_effect=error))
    await gate.call('same', AsyncMock(return_value='ok'))
    await gate.call('same', AsyncMock(return_value='ok'))
    assert waits == [12, 5]
    assert retry_after(error) == 12


@pytest.mark.asyncio
async def test_adapter_does_not_retry_permanent_errors(tmp_path):
    service, _, provider, _ = service_fixture(tmp_path)
    provider.get_embedding.side_effect = ValueError('invalid API key')
    with pytest.raises(ValueError):
        await EmbeddingProviderAdapter(service, provider).get_embedding('a')
    provider.get_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_with_same_model_does_not_requeue_entire_library(tmp_path):
    service, db, provider, _ = service_fixture(tmp_path, dimension=2)
    await db.insert_batch([{'path': 'a', 'hash': 'a', 'category': 'happy'}])
    await service._ensure_provider_ready()
    job = db.get_search_refresh_jobs()[0]
    db.finish_search_refresh('a', job['generation'])
    service._observed_identity = ''
    await service._ensure_provider_ready()
    assert db.get_search_refresh_jobs() == []


@pytest.mark.asyncio
async def test_cold_backend_generations_are_captured_before_refresh(tmp_path):
    db = DatabaseService(tmp_path / 'emoji.db')
    await db.insert_batch([{'path': 'a', 'hash': 'a', 'category': 'happy'}])
    state = {'ready': False, 'created': False}
    def available():
        if state['ready'] and not state['created']:
            state['created'] = True
            db.enqueue_search_refresh(['a'])
        return True
    async def prepare():
        state['ready'] = True
    async def delete(path):
        available()
    plugin = SimpleNamespace(db_service=db, plugin_config=SimpleNamespace(enable_embedding_search=True))
    embedding = EmbeddingService(plugin)
    embedding.is_available = available
    embedding._ensure_provider_ready = prepare
    embedding.delete_by_path = delete
    embedding.insert_emoji = AsyncMock(return_value=True)
    plugin.meme_selector = SimpleNamespace(embedding_service=embedding)
    assert await IndexManager(plugin).drain_search_refresh() == 1
    assert db.get_search_refresh_jobs() == []
    embedding.insert_emoji.assert_awaited_once()


def test_missing_retry_headers_use_default_cooldown():
    error = RuntimeError('429')
    error.headers = None
    assert retry_after(error) == 30


@pytest.mark.asyncio
async def test_gate_allows_parallel_background_and_prioritizes_realtime():
    gate = ModelRequestGate(SimpleNamespace(plugin_config=SimpleNamespace(model_request_interval=0)))
    order = []
    both_bg_running = asyncio.Event()
    release_first_wave = asyncio.Event()
    active = 0
    max_active = 0

    async def bg_op(name, hold=False):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        order.append(f'start:{name}')
        if active == 2:
            both_bg_running.set()
        if hold:
            await release_first_wave.wait()
        active -= 1
        return name

    async def run_bg(name, hold=False):
        with model_request_context(concurrency=2, priority=1):
            return await gate.call('vision', lambda: bg_op(name, hold=hold))

    t1 = asyncio.create_task(run_bg('bg1', hold=True))
    t2 = asyncio.create_task(run_bg('bg2', hold=True))
    await asyncio.wait_for(both_bg_running.wait(), timeout=1.0)
    assert max_active == 2

    t3 = asyncio.create_task(run_bg('bg3', hold=False))
    await asyncio.sleep(0)
    t_urgent = asyncio.create_task(gate.call('vision', lambda: bg_op('urgent', hold=False)))
    await asyncio.sleep(0)

    release_first_wave.set()
    await asyncio.gather(t1, t2, t3, t_urgent)
    assert order.index('start:urgent') < order.index('start:bg3')


@pytest.mark.asyncio
async def test_gate_staggers_starts_with_nonzero_interval_propagates_429_and_releases_cancelled_waiters():
    """验证非零发车间隔下仍支持流水线重叠并发、429 Retry-After 全局退避，以及排队取消不泄漏并发槽位。"""
    import time

    gate = ModelRequestGate(SimpleNamespace(plugin_config=SimpleNamespace(model_request_interval=0.04)))
    starts = {}
    active = 0
    max_active = 0

    class RateLimitedError(RuntimeError):
        status_code = 429
        headers = {'Retry-After': '0.09'}

    async def op(name, duration=0.10, fail_429=False):
        nonlocal active, max_active
        starts[name] = time.monotonic()
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(duration)
            if fail_429:
                raise RateLimitedError('rate limit')
            return name
        finally:
            active -= 1

    async def run_bg(name, **kwargs):
        with model_request_context(concurrency=2, priority=1):
            return await gate.call('vision', lambda: op(name, **kwargs))

    w1 = asyncio.create_task(run_bg('w1', duration=0.11))
    w2 = asyncio.create_task(run_bg('w2', duration=0.05, fail_429=True))
    await asyncio.sleep(0.01)
    # 排队中的第 3 个任务在中途取消，不应占用或泄漏槽位
    cancelled_waiter = asyncio.create_task(run_bg('cancelled', duration=0.01))
    await asyncio.sleep(0.01)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    w3 = asyncio.create_task(run_bg('w3', duration=0.01))
    assert await w1 == 'w1'
    with pytest.raises(RateLimitedError):
        await w2
    assert await w3 == 'w3'

    # w2 遵守 0.04s 最小发车间隔，且在 w1 尚未结束时已并发在途
    assert starts['w2'] - starts['w1'] >= 0.035
    assert max_active == 2
    # w2 抛出 Retry-After=0.09 后，w3 必须等待退避窗口结束才启动
    assert starts['w3'] - starts['w2'] >= 0.12
    assert gate._state('vision').in_flight == 0
