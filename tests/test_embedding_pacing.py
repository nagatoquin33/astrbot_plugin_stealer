import asyncio
import sys
from types import SimpleNamespace, ModuleType
from unittest.mock import AsyncMock

import pytest

from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.search.embedding_service import EmbeddingService
from core.search.embedding_provider_adapter import EmbeddingProviderAdapter
from core.util.model_requests import ModelRequestGate, retry_after


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
    embedding = SimpleNamespace(_embedding_enabled=lambda: True, is_available=available,
        _ensure_provider_ready=prepare, delete_by_path=delete, insert_emoji=AsyncMock(return_value=True))
    plugin = SimpleNamespace(db_service=db, meme_selector=SimpleNamespace(
        _smart_select_service=SimpleNamespace(_embedding_service=embedding)))
    assert await IndexManager(plugin).drain_search_refresh() == 1
    assert db.get_search_refresh_jobs() == []
    embedding.insert_emoji.assert_awaited_once()


def test_missing_retry_headers_use_default_cooldown():
    error = RuntimeError('429')
    error.headers = None
    assert retry_after(error) == 30
