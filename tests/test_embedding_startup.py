"""嵌入服务在核心 Provider 加载前后的行为回归测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.search import embedding_service as embedding_service_module
from core.search.embedding_service import EmbeddingService


def _service(context, provider_id="dashscope_embedding"):
    plugin = SimpleNamespace(
        plugin_config=SimpleNamespace(
            enable_embedding_search=True,
            embedding_provider_id=provider_id,
        ),
        context=context,
    )
    return EmbeddingService(plugin)


def _provider(provider_id="dashscope_embedding", dimension=1024):
    return SimpleNamespace(
        provider_config={"id": provider_id},
        get_embedding=lambda *args, **kwargs: None,
        get_dim=lambda: dimension,
    )


def test_provider_lookup_stays_silent_before_core_loaded():
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_provider_by_id=Mock(side_effect=AssertionError("core is not ready yet")),
        get_all_embedding_providers=Mock(return_value=[]),
    )
    service = _service(context)

    assert service.providers_loaded() is False
    assert service._get_provider() is None
    context.get_provider_by_id.assert_not_called()
    context.get_all_embedding_providers.assert_not_called()


def test_provider_lookup_resumes_after_core_loaded():
    provider = _provider()
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_provider_by_id=Mock(return_value=provider),
        get_all_embedding_providers=Mock(return_value=[provider]),
    )
    service = _service(context)

    assert service._get_provider() is None
    context.get_provider_by_id.assert_not_called()

    service.mark_providers_loaded()

    assert service._get_provider() is provider
    context.get_provider_by_id.assert_called_once_with("dashscope_embedding")


def test_reloaded_core_is_treated_as_ready_without_marking():
    provider = _provider()
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={"dashscope_embedding": provider}),
        get_provider_by_id=Mock(side_effect=AssertionError("inst_map already has it")),
        get_all_embedding_providers=Mock(return_value=[provider]),
    )
    service = _service(context)

    assert service.providers_loaded() is True
    assert service._get_provider() is provider


def test_provider_lookup_falls_back_to_public_api_when_id_missing():
    provider = _provider("other", 8)
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={"chat": object()}),
        get_provider_by_id=Mock(return_value=provider),
        get_all_embedding_providers=Mock(return_value=[]),
    )
    service = _service(context)
    service.mark_providers_loaded()

    assert service._get_provider() is provider
    context.get_provider_by_id.assert_called_once_with("dashscope_embedding")


def test_faiss_probe_before_core_loaded_does_not_cache_unavailable():
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_provider_by_id=Mock(return_value=None),
        get_all_embedding_providers=Mock(return_value=[]),
    )
    service = _service(context)

    assert service._init_faiss() is False
    assert service._faiss_available is None
    context.get_all_embedding_providers.assert_not_called()


def _recording_logger(records):
    return SimpleNamespace(
        info=lambda message, *args, **kwargs: records.append(message),
        warning=lambda message, *args, **kwargs: records.append(message),
        error=lambda message, *args, **kwargs: records.append(message),
        debug=lambda *args, **kwargs: None,
    )


@pytest.mark.asyncio
async def test_initialize_before_core_loaded_keeps_logs_quiet(monkeypatch):
    records = []
    monkeypatch.setattr(
        embedding_service_module, "logger", _recording_logger(records)
    )
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={}),
        get_provider_by_id=Mock(side_effect=AssertionError("core is not ready yet")),
        get_all_embedding_providers=Mock(return_value=[]),
    )
    service = _service(context)

    await service.initialize()

    assert records == []
    context.get_provider_by_id.assert_not_called()
    context.get_all_embedding_providers.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_after_core_loaded_reports_missing_provider(monkeypatch):
    records = []
    monkeypatch.setattr(
        embedding_service_module, "logger", _recording_logger(records)
    )
    context = SimpleNamespace(
        provider_manager=SimpleNamespace(inst_map={"chat": object()}),
        get_provider_by_id=Mock(return_value=None),
        get_all_embedding_providers=Mock(return_value=[]),
    )
    service = _service(context)
    service.mark_providers_loaded()

    await service.initialize()

    assert any("请在 AstrBot 后台配置 Embedding 模型" in message for message in records)
