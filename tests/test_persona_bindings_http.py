"""角色管理人格绑定的真实 HTTP 与配置写回。"""

from types import SimpleNamespace

import pytest
from quart import Quart

import astrbot_plugin_stealer.core.config.config as config_module
from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.db.database_service import DatabaseService
from astrbot_plugin_stealer.plugin_api import PluginAPI


@pytest.mark.asyncio
async def test_persona_binding_persists_and_can_override_same_name(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "StarTools", SimpleNamespace(get_data_dir=lambda _: tmp_path))
    backing = {"persona_character_map": {"old_persona": "atri"}}
    cfg = PluginConfig(backing)
    assert cfg.update_config({"characters": ["atri", "kipfel"], "character_info": {}})
    db = DatabaseService(tmp_path / "test.db")
    manager = SimpleNamespace(personas_v3=[{"name": "atri", "prompt": "secret"}, {"name": "fun", "prompt": "secret"}])
    plugin = SimpleNamespace(plugin_config=cfg, db_service=db, context=SimpleNamespace(persona_manager=manager))
    app = Quart(__name__)
    PluginAPI(plugin).register(SimpleNamespace(register_web_api=lambda path, handler, methods, desc:
        app.add_url_rule(path, endpoint=handler.__name__, view_func=handler, methods=methods)))
    client = app.test_client()
    url = "/astrbot_plugin_stealer/characters/personas"
    response = await client.get(url)
    data = await response.get_json()
    assert data == {"success": True, "personas": ["atri", "fun"], "mapping": {"old_persona": "atri"}, "isolation": False}
    assert "secret" not in str(data)
    assert (await client.post(url, json={"isolation": True})).status_code == 200
    assert cfg.persona_character_isolation is True
    assert (await client.post(url, json={"isolation": "true"})).status_code == 400
    assert (await client.post(url, json={"persona_id": "fun", "character": "kipfel"})).status_code == 200
    assert (await client.post(url, json={"persona_id": "atri", "character": ""})).status_code == 200
    assert PluginConfig(backing).persona_character_map == {"old_persona": "atri", "fun": "kipfel", "atri": ""}
    assert (await client.post(url, json={"persona_id": "missing", "character": "atri"})).status_code == 400
    assert (await client.post(url, json={"persona_id": "fun", "character": "missing"})).status_code == 400
    monkeypatch.setattr(PluginConfig, "update_config", lambda self, _: False)
    failed = await client.post(url, json={"persona_id": "fun", "character": "atri"})
    assert failed.status_code == 500
    assert cfg.persona_character_map["fun"] == "kipfel"
