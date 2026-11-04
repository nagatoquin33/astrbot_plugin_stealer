"""会话人格、角色归属与发送范围测试。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.events.persona_library import prepare_persona_scope
from core.search.meme_scope_service import MemeScopeService


@pytest.fixture
def env():
    cfg = SimpleNamespace(
        persona_character_isolation=True, enable_embedding_search=False, embedding_provider_id="",
        persona_character_map={"bot_fun": "revenant"},
        characters=[], character_info={},
        get_event_target=lambda event: ("group", event.group),
    )
    cfg.get_characters = lambda: cfg.characters
    def update(changes):
        for key, value in changes.items():
            setattr(cfg, key, value)
        return True
    cfg.update_config = update
    conv = SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value="c"),
                           get_conversation=AsyncMock(return_value=SimpleNamespace(persona_id="bound")))
    manager = SimpleNamespace(resolve_selected_persona=AsyncMock(return_value=("bot_fun", {}, None, False)))
    plugin = SimpleNamespace(plugin_config=cfg, context=SimpleNamespace(conversation_manager=conv, persona_manager=manager))
    event = SimpleNamespace(unified_msg_origin="adapter:group:1", group="1", get_platform_name=lambda: "onebot")
    return SimpleNamespace(plugin=plugin, event=event, cfg=cfg, manager=manager)


@pytest.mark.asyncio
async def test_persona_mapping_and_scope(env):
    await prepare_persona_scope(env.plugin, env.event)
    env.manager.resolve_selected_persona.assert_awaited_with(
        umo="adapter:group:1", conversation_persona_id="bound", platform_name="onebot"
    )
    scope = MemeScopeService(env.plugin)
    assert scope._is_entry_allowed_for_event({"character": "revenant"}, env.event)
    assert scope._is_entry_allowed_for_event({"character": ""}, env.event)
    assert not scope._is_entry_allowed_for_event({"character": "other"}, env.event)
    assert not scope._is_entry_allowed_for_event({"character": "revenant", "scope_mode": "local", "origin_target": "group:2"}, env.event)
    env.manager.resolve_selected_persona.return_value = ("other", {}, None, False)
    await prepare_persona_scope(env.plugin, env.event)
    assert not scope._is_entry_allowed_for_event({"character": "revenant"}, env.event)
    assert scope._is_entry_allowed_for_event({"character": "other"}, env.event)
    env.cfg.persona_character_isolation = False
    assert scope._is_entry_allowed_for_event({"character": "revenant"}, env.event)


@pytest.mark.asyncio
async def test_missing_persona_is_general(env):
    env.manager.resolve_selected_persona.return_value = ("[%None]", None, None, False)
    await prepare_persona_scope(env.plugin, env.event)
    assert not MemeScopeService(env.plugin)._is_entry_allowed_for_event({"character": "revenant"}, env.event)
    env.manager.resolve_selected_persona.side_effect = RuntimeError("not available")
    await prepare_persona_scope(env.plugin, env.event)
    assert not MemeScopeService(env.plugin)._is_entry_allowed_for_event({"character": "revenant"}, env.event)
    assert MemeScopeService(env.plugin)._is_entry_allowed_for_event({"character": ""}, env.event)


@pytest.mark.asyncio
async def test_search_ranks_current_character_and_hides_other(env, tmp_path):
    from core.search.meme_selector import MemeSelector

    env.cfg.get_keyword_map = lambda: {}
    env.cfg.category_info = {}
    env.cfg.character_info = {}
    env.cfg.categories = ["happy"]
    paths = [tmp_path / name for name in ("shared.png", "current.png", "other.png")]
    for path in paths:
        path.write_bytes(b"image")
    index = {
        str(path): {"category": "happy", "desc": "开心的猫咪大笑", "tags": ["猫咪"], "character": character}
        for path, character in zip(paths, ("", "revenant", "other"))
    }
    selector = MemeSelector(env.plugin)
    results = await selector.smart_search("开心的猫咪大笑", limit=5, idx=index, event=env.event)
    result_paths = [item[0] for item in results]
    assert result_paths == [str(paths[1])]
    env.manager.resolve_selected_persona.return_value = ("other", {}, None, False)
    results = await selector.smart_search("开心的猫咪大笑", limit=5, idx=index, event=env.event)
    result_paths = [item[0] for item in results]
    assert result_paths == [str(paths[2])]


@pytest.mark.asyncio
async def test_send_rechecks_persona_after_search(env, tmp_path):
    from core.search.meme_selector import MemeSelector

    path = tmp_path / "meme.png"
    path.write_bytes(b"image")
    env.plugin.db_service = SimpleNamespace(get_emoji=lambda _: {"character": "revenant"})
    env.plugin.is_send_enabled_for_event = lambda event: True
    state = SimpleNamespace(is_active_sent=lambda: False, is_auto_claimed=lambda: False)
    env.plugin._emoji_sender_engine = SimpleNamespace(emoji_turn_state=lambda event: state)
    selector = MemeSelector(env.plugin)
    selector.send_emoji_message = AsyncMock(return_value="image")
    await prepare_persona_scope(env.plugin, env.event)
    env.manager.resolve_selected_persona.return_value = ("other", {}, None, False)
    assert not await selector.send_emoji_with_text(env.event, str(path), "回复")
    selector.send_emoji_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_large_general_pool_does_not_displace_character(env, tmp_path):
    from core.search.meme_selector import MemeSelector

    env.cfg.get_keyword_map = lambda: {}
    env.cfg.category_info = {}
    env.cfg.categories = ["happy"]
    index = {}
    for number in range(100):
        path = tmp_path / f"general{number}.png"
        path.touch()
        index[str(path)] = {"category": "happy", "desc": "开心的猫咪大笑", "tags": ["猫咪"], "character": ""}
    for number in range(55):
        path = tmp_path / f"unrelated{number}.png"
        path.touch()
        index[str(path)] = {"category": "sad", "desc": "伤心哭泣", "tags": ["伤心"], "character": "revenant"}
    special = tmp_path / "special.png"
    special.touch()
    index[str(special)] = {"category": "happy", "desc": "开心的猫咪大笑", "tags": ["猫咪"], "character": "revenant"}
    selector = MemeSelector(env.plugin)
    found = await selector.smart_search("开心的猫咪大笑", limit=5, idx=index, event=env.event)
    assert [result[0] for result in found] == [str(special)]
    index.pop(str(special))
    for path in [path for path, data in index.items() if data["character"]]:
        index.pop(path)
    found = await selector.smart_search("开心的猫咪大笑", limit=5, idx=index, event=env.event)
    assert found and all(result[0] in index for result in found)


@pytest.mark.asyncio
async def test_random_selection_prefers_character_across_categories(env, tmp_path):
    from core.search.meme_selector import MemeSelector

    env.cfg.categories = ["happy", "sad"]
    env.cfg.category_info = {}
    env.cfg.smart_meme_selection = False
    shared, special = tmp_path / "shared.png", tmp_path / "special.png"
    shared.touch()
    special.touch()
    index = {str(shared): {"category": "happy", "character": ""},
             str(special): {"category": "sad", "character": "revenant"}}
    env.plugin.db_service = SimpleNamespace(get_index_cache_readonly=lambda: index)
    selector = MemeSelector(env.plugin)
    found = await selector.select_emoji("happy", event=env.event, extra_categories=["happy", "sad"])
    assert found == str(special)
    index.pop(str(special))
    found = await selector.select_emoji("happy", event=env.event, extra_categories=["happy", "sad"])
    assert found == str(shared)


@pytest.mark.asyncio
async def test_jev_tries_character_then_general_when_rejected(env, tmp_path):
    from core.search.meme_selector import MemeSelector

    shared, special = tmp_path / "shared.png", tmp_path / "special.png"
    shared.touch()
    special.touch()
    index = {str(shared): {"category": "happy", "character": ""},
             str(special): {"category": "happy", "character": "revenant"}}
    env.plugin.db_service = SimpleNamespace(get_entries=lambda paths: {p: index[p] for p in paths})
    env.plugin.is_send_enabled_for_event = lambda event: True
    selector = MemeSelector(env.plugin)
    selector._jev_selector.get_history = AsyncMock(return_value=[])
    selector._jev_selector.select = AsyncMock(side_effect=[None, str(shared)])
    selector._smart_select_service._rank_emoji_candidates = AsyncMock(return_value=[
        (str(shared), 0.9, 0, 0, 0, "happy"), (str(special), 0.6, 0, 0, 0, "happy")])
    assert await selector.select_emoji_with_jev(env.event, "回复") == str(shared)
    assert [call.args[0][0][0] for call in selector._jev_selector.select.await_args_list] == [str(special), str(shared)]


@pytest.mark.asyncio
async def test_menu_random_prefers_unseen_character_then_shared(env):
    from core.commands.meme_menu_command import MemeMenuCommand

    env.event._stealer_persona_character = "revenant"
    menu = MemeMenuCommand(env.plugin)
    menu._visible_index = AsyncMock(return_value={
        "shared": {"character": ""}, "special": {"character": "revenant"},
    })
    menu._send_path = AsyncMock()
    session = SimpleNamespace(sent=set())
    await menu._random_send(env.event, session)
    assert menu._send_path.await_args.args[2] == "special"
    session.sent.add("special")
    await menu._random_send(env.event, session)
    assert menu._send_path.await_args.args[2] == "shared"
