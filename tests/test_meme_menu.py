"""选单的原生负载、会话隔离和真实数据库发送约束。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from core.commands.image_mgmt_command import ImageManagementCommand
from core.commands.meme_menu_command import MemeMenuCommand
from core.commands.qq_callback_event import callback_event, source_sender
from core.commands.qq_panel_interactions import BUTTON_PREFIX
from core.commands.panel_actions import PANEL_ACTIONS
from core.commands.qq_menu import PanelButton, build_keyboard, panel_content, panel_rows, send_keyboard
from core.db.database_service import DatabaseService
from core.events.emoji_delivery import EmojiDeliveryUncertain
from core.search.meme_scope_service import MemeScopeService


class Result:
    def __init__(self, text="", image=""):
        self.text, self.image = text, image

    def file_image(self, path):
        self.image = path
        return self


class Event:
    def __init__(self, platform="qq_official", user="alice", session="group:one", admin=False):
        self.platform, self.user, self.admin = platform, user, admin
        self.unified_msg_origin = f"{platform}:{session}"
        self.session = session
        self.messages = []
        self.stopped = False
        self._extras = {}
        self.message_obj = SimpleNamespace(
            raw_message=SimpleNamespace(id="source-message", group_openid="one")
        )
        self.bot = SimpleNamespace(intents=0, api=SimpleNamespace(
            _http=SimpleNamespace(request=AsyncMock(return_value={"id":"reply"})),
            post_group_message=AsyncMock(return_value={"id": "menu-message"}),
            post_dms=AsyncMock(return_value={"id": "dm-message"}),
            post_message=AsyncMock(return_value={"id": "channel-message"}),
        ))
        self.post_c2c_message = AsyncMock(return_value={"id": "c2c-message"})

    def get_platform_name(self):
        return self.platform

    @property
    def admin(self):
        return self.role == "admin"

    @admin.setter
    def admin(self, value):
        self.role = "admin" if value else "member"

    def is_admin(self):
        return self.admin

    def get_extra(self, key, default=None):
        return self._extras.get(key, default)

    def get_sender_id(self):
        return self.user

    def is_private_chat(self):
        return self.session.startswith("user:")

    def plain_result(self, text):
        return Result(text=text)

    def make_result(self):
        return Result()

    def stop_event(self):
        self.stopped = True

    async def send(self, message):
        self.messages.append(message)


@pytest_asyncio.fixture
async def library(tmp_path):
    db = DatabaseService(tmp_path / "menu.db")
    rows = []
    for i in range(12):
        path = tmp_path / f"image-{i:02d}.png"
        path.write_bytes(b"image")
        rows.append({
            "path": str(path), "hash": f"hash-{i}", "category": "happy",
            "desc": f"图片 {i}", "overlay_text": f"下班好开心 {i}",
            "created_at": i, "scope_mode": "public",
        })
    private = tmp_path / "private.png"
    private.write_bytes(b"private-image")
    rows.append({
        "path": str(private), "hash": "private", "category": "happy",
        "created_at": 99, "scope_mode": "local", "origin_target": "group:other",
    })
    await db.insert_batch(rows)
    astrbot_config = {"wake_prefix": ["/"], "admins_id": []}
    plugin = SimpleNamespace(
        astrbot_config=astrbot_config,
        context=SimpleNamespace(
            get_config=lambda **kwargs: astrbot_config,
            platform_manager=SimpleNamespace(get_insts=lambda: []),
        ),
        db_service=db, enabled=True,
        plugin_config=SimpleNamespace(
            get_event_target=lambda event: event.session.split(":", 1),
            steal_meme=True, auto_send_meme=True, enable_natural_emotion_analysis=True,
        ),
        image_render_service=SimpleNamespace(
            render_emoji_list_page_file=AsyncMock(return_value=str(tmp_path / "preview.png"))
        ),
        _emoji_sender_engine=SimpleNamespace(mark_auto_emoji_sent=AsyncMock()),
    )
    scope = MemeScopeService(plugin)

    async def record(path, trigger):
        db.increment_usage_sync(path)

    plugin.is_send_enabled_for_event = lambda event: plugin.enabled
    plugin.meme_selector = SimpleNamespace(
        is_path_allowed_for_event=scope.is_path_allowed_for_event,
        smart_search=AsyncMock(return_value=[]),
        send_emoji_message=AsyncMock(return_value="file_image"),
        record_emoji_usage=AsyncMock(side_effect=record),
    )
    plugin.image_commands = ImageManagementCommand(plugin)
    plugin.menu_commands = MemeMenuCommand(plugin)
    yield plugin
    plugin.menu_commands.close()


def active_session(plugin):
    return next(iter(plugin.menu_commands._sessions.items()))


def commands(rows):
    return ["/" + PANEL_ACTIONS[button.action].command + (" token" if button.action in {"random", "close"} else "") for row in rows for button in row]


@pytest.mark.parametrize("section", ["overview", "settings", "maintenance"])
def test_panel_layout_has_valid_rows_and_no_per_image_sending(section):
    rows = panel_rows(section, True)
    assert 1 <= len(rows) <= 5
    assert all(1 <= len(row) <= 5 for row in rows)
    assert not any(command.startswith(("/meme search", "/meme pick", "/meme page")) for command in commands(rows))
    keyboard = build_keyboard(rows, "alice", "token")["content"]["rows"]
    assert [len(row["buttons"]) for row in keyboard] == [len(row) for row in rows]
    for row in keyboard:
        for button in row["buttons"]:
            assert len(button["render_data"]["label"]) <= 20
            assert button["action"]["type"] == 1
            assert "enter" not in button["action"]
            assert "/" not in button["action"]["data"]
            assert button["action"]["permission"] == {"type": 0, "specify_user_ids": ["alice"]}


def test_member_buttons_only_expose_view_random_and_panel_navigation():
    member_commands = commands(panel_rows("overview", False))
    assert set(member_commands) == {
        "/meme status", "/meme random token", "/meme menu", "/meme close token",
    }
    assert panel_rows("settings", False) == []
    assert panel_rows("maintenance", False) == []
    assert "/meme menu settings" in commands(panel_rows("overview", True))


@pytest.mark.asyncio
@pytest.mark.parametrize("scene", ["group", "c2c", "dm", "channel"])
async def test_keyboard_routes_native_payload_with_current_message(scene):
    event = Event(session="user:alice" if scene in {"c2c", "dm"} else "group:one")
    raw = {"id": "fresh-click-message"}
    if scene == "group":
        raw["group_openid"] = "one"
        method = event.bot.api.post_group_message
    elif scene == "c2c":
        raw["author"] = SimpleNamespace(user_openid="alice")
        method = event.post_c2c_message
    elif scene == "dm":
        raw["guild_id"] = "guild"
        method = event.bot.api.post_dms
    else:
        raw["channel_id"] = "channel"
        method = event.bot.api.post_message
    event.message_obj.raw_message = SimpleNamespace(**raw)
    keyboard = build_keyboard([[PanelButton("运行状态", "status", True)]], "alice", "token")
    await send_keyboard(event, "### 功能面板", keyboard)
    payload = method.await_args.kwargs
    assert payload["msg_id"] == "fresh-click-message"
    assert payload["keyboard"] == keyboard
    assert payload["markdown"] == {"content": "### 功能面板"}
    if scene in {"group", "c2c"}:
        assert payload["msg_type"] == 2
        assert 1 <= payload["msg_seq"] <= 10000


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
@pytest.mark.parametrize("admin", [False, True])
async def test_native_panels_match_current_role_and_state(library, platform, admin):
    event = Event(platform=platform, admin=admin)
    await library.menu_commands.show_panel(event)
    payload = event.bot.api.post_group_message.await_args.kwargs
    content = payload["markdown"]["content"]
    assert "功能面板" in content
    assert "收集：开" in content
    actions = [json.loads(b["action"]["data"])["action"] for row in payload["keyboard"]["content"]["rows"] for b in row["buttons"]]
    assert ("settings" in actions) is admin
    assert "random" in actions
    event.admin = True
    library.plugin_config.steal_meme = False
    await library.menu_commands.show_panel(event, "settings")
    content = event.bot.api.post_group_message.await_args.kwargs["markdown"]["content"]
    assert "收集：关" in content
    assert len(library.menu_commands._sessions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["settings", "maintenance"])
async def test_member_cannot_open_admin_sections_even_by_typing_command(library, section):
    event = Event()
    await library.menu_commands.show_panel(event, section)
    assert "仅限 AstrBot 管理员" in event.messages[-1].text
    assert not library.menu_commands._sessions
    event.bot.api.post_group_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_admin_panel_is_not_available_after_role_is_removed(library):
    event = Event(admin=True)
    await library.menu_commands.show_panel(event, "settings")
    event.admin = False
    event.bot.api.post_group_message.reset_mock()
    await library.menu_commands.show_panel(event, "settings")
    assert "仅限 AstrBot 管理员" in event.messages[-1].text
    event.bot.api.post_group_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "telegram"])
@pytest.mark.parametrize("admin", [False, True])
async def test_plain_fallback_lists_commands_without_privilege_leak(library, platform, admin):
    event = Event(platform=platform, admin=admin)
    event.bot.api.post_group_message.side_effect = RuntimeError("Markdown forbidden")
    section = "maintenance" if admin else "overview"
    await library.menu_commands.show_panel(event, section)
    text = event.messages[-1].text
    assert "功能指令" in text
    assert ("/meme list" in text) is admin
    assert "/meme delete" not in text
    assert "/meme pick" not in text and "/meme search" not in text
    if not admin:
        assert "/meme random" in text
        assert "/meme on" not in text


@pytest.mark.asyncio
async def test_invalid_section_and_empty_native_response_have_clear_fallback(library):
    event = Event()
    await library.menu_commands.show_panel(event, "wrong")
    assert "用法" in event.messages[-1].text
    assert not library.menu_commands._sessions
    event.bot.api.post_group_message.return_value = None
    await library.menu_commands.show_panel(event)
    assert "/meme status" in event.messages[-1].text


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["random", "close"])
@pytest.mark.parametrize("intruder", ["user", "session", "adapter"])
async def test_foreign_user_session_or_adapter_cannot_use_token(library, action, intruder):
    await library.menu_commands.show_panel(Event())
    token, _ = active_session(library)
    event = Event(
        user="bob" if intruder == "user" else "alice",
        session="group:other" if intruder == "session" else "group:one",
        platform="qq_official_webhook" if intruder == "adapter" else "qq_official",
    )
    if action == "random":
        await library.menu_commands.random_image(event, token)
    else:
        await library.menu_commands.dismiss(event, token)
    assert "其他用户或会话" in event.messages[-1].text
    assert token in library.menu_commands._sessions
    library.meme_selector.send_emoji_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_random_works_without_panel_respects_scope_and_deduplicates(library, monkeypatch):
    event = Event()
    offered = []

    def choose(paths):
        offered.append(list(paths))
        return paths[0]

    monkeypatch.setattr("core.commands.meme_menu_command.secrets.choice", choose)
    await library.menu_commands.random_image(event)
    token, session = active_session(library)
    first = library.meme_selector.send_emoji_message.await_args.args[1]
    assert len(offered[0]) == 12
    assert not any("private.png" in path for path in offered[0])
    await library.menu_commands.random_image(event, token)
    assert first not in offered[1]
    assert first in session.sent
    assert library.db_service.get_emoji(first)["use_count"] == 1
    assert library._emoji_sender_engine.mark_auto_emoji_sent.await_count == 2


@pytest.mark.asyncio
async def test_random_failure_can_retry_and_disabled_or_empty_library_does_not_send(library):
    event = Event()
    token = library.menu_commands._session_for(event)
    session = library.menu_commands._sessions[token]
    visible = await library.menu_commands._visible_index(event)
    only = next(iter(visible))
    session.sent.update(path for path in visible if path != only)
    library.meme_selector.send_emoji_message.return_value = None
    await library.menu_commands.random_image(event, token)
    assert only not in session.sent
    assert not session.busy
    library.meme_selector.send_emoji_message.return_value = "file_image"
    await library.menu_commands.random_image(event, token)
    assert only in session.sent
    await library.menu_commands.random_image(event, token)
    assert "暂无未试发" in event.messages[-1].text
    library.enabled = False
    await library.menu_commands.random_image(event, token)
    assert "禁用" in event.messages[-1].text
    assert library.meme_selector.send_emoji_message.await_count == 2


@pytest.mark.asyncio
async def test_random_delivery_timeout_skips_same_image_without_counting(library):
    event = Event(platform="aiocqhttp")
    token = library.menu_commands._session_for(event)
    session = library.menu_commands._sessions[token]
    visible = await library.menu_commands._visible_index(event)
    only = next(iter(visible))
    session.sent.update(path for path in visible if path != only)
    library.meme_selector.send_emoji_message.side_effect = EmojiDeliveryUncertain()

    await library.menu_commands.random_image(event, token)
    assert "可能已送达" in event.messages[-1].text
    await library.menu_commands.random_image(event, token)

    assert "暂无未试发" in event.messages[-1].text
    library.meme_selector.send_emoji_message.assert_awaited_once_with(event, only)
    library.meme_selector.record_emoji_usage.assert_not_awaited()
    library._emoji_sender_engine.mark_auto_emoji_sent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["disabled", "file", "scope", "deleted"])
async def test_random_rechecks_realtime_constraints_after_candidate_load(library, monkeypatch, change):
    event = Event()
    path = next(iter(await library.menu_commands._visible_index(event)))

    async def candidates(_event):
        if change == "disabled":
            library.enabled = False
        elif change == "file":
            from pathlib import Path
            Path(path).unlink()
        elif change == "scope":
            await library.db_service.update_path(path, {"scope_mode": "local", "origin_target": "group:other"})
        else:
            await library.db_service.delete_paths([path])
        return {path: {}}

    monkeypatch.setattr(library.menu_commands, "_visible_index", candidates)
    await library.menu_commands.random_image(event)
    library.meme_selector.send_emoji_message.assert_not_awaited()
    assert event.messages[-1].text


@pytest.mark.asyncio
async def test_busy_state_survives_refresh_and_blocks_concurrent_random(library, monkeypatch):
    event = Event()
    path = next(iter(await library.menu_commands._visible_index(event)))
    started, finish = asyncio.Event(), asyncio.Event()

    async def visible(_event):
        started.set()
        await finish.wait()
        return {path: {}}

    monkeypatch.setattr(library.menu_commands, "_visible_index", visible)
    task = asyncio.create_task(library.menu_commands.random_image(event))
    await started.wait()
    token, session = active_session(library)
    await library.menu_commands.show_panel(event)
    assert active_session(library)[0] == token
    await library.menu_commands.random_image(event)
    assert session.busy
    library.meme_selector.send_emoji_message.assert_not_awaited()
    finish.set()
    await task
    assert not session.busy
    library.meme_selector.send_emoji_message.assert_awaited_once_with(event, path)


@pytest.mark.asyncio
async def test_expiry_dismissal_and_unload_release_sessions(library):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    session.expires_at = 0
    await library.menu_commands.random_image(event, token)
    assert "过期" in event.messages[-1].text
    assert not library.menu_commands._sessions
    library.menu_commands.TTL_SECONDS = 0.01
    await library.menu_commands.show_panel(event)
    await asyncio.sleep(0.03)
    assert not library.menu_commands._owners
    library.menu_commands.TTL_SECONDS = 300
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    await library.menu_commands.dismiss(event, token)
    assert session.expiry.cancelled()
    await library.menu_commands.random_image(event, token)
    assert "已收起" in event.messages[-1].text
    await library.menu_commands.show_panel(event)
    _, session = active_session(library)
    library.menu_commands.close()
    assert session.expiry.cancelled()
    assert not library.menu_commands._sessions


@pytest.mark.asyncio
async def test_delivered_image_is_not_reported_failed_when_usage_update_fails(library):
    event = Event()
    library.meme_selector.record_emoji_usage.side_effect = RuntimeError("database busy")
    await library.menu_commands.random_image(event)
    _, session = active_session(library)
    assert len(session.sent) == 1
    assert not any("发送失败" in message.text for message in event.messages)


@pytest.mark.asyncio
async def test_registered_panel_handlers_delegate(library):
    from astrbot_plugin_stealer.main import Main
    event = Event()
    service = SimpleNamespace(show_panel=AsyncMock(), random_image=AsyncMock(), dismiss=AsyncMock())
    plugin = SimpleNamespace(menu_commands=service)
    await Main.menu_panel(plugin, event, "settings")
    await Main.random_menu_image(plugin, event)
    await Main.close_menu(plugin, event, "token")
    service.show_panel.assert_awaited_once_with(event, "settings")
    service.random_image.assert_awaited_once_with(event, "")
    service.dismiss.assert_awaited_once_with(event, "token")
    assert event.stopped


def test_panel_content_only_uses_current_config_and_fixed_function_descriptions():
    config = SimpleNamespace(steal_meme=False, auto_send_meme=True, enable_natural_emotion_analysis=False)
    content = panel_content("settings", True, config)
    assert "收集：关" in content
    assert "自动发送：开" in content
    assert "原文检索" in content


def click(token, action="status", *, click_id="click-1", user="alice", group="one"):
    return SimpleNamespace(
        id=click_id, event_id="fresh-interaction-event", group_member_openid=user,
        group_openid=group, user_openid=None, channel_id=None, guild_id=None,
        data=SimpleNamespace(resolved=SimpleNamespace(
            button_id=BUTTON_PREFIX + action,
            button_data=json.dumps({"panel": token, "action": action}),
            user_id=user,
        )),
        _api=SimpleNamespace(on_interaction_result=AsyncMock()),
    )


async def drain_callbacks(menu):
    if menu.interactions.tasks:
        await asyncio.gather(*list(menu.interactions.tasks))


@pytest.mark.asyncio
@pytest.mark.parametrize("prefixes", [["!"], ["小九 "], []])
async def test_fallback_follows_configured_prefix_and_native_payload_never_contains_commands(library, prefixes):
    library.astrbot_config["wake_prefix"] = prefixes
    event = Event(platform="telegram", admin=True)
    await library.menu_commands.show_panel(event, "maintenance")
    prefix = prefixes[0] if prefixes else ""
    assert f"{prefix}meme list" in event.messages[-1].text
    assert "/meme" not in event.messages[-1].text
    for row in build_keyboard(panel_rows("maintenance", True), "alice", "token")["content"]["rows"]:
        for button in row["buttons"]:
            assert button["action"]["type"] == 1
            assert "enter" not in button["action"]
            data = json.loads(button["action"]["data"])
            assert data == {"panel": "token", "action": button["id"].removeprefix(BUTTON_PREFIX)}


@pytest.mark.asyncio
async def test_callback_acknowledges_before_direct_service_execution_without_chat_routing(library):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    interaction = click(token)
    seen = []

    async def status(callback):
        interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 0)
        seen.append(callback)
        yield callback.plain_result("状态完成")

    library.command_handler = SimpleNamespace(status=status)
    assert await library.menu_commands.interactions.handle(interaction, event.bot)
    await drain_callbacks(library.menu_commands)
    assert event.messages[-1].text == "状态完成"
    assert seen[0] is not event
    assert seen[0].bot is not event.bot
    assert not session.busy
    assert interaction.id in session.handled


@pytest.mark.asyncio
@pytest.mark.parametrize("case,code", [("user", 4), ("group", 4), ("client", 4), ("expired", 3), ("busy", 2), ("invalid", 1), ("unknown", 1), ("missing_event_id", 1)])
async def test_invalid_or_foreign_callbacks_are_rejected(library, case, code):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    interaction = click(token)
    client = event.bot
    if case == "user":
        interaction.group_member_openid = "bob"
    elif case == "group":
        interaction.group_openid = "other"
    elif case == "client":
        client = SimpleNamespace()
    elif case == "expired":
        session.expires_at = 0
    elif case == "busy":
        session.busy = True
    elif case == "invalid":
        interaction.data.resolved.button_data = "[]"
    elif case == "unknown":
        interaction = click(token, "unknown")
    elif case == "missing_event_id":
        interaction.event_id = None
    assert await library.menu_commands.interactions.handle(interaction, client)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, code)
    assert not library.menu_commands.interactions.tasks


@pytest.mark.asyncio
async def test_callback_admin_actions_recheck_live_roles_and_api_constraints(library):
    event = Event(admin=True)
    await library.menu_commands.show_panel(event, "settings")
    token, session = active_session(library)
    interaction = click(token, "off")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 5)
    from core.commands.command_handler import CommandHandler
    library.command_handler = CommandHandler(library)

    def update(values):
        for key, value in values.items():
            setattr(library.plugin_config, key, value)
        return True

    library.update_config = update
    library.astrbot_config["admins_id"] = ["alice"]
    interaction = click(token, "off", click_id="allowed")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    await drain_callbacks(library.menu_commands)
    assert library.plugin_config.steal_meme is False
    event._extras["_api_key_allow_admin_role"] = False
    interaction = click(token, "on", click_id="api-denied")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 5)
    assert not session.busy


@pytest.mark.asyncio
async def test_random_callback_is_deduplicated_and_close_invalidates_all_buttons(library):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    interaction = click(token, "random")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    await drain_callbacks(library.menu_commands)
    await library.menu_commands.interactions.handle(interaction, event.bot)
    assert interaction._api.on_interaction_result.await_args.args == (interaction.id, 3)
    assert len(session.sent) == 1
    library.meme_selector.send_emoji_message.assert_awaited_once()
    close = click(token, "close", click_id="close")
    await library.menu_commands.interactions.handle(close, event.bot)
    await drain_callbacks(library.menu_commands)
    assert token not in library.menu_commands._sessions
    old = click(token, "status", click_id="old-status")
    await library.menu_commands.interactions.handle(old, event.bot)
    old._api.on_interaction_result.assert_awaited_once_with(old.id, 3)


@pytest.mark.asyncio
async def test_callback_registration_is_idempotent_and_preserves_other_plugins(library):
    event = Event()
    previous = AsyncMock()
    event.bot.on_interaction_create = previous
    router = library.menu_commands.interactions
    router.bind(event.bot)
    ours = event.bot.on_interaction_create
    router.bind(event.bot)
    assert event.bot.on_interaction_create is ours
    assert event.bot.intents & (1 << 26)
    foreign = click("foreign")
    foreign.data.resolved.button_id = "music_song_selection_1"
    await ours(foreign)
    previous.assert_awaited_once_with(foreign)

    async def later_plugin(interaction):
        await ours(interaction)

    event.bot.on_interaction_create = later_plugin
    router.close()
    assert event.bot.on_interaction_create is later_plugin
    await later_plugin(foreign)
    assert previous.await_count == 2


@pytest.mark.asyncio
async def test_unload_restores_handler_and_cancels_callback_jobs(library):
    event = Event()
    previous = AsyncMock()
    event.bot.on_interaction_create = previous
    await library.menu_commands.show_panel(event)
    token, _ = active_session(library)
    started = asyncio.Event()

    async def slow(_event):
        started.set()
        await asyncio.Event().wait()
        yield

    library.command_handler = SimpleNamespace(status=slow)
    await library.menu_commands.interactions.handle(click(token), event.bot)
    await started.wait()
    await library.menu_commands.aclose()
    assert event.bot.on_interaction_create is previous
    assert not library.menu_commands.interactions.tasks


@pytest.mark.asyncio
async def test_real_sdk_reply_requests_use_click_event_without_mutating_original(library):
    api_module = pytest.importorskip("botpy.api")
    from botpy.http import Route
    event = Event()
    http = SimpleNamespace(request=AsyncMock(return_value={"id": "reply"}))
    event.bot.api = api_module.BotAPI(http=http)
    interaction = click("token")
    callback = callback_event(event, interaction, False)
    await callback.bot.api.post_group_message(group_openid="one", content="状态", msg_id="old")
    assert http.request.await_args.kwargs["json"]["event_id"] == interaction.event_id
    assert "msg_id" not in http.request.await_args.kwargs["json"]
    assert http.request.await_args.kwargs["json"]["msg_seq"] == 1
    await callback.bot.api._http.request(Route("POST", "/v2/users/{openid}/messages", openid="alice"), json={"msg_id": "old", "content": "C2C"})
    assert http.request.await_args.kwargs["json"]["msg_seq"] == 2
    await callback.bot.api._http.request(Route("POST", "/v2/groups/{group_openid}/files", group_openid="one"), json={"file_type": 1})
    assert http.request.await_args.kwargs["json"] == {"file_type": 1}
    await event.bot.api.post_group_message(group_openid="one", content="原事件", msg_id="old")
    assert http.request.await_args.kwargs["json"]["msg_id"] == "old"
    assert event.bot.api._http is http


@pytest.mark.asyncio
async def test_sdk_parser_used_by_websocket_and_webhook_reaches_callback(library):
    connection = pytest.importorskip("botpy.connection")
    event = Event()
    await library.menu_commands.show_panel(event)
    token, _ = active_session(library)
    incoming = []
    state = connection.ConnectionState(lambda name, value: incoming.append((name, value)), event.bot.api)
    event.bot.api.on_interaction_result = AsyncMock()
    state.parse_interaction_create({"id": "webhook-event", "d": {
        "id": "interaction", "type": 11, "group_openid": "one", "group_member_openid": "alice",
        "data": {"type": 11, "resolved": {"button_id": BUTTON_PREFIX + "random", "button_data": json.dumps({"panel": token, "action": "random"})}},
    }})
    name, interaction = incoming[0]
    assert name == "interaction_create" and interaction.event_id == "webhook-event"
    await event.bot.on_interaction_create(interaction)
    await drain_callbacks(library.menu_commands)
    event.bot.api.on_interaction_result.assert_awaited_once_with("interaction", 0)
    library.meme_selector.send_emoji_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_platform_hook_registers_both_qq_adapters_and_preserves_existing_intents(library):
    from astrbot_plugin_stealer.main import Main
    clients = []
    platforms = []
    for name in ["qq_official", "qq_official_webhook", "telegram"]:
        client = SimpleNamespace(intents=1 << 25)
        clients.append(client)
        platforms.append(SimpleNamespace(
            meta=lambda name=name: SimpleNamespace(name=name),
            get_client=lambda client=client: client,
            intents=SimpleNamespace(interaction=False),
        ))
    library.context.platform_manager.get_insts = lambda: platforms
    await Main.register_menu_interactions(library)
    await Main.register_menu_interactions(library)
    for client in clients[:2]:
        assert client.intents & (1 << 25) and client.intents & (1 << 26)
        assert callable(client.on_interaction_create)
    assert not hasattr(clients[2], "on_interaction_create")
    assert len(library.menu_commands.interactions.bindings) == 2


@pytest.mark.asyncio
async def test_each_service_callback_uses_its_declared_handler_and_arguments(library):
    event = Event(admin=True)
    library.astrbot_config["admins_id"] = ["alice"]
    token = library.menu_commands._session_for(event)
    session = library.menu_commands._sessions[token]
    calls = []
    for action, spec in PANEL_ACTIONS.items():
        if not spec.service:
            continue

        async def handler(callback, *args, action=action):
            calls.append((action, callback, args))
            yield callback.plain_result("执行完成")

        service = getattr(library, spec.service, None)
        if service is None:
            service = SimpleNamespace()
            setattr(library, spec.service, service)
        setattr(service, spec.method, handler)
        await library.menu_commands.execute_action(event, token, action, session)
        assert calls[-1] == (action, event, spec.args)


@pytest.mark.asyncio
async def test_failed_callback_ack_does_not_execute_or_leave_panel_busy(library):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    interaction = click(token, "random")
    interaction._api.on_interaction_result.side_effect = RuntimeError("network error")
    with pytest.raises(RuntimeError):
        await library.menu_commands.interactions.handle(interaction, event.bot)
    assert not session.busy
    assert interaction.id not in session.handled
    library.meme_selector.send_emoji_message.assert_not_awaited()


@pytest.mark.parametrize("field", ["member_openid", "user_openid", "id"])
def test_button_identity_uses_raw_qq_sender_before_framework_alias(field):
    event = Event(user="framework-alias")
    event.message_obj.raw_message.author = SimpleNamespace(**{field: "actual-qq-openid"})
    assert source_sender(event) == ("actual-qq-openid", f"author.{field}")


@pytest.mark.asyncio
async def test_native_and_server_permissions_use_same_raw_identity_when_framework_maps_sender(library):
    event = Event(user="framework-alias")
    event.message_obj.raw_message.author = SimpleNamespace(member_openid="actual-qq-openid")
    await library.menu_commands.show_panel(event)
    payload = event.bot.api.post_group_message.await_args.kwargs
    for row in payload["keyboard"]["content"]["rows"]:
        for button in row["buttons"]:
            assert button["action"]["permission"] == {"type": 0, "specify_user_ids": ["actual-qq-openid"]}
    token, _ = active_session(library)
    seen = []

    async def status(callback):
        seen.append(callback.get_sender_id())
        yield callback.plain_result("状态完成")

    library.command_handler = SimpleNamespace(status=status)
    interaction = click(token, user="actual-qq-openid")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    await drain_callbacks(library.menu_commands)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 0)
    assert seen == ["framework-alias"]


@pytest.mark.asyncio
async def test_framework_alias_is_not_accepted_as_another_qq_account(library):
    event = Event(user="framework-alias")
    event.message_obj.raw_message.author = SimpleNamespace(member_openid="actual-qq-openid")
    await library.menu_commands.show_panel(event)
    token, _ = active_session(library)
    interaction = click(token, user="framework-alias")
    await library.menu_commands.interactions.handle(interaction, event.bot)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 4)
    assert not library.menu_commands.interactions.tasks


@pytest.mark.asyncio
async def test_native_permission_logs_separate_send_receive_and_denial_without_raw_ids(library, monkeypatch):
    from unittest.mock import Mock
    from core.commands import qq_panel_interactions
    info, warning = Mock(), Mock()
    monkeypatch.setattr(qq_panel_interactions.logger, "info", info)
    monkeypatch.setattr(qq_panel_interactions.logger, "warning", warning)
    owner = "owner-openid-0123456789abcdef"
    other = "other-openid-9876543210abcdef"
    event = Event(user=owner)
    event.message_obj.raw_message.author = SimpleNamespace(member_openid=owner)
    await library.menu_commands.show_panel(event)
    token, _ = active_session(library)
    interaction = click(token, user=other)
    await library.menu_commands.interactions.handle(interaction, event.bot)
    logs = "\n".join(call.args[0] for call in info.call_args_list + warning.call_args_list)
    assert "面板已发出" in logs
    assert "收到按钮回调" in logs
    assert "reason=sender" in logs
    assert "code=4" in logs
    assert owner not in logs and other not in logs and token not in logs


@pytest.mark.parametrize("section", ["overview", "settings", "maintenance"])
def test_panel_copy_is_compact_and_does_not_include_matching_statistics(section):
    config = SimpleNamespace(steal_meme=True, auto_send_meme=False, enable_natural_emotion_analysis=True)
    content = panel_content(section, True, config)
    assert len(content.splitlines()) == 3
    assert len(content) < 110
    for row in panel_rows(section, True):
        assert all(button.action != "emotion_stats" for button in row)
    assert "匹配统计" not in content


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unload", "dismiss", "expire", "cancel"])
async def test_inflight_ack_cannot_start_work_after_panel_or_router_is_invalid(library, change):
    event = Event()
    await library.menu_commands.show_panel(event)
    token, session = active_session(library)
    interaction = click(token, "random")
    started, finish = asyncio.Event(), asyncio.Event()

    async def acknowledge(*args):
        started.set()
        await finish.wait()

    interaction._api.on_interaction_result.side_effect = acknowledge
    task = asyncio.create_task(library.menu_commands.interactions.handle(interaction, event.bot))
    await started.wait()
    if change == "unload":
        await library.menu_commands.aclose()
    elif change == "dismiss":
        await library.menu_commands.dismiss(event, token)
    elif change == "expire":
        session.expires_at = 0
    else:
        task.cancel()
    finish.set()
    if change == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        await task
    assert not session.busy
    assert not library.menu_commands.interactions.tasks
    library.meme_selector.send_emoji_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,prefix", [("qq_official", "!"), ("qq_official_webhook", "九曜 "), ("aiocqhttp", "."), ("telegram", "/")])
async def test_adapters_use_their_own_profile_and_only_qq_uses_callbacks(library, platform, prefix):
    event = Event(platform=platform)
    profile = {"wake_prefix": [prefix], "admins_id": []}
    library.context.get_config = lambda *, umo: profile if umo == event.unified_msg_origin else {}
    await library.menu_commands.show_panel(event)
    if platform in {"qq_official", "qq_official_webhook"}:
        payload = event.bot.api.post_group_message.await_args.kwargs
        for row in payload["keyboard"]["content"]["rows"]:
            for button in row["buttons"]:
                data = json.loads(button["action"]["data"])
                assert set(data) == {"panel", "action"}
                assert "meme " not in button["action"]["data"]
    else:
        event.bot.api.post_group_message.assert_not_awaited()
        assert not hasattr(event.bot, "on_interaction_create")
        text = event.messages[-1].text
        assert prefix + "meme status" in text
        assert prefix + "meme random" in text


@pytest.mark.asyncio
async def test_same_qq_ids_on_different_adapter_instances_do_not_share_panel_tokens(library):
    first, second = Event(), Event()
    first.unified_msg_origin = "qq-instance-a:GroupMessage:one"
    second.unified_msg_origin = "qq-instance-b:GroupMessage:one"
    await library.menu_commands.show_panel(first)
    token = library.menu_commands._owners[library.menu_commands._owner(first)]
    await library.menu_commands.show_panel(second)
    other = library.menu_commands._owners[library.menu_commands._owner(second)]
    assert token != other
    interaction = click(token, "random")
    await library.menu_commands.interactions.handle(interaction, second.bot)
    interaction._api.on_interaction_result.assert_awaited_once_with(interaction.id, 4)
    library.meme_selector.send_emoji_message.assert_not_awaited()
