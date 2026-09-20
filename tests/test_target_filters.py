import asyncio
import types


from core.commands.target_filter_command import TargetFilterCommand
from core.config.config import PluginConfig


class DummyEvent:
    def __init__(self, group_id: str = "100", user_id: str = "42"):
        self._group_id = group_id
        self._user_id = user_id

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._user_id

    def plain_result(self, text):
        return text


async def _collect_asyncgen(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


def _build_config() -> PluginConfig:
    return PluginConfig({}, None)


def test_whitelist_first_allows_whitelisted_group_even_if_user_is_blacklisted():
    cfg = _build_config()
    cfg.send_target_whitelist = ["group:100"]
    cfg.send_target_blacklist = ["user:42"]
    cfg.send_target_filter_mode = "whitelist_first"

    assert cfg.is_action_allowed("send", DummyEvent()) is True


def test_blacklist_first_blocks_blacklisted_user_inside_whitelisted_group():
    cfg = _build_config()
    cfg.send_target_whitelist = ["group:100"]
    cfg.send_target_blacklist = ["user:42"]
    cfg.send_target_filter_mode = "blacklist_first"

    assert cfg.is_action_allowed("send", DummyEvent()) is False


def test_user_whitelist_can_match_group_event_sender():
    cfg = _build_config()
    cfg.steal_target_whitelist = ["user:42"]

    assert cfg.is_action_allowed("steal", DummyEvent(group_id="100", user_id="42")) is True
    assert cfg.is_action_allowed("steal", DummyEvent(group_id="100", user_id="99")) is False


def test_group_filter_priority_command_updates_mode():
    cfg = _build_config()
    plugin = types.SimpleNamespace(plugin_config=cfg)
    handler = TargetFilterCommand(plugin)

    results = asyncio.run(
        _collect_asyncgen(handler.group_filter(DummyEvent(), "send", "priority", "bl"))
    )

    assert cfg.send_target_filter_mode == "blacklist_first"
    assert results == ["已将发表情优先级设置为黑名单优先"]
