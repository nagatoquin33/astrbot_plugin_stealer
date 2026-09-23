"""Advertised /meme group forms must reach their intended operations."""

from types import SimpleNamespace

import pytest

from core.commands.target_filter_command import TargetFilterCommand
from core.config.config import PluginConfig


class _TargetConfig:
    def __init__(self):
        self.send_target_whitelist = []
        self.send_target_blacklist = []
        self.send_target_filter_mode = "whitelist_first"
        self.steal_target_whitelist = []
        self.steal_target_blacklist = []
        self.steal_target_filter_mode = "whitelist_first"

    normalize_target_entry = staticmethod(PluginConfig.normalize_target_entry)

    def get_event_target(self, _event):
        return "group", "42"

    def _get_action_lists(self, action):
        return (
            getattr(self, f"{action}_target_whitelist"),
            getattr(self, f"{action}_target_blacklist"),
        )

    def _get_action_filter_mode(self, action):
        return getattr(self, f"{action}_target_filter_mode")

    def update_config(self, updates):
        for key, value in updates.items():
            setattr(self, key, value)
        return True


@pytest.mark.asyncio
async def test_group_show_priority_and_short_form():
    config = _TargetConfig()
    command = TargetFilterCommand(SimpleNamespace(plugin_config=config))
    event = SimpleNamespace(plain_result=lambda text: text)

    async def run(*args):
        return [result async for result in command.group_filter(event, *args)][0]

    assert "发表情" in await run("send", "show")
    assert "偷表情" in await run("steal", "show")
    assert "当前优先级" in await run("send", "priority")
    assert "黑名单优先" in await run("send", "priority", "bl")
    assert config.send_target_filter_mode == "blacklist_first"

    assert "group:123" in await run("wl", "add", "group:123")
    assert config.send_target_whitelist == ["group:123"]
    assert "user:456" in await run("send", "bl", "add", "user", "456")
    assert config.send_target_blacklist == ["user:456"]
    assert "group:123" in await run("wl", "del", "group:123")
    assert config.send_target_whitelist == []

    assert "缺少目标" in await run("send", "wl", "add", "group")
    assert "缺少目标" in await run("send", "wl", "add", "user")
    assert config.send_target_whitelist == []
