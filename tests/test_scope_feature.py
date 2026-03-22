import asyncio
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    class AstrMessageEvent:
        pass

    class MessageChain(list):
        pass

    class Image:
        async def convert_to_file_path(self):
            return ""

    class Plain:
        def __init__(self, text: str = ""):
            self.text = text

    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = logger
    api_module.AstrBotConfig = object

    event_module = types.ModuleType("astrbot.api.event")
    event_module.AstrMessageEvent = AstrMessageEvent
    event_module.MessageChain = MessageChain

    star_module = types.ModuleType("astrbot.api.star")
    star_module.Context = object
    star_module.StarTools = object

    message_components_module = types.ModuleType("astrbot.api.message_components")
    message_components_module.Image = Image
    message_components_module.Plain = Plain

    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.star"] = star_module
    sys.modules["astrbot.api.message_components"] = message_components_module


_install_astrbot_stubs()

from core.command_handler import CommandHandler
from core.emoji_selector import EmojiSelector
from core.event_handler import EventHandler


class DummyCacheService:
    def __init__(self, index_map=None):
        self.index_map = dict(index_map or {})
        self.kv = {}

    def get_index_cache_readonly(self):
        return self.index_map

    async def update_index(self, updater):
        updater(self.index_map)

    async def set(self, cache_name, key, value, persist=False):
        self.kv[(cache_name, key)] = value


class DummyConfig:
    def __init__(self, target="group:100"):
        self.target = target
        self.categories_dir = None

    def get_event_target(self, event):
        raw_target = getattr(event, "target", self.target)
        scope, target_id = raw_target.split(":", 1)
        return scope, target_id


class DummyEvent:
    def __init__(self, target="group:100", messages=None):
        self.target = target
        self._messages = list(messages or [])
        self.sent_messages = []
        self.extras = {}
        self.message_obj = types.SimpleNamespace(raw_message=types.SimpleNamespace(message=[]))

    def get_messages(self):
        return list(self._messages)

    async def send(self, chain):
        self.sent_messages.append(chain)

    def plain_result(self, text):
        return text

    def image_result(self, url):
        return url

    def make_result(self):
        return self

    def base64_image(self, value):
        return value

    def file_image(self, value):
        return value

    def stop_event(self):
        return self

    def get_platform_name(self):
        return "aiocqhttp"

    def get_group_id(self):
        scope, target_id = self.target.split(":", 1)
        return target_id if scope == "group" else ""

    def get_sender_id(self):
        scope, target_id = self.target.split(":", 1)
        return target_id if scope == "user" else "42"

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)


class DummyPlugin:
    def __init__(self, index_map=None, target="group:100"):
        self.saved_index = None
        self.index_map = dict(index_map or {})
        self.cache_service = DummyCacheService(self.index_map)
        self.plugin_config = DummyConfig(target=target)
        self.smart_emoji_selection = False
        self.emoji_chance = 1.0
        self.steal_mode = "probability"
        self.steal_chance = 1.0
        self.image_processing_cooldown = 0
        self.steal_emoji = True
        self.base_dir = None
        self.categories_dir = None
        self.process_calls = []

    async def _load_index(self):
        return self.index_map

    async def _save_index(self, index_map):
        self.saved_index = dict(index_map)
        self.index_map = index_map
        self.cache_service.index_map = index_map

    async def _safe_remove_file(self, path):
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def _update_config_from_dict(self, updates):
        for key, value in updates.items():
            setattr(self, key, value)

    def is_send_enabled_for_event(self, event):
        return True

    def is_meme_enabled_for_event(self, event):
        return True

    def is_steal_enabled_for_event(self, event):
        return True

    def get_force_capture_entry(self, event):
        return None

    def consume_force_capture(self, event):
        return None

    async def _process_image(self, event, temp_path, is_temp=True, is_platform_emoji=True, extra_meta=None):
        self.process_calls.append(
            {
                "path": temp_path,
                "is_temp": is_temp,
                "is_platform_emoji": is_platform_emoji,
                "extra_meta": dict(extra_meta or {}),
            }
        )
        return True, {}


async def _collect_asyncgen(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


class ScopeFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_handler_records_origin_target_on_capture(self):
        plugin = DummyPlugin()
        handler = EventHandler(plugin)
        image = sys.modules["astrbot.api.message_components"].Image()
        event = DummyEvent(target="group:123", messages=[image])

        with tempfile.NamedTemporaryFile(delete=False) as fp:
            temp_path = fp.name

        async def fake_download(_img):
            return temp_path, False

        handler._download_original_image = fake_download
        handler._check_platform_emoji_metadata = lambda *args, **kwargs: True
        handler._should_process_image = lambda: True

        try:
            await handler.on_message(event)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        self.assertEqual(len(plugin.process_calls), 1)
        self.assertEqual(
            plugin.process_calls[0]["extra_meta"].get("origin_target"), "group:123"
        )

    async def test_scope_command_marks_entry_as_local(self):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            img_path = fp.name

        plugin = DummyPlugin(
            {
                img_path: {
                    "category": "happy",
                    "created_at": 10,
                    "hash": "abc",
                    "origin_target": "group:100",
                    "scope_mode": "public",
                }
            }
        )
        handler = CommandHandler(plugin)

        try:
            results = await _collect_asyncgen(
                handler.set_image_scope(DummyEvent(), "1", "local")
            )
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

        self.assertIn("仅来源群", results[0])
        self.assertEqual(plugin.index_map[img_path]["scope_mode"], "local")

    async def test_scope_command_rejects_local_without_origin_target(self):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            img_path = fp.name

        plugin = DummyPlugin(
            {
                img_path: {
                    "category": "happy",
                    "created_at": 10,
                    "hash": "abc",
                    "scope_mode": "public",
                }
            }
        )
        handler = CommandHandler(plugin)

        try:
            results = await _collect_asyncgen(
                handler.set_image_scope(DummyEvent(), "1", "local")
            )
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

        self.assertIn("缺少来源群信息", results[0])
        self.assertEqual(plugin.index_map[img_path]["scope_mode"], "public")

    async def test_search_filters_out_local_memes_from_other_groups(self):
        plugin = DummyPlugin(
            {
                "a.png": {
                    "category": "happy",
                    "desc": "猫猫开心",
                    "tags": ["猫猫"],
                    "created_at": 2,
                    "scope_mode": "local",
                    "origin_target": "group:100",
                },
                "b.png": {
                    "category": "happy",
                    "desc": "猫猫开心",
                    "tags": ["猫猫"],
                    "created_at": 1,
                    "scope_mode": "local",
                    "origin_target": "group:200",
                },
                "c.png": {
                    "category": "happy",
                    "desc": "猫猫开心",
                    "tags": ["猫猫"],
                    "created_at": 3,
                    "scope_mode": "public",
                },
            },
            target="group:100",
        )
        selector = EmojiSelector(plugin)

        results = await selector.search_images("猫猫", limit=10, event=DummyEvent("group:100"))
        paths = {item[0] for item in results}

        self.assertIn("a.png", paths)
        self.assertIn("c.png", paths)
        self.assertNotIn("b.png", paths)

    async def test_auto_select_returns_none_when_only_other_group_local_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "happy"
            category_dir.mkdir(parents=True, exist_ok=True)
            img_path = category_dir / "only.png"
            img_path.write_bytes(b"fake")

            plugin = DummyPlugin(
                {
                    str(img_path): {
                        "category": "happy",
                        "desc": "开心",
                        "created_at": 1,
                        "scope_mode": "local",
                        "origin_target": "group:200",
                    }
                },
                target="group:100",
            )
            plugin.plugin_config.categories_dir = tmpdir
            selector = EmojiSelector(plugin)

            result = await selector.select_emoji("happy", "开心", event=DummyEvent("group:100"))
            self.assertIsNone(result)

    async def test_rebuild_index_preserves_scope_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = str(Path(tmpdir) / "old.png")
            new_path = str(Path(tmpdir) / "new.png")
            Path(new_path).write_bytes(b"new")

            old_index = {
                old_path: {
                    "hash": "same-hash",
                    "desc": "旧描述",
                    "tags": ["旧标签"],
                    "origin_target": "group:100",
                    "scope_mode": "local",
                    "source": "qq_store",
                    "scenes": ["课堂"],
                }
            }
            rebuilt = {
                new_path: {
                    "hash": "same-hash",
                    "category": "happy",
                    "created_at": 1,
                }
            }

            plugin = DummyPlugin(old_index)
            plugin.base_dir = Path(tmpdir)
            plugin.categories_dir = Path(tmpdir)

            async def fake_rebuild():
                return dict(rebuilt)

            plugin._rebuild_index_from_files = fake_rebuild
            handler = CommandHandler(plugin)

            results = await _collect_asyncgen(handler.rebuild_index(DummyEvent()))

            self.assertTrue(any("重建完成" in str(item) for item in results))
            self.assertEqual(plugin.saved_index[new_path]["origin_target"], "group:100")
            self.assertEqual(plugin.saved_index[new_path]["scope_mode"], "local")
            self.assertEqual(plugin.saved_index[new_path]["source"], "qq_store")
            self.assertEqual(plugin.saved_index[new_path]["scenes"], ["课堂"])


if __name__ == "__main__":
    unittest.main()
