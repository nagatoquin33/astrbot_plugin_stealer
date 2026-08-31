import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_astrbot_stubs() -> None:
    # 使用 conftest.py 的共享 stubs 类
    from tests.conftest import (
        SHARED_IMAGE_CLASS,
        SHARED_PLAIN_CLASS,
        SHARED_MESSAGE_CHAIN_CLASS,
    )

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

    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    api_module.logger = logger
    api_module.AstrBotConfig = object

    event_module = types.ModuleType("astrbot.api.event")
    event_module.AstrMessageEvent = AstrMessageEvent
    event_module.MessageChain = SHARED_MESSAGE_CHAIN_CLASS

    star_module = types.ModuleType("astrbot.api.star")
    star_module.Context = object
    star_module.StarTools = object

    message_components_module = types.ModuleType("astrbot.api.message_components")
    message_components_module.Image = SHARED_IMAGE_CLASS
    message_components_module.Plain = SHARED_PLAIN_CLASS

    # 创建 astrbot.core.agent.message stub
    agent_message_module = types.ModuleType("astrbot.core.agent.message")
    agent_message_module.TextPart = object
    sys.modules["astrbot.core"] = types.ModuleType("astrbot.core")
    sys.modules["astrbot.core.agent"] = types.ModuleType("astrbot.core.agent")
    sys.modules["astrbot.core.agent.message"] = agent_message_module

    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.star"] = star_module
    sys.modules["astrbot.api.message_components"] = message_components_module


_install_astrbot_stubs()

from core.commands.command_handler import CommandHandler
from core.search.meme_selector import MemeSelector


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
        self.smart_meme_selection = False
        self.audit_required = True

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


class DummyDBService:
    """模拟数据库服务"""
    def __init__(self, index_map=None):
        self.index_map = dict(index_map or {})

    def count_total(self):
        return len(self.index_map)

    def get_index_cache_readonly(self):
        return dict(self.index_map)


class DummyPlugin:
    def __init__(self, index_map=None, target="group:100"):
        self.saved_index = None
        self.index_map = dict(index_map or {})
        self.cache_service = DummyCacheService(self.index_map)
        self.plugin_config = DummyConfig(target=target)
        self.smart_meme_selection = False
        self.emoji_chance = 1.0
        self.steal_mode = "probability"
        self.steal_chance = 1.0
        self.image_processing_cooldown = 0
        self.steal_meme = True
        self.base_dir = None
        self.categories_dir = None
        self.cache_dir = None  # 添加 cache_dir 属性
        self.process_calls = []
        self.natural_emotion_analysis_enabled = False
        self.categories = ["happy"]
        # 添加 db_service 属性（模拟数据库服务）
        self.db_service = DummyDBService(self.index_map)

        # v2.7.5+ 起调用方走 index_manager；通过 SimpleNamespace 引用方法，
        # 但绑定为 lambda 形式使其在测试中重新赋值 DummyPlugin._xxx 时能动态跟随。
        self._build_index_manager()

    def _build_index_manager(self) -> None:
        self.index_manager = types.SimpleNamespace(
            load_index=lambda: self._load_index(),
            save_index=lambda idx: self._save_index(idx),
            rebuild_index_from_files=lambda: self._rebuild_index_from_files(),
        )

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

    async def _rebuild_index_from_files(self):
        return {}

    def _update_config_from_dict(self, updates):
        for key, value in updates.items():
            setattr(self, key, value)

    def update_config(self, updates):
        # v2.7.5+：公开 API；兼容旧测试名
        return self._update_config_from_dict(updates)

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

    async def _process_image(
        self,
        event,
        temp_path,
        is_temp=True,
        is_platform_emoji=True,
        extra_meta=None,
        to_pending=False,
    ):
        self.process_calls.append(
            {
                "path": temp_path,
                "is_temp": is_temp,
                "is_platform_emoji": is_platform_emoji,
                "extra_meta": dict(extra_meta or {}),
                "to_pending": to_pending,
            }
        )
        return True, {}


async def _collect_asyncgen(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results


class ScopeFeatureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 使用 conftest.py 的共享 stubs 类并强制更新
        from tests.conftest import (
            SHARED_IMAGE_CLASS,
            SHARED_PLAIN_CLASS,
            SHARED_MESSAGE_CHAIN_CLASS,
        )
        sys.modules["astrbot.api.message_components"].Image = SHARED_IMAGE_CLASS
        sys.modules["astrbot.api.message_components"].Plain = SHARED_PLAIN_CLASS
        sys.modules["astrbot.api.event"].MessageChain = SHARED_MESSAGE_CHAIN_CLASS

    def _get_shared_image_class(self):
        """获取共享的 Image 类"""
        return sys.modules["astrbot.api.message_components"].Image

    def _create_handler(self, plugin):
        """创建使用最新 stubs 的 EventHandler"""
        # 强制重新导入以确保使用最新的 stubs
        from core.events.event_handler import EventHandler as FreshEventHandler
        return FreshEventHandler(plugin)

    async def test_event_handler_records_origin_target_on_capture(self):
        plugin = DummyPlugin()
        handler = self._create_handler(plugin)
        image_cls = self._get_shared_image_class()
        image = image_cls()
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

    async def test_non_emoji_image_does_not_consume_steal_cooldown(self):
        plugin = DummyPlugin()
        handler = self._create_handler(plugin)
        image_cls = self._get_shared_image_class()
        event = DummyEvent(target="group:123", messages=[image_cls()])
        checks = []
        handler._should_process_image = lambda: checks.append(True) or True
        handler._check_platform_emoji_metadata = lambda *args, **kwargs: False

        await handler.on_message(event)

        self.assertEqual(checks, [])

    async def test_steal_log_is_emitted_after_probability_gate(self):
        plugin = DummyPlugin()
        handler = self._create_handler(plugin)
        image_cls = self._get_shared_image_class()
        image = image_cls()
        image.url = "https://example.com/emoji.gif"
        event = DummyEvent(target="group:123", messages=[image])
        order = []

        class DummyQueue:
            async def submit_capture_async(self, _descriptors):
                order.append("queue")

        handler._background_queue = DummyQueue()
        handler._check_platform_emoji_metadata = lambda *args, **kwargs: order.append("detect") or True
        handler._should_process_image = lambda: order.append("probability") or True

        with patch(
            "core.events.event_handler.logger.info",
            side_effect=lambda *_args, **_kwargs: order.append("steal_log"),
        ):
            await handler.on_message(event)

        self.assertEqual(order, ["detect", "probability", "steal_log", "queue"])

    async def test_steal_log_is_skipped_when_probability_gate_rejects(self):
        plugin = DummyPlugin()
        handler = self._create_handler(plugin)
        image_cls = self._get_shared_image_class()
        event = DummyEvent(target="group:123", messages=[image_cls()])
        handler._check_platform_emoji_metadata = lambda *args, **kwargs: True
        handler._should_process_image = lambda: False

        with patch("core.events.event_handler.logger.info") as info_log:
            await handler.on_message(event)

        info_log.assert_not_called()

    async def test_force_capture_stays_synchronous_with_background_queue(self):
        plugin = DummyPlugin()
        tmp_dir = tempfile.TemporaryDirectory()
        plugin.base_dir = Path(tmp_dir.name)
        handler = self._create_handler(plugin)
        await handler.start_background_workers()

        image_cls = self._get_shared_image_class()
        event = DummyEvent(target="group:123", messages=[image_cls()])
        consumed = []
        plugin.get_force_capture_entry = lambda _event: {"until": time.time() + 30}
        plugin.consume_force_capture = lambda _event: consumed.append(True)

        with tempfile.NamedTemporaryFile(delete=False) as fp:
            temp_path = fp.name

        async def fake_download(_img):
            return temp_path, False

        handler._download_original_image = fake_download
        try:
            await handler.on_message(event)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await handler.stop_background_workers()
            tmp_dir.cleanup()

        self.assertEqual(consumed, [True])
        self.assertEqual(len(plugin.process_calls), 1)
        sent_texts = []
        for chain in event.sent_messages:
            for comp in chain:
                sent_texts.append(getattr(comp, "text", ""))
        self.assertTrue(any("已收录" in text for text in sent_texts))

    async def test_event_handler_merges_multi_image_results_before_save(self):
        class MultiImagePlugin(DummyPlugin):
            def __init__(self):
                super().__init__({"existing.png": {"category": "happy", "created_at": 1}})
                self._process_counter = 0

            async def _process_image(
                self,
                event,
                temp_path,
                is_temp=True,
                is_platform_emoji=True,
                extra_meta=None,
                to_pending=False,
            ):
                self._process_counter += 1
                return True, {
                    "existing.png": {"category": "happy", "created_at": 1},
                    f"new_{self._process_counter}.png": {
                        "category": "happy",
                        "created_at": self._process_counter + 1,
                        "origin_target": dict(extra_meta or {}).get("origin_target", ""),
                    },
                }

        plugin = MultiImagePlugin()
        handler = self._create_handler(plugin)
        image_cls = self._get_shared_image_class()
        event = DummyEvent(target="group:123", messages=[image_cls(), image_cls()])

        temp_paths = []
        cleanup_paths = []
        for _ in range(2):
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                temp_paths.append(fp.name)
                cleanup_paths.append(fp.name)

        async def fake_download(_img):
            return temp_paths.pop(0), False

        handler._download_original_image = fake_download
        handler._check_platform_emoji_metadata = lambda *args, **kwargs: True
        handler._should_process_image = lambda: True

        try:
            await handler.on_message(event)
        finally:
            for path_str in cleanup_paths:
                if os.path.exists(path_str):
                    os.remove(path_str)

        self.assertIsNotNone(plugin.saved_index)
        self.assertIn("existing.png", plugin.saved_index)
        self.assertIn("new_1.png", plugin.saved_index)
        self.assertIn("new_2.png", plugin.saved_index)

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

        self.assertIn("来源群", results[0])
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

        self.assertTrue(any("来源" in str(item) for item in results))
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
                    "desc": "公共猫猫",
                    "tags": ["猫猫"],
                    "created_at": 3,
                    "scope_mode": "public",
                },
            },
            target="group:100",
        )
        selector = MemeSelector(plugin)

        results = await selector.smart_search("猫猫", limit=10, event=DummyEvent(target="group:100"))
        paths = [item[0] for item in results]
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
                        "created_at": 1,
                        "scope_mode": "local",
                        "origin_target": "group:200",
                    }
                },
                target="group:100",
            )
            plugin.plugin_config.categories_dir = tmpdir
            selector = MemeSelector(plugin)

            result = await selector.select_emoji(
                "happy", event=DummyEvent(target="group:100")
            )
            self.assertIsNone(result)

    async def test_auto_select_returns_none_when_cache_index_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "happy"
            category_dir.mkdir(parents=True, exist_ok=True)
            img_path = category_dir / "only.png"
            img_path.write_bytes(b"fake")

            plugin = DummyPlugin({}, target="group:100")
            plugin.plugin_config.categories_dir = tmpdir
            selector = MemeSelector(plugin)

            result = await selector.select_emoji(
                "happy", event=DummyEvent(target="group:100")
            )
            self.assertIsNone(result)

    async def test_auto_select_returns_none_when_file_is_missing_from_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "happy"
            category_dir.mkdir(parents=True, exist_ok=True)
            img_path = category_dir / "only.png"
            img_path.write_bytes(b"fake")

            plugin = DummyPlugin({}, target="group:100")
            plugin.plugin_config.categories_dir = tmpdir
            selector = MemeSelector(plugin)

            result = await selector.select_emoji(
                "happy", event=DummyEvent(target="group:100")
            )
            self.assertIsNone(result)

    async def test_auto_select_returns_none_when_cache_service_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = Path(tmpdir) / "happy"
            category_dir.mkdir(parents=True, exist_ok=True)
            img_path = category_dir / "only.png"
            img_path.write_bytes(b"fake")

            plugin = DummyPlugin({}, target="group:100")
            plugin.cache_service = None
            plugin.plugin_config.categories_dir = tmpdir
            selector = MemeSelector(plugin)

            result = await selector.select_emoji(
                "happy", event=DummyEvent(target="group:100")
            )
            self.assertIsNone(result)

    async def test_blacklist_image_skips_empty_hash_cache_write(self):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            img_path = fp.name

        plugin = DummyPlugin(
            {
                img_path: {
                    "category": "happy",
                    "created_at": 10,
                    "hash": "",
                    "origin_target": "group:100",
                    "scope_mode": "public",
                }
            }
        )
        handler = CommandHandler(plugin)

        try:
            await _collect_asyncgen(handler.blacklist_image(DummyEvent(), "1"))
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)

        self.assertEqual(plugin.cache_service.kv, {})

    async def test_rebuild_index_preserves_scope_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_dir = Path(tmpdir) / "old_dir"
            old_dir.mkdir(parents=True, exist_ok=True)
            old_path = str(old_dir / "new.png")
            new_path = str(Path(tmpdir) / "new.png")
            Path(new_path).write_bytes(b"new")

            old_index = {
                old_path: {
                    "category": "happy",
                    "created_at": 1,
                    "hash": "abc",
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
                    "category": "happy",
                    "created_at": 2,
                    "hash": "def",
                    "desc": "新描述",
                    "tags": ["新标签"],
                }
            }

            plugin = DummyPlugin(old_index)
            plugin.base_dir = Path(tmpdir)
            plugin.categories_dir = Path(tmpdir)

            async def fake_rebuild():
                return rebuilt

            plugin._rebuild_index_from_files = fake_rebuild
            handler = CommandHandler(plugin)

            results = await _collect_asyncgen(handler.rebuild_index(DummyEvent()))
            self.assertTrue(any("重建" in str(item) for item in results))
            self.assertEqual(plugin.saved_index[new_path]["origin_target"], "group:100")
            self.assertEqual(plugin.saved_index[new_path]["scope_mode"], "local")
            self.assertEqual(plugin.saved_index[new_path]["source"], "qq_store")
            self.assertEqual(plugin.saved_index[new_path]["scenes"], ["课堂"])

    async def test_rebuild_index_recovers_from_backup_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            old_path = str(Path(tmpdir) / "legacy" / "new.png")
            new_path = str(Path(tmpdir) / "new.png")
            Path(new_path).write_bytes(b"new")

            backup_index = {
                old_path: {
                    "category": "happy",
                    "created_at": 1,
                    "hash": "abc",
                    "desc": "backup-desc",
                    "tags": ["backup-tag"],
                    "origin_target": "group:200",
                    "scope_mode": "local",
                    "source": "qq_store",
                    "scenes": ["backup-scene"],
                }
            }
            (cache_dir / "index_cache.json.backup").write_text(
                json.dumps(backup_index, ensure_ascii=False),
                encoding="utf-8",
            )

            rebuilt = {
                new_path: {
                    "category": "happy",
                    "created_at": 2,
                    "hash": "abc",
                }
            }

            plugin = DummyPlugin({})
            plugin.base_dir = Path(tmpdir)
            plugin.cache_dir = cache_dir
            plugin.categories_dir = Path(tmpdir)

            async def fake_rebuild():
                return rebuilt

            plugin._rebuild_index_from_files = fake_rebuild
            handler = CommandHandler(plugin)

            await _collect_asyncgen(handler.rebuild_index(DummyEvent()))
            self.assertEqual(plugin.saved_index[new_path]["desc"], "backup-desc")
            self.assertEqual(plugin.saved_index[new_path]["tags"], ["backup-tag"])
            self.assertEqual(plugin.saved_index[new_path]["origin_target"], "group:200")
            self.assertEqual(plugin.saved_index[new_path]["scope_mode"], "local")
            self.assertEqual(plugin.saved_index[new_path]["scenes"], ["backup-scene"])

    async def test_rebuild_index_prefers_database_metadata_over_backup_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            old_path = str(Path(tmpdir) / "legacy" / "new.png")
            new_path = str(Path(tmpdir) / "new.png")
            Path(new_path).write_bytes(b"new")

            backup_index = {
                old_path: {
                    "category": "happy",
                    "created_at": 1,
                    "hash": "abc",
                    "desc": "backup-desc",
                    "tags": ["backup-tag"],
                    "origin_target": "group:200",
                    "scope_mode": "local",
                    "scenes": ["backup-scene"],
                }
            }
            (cache_dir / "index_cache.json.backup").write_text(
                json.dumps(backup_index, ensure_ascii=False),
                encoding="utf-8",
            )

            current_index = {
                old_path: {
                    "category": "happy",
                    "created_at": 1,
                    "hash": "abc",
                    "desc": "current-desc",
                    "tags": ["current-tag"],
                    "origin_target": "group:100",
                    "scope_mode": "public",
                    "scenes": ["current-scene"],
                }
            }
            rebuilt = {
                new_path: {
                    "category": "happy",
                    "created_at": 2,
                    "hash": "abc",
                }
            }

            plugin = DummyPlugin(current_index)
            plugin.base_dir = Path(tmpdir)
            plugin.cache_dir = cache_dir
            plugin.categories_dir = Path(tmpdir)

            async def fake_rebuild():
                return rebuilt

            plugin._rebuild_index_from_files = fake_rebuild
            handler = CommandHandler(plugin)

            await _collect_asyncgen(handler.rebuild_index(DummyEvent()))
            self.assertEqual(plugin.saved_index[new_path]["desc"], "current-desc")
            self.assertEqual(plugin.saved_index[new_path]["tags"], ["current-tag"])
            self.assertEqual(plugin.saved_index[new_path]["origin_target"], "group:100")
            self.assertEqual(plugin.saved_index[new_path]["scope_mode"], "public")
            self.assertEqual(plugin.saved_index[new_path]["scenes"], ["current-scene"])

    async def test_rebuild_index_prefers_legacy_hash_match_over_current_name_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            current_same_name_path = str(Path(tmpdir) / "other" / "target.png")
            legacy_hash_path = str(Path(tmpdir) / "legacy" / "another.png")
            rebuilt_path = str(Path(tmpdir) / "target.png")
            Path(rebuilt_path).write_bytes(b"new")

            backup_index = {
                legacy_hash_path: {
                    "category": "happy",
                    "created_at": 1,
                    "hash": "expected-hash",
                    "desc": "legacy-hash-desc",
                    "tags": ["legacy-hash-tag"],
                    "origin_target": "group:999",
                    "scope_mode": "local",
                    "scenes": ["legacy-hash-scene"],
                }
            }
            (cache_dir / "index_cache.json.backup").write_text(
                json.dumps(backup_index, ensure_ascii=False),
                encoding="utf-8",
            )

            current_index = {
                current_same_name_path: {
                    "category": "happy",
                    "created_at": 2,
                    "hash": "other-hash",
                    "desc": "current-name-desc",
                    "tags": ["current-name-tag"],
                    "origin_target": "group:100",
                    "scope_mode": "public",
                    "scenes": ["current-name-scene"],
                }
            }
            rebuilt = {
                rebuilt_path: {
                    "category": "happy",
                    "created_at": 3,
                    "hash": "expected-hash",
                }
            }

            plugin = DummyPlugin(current_index)
            plugin.base_dir = Path(tmpdir)
            plugin.cache_dir = cache_dir
            plugin.categories_dir = Path(tmpdir)

            async def fake_rebuild():
                return rebuilt

            plugin._rebuild_index_from_files = fake_rebuild
            handler = CommandHandler(plugin)

            await _collect_asyncgen(handler.rebuild_index(DummyEvent()))
            self.assertEqual(plugin.saved_index[rebuilt_path]["desc"], "legacy-hash-desc")
            self.assertEqual(plugin.saved_index[rebuilt_path]["tags"], ["legacy-hash-tag"])
            self.assertEqual(plugin.saved_index[rebuilt_path]["origin_target"], "group:999")
            self.assertEqual(plugin.saved_index[rebuilt_path]["scope_mode"], "local")
            self.assertEqual(plugin.saved_index[rebuilt_path]["scenes"], ["legacy-hash-scene"])


if __name__ == "__main__":
    unittest.main()
