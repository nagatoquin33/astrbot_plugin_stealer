import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path


def install_astrbot_stubs() -> None:
    class Logger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class DummyDecoratorGroup:
        def __call__(self, func):
            func.command = self.command
            return func

        def command(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    def decorator_factory(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    filter_mod = types.ModuleType("astrbot.api.event.filter")
    filter_mod.EventMessageType = types.SimpleNamespace(ALL="ALL")
    filter_mod.PlatformAdapterType = types.SimpleNamespace(ALL="ALL")
    filter_mod.PermissionType = types.SimpleNamespace(ADMIN="ADMIN")
    filter_mod.command_group = lambda *args, **kwargs: DummyDecoratorGroup()
    filter_mod.permission_type = decorator_factory
    filter_mod.llm_tool = decorator_factory
    filter_mod.event_message_type = decorator_factory
    filter_mod.platform_adapter_type = decorator_factory
    filter_mod.on_llm_request = decorator_factory
    filter_mod.on_decorating_result = decorator_factory

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = type("AstrMessageEvent", (), {})
    event_mod.MessageChain = list
    event_mod.filter = filter_mod

    api_mod = types.ModuleType("astrbot.api")
    api_mod.AstrBotConfig = dict
    api_mod.logger = Logger()

    star_mod = types.ModuleType("astrbot.api.star")
    star_mod.Context = type("Context", (), {})
    star_mod.Star = type("Star", (), {})
    star_mod.StarTools = type("StarTools", (), {})

    message_components_mod = types.ModuleType("astrbot.api.message_components")

    class Image:
        def __init__(self, file=None):
            self.file = file

        @classmethod
        def fromBase64(cls, data):
            image = cls()
            image.data = data
            return image

    class Plain:
        def __init__(self, text=""):
            self.text = text

    message_components_mod.Image = Image
    message_components_mod.Plain = Plain

    agent_message_mod = types.ModuleType("astrbot.core.agent.message")

    class TextPart:
        def __init__(self, text=""):
            self.text = text

    agent_message_mod.TextPart = TextPart

    quart_mod = types.ModuleType("quart")
    quart_mod.jsonify = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    quart_mod.request = types.SimpleNamespace()
    quart_mod.send_file = lambda *args, **kwargs: None

    sys.modules.setdefault("astrbot", types.ModuleType("astrbot"))
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.event.filter"] = filter_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.message_components"] = message_components_mod
    sys.modules.setdefault("astrbot.core", types.ModuleType("astrbot.core"))
    sys.modules.setdefault("astrbot.core.agent", types.ModuleType("astrbot.core.agent"))
    sys.modules["astrbot.core.agent.message"] = agent_message_mod
    sys.modules["quart"] = quart_mod


install_astrbot_stubs()

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from astrbot_plugin_stealer.core.events import emoji_sender_engine as engine_module
from astrbot_plugin_stealer.core.events.emoji_sender_engine import EmojiSenderEngine
from astrbot_plugin_stealer.core.search.emoji_smart_select_service import (
    EmojiSmartSelectService,
)
from astrbot_plugin_stealer.main import Main


class FakeEvent:
    def __init__(self, text="user asks"):
        self.extras = {}
        self.text = text
        self.result = None
        self.sent = []

    def get_extra(self, key, default=None):
        return self.extras.get(key, default)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_session_id(self):
        return "session-1"

    def get_message_str(self):
        return self.text

    def get_result(self):
        return self.result

    async def send(self, message):
        self.sent.append(message)


class FakeResult:
    result_content_type = "text"
    chain = []

    def __init__(self, text):
        self.text = text
        self.cleaned_text = text

    def is_llm_result(self):
        return True

    def get_plain_text(self):
        return self.text


class FakePlugin:
    def __init__(self, *, chance=1.0, natural=False, selector=None, matcher=None):
        self.auto_send = True
        self.emoji_chance = chance
        self.enable_natural_emotion_analysis = natural
        self.emoji_send_delay = 0
        self.emoji_send_delay_random = 0
        self.emoji_selector = selector
        self.smart_emotion_matcher = matcher

    def is_send_enabled_for_event(self, event):
        return True


class RecordingSelector:
    def __init__(self, result=True):
        self.result = result
        self.calls = []
        self.sent = []

    async def try_send_emoji(self, event, emotions, cleaned_text):
        self.calls.append((list(emotions), cleaned_text))
        return self.result

    def is_path_allowed_for_event(self, path, event):
        return True

    async def send_emoji_message(self, event, path):
        self.sent.append(path)
        return "file_image" if self.result else None

    async def record_emoji_usage(self, path, trigger="auto"):
        pass


class RecordingMatcher:
    def __init__(self, emotion="happy"):
        self.emotion = emotion
        self.calls = []

    async def analyze_and_match_emotion(
        self, event, text, *, use_natural_analysis=False, user_query=""
    ):
        self.calls.append((text, use_natural_analysis, user_query))
        return self.emotion


class CacheStub:
    def __init__(self, idx):
        self.idx = idx

    def get_index_cache_readonly(self):
        return self.idx


class DatabaseStub:
    def count_total(self):
        return 0

    def get_index_cache_readonly(self):
        return {}


def make_main(*, chance=1.0, natural=True, selector=None, matcher=None):
    instance = Main.__new__(Main)
    instance.auto_send = True
    instance.emoji_chance = chance
    instance.enable_natural_emotion_analysis = natural
    instance.emoji_send_delay = 0
    instance.emoji_send_delay_random = 0
    instance.emoji_selector = selector or RecordingSelector()
    instance.smart_emotion_matcher = matcher or RecordingMatcher()
    instance._emoji_sender_engine = EmojiSenderEngine(instance)
    instance.is_send_enabled_for_event = lambda event: True
    instance._update_result_with_cleaned_text_safe = (
        lambda event, result, cleaned_text: setattr(result, "cleaned_text", cleaned_text)
    )
    return instance


async def consume_async_generator(generator):
    return [item async for item in generator]


class AutoEmojiChanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_permission_reuses_decision_without_rerolling(self):
        plugin = FakePlugin(chance=0.2)
        engine = EmojiSenderEngine(plugin)
        event = FakeEvent()
        calls = 0
        original_random = engine_module.random.random

        def fake_random():
            nonlocal calls
            calls += 1
            return 0.1

        engine_module.random.random = fake_random
        try:
            self.assertTrue(await engine.resolve_auto_emoji_turn_permission(event))
            self.assertTrue(await engine.resolve_auto_emoji_turn_permission(event))
        finally:
            engine_module.random.random = original_random

        self.assertEqual(calls, 1)
        self.assertEqual(event.get_extra("stealer_auto_emoji_turn_reason"), "chance_hit")

    async def test_chance_zero_blocks_intelligent_auto_send(self):
        instance = make_main(chance=0, natural=True)
        event = FakeEvent()
        event.result = FakeResult("hello")
        tasks = []
        instance._safe_create_task = lambda task, name=None: tasks.append((task, name))

        handled = await Main._prepare_emoji_response(instance, event)

        self.assertFalse(handled)
        self.assertEqual(tasks, [])
        self.assertEqual(event.sent, [])
        self.assertFalse(instance._emoji_turn_state(event).is_active_sent())

    async def test_chance_zero_blocks_explicit_ast_emoji(self):
        instance = make_main(chance=0, natural=True)
        event = FakeEvent()
        event.result = FakeResult("[ast_emoji:path/to/emoji.png] hello")
        instance._extract_emotions_from_text = lambda event, text: asyncio.sleep(
            0, result=([], text.strip())
        )

        handled = await Main._prepare_emoji_response(instance, event)

        self.assertTrue(handled)
        self.assertEqual(event.sent, [])
        self.assertEqual(event.result.cleaned_text, "hello")
        self.assertFalse(instance._emoji_turn_state(event).is_active_sent())

    async def test_chance_zero_blocks_llm_tool_send_emoji_by_id(self):
        selector = RecordingSelector()
        instance = make_main(chance=0, selector=selector)
        event = FakeEvent()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emoji.png"
            path.write_bytes(b"fake")
            instance._emoji_turn_state(event).set_candidates(
                [{"path": str(path), "desc": "smile", "emotion": "happy"}]
            )

            replies = await consume_async_generator(
                Main.send_emoji_by_id(instance, event, 1)
            )

        self.assertEqual(selector.sent, [])
        self.assertIn("未触发表情包发送条件", replies[0])
        self.assertFalse(instance._emoji_turn_state(event).is_active_sent())

    async def test_chance_one_allows_llm_tool_send_emoji_by_id(self):
        selector = RecordingSelector()
        instance = make_main(chance=1, selector=selector)
        event = FakeEvent()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emoji.png"
            path.write_bytes(b"fake")
            instance._emoji_turn_state(event).set_candidates(
                [{"path": str(path), "desc": "smile", "emotion": "happy"}]
            )

            replies = await consume_async_generator(
                Main.send_emoji_by_id(instance, event, 1)
            )

        self.assertEqual(selector.sent, [str(path)])
        self.assertIn("发送成功", replies[0])
        self.assertTrue(instance._emoji_turn_state(event).is_active_sent())
        self.assertIn("session-1", instance._emoji_sender_engine._auto_emoji_cooldowns)

    async def test_fractional_chance_uses_random_threshold(self):
        plugin = FakePlugin(chance=0.2)
        engine = EmojiSenderEngine(plugin)
        original_random = engine_module.random.random
        rolls = iter([0.19, 0.21])

        engine_module.random.random = lambda: next(rolls)
        try:
            self.assertTrue(await engine.resolve_auto_emoji_turn_permission(FakeEvent()))
            self.assertFalse(await engine.resolve_auto_emoji_turn_permission(FakeEvent()))
        finally:
            engine_module.random.random = original_random

    async def test_cooldown_only_written_after_success_and_pruned(self):
        selector = RecordingSelector(result=False)
        plugin = FakePlugin(selector=selector)
        engine = EmojiSenderEngine(plugin)
        event = FakeEvent()

        await engine.async_analyze_and_send_emoji(event, "reply", ["happy"])
        self.assertNotIn("session-1", engine._auto_emoji_cooldowns)

        selector.result = True
        old_key = "old-session"
        now = asyncio.get_event_loop().time()
        engine._auto_emoji_cooldowns[old_key] = (
            now - engine.AUTO_EMOJI_COOLDOWN_SECONDS * 3
        )
        await engine.async_analyze_and_send_emoji(event, "reply", ["happy"])

        self.assertIn("session-1", engine._auto_emoji_cooldowns)
        self.assertNotIn(old_key, engine._auto_emoji_cooldowns)
        self.assertFalse(await engine.resolve_auto_emoji_turn_permission(FakeEvent()))

    async def test_search_emoji_sets_candidates_without_marking_active_sent(self):
        instance = make_main(chance=0)
        event = FakeEvent()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "emoji.png"
            path.write_bytes(b"fake")
            instance.db_service = DatabaseStub()
            instance.cache_service = CacheStub({str(path): {}})
            instance.categories = ["happy"]
            instance.MAX_SEARCH_RESULTS = 5
            instance._load_index = lambda: asyncio.sleep(0)
            instance._search_emoji_candidates = lambda *args, **kwargs: asyncio.sleep(
                0, result=[(str(path), "smile", "happy", ["tag"])]
            )
            instance._find_similar_categories = lambda query, top_n=3: []

            replies = await consume_async_generator(Main.search_emoji(instance, event, "happy"))

        state = instance._emoji_turn_state(event)
        self.assertFalse(state.is_active_sent())
        self.assertEqual(len(state.get_candidates()), 1)
        self.assertIn("找到 1 个匹配的表情包", replies[0])

    async def test_prepare_passive_mode_extracts_emotions_and_cleans_text(self):
        instance = make_main(chance=1, natural=False)
        event = FakeEvent()
        event.result = FakeResult("&&happy&& hello")
        calls = []
        tasks = []

        async def extract_emotions(event, text):
            self.assertEqual(text, "&&happy&& hello")
            return ["happy"], "hello"

        def analyze(event, text, emotions, **kwargs):
            calls.append((text, list(emotions), kwargs))
            return object()

        instance._extract_emotions_from_text = extract_emotions
        instance._async_analyze_and_send_emoji = analyze
        instance._safe_create_task = lambda task, name=None: tasks.append((task, name))

        handled = await Main._prepare_emoji_response(instance, event)

        self.assertTrue(handled)
        self.assertEqual(event.result.cleaned_text, "hello")
        self.assertEqual(calls, [("hello", ["happy"], {"user_query": "user asks"})])
        self.assertEqual(tasks[0][1], "emoji_analyze_passive")

    async def test_smart_select_claimed_turn_can_send_but_active_sent_cannot(self):
        plugin = FakePlugin()
        engine = EmojiSenderEngine(plugin)
        event = FakeEvent()
        await engine.resolve_auto_emoji_turn_permission(event)
        self.assertTrue(engine.claim_auto_emoji_turn(event))

        plugin._emoji_turn_state = engine.emoji_turn_state
        plugin.emoji_selector = types.SimpleNamespace()
        service = EmojiSmartSelectService(plugin)
        sent = []

        async def select_emoji(emotion, cleaned_text, event=None):
            return "emoji.png"

        async def send_emoji_message(event, emoji_path):
            sent.append((emoji_path, event.get_extra("stealer_auto_emoji_turn_claimed")))
            return "file_image"

        async def record_emoji_usage(emoji_path, trigger="auto"):
            pass

        plugin.emoji_selector.select_emoji = select_emoji
        service._check_group_allowed = lambda event: True
        service.send_emoji_message = send_emoji_message
        service.record_emoji_usage = record_emoji_usage

        self.assertTrue(await service.try_send_emoji(event, ["happy"], "reply"))
        self.assertEqual(sent, [("emoji.png", True)])

        engine.emoji_turn_state(event).mark_active_sent()
        self.assertFalse(await service.try_send_emoji(event, ["happy"], "reply"))


if __name__ == "__main__":
    unittest.main()
