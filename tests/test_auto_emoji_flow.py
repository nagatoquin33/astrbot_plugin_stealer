import asyncio
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

from astrbot_plugin_stealer.core.events.meme_sender_engine import MemeSenderEngine


def _install_stubs() -> str:
    # 检查是否已安装兼容的 stubs (由 conftest.py 安装)
    if "astrbot.api.message_components" in sys.modules:
        existing_image = sys.modules["astrbot.api.message_components"].Image
        if hasattr(existing_image, "fromBase64") and hasattr(
            existing_image, "convert_to_file_path"
        ):
            test_result = existing_image.fromBase64("test")
            if test_result == "b64:test":
                package_name = Path(__file__).resolve().parents[1].name
                return package_name  # stubs 已兼容，跳过安装

    package_name = Path(__file__).resolve().parents[1].name
    package_prefix = f"{package_name}."

    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    def _decorator(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper

    class _CommandGroup:
        def __call__(self, func):
            return self

        def command(self, *args, **kwargs):
            return _decorator(*args, **kwargs)

    filter_stub = types.SimpleNamespace(
        on_decorating_result=_decorator,
        on_llm_tool_respond=_decorator,
        command_group=lambda *args, **kwargs: _CommandGroup(),
        permission_type=_decorator,
        llm_tool=_decorator,
        event_message_type=_decorator,
        platform_adapter_type=_decorator,
    )

    class Star:
        def __init__(self, context=None):
            self.context = context

    class MessageChain(list):
        pass

    class Plain:
        def __init__(self, text: str = ""):
            self.text = text

    class Image:
        @classmethod
        def fromBase64(cls, value):
            return f"b64:{value}"

        async def convert_to_file_path(self):
            return ""

    api_module = types.ModuleType("astrbot.api")
    api_module.logger = logger
    api_module.AstrBotConfig = object

    event_module = types.ModuleType("astrbot.api.event")
    event_module.AstrMessageEvent = object
    event_module.MessageChain = MessageChain
    event_module.filter = filter_stub

    event_filter_module = types.ModuleType("astrbot.api.event.filter")
    event_filter_module.EventMessageType = types.SimpleNamespace(ALL="ALL")
    event_filter_module.PermissionType = types.SimpleNamespace(ADMIN="ADMIN")
    event_filter_module.PlatformAdapterType = types.SimpleNamespace(ALL="ALL")

    message_components_module = types.ModuleType("astrbot.api.message_components")
    message_components_module.Image = Image
    message_components_module.Plain = Plain

    star_module = types.ModuleType("astrbot.api.star")
    star_module.Context = object
    star_module.Star = Star

    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.event.filter"] = event_filter_module
    sys.modules["astrbot.api.message_components"] = message_components_module
    sys.modules["astrbot.api.star"] = star_module

    def _stub_module(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    _stub_module(package_prefix + "cache_service", CacheService=type("CacheService", (), {}))
    _stub_module(
        package_prefix + "task_scheduler",
        TaskScheduler=type("TaskScheduler", (), {}),
    )
    _stub_module(package_prefix + "web_server", WebServer=type("WebServer", (), {}))
    _stub_module(
        package_prefix + "core.commands.command_handler",
        CommandHandler=type("CommandHandler", (), {}),
    )
    _stub_module(
        package_prefix + "core.config.config",
        PluginConfig=type("PluginConfig", (), {}),
    )
    _stub_module(
        package_prefix + "core.search.meme_selector",
        MemeSelector=type("MemeSelector", (), {}),
    )
    _stub_module(
        package_prefix + "core.events.event_handler",
        EventHandler=type("EventHandler", (), {}),
    )
    _stub_module(
        package_prefix + "core.processing.image_processor_service",
        ImageProcessorService=type("ImageProcessorService", (), {}),
    )
    _stub_module(
        package_prefix + "core.processing.natural_emotion_analyzer",
        SmartEmotionMatcher=type("SmartEmotionMatcher", (), {}),
    )
    _stub_module(
        package_prefix + "core.db.database_service",
        DatabaseService=type("DatabaseService", (), {}),
    )
    _stub_module(
        package_prefix + "core.db.index_manager",
        IndexManager=type("IndexManager", (), {}),
    )
    _stub_module(
        package_prefix + "plugin_api",
        PluginAPI=type("PluginAPI", (), {}),
    )

    return package_name


PACKAGE_NAME = _install_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

Main = importlib.import_module(f"{PACKAGE_NAME}.main").Main


class DummyPlain:
    def __init__(self, text: str = ""):
        self.text = text


class DummyResult:
    def __init__(self, text: str):
        self._text = text
        self.chain = [DummyPlain(text)]

    def is_llm_result(self):
        return True

    def get_plain_text(self):
        return "".join(str(getattr(comp, "text", "")) for comp in self.chain)


class DummyEvent:
    def __init__(self, text: str = "hello"):
        self._result = DummyResult(text)
        self._extras = {}
        self.sent = []

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def get_extra(self, key=None, default=None):
        if key is None:
            return dict(self._extras)
        return self._extras.get(key, default)

    def set_extra(self, key, value):
        self._extras[key] = value

    def get_session_id(self):
        return "session-1"

    def get_message_str(self):
        return "user message"

    def get_messages(self):
        return []

    async def send(self, message):
        self.sent.append(message)


def _build_main(chance: float) -> Main:
    main = Main.__new__(Main)
    main.auto_send_meme = True
    main.meme_chance = chance
    # v2.7.5+ 起配置通过 plugin_config 读取；测试桩用 SimpleNamespace 模拟
    main.plugin_config = types.SimpleNamespace(
        enable_natural_emotion_analysis=False,
        steal_meme=True,
        auto_send_meme=True,
    )
    # v2.7.5+ 起调用方走 index_manager；测试桩挂一个空实现，
    # 后续各测试可按需覆盖 load_index / save_index / rebuild_index_from_files

    async def _async_empty_dict():
        return {}

    async def _async_none(_idx=None):
        pass

    main.index_manager = types.SimpleNamespace(
        load_index=_async_empty_dict,
        save_index=_async_none,
        rebuild_index_from_files=_async_empty_dict,
    )
    main.update_config = lambda updates: None
    main._auto_emoji_cooldowns = {}
    main._auto_emoji_cooldowns_lock = asyncio.Lock()
    main._auto_emoji_cooldowns_max = 100
    main._should_skip_auto_emoji_by_gate = lambda text: False
    main.is_send_enabled_for_event = lambda event: True
    main._emoji_sender_engine = MemeSenderEngine(main)
    return main


class AutoEmojiFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_state_wraps_event_extras(self):
        main = _build_main(1.0)
        event = DummyEvent()
        state = main._emoji_turn_state(event)

        self.assertFalse(state.is_active_sent())
        state.mark_active_sent()
        self.assertTrue(state.is_active_sent())

        self.assertFalse(state.is_auto_decided())
        state.set_auto_decision(allowed=True, reason="chance_hit")
        self.assertTrue(state.is_auto_decided())
        self.assertTrue(state.get_auto_allowed())
        self.assertEqual(state.get_auto_reason(), "chance_hit")

        self.assertTrue(state.claim_auto_send_meme())
        self.assertFalse(state.claim_auto_send_meme())

        candidates = [{"path": "a.gif"}]
        state.set_candidates(candidates)
        self.assertEqual(state.get_candidates(), candidates)

    async def test_turn_permission_respects_zero_probability(self):
        main = _build_main(0.0)
        event = DummyEvent()

        allowed = await main._resolve_auto_emoji_turn_permission(event)

        self.assertFalse(allowed)
        self.assertTrue(event.get_extra("stealer_auto_emoji_turn_decided"))
        self.assertFalse(event.get_extra("stealer_auto_emoji_turn_allowed"))
        self.assertEqual(
            event.get_extra("stealer_auto_emoji_turn_reason"), "chance_zero"
        )

    async def test_prepare_emoji_response_leaves_text_unchanged_when_probability_misses(self):
        main = _build_main(0.0)
        event = DummyEvent("hello")

        handled = await main._prepare_emoji_response(event)

        self.assertFalse(handled)
        self.assertEqual(event.get_result().get_plain_text(), "hello")
        self.assertFalse(event.get_extra("stealer_auto_emoji_turn_claimed", False))

    async def test_prepare_emoji_response_only_claims_once_per_turn(self):
        main = _build_main(1.0)
        scheduled = []

        def _safe_create_task(coro, name):
            scheduled.append(name)
            coro.close()

        main._safe_create_task = _safe_create_task
        event = DummyEvent("hello")

        first = await main._prepare_emoji_response(event)
        second = await main._prepare_emoji_response(event)

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(scheduled, ["emoji_analyze_passive"])
        self.assertTrue(event.get_extra("stealer_auto_emoji_turn_claimed"))

    async def test_search_tool_bypasses_probability_gate(self):
        main = _build_main(0.0)
        main.categories = ["happy"]
        main.db_service = types.SimpleNamespace(
            count_total=lambda: 1,
            get_index_cache_readonly=lambda: {__file__: {"source": "local"}}
        )

        async def _search_candidates(_event, _query, limit, idx):
            return [(__file__, "desc", "happy", "tag")]

        main._load_index = lambda: None
        main._search_meme_candidates = _search_candidates
        event = DummyEvent()

        results = []
        async for chunk in main.search_meme(event, "happy"):
            results.append(chunk)

        self.assertTrue(results)
        self.assertNotIn("Auto emoji is disabled for this turn.", results)
        self.assertIsNotNone(main._emoji_turn_state(event).get_candidates())
        result_text = "\n".join(results)
        self.assertIn("send_meme(emoji_id=", result_text)
        self.assertNotIn("send_emoji_by_id", result_text)

    async def test_generic_image_tool_send_suppresses_passive_meme(self):
        main = _build_main(1.0)
        event = DummyEvent("final reply")
        event._has_send_oper = True
        tool = types.SimpleNamespace(name="send_message_to_user")
        tool_result = types.SimpleNamespace(
            content=[types.SimpleNamespace(text="Message sent to session session-1")]
        )

        await main._track_external_image_delivery(
            event,
            tool,
            {"messages": [{"type": "image", "path": "meme.png"}]},
            tool_result,
        )

        scheduled = []

        def _safe_create_task(coro, name):
            scheduled.append(name)
            coro.close()

        main._safe_create_task = _safe_create_task
        handled = await main._prepare_emoji_response(event)

        self.assertTrue(main._emoji_turn_state(event).is_active_sent())
        self.assertFalse(handled)
        self.assertEqual(scheduled, [])

    def test_tool_documentation_uses_exposed_names(self):
        search_doc = Main.search_meme.__doc__ or ""
        send_doc = Main.send_meme.__doc__ or ""

        self.assertIn("send_meme", search_doc)
        self.assertIn("search_meme", send_doc)
        self.assertNotIn("send_emoji_by_id", search_doc)
        self.assertNotIn("search_emoji", send_doc)

    async def test_emoji_selector_no_probability_check(self):
        """验证 MemeSelector.try_send_emoji 不再检查概率，由 Main 在调用前完成判定"""
        main = _build_main(0.0)  # 概率为 0
        event = DummyEvent()

        # 模拟 MemeSelector
        class MockEmojiSelector:
            def __init__(self):
                self.called = False
                self.call_args = None

            async def try_send_emoji(self, event, emotions, cleaned_text):
                # 不应该检查概率，直接尝试选图发图
                self.called = True
                self.call_args = (emotions, cleaned_text)
                return False  # 模拟没有匹配的表情包

        main.meme_selector = MockEmojiSelector()

        # 即使概率为 0，如果 Main 已经判定允许（通过 turn_state），MemeSelector 应该尝试发送
        state = main._emoji_turn_state(event)
        state.set_auto_decision(allowed=True, reason="forced_by_test")

        # 调用 _try_send_emoji
        await main._try_send_emoji(event, ["happy"], "hello")

        # MemeSelector 应被调用，且不检查概率
        self.assertTrue(main.meme_selector.called)
        self.assertEqual(main.meme_selector.call_args, (["happy"], "hello"))

    async def test_auto_decision_made_before_selector_call(self):
        """验证自动发送判定在调用 MemeSelector 前完成"""
        main = _build_main(1.0)
        event = DummyEvent()

        # 判定应该存储在 turn_state 中
        allowed = await main._resolve_auto_emoji_turn_permission(event)
        state = main._emoji_turn_state(event)

        # 验证判定已存储
        self.assertTrue(state.is_auto_decided())
        self.assertEqual(state.get_auto_allowed(), allowed)
        self.assertIsNotNone(state.get_auto_reason())

    async def test_candidates_passed_correctly_to_send_tool(self):
        """验证候选列表通过 turn_state 正确传递"""
        main = _build_main(1.0)
        main.categories = ["happy"]
        event = DummyEvent()

        # 设置候选列表
        candidates = [
            {"path": "/path/a.gif", "desc": "happy", "emotion": "happy"},
            {"path": "/path/b.gif", "desc": "smile", "emotion": "happy"},
        ]
        state = main._emoji_turn_state(event)
        state.set_candidates(candidates)

        # 验证可以正确获取
        retrieved = state.get_candidates()
        self.assertEqual(retrieved, candidates)
        self.assertEqual(len(retrieved), 2)

    async def test_steal_tool_resolves_current_message_image_when_ref_empty(self):
        main = _build_main(1.0)
        message_image_cls = sys.modules[Main.__module__].MessageImage

        class TestImage(message_image_cls):
            pass

        image = TestImage()
        image.url = "https://example.test/a.png"

        class ImageEvent(DummyEvent):
            def get_messages(self):
                return [image]

        class FakeEventHandler:
            def _extract_store_emoji_urls(self, _event):
                return []

        image_ref, source = await main._resolve_steal_image_ref(
            ImageEvent(), "", FakeEventHandler()
        )

        self.assertEqual(image_ref, "https://example.test/a.png")
        self.assertEqual(source, "llm_tool")

    async def test_steal_tool_resolves_relative_path_to_message_image(self):
        """issue #88: LLM 传 ./image.png 相对路径时应当映射回当前消息中的真实 URL。"""
        main = _build_main(1.0)
        message_image_cls = sys.modules[Main.__module__].MessageImage

        class TestImage(message_image_cls):
            pass

        image = TestImage()
        image.url = "https://example.test/abc-123.png"

        class ImageEvent(DummyEvent):
            def get_messages(self):
                return [image]

        class FakeEventHandler:
            def _extract_store_emoji_urls(self, _event):
                return []

        # 相对路径 + basename 都应解析为消息组件的 url
        for ref in (
            "./abc-123.png",
            "abc-123.png",
            ".\\abc-123.png",
            "https://example.test/abc-123.png",  # 完整 URL 维持原样
        ):
            resolved, source = await main._resolve_steal_image_ref(
                ImageEvent(), ref, FakeEventHandler()
            )
            if ref.startswith("http"):
                self.assertEqual(resolved, ref)
            else:
                self.assertEqual(resolved, "https://example.test/abc-123.png")
            self.assertEqual(source, "llm_tool")

    async def test_steal_tool_resolves_absolute_path_via_convert_to_file_path(self):
        """绝对本地路径 + convert_to_file_path 命中应返回组件本地路径。"""
        main = _build_main(1.0)
        message_image_cls = sys.modules[Main.__module__].MessageImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            local_path = fp.name
            fp.write(b"fake png")

        try:

            class TestImage(message_image_cls):
                async def convert_to_file_path(self_inner):
                    return local_path

            image = TestImage()
            image.url = "https://example.test/x.png"
            image.file = local_path
            image.path = local_path

            class ImageEvent(DummyEvent):
                def get_messages(self):
                    return [image]

            class FakeEventHandler:
                def _extract_store_emoji_urls(self, _event):
                    return []

            basename = Path(local_path).name
            resolved, source = await main._resolve_steal_image_ref(
                ImageEvent(), basename, FakeEventHandler()
            )
            self.assertEqual(resolved, local_path)
            self.assertEqual(source, "llm_tool")
        finally:
            Path(local_path).unlink(missing_ok=True)

    async def test_steal_tool_keeps_relative_path_when_no_image_in_message(self):
        """消息内无 Image 组件时，ref 原样回传给下游以提供准确错误提示。"""
        main = _build_main(1.0)

        class EmptyEvent(DummyEvent):
            def get_messages(self):
                return []

        class FakeEventHandler:
            def _extract_store_emoji_urls(self, _event):
                return []

        resolved, source = await main._resolve_steal_image_ref(
            EmptyEvent(), "./orphan.png", FakeEventHandler()
        )
        self.assertEqual(resolved, "./orphan.png")
        self.assertEqual(source, "llm_tool")

    async def test_steal_tool_passes_origin_metadata_to_processor(self):
        main = _build_main(1.0)
        main.steal_meme = True
        main.is_steal_enabled_for_event = lambda event: True
        main.get_event_target = lambda event: ("group", "123")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
            temp_path = fp.name

        class FakeEventHandler:
            def _extract_store_emoji_urls(self, _event):
                return []

            async def _download_to_temp(self, _url, *, log_download=False):
                return temp_path, False

        captured = {}
        saved = []
        main._get_event_handler = lambda **kwargs: FakeEventHandler()
        main._precheck_image_file = lambda path: (True, "")

        async def _load_index():
            return {}

        async def _save_index(idx):
            saved.append(idx)

        async def _process_image(event, file_path, is_temp=False, extra_meta=None, **kwargs):
            captured["extra_meta"] = dict(extra_meta or {})
            return True, {
                file_path: {
                    "category": "happy",
                    "tags": ["tag"],
                    "desc": "desc",
                    "scenes": ["scene"],
                }
            }

        main._load_index = _load_index
        main._save_index = _save_index
        main.index_manager.load_index = _load_index
        main.index_manager.save_index = _save_index
        main._process_image = _process_image

        try:
            results = []
            async for chunk in main.steal_sticker(DummyEvent(), "https://example.test/a.png"):
                results.append(chunk)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        self.assertTrue(any("偷取成功" in item for item in results), f"results={results}")
        self.assertEqual(captured["extra_meta"]["origin_target"], "group:123")
        self.assertEqual(captured["extra_meta"]["origin_url"], "https://example.test/a.png")
        self.assertEqual(captured["extra_meta"]["source"], "llm_tool")
        self.assertTrue(saved)

    async def test_state_keys_are_centralized(self):
        """验证所有状态键都通过 _MemeTurnState 集中管理"""
        main = _build_main(1.0)
        event = DummyEvent()
        state = main._emoji_turn_state(event)

        # 测试所有状态键的封装
        # active_sent
        self.assertFalse(state.is_active_sent())
        state.mark_active_sent()
        self.assertTrue(state.is_active_sent())

        # auto_decision
        state.set_auto_decision(allowed=True, reason="test")
        self.assertTrue(state.is_auto_decided())
        self.assertTrue(state.get_auto_allowed())
        self.assertEqual(state.get_auto_reason(), "test")

        # claim
        self.assertFalse(state.is_auto_claimed())
        self.assertTrue(state.claim_auto_send_meme())
        self.assertTrue(state.is_auto_claimed())
        self.assertFalse(state.claim_auto_send_meme())  # 第二次 claim 失败

        # candidates
        state.set_candidates([{"path": "x"}])
        self.assertEqual(state.get_candidates(), [{"path": "x"}])

    async def test_main_responsibility_boundary(self):
        """验证 Main 和 MemeSelector 的职责边界：Main 负责判定，Selector 负责选图发图"""
        main = _build_main(0.5)
        event = DummyEvent()

        # 记录调用顺序
        call_log = []

        # 模拟判定流程
        async def mock_resolve(event):
            call_log.append("resolve_permission")
            return True  # 假设通过判定

        # 模拟 MemeSelector 发送
        class MockSelector:
            async def try_send_emoji(self, event, emotions, text):
                call_log.append("selector_send")
                return True

        main._resolve_auto_emoji_turn_permission = mock_resolve
        main.meme_selector = MockSelector()

        # 执行完整流程
        allowed = await main._resolve_auto_emoji_turn_permission(event)
        if allowed:
            await main._try_send_emoji(event, ["happy"], "hello")

        # 验证调用顺序：判定先于发送
        self.assertEqual(call_log, ["resolve_permission", "selector_send"])
