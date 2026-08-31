"""
pytest 配置文件 - 统一管理 astrbot API stubs
所有测试必须使用此文件定义的共享 stubs 类
"""

import sys
import types
from pathlib import Path

# 确保项目路径可用
sys.path.insert(0, str(Path(__file__).parent.parent))


# 定义共享的 stubs 类（所有测试必须使用这些类）
class _SharedImage:
    """所有测试共享的 Image stub 类"""

    @classmethod
    def fromBase64(cls, value):
        return f"b64:{value}"

    async def convert_to_file_path(self):
        return ""


class _SharedPlain:
    """所有测试共享的 Plain stub 类"""

    def __init__(self, text: str = ""):
        self.text = text


class _SharedMessageChain(list):
    """所有测试共享的 MessageChain stub 类"""
    pass


class _SharedAstrMessageEvent:
    """所有测试共享的 AstrMessageEvent stub 类"""
    pass


def _install_compatible_stubs() -> None:
    """安装兼容所有测试的 astrbot API stubs"""
    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    # 创建模块
    astrbot_module = types.ModuleType("astrbot")
    api_module = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    event_filter_module = types.ModuleType("astrbot.api.event.filter")
    message_components_module = types.ModuleType("astrbot.api.message_components")
    star_module = types.ModuleType("astrbot.api.star")

    # 设置属性
    api_module.logger = logger
    api_module.AstrBotConfig = object

    event_module.AstrMessageEvent = _SharedAstrMessageEvent
    event_module.MessageChain = _SharedMessageChain

    # filter stub
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
    event_module.filter = filter_stub

    event_filter_module.EventMessageType = types.SimpleNamespace(ALL="ALL")
    event_filter_module.PermissionType = types.SimpleNamespace(ADMIN="ADMIN")
    event_filter_module.PlatformAdapterType = types.SimpleNamespace(ALL="ALL")

    # 使用共享的 stubs 类
    message_components_module.Image = _SharedImage
    message_components_module.Plain = _SharedPlain

    class Star:
        def __init__(self, context=None):
            self.context = context

    star_module.Context = object
    star_module.Star = Star
    star_module.StarTools = object

    # 创建 astrbot.core.agent.message stub
    agent_message_module = types.ModuleType("astrbot.core.agent.message")
    agent_message_module.TextPart = object
    sys.modules["astrbot.core"] = types.ModuleType("astrbot.core")
    sys.modules["astrbot.core.agent"] = types.ModuleType("astrbot.core.agent")
    sys.modules["astrbot.core.agent.message"] = agent_message_module

    # 创建 astrbot.core.message stub（meme_sender_engine 等需要）
    core_message_module = types.ModuleType("astrbot.core.message")
    sys.modules["astrbot.core.message"] = core_message_module
    core_message_event_result_module = types.ModuleType(
        "astrbot.core.message.message_event_result"
    )
    core_message_event_result_module.MessageChain = list
    sys.modules[
        "astrbot.core.message.message_event_result"
    ] = core_message_event_result_module

    # 创建 astrbot.core.agent.run_context stub（v4.26+）
    run_context_module = types.ModuleType("astrbot.core.agent.run_context")

    class _SharedContextWrapper:
        """v4.26+ LLM 工具接收 ContextWrapper，context.event 为 AstrMessageEvent。"""

        def __init__(self, event=None):
            self.context = types.SimpleNamespace(event=event)

    run_context_module.ContextWrapper = _SharedContextWrapper
    sys.modules["astrbot.core.agent.run_context"] = run_context_module

    # 创建 quart stub（plugin_api 需要）
    if "quart" not in sys.modules:
        quart_module = types.ModuleType("quart")
        quart_module.request = object()
        quart_module.jsonify = lambda *args, **kwargs: {"jsonified": True}
        quart_module.send_file = lambda *args, **kwargs: {"send_file": True}
        sys.modules["quart"] = quart_module

    # 创建 aiofiles stub（main.py 可选导入）
    if "aiofiles" not in sys.modules:
        aiofiles_module = types.ModuleType("aiofiles")
        sys.modules["aiofiles"] = aiofiles_module

    # 注册模块（强制覆盖）
    sys.modules["astrbot"] = astrbot_module
    sys.modules["astrbot.api"] = api_module
    sys.modules["astrbot.api.event"] = event_module
    sys.modules["astrbot.api.event.filter"] = event_filter_module
    sys.modules["astrbot.api.message_components"] = message_components_module
    sys.modules["astrbot.api.star"] = star_module

    # 重新加载已导入的模块以使用新的 stubs
    import importlib

    package_name = Path(__file__).parent.parent.name
    modules_to_reload = [
        f"{package_name}.core.events.event_handler",
        f"{package_name}.core.events.platform_detector",
        f"{package_name}.core.events.image_download_service",
        f"{package_name}.core.events.meme_sender_engine",
        f"{package_name}.core.search.meme_selector",
        f"{package_name}.core.search.meme_search_engine",
        f"{package_name}.core.search.meme_selection_strategy",
        f"{package_name}.core.search.meme_smart_select_service",
        f"{package_name}.core.search.meme_scope_service",
        f"{package_name}.core.search.text_similarity",
        f"{package_name}.core.commands.command_handler",
        f"{package_name}.core.commands.target_filter_command",
        f"{package_name}.core.commands.image_mgmt_command",
        f"{package_name}.core.commands.index_rebuild_command",
        f"{package_name}.core.processing.image_processor_service",
        f"{package_name}.core.processing.image_render_service",
        f"{package_name}.core.processing.phash_dedup_service",
        f"{package_name}.core.processing.prompt_manager",
        f"{package_name}.core.processing.classification_parser",
        f"{package_name}.core.processing.vlm_call_service",
        f"{package_name}.core.processing.natural_emotion_analyzer",
        f"{package_name}.core.db.database_service",
        f"{package_name}.core.db.index_manager",
        f"{package_name}.core.config.config",
        f"{package_name}.api.image_handler",
        f"{package_name}.api.batch_handler",
        f"{package_name}.api.category_handler",
        f"{package_name}.plugin_api",
        f"{package_name}.cache_service",
        f"{package_name}.task_scheduler",
        f"{package_name}.main",
    ]
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
            except Exception:
                pass


def pytest_configure(config):
    """pytest hook - 在收集测试前安装 stubs"""
    _install_compatible_stubs()


# 导出共享类供其他测试模块使用
SHARED_IMAGE_CLASS = _SharedImage
SHARED_PLAIN_CLASS = _SharedPlain
SHARED_MESSAGE_CHAIN_CLASS = _SharedMessageChain


# 立即安装 stubs（在 pytest_configure 之前）
_install_compatible_stubs()
