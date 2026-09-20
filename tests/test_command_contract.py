"""Static command registration and delegation contract checks."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_real_startup_registers_api_and_list_command_works_with_empty_database():
    from astrbot_plugin_stealer.main import Main

    routes = {}

    def register(path, handler, methods, description):
        routes[path] = handler

    plugin = Main(SimpleNamespace(register_web_api=register))
    messages = [
        message async for message in plugin.list_images(
            SimpleNamespace(plain_result=lambda text: text)
        )
    ]
    assert routes["/astrbot_plugin_stealer/analyze"] == plugin.plugin_api.handle_analyze_image
    assert messages and "暂无" in messages[0]
    assert plugin.db_service.count_total() == 0


def _main_tree() -> ast.Module:
    return ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))


def test_meme_command_group_and_expected_subcommands_are_registered():
    tree = _main_tree()
    groups = set()
    commands = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            arg = decorator.args[0]
            if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
                continue
            func = decorator.func
            if isinstance(func, ast.Attribute) and func.attr == "command_group":
                groups.add(arg.value)
            elif isinstance(func, ast.Attribute) and func.attr == "command":
                commands.add(arg.value)

    assert "meme" in groups
    assert commands == {
        "on", "off", "auto_on", "auto_off", "group", "偷",
        "natural_analysis", "emotion_stats", "clear_emotion_cache", "status",
        "tag_stats", "clean", "capacity", "list", "delete", "blacklist",
        "scope", "rebuild_index",
    }


def test_registered_command_methods_route_to_their_handlers():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    delegated = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if any(f"self.{name}." in segment for name in ("command_handler", "image_commands", "index_commands", "target_commands")):
            delegated.add(node.name)
    assert {
        "meme_on", "meme_off", "auto_on", "auto_off", "group_filter", "capture",
        "toggle_natural_analysis", "emotion_analysis_stats", "clear_emotion_cache",
        "status", "tag_stats", "clean", "enforce_capacity", "list_images",
        "delete_image", "blacklist_image", "set_image_scope", "rebuild_index",
    } <= delegated
