import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from astrbot_plugin_stealer.core.config.config import PluginConfig
from astrbot_plugin_stealer.core.events import emoji_delivery


@pytest.fixture
def qq_delivery(monkeypatch, tmp_path):
    class QQEvent:
        pass

    module_name = "astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event"
    module = types.ModuleType(module_name)
    module.AiocqhttpMessageEvent = QQEvent
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(emoji_delivery, "Image", lambda **kwargs: kwargs)
    monkeypatch.setattr(emoji_delivery, "MessageChain", list)
    event = QQEvent()
    event._parse_onebot_json = AsyncMock(
        return_value=[{"type": "image", "data": {"file": "base64://image"}}]
    )
    event.bot = types.SimpleNamespace(send=AsyncMock())
    event.message_obj = types.SimpleNamespace(raw_message={"group_id": 123})
    path = tmp_path / "meme.png"
    path.write_bytes(b"image")
    return event, str(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [True, None])
@pytest.mark.parametrize("as_gif", [False, True])
async def test_qq_sticker_type_with_enabled_or_legacy_config(
    qq_delivery, setting, as_gif
):
    event, path = qq_delivery
    renderer = types.SimpleNamespace(file_to_gif_base64=AsyncMock(return_value="gif"))
    plugin = types.SimpleNamespace(
        send_meme_as_gif=as_gif, image_render_service=renderer
    )
    if setting is not None:
        plugin.send_meme_as_qq_sticker = setting

    assert await emoji_delivery.send_qq_image_as_sticker(event, path, plugin=plugin)

    event.bot.send.assert_awaited_once_with(
        event.message_obj.raw_message,
        [
            {
                "type": "image",
                "data": {
                    "file": "base64://image",
                    "summary": "[动画表情]",
                    "sub_type": 1,
                },
            }
        ],
    )
    expected_source = "base64://gif" if as_gif else path
    event._parse_onebot_json.assert_awaited_once_with([{"file": expected_source}])


@pytest.mark.asyncio
async def test_disabled_sticker_returns_control_to_normal_image_delivery(qq_delivery):
    event, path = qq_delivery
    plugin = types.SimpleNamespace(send_meme_as_qq_sticker=False)

    assert not await emoji_delivery.send_qq_image_as_sticker(event, path, plugin=plugin)
    event._parse_onebot_json.assert_not_awaited()
    event.bot.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_failure_logs_and_allows_fallback(qq_delivery, monkeypatch):
    event, path = qq_delivery
    event.bot.send.side_effect = RuntimeError("send failed")
    logger = Mock()
    monkeypatch.setattr(emoji_delivery, "logger", logger)

    assert not await emoji_delivery.send_qq_image_as_sticker(event, path)
    logger.warning.assert_called_once()
    assert "send failed" in logger.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_non_qq_platform_keeps_existing_delivery(qq_delivery):
    _, path = qq_delivery
    event = types.SimpleNamespace(bot=types.SimpleNamespace(send=AsyncMock()))

    assert not await emoji_delivery.send_qq_image_as_sticker(event, path)
    event.bot.send.assert_not_awaited()


def test_sticker_setting_schema_and_config_default():
    root = Path(__file__).parents[1]
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["send_meme_as_qq_sticker"]["type"] == "bool"
    assert schema["send_meme_as_qq_sticker"]["default"] is True
    assert PluginConfig.model_fields["send_meme_as_qq_sticker"].default is True
    for locale in ("zh-CN", "en-US", "ru-RU"):
        messages = json.loads(
            (root / ".astrbot-plugin" / "i18n" / f"{locale}.json").read_text(
                encoding="utf-8"
            )
        )
        assert "send_meme_as_qq_sticker" in messages["config"]
