"""平台相关的表情包投递优化。"""

import os
from typing import Any

from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image


async def send_qq_image_as_sticker(
    event: AstrMessageEvent,
    file_path: str,
    summary: str = "[动画表情]",
    plugin: Any = None,
) -> bool:
    """在 QQ (aiocqhttp) 平台发送表情包时修改 summary 外显。"""
    try:
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
            AiocqhttpMessageEvent,
        )
    except ImportError:
        return False
    if not isinstance(event, AiocqhttpMessageEvent):
        return False
    if not file_path or not os.path.exists(file_path):
        return False

    # 这里使用适配器内部接口做 QQ 专属优化；任何失败都交给调用方的标准
    # event.send 路径回退，避免平台版本变化中断回复。
    try:
        file_source: str = file_path
        if plugin and getattr(plugin, "send_meme_as_gif", False):
            image_processor = getattr(plugin, "image_processor_service", None)
            if image_processor:
                encoded = await image_processor._file_to_gif_base64(file_path)
                if encoded:
                    file_source = f"base64://{encoded}"
        chain = MessageChain(chain=[Image(file=file_source)])
        onebot_message = await event._parse_onebot_json(chain)
        onebot_message[0]["data"]["summary"] = summary
        await event.bot.send(event.message_obj.raw_message, onebot_message)
        return True
    except Exception:
        return False
