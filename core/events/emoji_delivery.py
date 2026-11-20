"""平台相关的表情包投递优化。"""

import os
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image


async def send_qq_image_as_sticker(
    event: AstrMessageEvent,
    file_path: str,
    summary: str = "[动画表情]",
    plugin: Any = None,
) -> bool:
    """在 QQ (aiocqhttp) 平台以自定义表情类型发送图片。"""
    if plugin is not None and not plugin.plugin_config.send_meme_as_qq_sticker:
        return False
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

    try:
        file_source: str = file_path
        if plugin and plugin.plugin_config.send_meme_as_gif:
            renderer = getattr(plugin, "image_render_service", None)
            if renderer:
                encoded = await renderer.file_to_gif_base64(file_path)
                if encoded:
                    file_source = f"base64://{encoded}"
        chain = MessageChain([Image(file=file_source)])
        onebot_message = await event._parse_onebot_json(chain)
        onebot_message[0]["data"]["summary"] = summary
        # NapCat: 0 为普通图片，1 为自定义表情；summary 只修改摘要。
        onebot_message[0]["data"]["sub_type"] = 1
        await event.bot.send(event.message_obj.raw_message, onebot_message)
        return True
    except Exception as e:
        if "waitForSelfEcho timeout" in str(e):
            # 回执超时时消息可能已送达，继续回退会重复发送。
            logger.warning("[Stealer] QQ 表情发送回执超时，发送状态待确认，停止补发")
            raise
        logger.warning(f"[Stealer] QQ 表情发送失败，改用通用图片接口发送: {e}")
        return False
