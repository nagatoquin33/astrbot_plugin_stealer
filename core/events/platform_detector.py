"""平台检测器：负责平台识别、元信息提取和 URL 解析。"""

from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class PlatformDetector:
    """负责检测消息来源平台并提取表情包相关的元信息。"""

    def __init__(self, plugin_instance: Any = None) -> None:
        self.plugin = plugin_instance

    @staticmethod
    def _normalize_str(value: object) -> str:
        """规范化字符串值。"""
        if value is None:
            return ""
        try:
            s = str(value)
        except Exception:
            return ""
        s = s.strip()
        if s.startswith("`") and s.endswith("`") and len(s) >= 2:
            s = s[1:-1].strip()
        return s

    def get_platform_name(self, event: AstrMessageEvent | None = None) -> str:
        """获取事件平台名（小写），失败时返回空字符串。"""
        if event is None:
            return ""

        for getter in ("get_platform_name", "get_platform_id"):
            fn = getattr(event, getter, None)
            if callable(fn):
                try:
                    name = self._normalize_str(fn()).lower()
                    if name:
                        return name
                except Exception:
                    pass

        return ""

    def is_telegram_event(self, event: AstrMessageEvent | None = None) -> bool:
        """判断事件是否来自 Telegram 平台。"""
        platform_name = self.get_platform_name(event)
        return platform_name == "telegram"

    def check_platform_emoji_metadata(
        self,
        img: object,
        event: AstrMessageEvent | None = None,
        img_index: int | None = None,
        image_segments: list[dict] | None = None,
        image_file_map: dict[str, dict] | None = None,
    ) -> bool:
        """检查图片元信息，判断是否为平台标记的表情包。

        支持的平台特征：
        - NapCat/OneBot: subType=1 或 sub_type=1 表示表情包
        - QQ: summary包含"表情"关键词

        Args:
            img: 图片组件
            event: 消息事件对象（可选），用于访问原始消息数据

        Returns:
            bool: 是否为平台标记的表情包
        """
        try:

            def is_emoji_summary(summary: object) -> bool:
                s = self._normalize_str(summary)
                if not s:
                    return False
                s_lower = s.lower()
                return "表情" in s or "emoji" in s_lower or "sticker" in s_lower

            def is_sub_type_emoji(sub_type: object) -> bool:
                if sub_type is None:
                    return False
                if sub_type == 1 or sub_type == "1":
                    return True
                try:
                    return int(sub_type) == 1
                except Exception:
                    return False

            # Telegram 兼容：优先识别 sticker，再兜底 .webp 贴纸文件
            if self.is_telegram_event(event):
                try:
                    raw_event = getattr(getattr(event, "message_obj", None), "raw_message", None)
                    tg_message = getattr(raw_event, "message", None)
                    tg_sticker = getattr(tg_message, "sticker", None) if tg_message else None
                    if tg_sticker is not None:
                        is_animated = bool(getattr(tg_sticker, "is_animated", False))
                        is_video = bool(getattr(tg_sticker, "is_video", False))
                        if is_animated or is_video:
                            logger.debug("检测到 Telegram 动态贴纸，当前跳过收录")
                            return False
                        logger.debug("检测到 Telegram 静态贴纸")
                        return True
                except Exception:
                    pass

                img_file = self._normalize_str(getattr(img, "file", "")).lower()
                img_url = self._normalize_str(getattr(img, "url", "")).lower()
                if img_file.endswith(".webp") or img_url.endswith(".webp"):
                    logger.debug("检测到 Telegram WebP 贴纸文件")
                    return True

            # 方式0: 从原始事件中查找 sub_type (最可靠的方法)
            if (
                image_segments is None
                and event
                and hasattr(event, "message_obj")
                and hasattr(event.message_obj, "raw_message")
            ):
                raw_event = event.message_obj.raw_message
                logger.debug(f"[EmojiCheck] 方式0: raw_event type={type(raw_event).__name__}, is_dict={isinstance(raw_event, dict)}")
                # raw_event 可能是 dict（aiocqhttp）或对象，需要兼容两种类型
                if isinstance(raw_event, dict):
                    raw_message = raw_event.get("message")
                elif raw_event is not None:
                    raw_message = getattr(raw_event, "message", None)
                logger.debug(f"[EmojiCheck] 方式0: raw_message type={type(raw_message).__name__ if raw_message else 'None'}")
                if isinstance(raw_message, list):
                    image_segments = [
                        seg
                        for seg in raw_message
                        if isinstance(seg, dict) and seg.get("type") == "image"
                    ]
                    logger.debug(f"[EmojiCheck] 方式0: 提取到 {len(image_segments)} 个图片段")

            if image_segments:
                logger.debug(f"[EmojiCheck] 有 {len(image_segments)} 个 image_segments 可匹配, img_index={img_index}")
                matched_data: dict[str, object] | None = None

                if (
                    img_index is not None
                    and 0 <= img_index < len(image_segments)
                    and isinstance(image_segments[img_index], dict)
                ):
                    matched_data = image_segments[img_index].get("data", {}) or {}
                else:
                    img_file = self._normalize_str(getattr(img, "file", ""))
                    img_url = self._normalize_str(getattr(img, "url", ""))
                    img_file_unique = self._normalize_str(getattr(img, "file_unique", ""))

                    if image_file_map and img_file:
                        matched_data = image_file_map.get(img_file)
                        if matched_data is None and img_file_unique:
                            matched_data = image_file_map.get(img_file_unique)

                    if matched_data is None:
                        for seg in image_segments:
                            if not isinstance(seg, dict):
                                continue
                            data = seg.get("data", {}) or {}
                            if not isinstance(data, dict):
                                continue
                            seg_file = self._normalize_str(data.get("file", ""))
                            seg_url = self._normalize_str(data.get("url", ""))

                            if seg_file and (
                                seg_file == img_file
                                or (img_file_unique and seg_file == img_file_unique)
                                or (img_url and seg_file in img_url)
                                or (img_file and seg_file in img_file)
                            ):
                                matched_data = data
                                break

                            if seg_url and (
                                (img_url and seg_url == img_url)
                                or (img_file and seg_url in img_file)
                            ):
                                matched_data = data
                                break

                if matched_data is not None:
                    logger.debug(f"[EmojiCheck] matched_data keys: {list(matched_data.keys())}")
                    sub_type = matched_data.get("sub_type")
                    if is_sub_type_emoji(sub_type):
                        logger.debug(f"检测到表情包标记: sub_type={sub_type} (从原始事件)")
                        return True

                    summary = matched_data.get("summary", "")
                    if is_emoji_summary(summary):
                        logger.debug(f"检测到表情包标记: summary='{summary}' (从原始事件)")
                        return True

                    # QQ 官方商城表情包（raw data 常见字段：emoji_id / emoji_package_id / key）
                    if matched_data.get("emoji_id") or matched_data.get("emoji_package_id"):
                        logger.debug("检测到表情包标记: emoji_id/emoji_package_id (从原始事件)")
                        return True

                    url = self._normalize_str(matched_data.get("url", ""))
                    if "vip.qq.com/club/item/parcel" in url or "gxh.vip.qq.com" in url:
                        logger.debug("检测到表情包标记: QQ 商城 CDN URL (从原始事件)")
                        return True

            # 方式1: 检查 Image 对象的 subType 字段
            if hasattr(img, "subType") and img.subType:
                if is_sub_type_emoji(img.subType):
                    logger.debug(f"检测到表情包标记: subType={img.subType}")
                    return True

            # 方式2: 检查 __dict__ 中的 sub_type
            if hasattr(img, "__dict__"):
                img_dict = img.__dict__
                sub_type_underscore = img_dict.get("sub_type")
                if is_sub_type_emoji(sub_type_underscore):
                    logger.debug(f"检测到表情包标记: sub_type={sub_type_underscore} (从__dict__)")
                    return True

            # 方式3: 通过 toDict() 检查
            try:
                raw_data = img.toDict()
                if isinstance(raw_data, dict) and "data" in raw_data:
                    data = raw_data["data"]

                    sub_type = data.get("sub_type") or data.get("subType")
                    if is_sub_type_emoji(sub_type):
                        logger.debug(f"检测到表情包标记: sub_type={sub_type} (从toDict)")
                        return True

                    summary = data.get("summary", "")
                    if is_emoji_summary(summary):
                        logger.debug(f"检测到表情包标记: summary='{summary}'")
                        return True

                    if data.get("emoji_id") or data.get("emoji_package_id"):
                        logger.debug("检测到表情包标记: emoji_id/emoji_package_id (从toDict)")
                        return True

                    img_type = data.get("type") or data.get("imageType") or data.get("image_type")
                    if img_type in ["emoji", "sticker", "face", "meme"]:
                        logger.debug(f"检测到表情包标记: type='{img_type}'")
                        return True
            except Exception as e:
                logger.debug(f"无法获取图片字典数据: {e}")

            return False

        except Exception as e:
            logger.debug(f"检查平台表情包元信息失败: {e}")
            return False

    def extract_store_emoji_urls(self, event: AstrMessageEvent) -> list[str]:
        """从 OneBot raw_message 里提取 QQ 商城表情（marketface/mface）的可下载 URL。"""
        urls: list[str] = []
        seen: set[str] = set()
        try:
            raw_event = getattr(getattr(event, "message_obj", None), "raw_message", None)
            raw_message = getattr(raw_event, "message", None)
            if not isinstance(raw_message, list):
                return []

            for seg in raw_message:
                if not isinstance(seg, dict):
                    continue
                seg_type = self._normalize_str(seg.get("type", "")).lower()
                if seg_type not in {"marketface", "mface"}:
                    continue
                data = seg.get("data", {}) or {}
                if not isinstance(data, dict):
                    continue

                # 常见字段优先
                for key in (
                    "url",
                    "cdnurl",
                    "cdn_url",
                    "raw_url",
                    "origin_url",
                    "original_url",
                    "thumb",
                    "thumb_url",
                ):
                    v = data.get(key)
                    s = self._normalize_str(v)
                    if s.startswith("http://") or s.startswith("https://"):
                        if s not in seen:
                            seen.add(s)
                            urls.append(s)

                # 再兜底扫描一遍所有字符串值
                if not urls:
                    for v in data.values():
                        s = self._normalize_str(v)
                        if s.startswith("http://") or s.startswith("https://"):
                            if s not in seen:
                                seen.add(s)
                                urls.append(s)
        except Exception:
            return urls

        return urls
