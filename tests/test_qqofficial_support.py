"""
QQ 官方平台（qq_official）表情判定测试

QQ 官方（官方机器人 API）的 raw_message 是 botpy SDK 对象而非 OneBot dict 段列表，
无法提取 sub_type/summary 标记。本测试验证 PlatformDetector 的 QQ 官方分支：
按图片 URL 的 CDN 特征 / qqofficial_steal_all_images 配置判定表情。
"""

from types import SimpleNamespace

from astrbot_plugin_stealer.core.events.platform_detector import PlatformDetector


class _FakePlugin:
    """模拟插件实例（仅提供 plugin_config）。"""

    def __init__(self, steal_all_images: bool = False):
        self.plugin_config = SimpleNamespace(qqofficial_steal_all_images=steal_all_images)


class _FakeEvent:
    """模拟 QQ 官方平台事件（raw_message 是对象而非 dict 列表）。"""

    def __init__(self):
        self.message_obj = SimpleNamespace(raw_message=object())

    def get_platform_name(self) -> str:
        return "qq_official"


class _FakeImage:
    """模拟 Image 组件（QQ 官方为 Image.fromURL，URL 存在 file 字段）。"""

    def __init__(self, file: str = "", url: str = ""):
        self.file = file
        self.url = url


def _detector(steal_all_images: bool = False) -> PlatformDetector:
    return PlatformDetector(_FakePlugin(steal_all_images))


class TestQQOfficialEmojiDetection:
    """QQ 官方平台表情判定"""

    def test_emoji_cdn_url_detected(self):
        """URL 带 QQ 表情 CDN 特征 → 判定为表情"""
        det = _detector(steal_all_images=False)
        img = _FakeImage(file="https://gxh.vip.qq.com/xxx/parcel.jpg")
        assert det.check_platform_emoji_metadata(img, _FakeEvent()) is True

    def test_vip_parcel_url_detected(self):
        """vip.qq.com/club/item/parcel 特征 → 判定为表情"""
        det = _detector()
        img = _FakeImage(file="https://vip.qq.com/club/item/parcel/123/abc.png")
        assert det.check_platform_emoji_metadata(img, _FakeEvent()) is True

    def test_plain_image_without_steal_all(self):
        """普通图片 URL + 未开启全收 → 判定非表情"""
        det = _detector(steal_all_images=False)
        img = _FakeImage(file="https://example.com/a.png")
        assert det.check_platform_emoji_metadata(img, _FakeEvent()) is False

    def test_plain_image_with_steal_all(self):
        """普通图片 URL + 开启全收 → 判定为表情"""
        det = _detector(steal_all_images=True)
        img = _FakeImage(file="https://example.com/a.png")
        assert det.check_platform_emoji_metadata(img, _FakeEvent()) is True

    def test_empty_ref(self):
        """无 file/url → 判定非表情"""
        det = _detector(steal_all_images=True)
        img = _FakeImage()
        assert det.check_platform_emoji_metadata(img, _FakeEvent()) is False

    def test_non_qqofficial_platform_untouched(self):
        """非 QQ 官方平台不受新分支影响（走原 OneBot 逻辑）"""

        class _OneBotEvent:
            def get_platform_name(self) -> str:
                return "aiocqhttp"

            def __init__(self):
                self.message_obj = SimpleNamespace(
                    raw_message={"message": [{"type": "image", "data": {"sub_type": 1}}]}
                )

        det = _detector(steal_all_images=False)
        img = _FakeImage(file="https://example.com/a.png")
        # OneBot sub_type=1 仍是表情（与 event_handler 一致：传 img_index 匹配原始段）
        assert det.check_platform_emoji_metadata(img, _OneBotEvent(), img_index=0) is True
