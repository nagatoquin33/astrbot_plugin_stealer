import random
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image


class CommandHandler:
    """命令处理服务类，负责处理所有与插件相关的命令操作。"""

    def __init__(self, plugin_instance):
        """初始化命令处理服务。

        Args:
            plugin_instance: StealerPlugin 实例，用于访问插件的配置和服务
        """
        self.plugin = plugin_instance

    async def meme_on(self, event: AstrMessageEvent):
        """开启偷表情包功能。"""
        self.plugin.steal_emoji = True
        self.plugin._persist_config()
        return event.plain_result("已开启偷表情包")

    async def meme_off(self, event: AstrMessageEvent):
        """关闭偷表情包功能。"""
        self.plugin.steal_emoji = False
        self.plugin._persist_config()
        return event.plain_result("已关闭偷表情包")

    async def auto_on(self, event: AstrMessageEvent):
        """开启自动发送功能。"""
        self.plugin.auto_send = True
        self.plugin._persist_config()
        return event.plain_result("已开启自动发送")

    async def auto_off(self, event: AstrMessageEvent):
        """关闭自动发送功能。"""
        self.plugin.auto_send = False
        self.plugin._persist_config()
        return event.plain_result("已关闭自动发送")

    async def set_vision(self, event: AstrMessageEvent, provider_id: str = ""):
        """设置视觉模型。"""
        if not provider_id:
            return event.plain_result("请提供视觉模型的 provider_id")
        self.plugin.vision_provider_id = provider_id
        self.plugin._persist_config()
        return event.plain_result(f"已设置视觉模型: {provider_id}")

    async def show_providers(self, event: AstrMessageEvent):
        """显示当前视觉模型。"""
        vp = self.plugin.vision_provider_id or "当前会话"
        return event.plain_result(f"视觉模型: {vp}")



    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        st_on = "开启" if self.plugin.enabled else "关闭"
        st_auto = "开启" if self.plugin.auto_send else "关闭"

        idx = await self.plugin._load_index()
        # 添加视觉模型信息
        vision_model = self.plugin.vision_provider_id or "未设置（将使用当前会话默认模型）"
        return event.plain_result(
            f"偷取: {st_on}\n自动发送: {st_auto}\n已注册数量: {len(idx)}\n概率: {self.plugin.emoji_chance}\n上限: {self.plugin.max_reg_num}\n替换: {self.plugin.do_replace}\n维护周期: {self.plugin.maintenance_interval}min\n审核: {self.plugin.content_filtration}\n视觉模型: {vision_model}"
        )

    async def push(self, event: AstrMessageEvent, category: str = "", alias: str = ""):
        """手动推送指定分类的表情包。"""
        if not self.plugin.base_dir:
            return event.plain_result("插件未正确配置，缺少图片存储目录")
        if alias:
            aliases = await self.plugin._load_aliases()
            if alias in aliases:
                aliases[alias]
            else:
                return event.plain_result("别名不存在")
        cat = category or (self.plugin.categories[0] if self.plugin.categories else "happy")
        cat_dir = self.plugin.base_dir / "categories" / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            return event.plain_result(f"分类 {cat} 不存在")
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            return event.plain_result("该分类暂无表情包")
        pick = random.choice(files)
        b64 = await self.plugin._file_to_base64(pick.as_posix())
        chain = event.make_result().base64_image(b64).message_chain
        return event.result_with_message_chain(chain)

    async def debug_image(self, event: AstrMessageEvent):
        """调试命令：处理当前消息中的图片并显示详细信息。"""
        # 收集所有图片组件
        imgs = [comp for comp in event.message_obj.message if isinstance(comp, Image)]

        if not imgs:
            return event.plain_result("当前消息中没有图片")

        # 处理第一张图片
        img = imgs[0]
        try:
            # 转换图片到临时文件路径
            temp_path = await img.convert_to_file_path()

            # 检查路径安全性
            is_safe, safe_path = self.plugin._is_safe_path(temp_path)
            if not is_safe:
                return event.plain_result("图片路径不安全")

            temp_path = safe_path

            # 确保临时文件存在且可访问
            if not Path(temp_path).exists():
                return event.plain_result("临时文件不存在")

            # 开始调试处理
            result_msg = "=== 图片调试信息 ===\n"

            # 1. 基本信息
            file_path = Path(temp_path)
            size = file_path.stat().st_size
            result_msg += f"文件大小: {size / 1024:.2f} KB\n"

            # 2. 元数据过滤结果
            # 直接使用plugin中的PILImage引用
            if self.plugin.PILImage is not None:
                try:
                    with self.plugin.PILImage.open(temp_path) as im:
                        width, height = im.size
                    result_msg += f"分辨率: {width}x{height}\n"
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                    result_msg += f"宽高比: {aspect_ratio:.2f}\n"
                except Exception as e:
                    result_msg += f"获取图片信息失败: {e}\n"

            # 3. 多模态分析结果
            result_msg += "\n=== 多模态分析结果 ===\n"

            # 处理图片
            success, idx = await self.plugin._process_image(event, temp_path, is_temp=True, idx=None)
            if success and idx:
                for file_path, info in idx.items():
                    if isinstance(info, dict):
                        result_msg += f"分类: {info.get('category', '未知')}\n"
                        result_msg += f"情绪: {info.get('emotion', '未知')}\n"
                        result_msg += f"标签: {info.get('tags', [])}\n"
                        result_msg += f"描述: {info.get('desc', '无')}\n"
            else:
                result_msg += "图片处理失败\n"

            return event.plain_result(result_msg)

        except Exception as e:
            logger.error(f"调试图片失败: {e}")
            return event.plain_result(f"调试失败: {str(e)}")
