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
        vision_provider = self.plugin.vision_provider_id or "当前会话"
        return event.plain_result(f"视觉模型: {vision_provider}")

    async def status(self, event: AstrMessageEvent):
        """显示当前偷取状态与后台标识。"""
        stealing_status = "开启" if self.plugin.enabled else "关闭"
        auto_send_status = "开启" if self.plugin.auto_send else "关闭"

        image_index = await self.plugin._load_index()
        # 添加视觉模型信息
        vision_model = (
            self.plugin.vision_provider_id or "未设置（将使用当前会话默认模型）"
        )
        return event.plain_result(
            f"偷取: {stealing_status}\n自动发送: {auto_send_status}\n已注册数量: {len(image_index)}\n概率: {self.plugin.emoji_chance}\n上限: {self.plugin.max_reg_num}\n替换: {self.plugin.do_replace}\n维护周期: {self.plugin.maintenance_interval}min\n审核: {self.plugin.content_filtration}\n视觉模型: {vision_model}"
        )

    async def push(self, event: AstrMessageEvent, category: str = "", alias: str = ""):
        """手动推送指定分类的表情包。支持使用分类名称或别名。"""
        if not self.plugin.base_dir:
            return event.plain_result("插件未正确配置，缺少图片存储目录")

        # 初始化目标分类变量
        target_category = None

        # 如果提供了别名，优先使用别名查找实际分类
        if alias:
            aliases = await self.plugin._load_aliases()
            if alias in aliases:
                # 别名存在，映射到实际分类名称
                target_category = aliases[alias]
            else:
                return event.plain_result("未找到指定的别名")

        # 如果没有提供别名或别名不存在，使用分类参数
        # 如果分类参数也为空，则使用默认分类
        target_category = target_category or category or (
            self.plugin.categories[0] if self.plugin.categories else "happy"
        )

        # 将目标分类赋值给cat变量，保持后续代码兼容性
        cat = target_category
        cat_dir = self.plugin.base_dir / "categories" / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            return event.plain_result(f"分类 {cat} 不存在")
        files = [p for p in cat_dir.iterdir() if p.is_file()]
        if not files:
            return event.plain_result("该分类暂无表情包")
        pick = random.choice(files)
        b64 = await self.plugin.image_processor_service._file_to_base64(pick.as_posix())
        chain = event.make_result().base64_image(b64).message_chain
        return event.result_with_message_chain(chain)

    async def debug_image(self, event: AstrMessageEvent):
        """调试命令：处理当前消息中的图片并显示详细信息。"""
        # 收集所有图片组件
        image_components = [comp for comp in event.message_obj.message if isinstance(comp, Image)]

        if not image_components:
            return event.plain_result("当前消息中没有图片")

        # 处理第一张图片
        first_image = image_components[0]
        try:
            # 转换图片到临时文件路径
            temp_file_path = await first_image.convert_to_file_path()

            # 检查路径安全性
            is_safe, safe_file_path = self.plugin._is_safe_path(temp_file_path)
            if not is_safe:
                return event.plain_result("图片路径不安全")

            temp_file_path = safe_file_path

            # 确保临时文件存在且可访问
            if not Path(temp_file_path).exists():
                return event.plain_result("临时文件不存在")

            # 开始调试处理
            result_message = "=== 图片调试信息 ===\n"

            # 1. 基本信息
            image_path = Path(temp_file_path)
            file_size = image_path.stat().st_size
            result_message += f"文件大小: {file_size / 1024:.2f} KB\n"

            # 2. 元数据过滤结果
            # 直接使用plugin中的PILImage引用
            if self.plugin.PILImage is not None:
                try:
                    with self.plugin.PILImage.open(temp_file_path) as image:
                        width, height = image.size
                    result_message += f"分辨率: {width}x{height}\n"
                    aspect_ratio = (
                        max(width, height) / min(width, height)
                        if min(width, height) > 0
                        else 0
                    )
                    result_message += f"宽高比: {aspect_ratio:.2f}\n"
                except Exception as e:
                    result_message += f"获取图片信息失败: {e}\n"

            # 3. 多模态分析结果
            result_message += "\n=== 多模态分析结果 ===\n"

            # 处理图片
            success, image_index = await self.plugin._process_image(
                event, temp_file_path, is_temp=True, idx=None
            )
            if success and image_index:
                for processed_file_path, image_info in image_index.items():
                    if isinstance(image_info, dict):
                        result_message += f"分类: {image_info.get('category', '未知')}\n"
                        result_message += f"情绪: {image_info.get('emotion', '未知')}\n"
                        result_message += f"标签: {image_info.get('tags', [])}\n"
                        result_message += f"描述: {image_info.get('desc', '无')}\n"
            else:
                result_message += "图片处理失败\n"

            return event.plain_result(result_message)

        except Exception as e:
            logger.error(f"调试图片失败: {e}")
            return event.plain_result(f"调试失败: {str(e)}")

    async def clean(self, event: AstrMessageEvent):
        """手动触发清理操作，清理过期的原始图片文件。"""
        try:
            # 加载图片索引
            image_index = await self.plugin._load_index()
            
            # 执行容量控制
            await self.plugin._enforce_capacity(image_index)
            await self.plugin._save_index(image_index)
            
            # 执行raw目录清理
            await self.plugin._clean_raw_directory()
            
            return event.plain_result("手动清理完成")
        except Exception as e:
            logger.error(f"手动清理失败: {e}")
            return event.plain_result(f"清理失败: {str(e)}")
