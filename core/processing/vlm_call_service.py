"""VLM 调用服务：负责调用视觉模型分析图片。"""

import asyncio
import os
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

try:
    import numpy as np
except Exception:
    np = None


class VLMCallService:
    """负责调用视觉模型进行图片分析。"""

    def __init__(self, plugin_instance) -> None:
        self.plugin = plugin_instance
        self.plugin_config = getattr(plugin_instance, "plugin_config", None)
        self.vision_provider_id = (
            str(getattr(self.plugin_config, "vision_provider_id", "") or "")
            if self.plugin_config
            else ""
        )
        self._cached_framework_vlm_id: str | None = None

    async def _resolve_vision_provider(self, event=None) -> str | None:
        """统一的视觉模型 provider 解析逻辑。

        优先级：
        1. 插件配置的 vision_provider_id
        2. AstrBot 框架配置的 default_image_caption_provider_id
        """
        if self.vision_provider_id:
            return self.vision_provider_id

        if self._cached_framework_vlm_id is not None:
            return self._cached_framework_vlm_id or None

        framework_vlm_id = ""
        try:
            if hasattr(self.plugin, "context"):
                astrbot_config = self.plugin.context.get_config()
                provider_settings = astrbot_config.get("provider_settings", {})
                framework_vlm_id = str(
                    provider_settings.get("default_image_caption_provider_id", "") or ""
                )
        except Exception as e:
            logger.debug(f"读取框架视觉模型配置失败: {e}")

        self._cached_framework_vlm_id = framework_vlm_id

        if framework_vlm_id:
            logger.info(f"使用框架全局图片描述模型: {framework_vlm_id}")
            return framework_vlm_id

        logger.warning(
            "未配置视觉模型，无法进行图片分类。"
            "请在插件配置中设置 vision_provider_id，"
            "或在 AstrBot 全局配置中设置 default_image_caption_provider_id。"
        )
        return None

    async def _call_vision_model(
        self, event: AstrMessageEvent | None, img_path: str, prompt: str
    ) -> str:
        """调用视觉模型分析图片。

        使用 context.llm_generate 调用指定的视觉模型 provider，
        支持指数退避重试。对于 GIF 动图，会提取关键帧拼接后分析。

        Args:
            event: 消息事件（用于 provider 解析）
            img_path: 图片绝对路径（调用方需保证已验证）
            prompt: 提示词

        Returns:
            str: 模型响应文本

        Raises:
            ValueError: 未配置视觉模型
            FileNotFoundError: 图片文件不存在
            Exception: 模型调用失败（已重试）
        """
        # 路径规范化
        img_path_obj = Path(img_path)
        if not img_path_obj.is_absolute():
            data_dir = getattr(self.plugin_config, "data_dir", None)
            img_path_obj = (
                (Path(data_dir) / img_path).resolve() if data_dir else img_path_obj.resolve()
            )
        img_path = str(img_path_obj)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片文件不存在: {img_path}")

        # 解析 provider
        provider_id = await self._resolve_vision_provider(event)
        if not provider_id:
            raise ValueError(
                "未配置视觉模型(vision_provider_id)，无法进行图片分析。"
                "请在插件配置或 AstrBot 全局配置中设置。"
            )

        # 处理 GIF 动图：提取多帧拼接
        temp_file = None
        try:
            actual_img_path, is_animated = await self._prepare_image_for_vlm(img_path)
            if actual_img_path != img_path:
                temp_file = actual_img_path  # 标记为临时文件，分析后删除

            # 直接传入本地绝对路径，框架内部会自动处理路径转换
            resolved_img_path = str(Path(actual_img_path).resolve())

            # 如果是动图拼接，添加专用提示词前缀
            actual_prompt = prompt
            if is_animated:
                animated_prefix = (
                    "[动图帧序列] 这是一个动态图表情包，每一帧从左到右按时间顺序展示了动画过程。"
                    "黑色背景代表透明区域。"
                    "请理解这些帧是连贯的动画，分析整体动作和情绪变化，从互联网梗/meme的角度分析。"
                    "请特别识别并理解画面中的文字（字幕、弹幕、对话框、贴纸文字），不要忽略文字语义。\n\n"
                )
                actual_prompt = animated_prefix + prompt

            return await self._do_vlm_call(provider_id, actual_prompt, resolved_img_path)
        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"已清理临时拼接图: {temp_file}")
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")

    async def _prepare_image_for_vlm(self, img_path: str) -> tuple[str, bool]:
        """为 VLM 分析准备图片，对动图提取多帧拼接。

        Args:
            img_path: 原始图片路径

        Returns:
            tuple[str, bool]: (准备好的图片路径, 是否为动图拼接)
        """
        # 只处理 GIF 文件
        if not img_path.lower().endswith(".gif"):
            return img_path, False

        if PILImage is None or np is None:
            return img_path, False

        try:
            # 检测是否为动图
            def _check_animated(fp: str) -> tuple[bool, int, int, int]:
                with PILImage.open(fp) as im:
                    is_animated = bool(getattr(im, "is_animated", False))
                    n_frames = int(getattr(im, "n_frames", 1) or 1)
                    width, height = im.size
                    return is_animated, n_frames, width, height

            is_animated, n_frames, width, height = await asyncio.to_thread(
                _check_animated, img_path
            )

            # 非动图或帧数太少，直接返回原路径
            if not is_animated or n_frames <= 1:
                return img_path, False

            # 动图处理：提取关键帧并横向拼接
            MAX_FRAMES = 12  # 最多提取 12 帧
            TARGET_HEIGHT = 480  # 输出高度（提高以保留小字可读性）
            SIMILARITY_THRESHOLD = 1000.0  # 相似帧过滤阈值 (MSE)

            # 计算缩放比例
            scale = TARGET_HEIGHT / height if height > TARGET_HEIGHT else 1.0
            frame_width = int(width * scale)
            frame_height = TARGET_HEIGHT

            def _extract_and_combine(fp: str) -> tuple[str, int]:
                frames = []
                last_selected_np = None

                with PILImage.open(fp) as im:
                    # 先提取所有帧
                    all_frames = []
                    for idx in range(n_frames):
                        im.seek(idx)
                        frame = im.convert("RGBA")
                        if scale < 1.0:
                            frame = frame.resize((frame_width, frame_height), PILImage.LANCZOS)
                        all_frames.append(frame)

                # 相似帧过滤
                for frame in all_frames:
                    frame_np = np.array(frame, dtype=np.float32)

                    if last_selected_np is None:
                        # 第一帧直接加入
                        frames.append(frame)
                        last_selected_np = frame_np
                    else:
                        # 计算均方误差 (MSE)
                        mse = np.mean((frame_np - last_selected_np) ** 2)

                        # 差异够大才选中
                        if mse > SIMILARITY_THRESHOLD:
                            frames.append(frame)
                            last_selected_np = frame_np

                # 如果过滤后帧数太少，保留更多帧
                if len(frames) < 3 and len(all_frames) >= 3:
                    # 均匀采样
                    step = max(1, len(all_frames) // 6)
                    frames = [all_frames[i] for i in range(0, len(all_frames), step)][:6]

                # 限制最大帧数
                if len(frames) > MAX_FRAMES:
                    # 均匀抽取
                    step = len(frames) / MAX_FRAMES
                    frames = [frames[int(i * step)] for i in range(MAX_FRAMES)]

                # 横向拼接所有帧，黑色背景代表透明
                total_width = frame_width * len(frames)
                combined = PILImage.new("RGBA", (total_width, frame_height), (0, 0, 0, 255))

                for i, frame in enumerate(frames):
                    combined.paste(frame, (i * frame_width, 0), frame)  # 使用帧的 alpha 通道

                # 保存临时文件
                import tempfile

                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                # 使用无损 PNG，避免 JPEG 压缩导致小字发糊。
                combined.save(temp_path, "PNG", optimize=True)

                return temp_path, len(frames)

            temp_path, actual_frames = await asyncio.to_thread(_extract_and_combine, img_path)
            logger.debug(
                f"GIF 动图拼接完成: {n_frames} 帧 -> {actual_frames} 帧, "
                f"输出尺寸: {frame_width * actual_frames}x{frame_height}"
            )
            return temp_path, True

        except Exception as e:
            logger.warning(f"GIF 动图帧提取失败，使用原图: {e}")
            return img_path, False

    async def _do_vlm_call(self, provider_id: str, prompt: str, file_url: str) -> str:
        """执行 VLM 调用（带重试）。

        Args:
            provider_id: 提供商 ID
            prompt: 提示词
            file_url: 文件 URL

        Returns:
            str: 模型响应文本
        """
        # 重试配置
        try:
            max_retries = int(getattr(self.plugin, "vision_max_retries", 3))
        except (TypeError, ValueError):
            max_retries = 3
        try:
            retry_delay = float(getattr(self.plugin, "vision_retry_delay", 1.0))
        except (TypeError, ValueError):
            retry_delay = 1.0
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"调用VLM (尝试 {attempt + 1}/{max_retries}), "
                    f"provider={provider_id}, 图片={file_url}"
                )
                result = await self._llm_generate_with_image_compat(
                    provider_id=provider_id,
                    prompt=prompt,
                    file_url=file_url,
                )

                # LLMResponse.completion_text 是 @property，自动处理 result_chain
                text = (result.completion_text or "").strip() if result else ""
                if text:
                    logger.debug(f"VLM响应: {text[:200]}")
                    return text

                logger.warning("VLM返回空响应")
                last_error = Exception("VLM返回空响应")

            except Exception as e:
                last_error = e
                error_msg = str(e)
                is_rate_limit = any(
                    kw in error_msg
                    for kw in (
                        "429",
                        "RateLimit",
                        "exceeded your current request limit",
                    )
                )
                is_provider_error = "Provider" in error_msg or "提供商" in error_msg
                if is_rate_limit:
                    logger.warning(f"VLM请求被限流 ({attempt + 1}/{max_retries})")
                elif is_provider_error:
                    logger.error(
                        f"VLM模型提供商错误 ({attempt + 1}/{max_retries}): {e}\n"
                        f"  当前provider_id: {provider_id}\n"
                        f"  提示: 请检查插件配置中的'视觉模型'是否有效，"
                        f"  或尝试清空该配置使用框架全局的图片描述模型"
                    )
                else:
                    logger.error(f"VLM调用失败 ({attempt + 1}/{max_retries}): {e}")

            # 指数退避
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))

        raise Exception(f"视觉模型调用失败（已重试{max_retries}次）: {last_error}") from last_error

    async def _llm_generate_with_image_compat(self, provider_id: str, prompt: str, file_url: str):
        """兼容不同 AstrBot 版本对 image_urls 参数形态的处理差异。"""
        try:
            # 优先使用列表形态，避免部分版本把字符串按字符拆分成“多张图片”。
            return await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=[file_url],
            )
        except Exception as e:
            # 仅在参数形态/类型不兼容时回退，避免对无关错误重复请求。
            err = str(e).lower()
            fallback_markers = (
                "list object",
                "startswith",
                "image_urls",
                "expected list",
                "expected str",
                "typeerror",
            )
            if not any(marker in err for marker in fallback_markers):
                raise

            logger.warning(f"VLM image_urls 参数形态不兼容，回退为字符串模式重试一次: {e}")
            return await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                image_urls=file_url,
            )
