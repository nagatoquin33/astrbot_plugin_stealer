"""图片下载服务：负责 HTTP 下载、连接池管理和超时处理。"""

import asyncio
import os
import shutil
import tempfile
from typing import Any

import aiohttp

from astrbot.api import logger


class ImageDownloadService:
    """基于 aiohttp 的图片下载服务，复用连接池。"""

    HTTP_TIMEOUT_SECONDS = 30
    HTTP_CONNECTOR_LIMIT = 10
    HTTP_CONNECTOR_LIMIT_PER_HOST = 5
    HTTP_DNS_CACHE_SECONDS = 300

    def __init__(self, plugin_instance: Any = None):
        self.plugin = plugin_instance
        self._aiohttp_session: aiohttp.ClientSession | None = None

    async def get_session(self) -> aiohttp.ClientSession:
        """获取或创建共享的 aiohttp session。"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.HTTP_CONNECTOR_LIMIT,
                limit_per_host=self.HTTP_CONNECTOR_LIMIT_PER_HOST,
                ttl_dns_cache=self.HTTP_DNS_CACHE_SECONDS,
                use_dns_cache=True,
            )
            timeout = aiohttp.ClientTimeout(total=self.HTTP_TIMEOUT_SECONDS)
            self._aiohttp_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )
        return self._aiohttp_session

    async def close(self) -> None:
        """关闭共享的 aiohttp session。"""
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    def build_headers(self) -> dict[str, str]:
        """构建下载请求头。

        历史原因：这里曾为 NapCat 下发 Authorization header，但 NapCat 的
        图片 URL 已经是本地可访问路径或不需要该 token，且把 token 附加到
        任意第三方图片 URL 会导致安全/兼容问题，因此不再附加任何认证头。
        """
        return {}

    @staticmethod
    def detect_file_type(content_type: str, content: bytes) -> tuple[str, bool]:
        content_type = str(content_type or "").lower()
        is_gif = "gif" in content_type or content[:6] in (b"GIF89a", b"GIF87a")
        if is_gif:
            return ".gif", True
        if "png" in content_type or content[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png", False
        if "webp" in content_type or (
            content[:4] == b"RIFF" and content[8:12] == b"WEBP"
        ):
            return ".webp", False
        if "jpeg" in content_type or "jpg" in content_type:
            return ".jpg", False
        return ".jpg", False

    async def download_to_temp(
        self, url: str, *, log_download: bool = False
    ) -> tuple[str | None, bool]:
        """从 URL 下载文件到临时文件。

        Returns:
            tuple[str | None, bool]: (临时文件路径, 是否为GIF动图)，失败返回 (None, False)
        """
        if not url or not isinstance(url, str):
            return None, False

        try:
            session = await self.get_session()
            async with session.get(
                url,
                headers=self.build_headers(),
                timeout=aiohttp.ClientTimeout(total=self.HTTP_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"下载图片失败: HTTP {resp.status}")
                    return None, False

                content_type = resp.headers.get("Content-Type", "").lower()
                content = await resp.read()
                ext, is_gif = self.detect_file_type(content_type, content)

                temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
                try:
                    os.write(temp_fd, content)
                    if log_download:
                        logger.debug(
                            f"已下载原始图片: {temp_path} ({len(content)} bytes, "
                            f"type={content_type}, is_gif={is_gif})"
                        )
                    return temp_path, is_gif
                finally:
                    os.close(temp_fd)
        except asyncio.TimeoutError:
            logger.warning("下载图片超时")
            return None, False
        except Exception as e:
            logger.warning(f"下载图片失败: {e}")
            return None, False

    async def download_original_image(self, img: Any) -> tuple[str | None, bool]:
        """下载原始图片文件。

        优先从图片组件的本地路径复制出插件自有临时文件，
        仅对远程 HTTP URL 发起下载请求。调用方会把返回路径作为临时文件
        移动或删除，因此不能直接返回由 AstrBot 事件生命周期管理的原路径。

        Args:
            img: 图片组件

        Returns:
            tuple[str | None, bool]: (临时文件路径, 是否为GIF动图)，失败返回 (None, False)
        """
        img_url = getattr(img, "url", "") or ""
        img_file = getattr(img, "file", "") or ""
        img_path = getattr(img, "path", "") or ""

        # 检查是否已经是本地文件路径
        for candidate in (img_path, img_file, img_url):
            if candidate and os.path.exists(candidate):
                content_type = ""
                try:
                    with open(candidate, "rb") as f:
                        header = f.read(12)
                    if header[:6] in (b"GIF89a", b"GIF87a"):
                        content_type = "image/gif"
                    elif header[:8] == b"\x89PNG\r\n\x1a\n":
                        content_type = "image/png"
                    elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
                        content_type = "image/webp"
                except OSError:
                    logger.warning(f"无法读取本地图片文件: {candidate}")
                    continue

                ext, is_gif = self.detect_file_type(content_type, header)
                temp_fd, temp_path = tempfile.mkstemp(suffix=ext)
                os.close(temp_fd)
                try:
                    await asyncio.to_thread(shutil.copyfile, candidate, temp_path)
                except asyncio.CancelledError:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    raise
                except OSError as e:
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    logger.warning(f"复制本地图片失败: {candidate}, {e}")
                    continue
                logger.debug(
                    f"已复制本地图片到插件临时文件: {candidate} -> {temp_path} "
                    f"(is_gif={is_gif})"
                )
                return temp_path, is_gif

        # 回退到 HTTP 下载
        url = img_url or img_file
        return await self.download_to_temp(url, log_download=True)

    async def download_url_to_temp(self, url: str) -> tuple[str | None, bool]:
        """从 URL 下载文件到临时文件，返回 (temp_path, is_gif)。"""
        return await self.download_to_temp(url)
