"""感知哈希去重服务：计算 pHash、比较汉明距离并定位库内视觉重复。"""

import asyncio
import io
import os
from typing import Any

from astrbot.api import logger

try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None


class PHashDedupService:
    """基于感知哈希 (pHash) 的图片视觉去重服务。"""

    PHASH_SIZE = 16  # 感知哈希图片缩放尺寸（16x16 = 256 bit hash）
    PHASH_HAMMING_THRESHOLD = 20  # 汉明距离阈值，低于此值视为重复

    def __init__(self, plugin_instance: Any = None) -> None:
        self.plugin = plugin_instance

    async def compute_phash(self, file_path: str) -> str:
        """计算图片文件的感知哈希 (pHash)，用于视觉去重。

        使用 difference hash (dHash) 算法：
        - 将图片缩放到 (PHASH_SIZE+1) x PHASH_SIZE 的灰度图
        - 比较相邻像素的亮度差异生成哈希位
        - 动图仅使用 SHA256 去重，避免首帧相同的不同动画被丢弃

        Args:
            file_path: 图片文件路径

        Returns:
            str: 十六进制感知哈希字符串，失败返回空字符串
        """
        if PILImage is None:
            return ""
        return await asyncio.to_thread(self._phash_from_path, file_path)

    async def compute_phash_bytes(self, raw: bytes) -> str:
        """计算内存中图片字节的感知哈希，用于入库前的视觉去重。

        Args:
            raw: 图片原始字节

        Returns:
            str: 十六进制感知哈希字符串，动图或解析失败返回空字符串
        """
        if PILImage is None or not raw:
            return ""
        return await asyncio.to_thread(self._phash_from_bytes, raw)

    @classmethod
    def _phash_from_path(cls, file_path: str) -> str:
        try:
            with PILImage.open(file_path) as im:
                return cls._phash_from_image(im)
        except Exception as e:
            logger.warning(f"计算感知哈希失败: {e}")
            return ""

    @classmethod
    def _phash_from_bytes(cls, raw: bytes) -> str:
        try:
            with PILImage.open(io.BytesIO(raw)) as im:
                return cls._phash_from_image(im)
        except Exception as e:
            logger.warning(f"计算感知哈希失败: {e}")
            return ""

    @classmethod
    def _phash_from_image(cls, im: Any) -> str:
        size = cls.PHASH_SIZE
        if getattr(im, "is_animated", False):
            return ""
        # 转灰度并缩放到 (size+1) x size
        gray = im.convert("L").resize((size + 1, size), PILImage.BILINEAR)
        pixels = list(gray.getdata())

        # dHash: 比较每行相邻像素
        hash_int = 0
        for row in range(size):
            for col in range(size):
                idx = row * (size + 1) + col
                hash_int = (hash_int << 1) | (1 if pixels[idx] > pixels[idx + 1] else 0)

        hex_len = (size * size + 3) // 4  # 向上取整
        return format(hash_int, f"0{hex_len}x")

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """计算两个十六进制哈希字符串的汉明距离。

        Args:
            hash1: 第一个十六进制哈希
            hash2: 第二个十六进制哈希

        Returns:
            int: 汉明距离（不同位的数量），如果无法计算返回极大值
        """
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 999
        try:
            xor = int(hash1, 16) ^ int(hash2, 16)
            return xor.bit_count()
        except (ValueError, TypeError):
            return 999

    @staticmethod
    def is_informative(phash_val: str) -> bool:
        """判断 dHash 是否包含足够结构信息。

        近乎纯色的图片（如纯色表情）哈希位几乎全 0 或全 1，彼此之间没有
        区分度；这类图片只按 SHA256 去重，避免把视觉不同的纯色图判成重复。
        """
        if not phash_val:
            return False
        try:
            bits = int(phash_val, 16).bit_count()
        except (ValueError, TypeError):
            return False
        total = len(phash_val) * 4
        return total > 0 and total * 0.1 <= bits <= total * 0.9

    async def find_duplicate(
        self,
        phash_val: str,
        db: Any,
        *,
        include_pending: bool = False,
    ) -> tuple[str, bool] | None:
        """在正式库（可选待审核池）中查找与给定感知哈希视觉重复的图片。

        Args:
            phash_val: 待判定图片的感知哈希
            db: 数据库服务
            include_pending: 是否同时比对待审核池

        Returns:
            (匹配到的图片路径, 是否来自待审核池)；没有匹配返回 None。
        """
        if not phash_val or db is None or not self.is_informative(phash_val):
            return None

        candidates: list[tuple[str, str, bool]] = []
        try:
            candidates.extend(
                (path, value, False) for path, value in db.get_phash_map().items()
            )
            if include_pending and hasattr(db, "get_pending_phash_map"):
                candidates.extend(
                    (path, value, True)
                    for path, value in db.get_pending_phash_map().items()
                )
        except Exception as e:
            logger.warning(f"读取感知哈希索引失败: {e}")
            return None

        for path, existing_phash, pending in candidates:
            if not self.is_informative(existing_phash):
                continue
            distance = self.hamming_distance(phash_val, existing_phash)
            if distance > self.PHASH_HAMMING_THRESHOLD:
                continue
            # 旧库可能存了动图首帧哈希，重新计算确认后再判定重复。
            if not os.path.isfile(path):
                continue
            if not await self.compute_phash(path):
                continue
            logger.debug(f"[去重] 感知哈希命中: 距离={distance} pending={pending}")
            return path, pending
        return None
