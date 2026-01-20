import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from astrbot.api import logger


class CacheService:
    """缓存服务类，负责管理各种类型的缓存。"""

    # 缓存最大大小
    _CACHE_MAX_SIZE = 100

    def __init__(self, cache_dir: str | Path = None):
        """初始化缓存服务。

        Args:
            cache_dir: 缓存文件存储目录，如果为None则使用默认目录
        """
        if not cache_dir:
            from astrbot.api.star import StarTools

            cache_dir = Path(StarTools.get_data_dir("astrbot_plugin_stealer")) / "cache"

        self._cache_dir = Path(cache_dir)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"缓存目录创建成功: {self._cache_dir}")
        except Exception as e:
            logger.error(f"创建缓存目录 {self._cache_dir} 失败: {e}")
            raise Exception(f"无法创建缓存目录: {e}") from e

        # 初始化不同类型的缓存
        self._caches: dict[str, dict[str, Any]] = {
            "image_cache": {},  # 图片分类缓存
            "text_cache": {},  # 文本情绪分类缓存
            "index_cache": {},  # 索引缓存
            "desc_cache": {},  # 描述缓存
        }

        # 加载持久化的缓存
        self._load_caches()

    def _load_caches(self):
        """加载持久化的缓存文件。"""
        for cache_name in self._caches.keys():
            cache_file = self._cache_dir / f"{cache_name}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, encoding="utf-8") as f:
                        cached_data = json.load(f)
                        if isinstance(cached_data, dict):
                            self._caches[cache_name] = cached_data
                            logger.info(f"[load_caches] loaded {len(cached_data)} items for {cache_name} from {cache_file}")
                except Exception as e:
                    logger.error(f"加载缓存文件 {cache_file} 失败: {e}")
            else:
                logger.debug(f"[load_caches] cache file not found: {cache_file}")

    def _save_cache(self, cache_name: str):
        """保存指定类型的缓存到文件。

        Args:
            cache_name: 缓存类型名称
        """
        if cache_name not in self._caches:
            logger.warning(f"缓存类型 {cache_name} 不存在，无法保存")
            return

        try:
            # 确保缓存目录存在
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / f"{cache_name}.json"

            # 记录保存前的数据量
            data_size = len(self._caches[cache_name])
            logger.debug(f"准备保存缓存 {cache_name}，数据量: {data_size}")

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self._caches[cache_name], f, ensure_ascii=False, indent=2)
            logger.info(f"缓存文件 {cache_file} 保存成功，数据量: {data_size}")
        except Exception as e:
            logger.error(f"保存缓存文件 {cache_file} 失败: {e}", exc_info=True)

    def _clean_cache(self, cache: dict[str, Any]) -> None:
        """清理缓存，保持在最大大小以下。

        Args:
            cache: 要清理的缓存字典
        """
        if len(cache) > self._CACHE_MAX_SIZE:
            keys_to_keep = list(cache.keys())[-self._CACHE_MAX_SIZE :]
            items_to_keep = {k: cache[k] for k in keys_to_keep}
            cache.clear()
            cache.update(items_to_keep)

    def get(self, cache_name: str, key: str) -> Any | None:
        """从指定类型的缓存中获取数据。

        Args:
            cache_name: 缓存类型名称
            key: 缓存键

        Returns:
            缓存的值，如果不存在则返回None
        """
        if cache_name in self._caches:
            return self._caches[cache_name].get(key)
        return None

    def set(self, cache_name: str, key: str, value: Any, persist: bool = False) -> None:
        """设置指定类型缓存的数据。

        Args:
            cache_name: 缓存类型名称
            key: 缓存键
            value: 缓存值
            persist: 是否立即持久化到文件
        """
        if cache_name not in self._caches:
            return

        # 设置缓存值
        self._caches[cache_name][key] = value

        # 清理缓存，保持在最大大小以下
        self._clean_cache(self._caches[cache_name])

        # 如果需要立即持久化
        if persist:
            self._save_cache(cache_name)

    def delete(self, cache_name: str, key: str, persist: bool = False) -> None:
        """从指定类型的缓存中删除数据。

        Args:
            cache_name: 缓存类型名称
            key: 缓存键
            persist: 是否立即持久化到文件
        """
        if cache_name in self._caches:
            if key in self._caches[cache_name]:
                del self._caches[cache_name][key]

                # 如果需要立即持久化
                if persist:
                    self._save_cache(cache_name)

    def clear(self, cache_name: str | None = None, persist: bool = False) -> None:
        """清空缓存。

        Args:
            cache_name: 缓存类型名称，如果为None则清空所有缓存
            persist: 是否立即持久化到文件
        """
        if cache_name:
            # 清空指定类型的缓存
            if cache_name in self._caches:
                self._caches[cache_name].clear()
                if persist:
                    self._save_cache(cache_name)
        else:
            # 清空所有缓存
            for name in self._caches.keys():
                self._caches[name].clear()
                if persist:
                    self._save_cache(name)

    def get_cache_size(self, cache_name: str) -> int:
        """获取指定类型缓存的大小。

        Args:
            cache_name: 缓存类型名称

        Returns:
            缓存中的键值对数量
        """
        if cache_name in self._caches:
            return len(self._caches[cache_name])
        return 0

    def compute_hash(self, data: str | bytes) -> str:
        """计算数据的哈希值，用于生成缓存键。

        Args:
            data: 要计算哈希的数据

        Returns:
            数据的SHA256哈希值
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        return hash_obj.hexdigest()

    def persist_all(self):
        """将所有缓存持久化到文件。"""
        for cache_name in self._caches.keys():
            self._save_cache(cache_name)

    def get_cache(self, cache_name: str) -> dict[str, Any]:
        """获取指定类型的缓存字典（只读视图）。

        Args:
            cache_name: 缓存类型名称

        Returns:
            只读视图字典，如果不存在则返回空字典
        """
        if cache_name in self._caches:
            proxy = MappingProxyType(self._caches[cache_name])
            logger.debug(f"[get_cache] {cache_name}: {len(proxy)} items in cache")
            return proxy
        logger.debug(f"[get_cache] {cache_name}: not found, returning empty dict")
        return {}

    def set_cache(
        self, cache_name: str, cache_data: dict[str, Any], persist: bool = True
    ) -> None:
        """设置指定类型的缓存字典（完全替换模式）。

        Args:
            cache_name: 缓存类型名称
            cache_data: 要设置的缓存数据（完全替换现有数据）
            persist: 是否立即持久化到文件
        """
        if cache_name in self._caches:
            old_len = len(self._caches[cache_name])
            new_len = len(cache_data)
            self._caches[cache_name].clear()
            self._caches[cache_name].update(cache_data)
            logger.debug(f"[set_cache] {cache_name}: {old_len} -> {new_len} items")
            if persist:
                self._save_cache(cache_name)

    def update_config(self, max_cache_size: int | None = None):
        """更新缓存配置。

        Args:
            max_cache_size: 缓存最大大小，如果为None则使用默认值
        """
        if max_cache_size is not None:
            self._CACHE_MAX_SIZE = max_cache_size
            # 清理所有缓存，确保不超过新的最大大小
            for cache in self._caches.values():
                self._clean_cache(cache)
            # 持久化更新后的缓存
            self.persist_all()

    def cleanup(self):
        """清理资源（持久化所有缓存）。"""
        self.persist_all()
