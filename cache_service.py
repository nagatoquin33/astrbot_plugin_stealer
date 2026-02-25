import asyncio
import copy
import hashlib
import inspect
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
        self._ensure_cache_dir()

        # 初始化不同类型的缓存
        self._caches: dict[str, dict[str, Any]] = {
            "image_cache": {},  # 图片分类缓存
            "text_cache": {},  # 文本情绪分类缓存
            "index_cache": {},  # 索引缓存
            "desc_cache": {},  # 描述缓存
            "blacklist_cache": {},  # 黑名单缓存
        }

        # 加载持久化的缓存
        self._load_caches()

        # 锁保护
        self._index_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()  # 通用缓存锁

    def _load_caches(self):
        """加载持久化的缓存文件。"""
        for cache_name in self._caches.keys():
            cache_file = self._get_cache_file(cache_name)
            if cache_file.exists():
                try:
                    with open(cache_file, encoding="utf-8") as f:
                        cached_data = json.load(f)
                        if isinstance(cached_data, dict):
                            self._caches[cache_name] = cached_data
                            logger.info(
                                f"[load_caches] loaded {len(cached_data)} items for {cache_name} from {cache_file}"
                            )
                except Exception as e:
                    logger.error(f"加载缓存文件 {cache_file} 失败: {e}")
            else:
                logger.debug(f"[load_caches] cache file not found: {cache_file}")

    async def _save_cache_async(self, cache_name: str):
        """异步保存指定类型的缓存到文件。

        Args:
            cache_name: 缓存类型名称
        """
        if cache_name not in self._caches:
            logger.warning(f"缓存类型 {cache_name} 不存在，无法保存")
            return

        try:
            self._ensure_cache_dir()
            cache_file = self._get_cache_file(cache_name)

            # 记录保存前的数据量
            data_size = len(self._caches[cache_name])
            logger.debug(f"准备异步保存缓存 {cache_name}，数据量: {data_size}")

            # 深拷贝数据，避免 to_thread 执行 json.dump 期间主线程修改字典导致 RuntimeError
            data_to_save = copy.deepcopy(self._caches[cache_name])

            def write_file():
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            await asyncio.to_thread(write_file)
            logger.debug(f"缓存文件 {cache_file} 保存成功 (Async)，数据量: {data_size}")
        except Exception as e:
            logger.error(f"保存缓存文件 {cache_file} 失败: {e}", exc_info=True)

    def _save_cache(self, cache_name: str):
        """同步保存指定类型的缓存到文件（仅用于非异步上下文）。

        尽量使用 _save_cache_async。
        """
        if cache_name not in self._caches:
            logger.warning(f"缓存类型 {cache_name} 不存在，无法保存")
            return

        try:
            self._ensure_cache_dir()
            cache_file = self._get_cache_file(cache_name)

            # 记录保存前的数据量
            data_size = len(self._caches[cache_name])

            # 深拷贝，防止写入过程中被其他协程修改
            data_snapshot = copy.deepcopy(self._caches[cache_name])
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data_snapshot, f, ensure_ascii=False, indent=2)
            logger.debug(f"缓存文件 {cache_file} 保存成功，数据量: {data_size}")
        except Exception as e:
            logger.error(f"保存缓存文件 {cache_file} 失败: {e}", exc_info=True)

    def _ensure_cache_dir(self) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"创建缓存目录 {self._cache_dir} 失败: {e}")
            raise Exception(f"无法创建缓存目录: {e}") from e

    def _get_cache_file(self, cache_name: str) -> Path:
        return self._cache_dir / f"{cache_name}.json"

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

    async def set(
        self, cache_name: str, key: str, value: Any, persist: bool = False
    ) -> None:
        """设置指定类型缓存的数据。

        Args:
            cache_name: 缓存类型名称
            key: 缓存键
            value: 缓存值
            persist: 是否立即持久化到文件
        """
        if cache_name not in self._caches:
            return

        async with self._cache_lock:
            # 设置缓存值
            self._caches[cache_name][key] = value

            # 清理缓存，保持在最大大小以下
            self._clean_cache(self._caches[cache_name])

        # 如果需要立即持久化
        if persist:
            await self._save_cache_async(cache_name)

    async def delete(self, cache_name: str, key: str, persist: bool = False) -> None:
        """从指定类型的缓存中删除数据。

        Args:
            cache_name: 缓存类型名称
            key: 缓存键
            persist: 是否立即持久化到文件
        """
        if cache_name in self._caches:
            async with self._cache_lock:
                if key in self._caches[cache_name]:
                    del self._caches[cache_name][key]

            # 如果需要立即持久化
            if persist:
                await self._save_cache_async(cache_name)

    async def clear(self, cache_name: str | None = None, persist: bool = False) -> None:
        """清空缓存。

        Args:
            cache_name: 缓存类型名称，如果为None则清空所有缓存
            persist: 是否立即持久化到文件
        """
        async with self._cache_lock:
            if cache_name:
                # 清空指定类型的缓存
                if cache_name in self._caches:
                    self._caches[cache_name].clear()
            else:
                # 清空所有缓存
                for name in self._caches.keys():
                    self._caches[name].clear()

        # 持久化在锁外执行
        if persist:
            if cache_name:
                await self._save_cache_async(cache_name)
            else:
                for name in self._caches.keys():
                    await self._save_cache_async(name)

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

    async def persist_all(self):
        """将所有缓存持久化到文件。"""
        for cache_name in self._caches.keys():
            await self._save_cache_async(cache_name)

    def get_index_cache(self) -> dict[str, Any]:
        """获取索引缓存的便捷方法。

        Returns:
            dict[str, Any]: 索引缓存字典（可变副本）
        """
        return dict(self.get_cache("index_cache"))

    def get_index_cache_readonly(self) -> MappingProxyType:
        """获取索引缓存的只读视图（零拷贝，适用于只读遍历场景）。

        Returns:
            MappingProxyType: 索引缓存的只读视图
        """
        return self.get_cache("index_cache")

    def get_cache(self, cache_name: str) -> MappingProxyType:
        """获取指定类型的缓存字典（只读视图）。

        Args:
            cache_name: 缓存类型名称

        Returns:
            只读视图 MappingProxyType，如果不存在则返回空的只读视图
        """
        if cache_name in self._caches:
            return MappingProxyType(self._caches[cache_name])
        return MappingProxyType({})

    async def set_cache(
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
                await self._save_cache_async(cache_name)

    async def update_config(self, max_cache_size: int | None = None):
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
            await self.persist_all()

    async def load_index(self) -> dict[str, Any]:
        """加载分类索引文件。

        Returns:
            Dict[str, Any]: 键为文件路径，值为包含 category 与 tags 的字典。
        """
        try:
            async with self._index_lock:
                cache_data = self.get_index_cache()
                index_data = dict(cache_data) if cache_data else {}
                return index_data
        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def save_index(self, idx: dict[str, Any]):
        """保存分类索引文件。"""
        try:
            async with self._index_lock:
                await self.set_cache("index_cache", idx, persist=True)
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}", exc_info=True)

    async def update_index(self, updater) -> dict[str, Any]:
        try:
            async with self._index_lock:
                current = self.get_index_cache()
                result = updater(current)
                if inspect.isawaitable(result):
                    await result
                await self.set_cache("index_cache", current, persist=True)
                return current
        except Exception as e:
            logger.error(f"更新索引失败: {e}", exc_info=True)
            try:
                return self.get_index_cache()
            except Exception:
                return {}

    async def migrate_legacy_data(self, base_dir) -> dict[str, Any]:
        """迁移旧版本数据到新版本。

        Args:
            base_dir: 插件基础目录

        Returns:
            Dict[str, Any]: 迁移后的索引数据
        """
        try:
            import shutil
            from pathlib import Path

            logger.info("开始检查和迁移旧版本数据...")

            base_dir_path = Path(base_dir) if base_dir else None
            possible_paths = []
            if base_dir_path:
                possible_paths.extend(
                    [
                        base_dir_path / "index.json",
                        base_dir_path / "image_index.json",
                        base_dir_path / "cache" / "index.json",
                    ]
                )

            migrated_data = {}

            # 同步文件操作放入 executor
            def load_old_file(path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)

            for old_path in possible_paths:
                if old_path.exists():
                    try:
                        logger.info(f"发现旧版本索引文件: {old_path}")

                        old_data = await asyncio.to_thread(load_old_file, old_path)

                        if isinstance(old_data, dict) and old_data:
                            logger.info(
                                f"从 {old_path} 加载了 {len(old_data)} 条旧记录"
                            )
                            migrated_data.update(old_data)

                            # 备份旧文件
                            backup_path = old_path.with_suffix(".json.backup")
                            await asyncio.to_thread(shutil.copy2, old_path, backup_path)
                            logger.info(f"已备份旧索引文件到: {backup_path}")

                    except Exception as e:
                        logger.error(f"迁移文件 {old_path} 失败: {e}")
                        continue

            if not migrated_data:
                logger.info("未发现需要迁移的旧版本数据文件")
                return {}

            # 智能合并逻辑
            try:
                current_index = self.get_index_cache()
            except Exception as e:
                logger.warning(f"[migrate] 获取当前索引失败，将使用空索引: {e}")
                current_index = {}

            # 建立当前索引的哈希映射
            current_hash_map = {}
            for k, v in current_index.items():
                if isinstance(v, dict) and v.get("hash"):
                    current_hash_map[v["hash"]] = k

            merged_count = 0

            # 遍历旧数据，尝试合并到当前索引
            for old_path, old_info in migrated_data.items():
                if not isinstance(old_info, dict):
                    continue

                target_path = None

                # 1. 路径完全匹配
                if old_path in current_index:
                    target_path = old_path
                # 2. 哈希匹配（处理路径变更）
                elif old_info.get("hash") in current_hash_map:
                    target_path = current_hash_map[old_info["hash"]]

                # 如果找到了对应的目标记录，且旧数据有描述/标签，保留之
                if target_path:
                    target_info = current_index[target_path]
                    updated = False

                    if old_info.get("desc") and not target_info.get("desc"):
                        target_info["desc"] = old_info["desc"]
                        updated = True

                    if old_info.get("tags") and not target_info.get("tags"):
                        target_info["tags"] = old_info["tags"]
                        updated = True

                    if updated:
                        merged_count += 1

            # 保存合并后的索引
            logger.info(f"成功从旧数据中恢复了 {merged_count} 条记录的元数据")
            await self.save_index(current_index)

            return current_index

        except Exception as e:
            logger.error(f"数据迁移失败: {e}", exc_info=True)
            return {}

    async def rebuild_index_from_files(
        self, base_dir, categories_dir
    ) -> dict[str, Any]:
        """从现有的分类文件重建索引。

        Args:
            base_dir: 插件基础目录
            categories_dir: 分类目录路径

        Returns:
            Dict[str, Any]: 重建的索引数据
        """
        try:
            from pathlib import Path

            rebuilt_index = {}

            if not categories_dir.exists():
                return rebuilt_index

            # 遍历所有分类目录
            for category_dir in categories_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name
                logger.info(f"重建分类 '{category_name}' 的索引...")

                # 遍历分类目录中的图片文件
                for img_file in category_dir.iterdir():
                    if not img_file.is_file():
                        continue

                    # 检查是否是图片文件
                    if img_file.suffix.lower() not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".gif",
                        ".webp",
                    ]:
                        continue

                    # 尝试找到对应的raw文件
                    raw_dir = Path(base_dir) / "raw"
                    raw_path = None
                    if raw_dir.exists():
                        potential_raw = raw_dir / img_file.name
                        if potential_raw.exists():
                            raw_path = str(potential_raw)
                        else:
                            for raw_file in raw_dir.iterdir():
                                if (
                                    raw_file.is_file()
                                    and raw_file.stem == img_file.stem
                                ):
                                    raw_path = str(raw_file)
                                    break

                    # 如果没找到raw文件，使用categories中的文件路径
                    if not raw_path:
                        raw_path = str(img_file)

                    # 计算文件哈希
                    try:
                        file_hash = self.compute_hash(Path(img_file).read_bytes())
                    except Exception as e:
                        logger.debug(f"计算文件哈希失败: {e}")
                        file_hash = ""

                    # 创建索引记录
                    rebuilt_index[raw_path] = {
                        "hash": file_hash,
                        "category": category_name,
                        "created_at": int(img_file.stat().st_mtime),
                        "migrated": True,
                    }

            logger.info(f"从文件重建了 {len(rebuilt_index)} 条索引记录")
            return rebuilt_index

        except Exception as e:
            logger.error(f"从文件重建索引失败: {e}", exc_info=True)
            return {}

    async def cleanup(self):
        """清理资源（持久化所有缓存）。"""
        await self.persist_all()
