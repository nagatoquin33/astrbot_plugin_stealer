"""索引管理器：负责索引加载、持久化、重建和迁移。"""

from typing import Any

from astrbot.api import logger


class IndexManager:
    """管理表情包索引的生命周期（加载、保存、重建、迁移）。"""

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance
        self.db_service = getattr(plugin_instance, "db_service", None)
        self.cache_service = getattr(plugin_instance, "cache_service", None)
        self.base_dir = getattr(plugin_instance, "base_dir", None)
        self.categories_dir = getattr(plugin_instance, "categories_dir", None)
        self.cache_dir = getattr(plugin_instance, "cache_dir", None)
        self._migration_done: bool = False

    async def load_index(self) -> dict[str, Any]:
        """从数据库加载索引；DB 为空时尝试从旧 JSON 一次性迁移。

        Returns:
            dict[str, Any]: 索引数据（兼容旧接口的字典格式）
        """
        try:
            idx: dict[str, Any] = {}
            if self.db_service is None:
                return idx

            db_count = self.db_service.count_total()

            if db_count > 0:
                logger.debug(f"[DB] 从数据库加载 {db_count} 条索引")
                return self.db_service.get_index_cache_readonly()

            if self._migration_done:
                return idx

            # 旧实例迁移：cache/index_cache.json 存在时一次性导入 DB 后删除
            old_json_path = self.cache_dir / "index_cache.json" if self.cache_dir else None
            if old_json_path and old_json_path.exists():
                migrated = await self.db_service.migrate_from_json(old_json_path)
                if migrated > 0:
                    self._migration_done = True
                    logger.info(f"[DB] 迁移了 {migrated} 条旧记录到数据库")
                    return self.db_service.get_index_cache_readonly()

            # 兜底：从更老的 JSON 位置（base_dir/index.json 等）合并
            if self.cache_service and self.base_dir:
                legacy_data = await self.cache_service.migrate_legacy_data(self.base_dir)
                if legacy_data:
                    await self.db_service.save_index(legacy_data)
                    self._migration_done = True
                    logger.info("[DB] 迁移旧数据到数据库完成")
                    return self.db_service.get_index_cache_readonly()

            self._migration_done = True
            return idx

        except Exception as e:
            logger.error(f"加载索引失败: {e}", exc_info=True)
            return {}

    async def rebuild_index_from_files(self) -> dict[str, Any]:
        """从文件重建基础索引（不保存到数据库，等待合并后保存）。"""
        if self.cache_service is None or self.base_dir is None:
            return {}
        return await self.cache_service.rebuild_index_from_files(self.base_dir, self.categories_dir)

    async def save_index(self, idx: dict[str, Any]) -> None:
        """将索引同步到数据库，并失效 BM25 缓存。"""
        if self.db_service:
            await self.db_service.sync_index(idx)
        selector = getattr(self.plugin, "meme_selector", None)
        if selector and hasattr(selector, "_invalidate_bm25_index"):
            try:
                selector._invalidate_bm25_index()
            except Exception as e:
                logger.debug(f"[IndexManager] invalidate BM25 failed: {e}")
