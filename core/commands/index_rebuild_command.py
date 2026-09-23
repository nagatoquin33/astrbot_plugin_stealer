"""索引重建命令：负责从文件重建索引并合并旧元数据。"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from astrbot.api import logger

from astrbot.api.event import AstrMessageEvent

from ..db.index_metadata_merge import has_meaningful_metadata, restore_rebuilt_metadata
from ..db.index_manager import delete_index_paths
from ..maintenance.retention import library_counts


class IndexRebuildCommand:
    """负责索引重建、元数据合并和迁移。"""

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance

    async def rebuild_index(self, event: AstrMessageEvent):
        """重建索引命令，用于从旧版本迁移或修复索引。

        扫描 categories 目录中的所有图片文件，重新构建索引。
        """
        try:
            yield event.plain_result("🔄 开始重建索引，请稍候...")

            # 调用插件的重建索引方法（只重建基础索引，不保存到数据库）
            rebuilt_index = await self.plugin.index_manager.rebuild_index_from_files()

            if not rebuilt_index:
                yield event.plain_result(
                    "⚠️ 未找到可重建的图片文件。\n"
                    f"请确保 categories 目录中存在图片文件:\n"
                    f"{self.plugin.plugin_config.categories_dir}"
                )
                return

            # --- 收集所有旧数据源（JSON + 数据库）---
            # 1. 尝试从数据库加载（如果有数据）
            old_index = {}
            db_count = self.plugin.db_service.count_total()
            if db_count > 0:
                old_index = self.plugin.db_service.get_index_cache_readonly()
                logger.info(f"[rebuild_index] 从数据库加载 {len(old_index)} 条旧记录")

            # 2. 尝试加载所有可能的旧版 JSON 文件（不依赖 _load_index 的迁移逻辑）
            legacy_metadata_count = 0
            legacy_data_map = {}
            possible_legacy_paths = []

            # 添加所有可能的 JSON 文件路径（包括迁移后的备份文件）
            if self.plugin.cache_dir:
                possible_legacy_paths.extend(
                    [
                        self.plugin.cache_dir / "index_cache.json",
                        self.plugin.cache_dir / "index_cache.json.backup",
                        self.plugin.cache_dir / "index_cache.json.migrated",  # 迁移后的备份
                    ]
                )
            if self.plugin.base_dir:
                possible_legacy_paths.extend(
                    [
                        self.plugin.base_dir / "index.json",
                        self.plugin.base_dir / "index.json.backup",
                        self.plugin.base_dir / "index.json.migrated",  # 迁移后的备份
                        self.plugin.base_dir / "image_index.json",
                        self.plugin.base_dir / "image_index.json.backup",
                        self.plugin.base_dir / "image_index.json.migrated",
                        self.plugin.base_dir / "cache" / "index.json",
                        self.plugin.base_dir / "cache" / "index.json.backup",
                        self.plugin.base_dir / "cache" / "index.json.migrated",
                        self.plugin.base_dir / "cache" / "index_cache.json",
                        self.plugin.base_dir / "cache" / "index_cache.json.backup",
                        self.plugin.base_dir / "cache" / "index_cache.json.migrated",
                    ]
                )

            for legacy_path in possible_legacy_paths:
                if legacy_path.exists():
                    try:
                        with open(legacy_path, encoding="utf-8") as f:
                            legacy_data = json.load(f)
                            if isinstance(legacy_data, dict) and legacy_data:
                                legacy_data_map.update(legacy_data)
                                legacy_metadata_count += len(legacy_data)
                                logger.info(
                                    f"[rebuild_index] 从 JSON 加载 {len(legacy_data)} 条: {legacy_path}"
                                )
                    except Exception as e:
                        logger.warning(f"[rebuild_index] 加载 JSON 失败 {legacy_path}: {e}")

            old_count = len(old_index) + len(legacy_data_map)
            restore_rebuilt_metadata(rebuilt_index, old_index, legacy_data_map)

            # 3. 使用新的索引作为最终索引（自动清理了不存在的文件记录）
            final_index = rebuilt_index

            # 重建后若超过容量限制，先执行容量控制清理
            max_reg = getattr(self.plugin.plugin_config, "max_reg_num", getattr(self.plugin, "max_reg_num", 0))
            if max_reg > 0 and library_counts(final_index)["automatic"] > max_reg:
                logger.info(
                    f"[rebuild_index] 重建后数量 {len(final_index)} 超过限制 {max_reg}，"
                    f"执行容量控制清理"
                )
                await self.plugin.event_handler._enforce_capacity(final_index)

            # 保存合并后的索引
            await self.plugin.index_manager.save_index(final_index)

            # save_index 是增量同步，明确移除旧数据库中已丢失的文件记录。
            stale_paths = [
                path for path in old_index
                if path not in final_index
                and Path(path).is_absolute()
                and not Path(path).exists()
            ]
            if stale_paths:
                await delete_index_paths(self.plugin, stale_paths)

            # 增量同步还会保留扫描目录外的有效记录，统计以数据库为准。
            persisted_index = self.plugin.db_service.get_index_cache_readonly()
            new_count = len(persisted_index)
            recovered_count = sum(
                1 for meta in final_index.values()
                if isinstance(meta, dict) and has_meaningful_metadata(meta)
            )

            # 按分类统计
            category_stats = Counter(
                img_info.get("category", "未分类")
                for img_info in persisted_index.values()
                if isinstance(img_info, dict)
            )

            # 构建结果消息
            result_msg = "✅ 索引重建完成！\n\n"
            result_msg += "📊 统计信息:\n"
            result_msg += f"  重建前旧数据: {old_count} 条\n"
            if legacy_metadata_count > 0:
                result_msg += f"  旧版备份数据: {legacy_metadata_count} 条\n"
            result_msg += f"  重建后索引/文件: {new_count} 个\n"
            result_msg += f"  已恢复元数据: {recovered_count} 条\n"

            if category_stats:
                result_msg += "\n📂 分类统计:\n"
                for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                    result_msg += f"  {cat}: {count}张\n"

            yield event.plain_result(result_msg)

        except Exception as e:
            logger.error(f"重建索引失败: {e}", exc_info=True)
            yield event.plain_result(f"❌ 重建索引失败: {str(e)}")
