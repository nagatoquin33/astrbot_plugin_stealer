"""索引重建命令：负责从文件重建索引并合并旧元数据。"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


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

            # --- 智能合并逻辑开始 ---
            def _has_meaningful_metadata(meta: dict[str, Any] | None) -> bool:
                if not isinstance(meta, dict):
                    return False
                return bool(meta.get("tags") or meta.get("desc") or meta.get("scenes"))

            def _build_lookup_maps(
                index_map: dict[str, Any],
            ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
                hash_map: dict[str, dict[str, Any]] = {}
                name_map: dict[str, dict[str, Any]] = {}
                for path_key, meta in index_map.items():
                    if not isinstance(meta, dict):
                        continue

                    if meta.get("hash"):
                        hash_val = str(meta["hash"])
                        existing = hash_map.get(hash_val)
                        if existing is None or (
                            not _has_meaningful_metadata(existing)
                            and _has_meaningful_metadata(meta)
                        ):
                            hash_map[hash_val] = meta

                    path_obj = Path(path_key)
                    for name_key in (path_obj.name, path_obj.stem):
                        existing = name_map.get(name_key)
                        if existing is None or (
                            not _has_meaningful_metadata(existing)
                            and _has_meaningful_metadata(meta)
                        ):
                            name_map[name_key] = meta
                return hash_map, name_map

            def _first_casefold_stem_match(
                new_path_obj: Path, *source_indexes: dict[str, Any]
            ) -> dict[str, Any] | None:
                needle = new_path_obj.stem.lower()
                for source_index in source_indexes:
                    for old_path, old_val in source_index.items():
                        if not isinstance(old_val, dict):
                            continue
                        if Path(old_path).stem.lower() == needle:
                            return old_val
                return None

            def _resolve_old_metadata(
                new_path: str,
                new_data: dict[str, Any],
            ) -> dict[str, Any] | None:
                new_path_obj = Path(new_path)
                new_hash = new_data.get("hash")

                exact_candidates = (
                    old_index.get(new_path),
                    current_hash_map.get(new_hash),
                    legacy_data_map.get(new_path),
                    legacy_hash_map.get(new_hash),
                    current_name_map.get(new_path_obj.name),
                    current_name_map.get(new_path_obj.stem),
                    legacy_name_map.get(new_path_obj.name),
                    legacy_name_map.get(new_path_obj.stem),
                )
                for candidate in exact_candidates:
                    if isinstance(candidate, dict):
                        return candidate

                return _first_casefold_stem_match(new_path_obj, old_index, legacy_data_map)

            def _restore_metadata(target_data: dict[str, Any], source_data: dict[str, Any]) -> bool:
                if not isinstance(source_data, dict):
                    return False

                restored = False
                if source_data.get("desc"):
                    target_data["desc"] = source_data["desc"]
                    restored = True
                if source_data.get("tags"):
                    target_data["tags"] = source_data["tags"]
                    restored = True

                for key in (
                    "source_message",
                    "source",
                    "origin_target",
                    "scope_mode",
                    "qq_emoji_id",
                    "qq_emoji_package_id",
                    "origin_url",
                    "qq_key",
                    "scenes",
                    "scene",
                ):
                    if key in source_data:
                        target_data[key] = source_data[key]
                        restored = True
                return restored

            current_hash_map, current_name_map = _build_lookup_maps(old_index)
            legacy_hash_map, legacy_name_map = _build_lookup_maps(legacy_data_map)

            old_count = len(old_index) + len(legacy_data_map)

            recovered_count = 0

            # 2. 遍历重建的索引，尝试恢复元数据
            for new_path, new_data in rebuilt_index.items():
                old_data = _resolve_old_metadata(new_path, new_data)
                if _restore_metadata(new_data, old_data):
                    recovered_count += 1

            # 3. 使用新的索引作为最终索引（自动清理了不存在的文件记录）
            final_index = rebuilt_index
            # --- 智能合并逻辑结束 ---

            # 重建后若超过容量限制，先执行容量控制清理
            max_reg = getattr(self.plugin, "max_reg_num", 0)
            if max_reg > 0 and len(final_index) > max_reg:
                logger.info(
                    f"[rebuild_index] 重建后数量 {len(final_index)} 超过限制 {max_reg}，"
                    f"执行容量控制清理"
                )
                await self.plugin.event_handler._enforce_capacity(final_index)

            # 保存合并后的索引
            await self.plugin.index_manager.save_index(final_index)

            # 统计信息（容量控制后重新统计，只算最终保留的文件）
            new_count = len(final_index)
            recovered_count = sum(
                1 for meta in final_index.values()
                if isinstance(meta, dict) and _has_meaningful_metadata(meta)
            )

            # 按分类统计
            category_stats = Counter(
                img_info.get("category", "未分类")
                for img_info in final_index.values()
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
