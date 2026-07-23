"""表情包选择策略：负责最近使用记录管理和候选分类获取。"""

import os
from typing import Any

from astrbot.api import logger

from .text_similarity import calculate_hybrid_similarity


class MemeSelectionStrategy:
    """管理表情包选择的策略状态（最近使用记录等）。"""

    MAX_RECENT_USAGE = 10  # 最近使用记录最大数量
    MIN_RECENT_USAGE = 3  # 最近使用记录最小数量

    def __init__(self, plugin_instance: Any, selector: Any) -> None:
        self.plugin = plugin_instance
        self.selector = selector
        self._recent_usage: dict[str, list[str]] = {}  # category -> [canon_path, ...]

    def _canon_path(self, path: str) -> str:
        """规范化路径用于比较去重（仅大小写规范化，不转换斜杠）。"""
        return os.path.normcase(str(path or "")).replace("\\", "/")

    def _get_recent_usage(self, category: str) -> list[str]:
        return list(self._recent_usage.get(category, []))

    def _set_recent_usage(self, category: str, recent_usage: list[str]) -> None:
        self._recent_usage[category] = recent_usage

    def _update_recent_usage(self, category: str, path: str) -> None:
        canon_path = self._canon_path(path)
        recent_usage = [
            p for p in self._get_recent_usage(category) if self._canon_path(p) != canon_path
        ]
        recent_usage.append(path)
        if len(recent_usage) > self.MAX_RECENT_USAGE:
            recent_usage = recent_usage[-self.MAX_RECENT_USAGE :]
        self._set_recent_usage(category, recent_usage)
        logger.debug(f"[去重] 更新历史: 分类={category}, 路径={path}, 新历史={recent_usage}")

    def _calculate_recent_penalty(self, category: str, path: str) -> float:
        canon_path = self._canon_path(path)
        recent_usage = self._get_recent_usage(category)
        recent_paths_canon = [self._canon_path(p) for p in recent_usage]
        if not recent_usage or canon_path not in recent_paths_canon:
            return 0.0

        recency_rank = len(recent_usage) - 1 - recent_paths_canon.index(canon_path)
        penalty_steps = [0.55, 0.38, 0.24, 0.14]
        if recency_rank < len(penalty_steps):
            penalty = penalty_steps[recency_rank]
        else:
            penalty = max(0.06, 0.14 - (recency_rank - len(penalty_steps) + 1) * 0.02)
        logger.debug(
            f"[去重] 分类={category}, 路径={path}, 历史={recent_usage}, 排名={recency_rank}, 惩罚={penalty:.2f}"
        )
        return penalty

    def _get_candidate_categories(self, category: str, limit: int = 3) -> list[str]:
        normalized = self.selector.normalize_category(category) or category.lower().strip()
        if not normalized:
            return []

        cfg = self.plugin.plugin_config
        info_map = getattr(cfg, "category_info", {}) if cfg else {}
        scored: list[tuple[float, str]] = []

        for current in self.selector.categories:
            if current == normalized:
                scored.append((10.0, current))
                continue

            score = calculate_hybrid_similarity(normalized, current)
            info = info_map.get(current, {}) if isinstance(info_map, dict) else {}
            name = str(info.get("name", "") or "")
            desc = str(info.get("desc", "") or "")
            if name:
                score = max(score, calculate_hybrid_similarity(normalized, name))
            if desc:
                score = max(score, calculate_hybrid_similarity(normalized, desc))
            if score >= 0.18:
                scored.append((score, current))

        scored.sort(key=lambda item: item[0], reverse=True)
        result: list[str] = []
        for _, current in scored:
            if current not in result:
                result.append(current)
            if len(result) >= max(1, limit):
                break
        return result or [normalized]
