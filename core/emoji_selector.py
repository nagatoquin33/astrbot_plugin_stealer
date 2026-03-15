import asyncio
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

# 尝试导入文本相似度计算模块
try:
    from .text_similarity import (
        _extract_words,
        calculate_hybrid_similarity,
    )
except ImportError:
    # 如果导入失败，提供简单的降级实现
    def _extract_words(text: str) -> set[str]:
        return set(text.split())

    def calculate_hybrid_similarity(text1: str, text2: str) -> float:
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        if text2_lower in text1_lower or text1_lower in text2_lower:
            return 0.5
        words1 = set(text1_lower.split())
        words2 = set(text2_lower.split())
        if words1 and words2:
            return len(words1 & words2) / max(len(words1), len(words2))
        return 0.0


class EmojiSelector:
    """表情包选择器，负责查找、筛选和选择表情包。"""

    # 正则表达式模式常量
    HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:&&|\\&\\&)")
    INCOMPLETE_HEX_PATTERN = re.compile(
        r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:[|]|\n|$)"
    )
    SINGLE_HEX_PATTERN = re.compile(r"&([^&\s]+?)&")

    # 选择器常量
    MAX_RECENT_USAGE = 10  # 最近使用记录最大数量
    MIN_RECENT_USAGE = 3  # 最近使用记录最小数量

    # 相似度阈值常量（v2 n-gram 改进后适当降低阈值以提升召回率）
    SIMILARITY_THRESHOLD = 0.45  # 模糊匹配相似度阈值

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self.categories: list[str] = getattr(plugin_instance, "categories", [])
        self._selection_lock = asyncio.Lock()
        self._recent_usage: dict[str, list[str]] = {}  # category -> [canon_path, ...]

    def _check_group_allowed(self, event: AstrMessageEvent) -> bool:
        """检查当前群组是否允许使用表情包功能。

        Args:
            event: 消息事件对象

        Returns:
            bool: True 表示允许，False 表示不允许
        """
        return self.plugin.is_send_enabled_for_event(event)

    def _canon_path(self, path: str) -> str:
        """规范化路径用于比较去重。"""
        try:
            return os.path.normcase(os.path.abspath(os.path.normpath(path)))
        except Exception:
            return str(path or "")

    def _get_category_from_data(self, data: dict | None) -> str:
        """从数据字典中获取小写的分类名。

        Args:
            data: 图片元数据字典

        Returns:
            str: 小写的分类名，如果不存在则返回空字符串
        """
        if not isinstance(data, dict):
            return ""
        return str(data.get("category", "")).lower()

    @staticmethod
    @lru_cache(maxsize=4096)
    def _collect_phrase_words(items: tuple[str, ...]) -> frozenset[str]:
        words = set()
        for item in items:
            words.update(_extract_words(item))
        return frozenset(words)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _prepare_entry_text_features(
        category: str,
        desc: str,
        tags: tuple[str, ...],
        scenes: tuple[str, ...] = (),
    ) -> tuple[str, frozenset[str], frozenset[str], frozenset[str], str]:
        desc_lower = str(desc or "").lower()
        tag_words = EmojiSelector._collect_phrase_words(tags)
        scene_words = EmojiSelector._collect_phrase_words(scenes)
        all_text = " ".join(
            part for part in [str(category or ""), desc_lower, " ".join(tags)] if part
        )
        all_words = _extract_words(all_text)
        return desc_lower, tag_words, scene_words, all_words, all_text

    def _get_recent_usage(self, category: str) -> list[str]:
        return list(self._recent_usage.get(category, []))

    def _set_recent_usage(self, category: str, recent_usage: list[str]) -> None:
        self._recent_usage[category] = recent_usage

    def _update_recent_usage(self, category: str, path: str) -> None:
        canon_path = self._canon_path(path)
        recent_usage = [p for p in self._get_recent_usage(category) if p != canon_path]
        recent_usage.append(canon_path)
        if len(recent_usage) > self.MAX_RECENT_USAGE:
            recent_usage = recent_usage[-self.MAX_RECENT_USAGE :]
        self._set_recent_usage(category, recent_usage)

    def _calculate_recent_penalty(self, category: str, path: str) -> float:
        canon_path = self._canon_path(path)
        recent_usage = self._get_recent_usage(category)
        if not recent_usage or canon_path not in recent_usage:
            return 0.0

        # 越接近列表末尾表示越新近使用，惩罚越重，避免连续发送相同表情。
        recency_rank = len(recent_usage) - 1 - recent_usage.index(canon_path)
        penalty_steps = [0.55, 0.38, 0.24, 0.14]
        if recency_rank < len(penalty_steps):
            return penalty_steps[recency_rank]

        return max(0.06, 0.14 - (recency_rank - len(penalty_steps) + 1) * 0.02)

    def _get_candidate_categories(self, category: str, limit: int = 3) -> list[str]:
        normalized = self.normalize_category(category) or category.lower().strip()
        if not normalized:
            return []

        cfg = self.plugin.plugin_config
        info_map = getattr(cfg, "category_info", {}) if cfg else {}
        scored: list[tuple[float, str]] = []

        for current in self.categories:
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

    async def record_emoji_usage(self, emoji_path: str, trigger: str = "auto") -> None:
        cache_service = getattr(self.plugin, "cache_service", None)
        if not cache_service or not emoji_path:
            return

        target_path = self._canon_path(emoji_path)
        now = int(time.time())

        def _updater(current: dict) -> None:
            for stored_path, meta in current.items():
                if self._canon_path(stored_path) != target_path:
                    continue
                if not isinstance(meta, dict):
                    continue

                meta["use_count"] = int(meta.get("use_count", 0) or 0) + 1
                meta["last_used_at"] = now
                break

        try:
            await cache_service.update_index(_updater)
        except Exception as e:
            logger.debug(f"[Stealer] 更新表情使用统计失败: {e}")

    def normalize_category(self, category: str) -> str:
        """归一化分类名称，返回有效分类或空字符串。"""
        if not category:
            return ""
        cfg = self.plugin.plugin_config
        if not cfg:
            return ""
        try:
            result = cfg.normalize_category_strict(category)
            return result or ""
        except Exception:
            return ""

    async def extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.categories)

            # 三种标记格式依次提取: &&tag&&, 残缺 &&tag, 单个 &tag&
            patterns = [
                (self.HEX_PATTERN, True),  # 完整标记：总是清理
                (self.INCOMPLETE_HEX_PATTERN, False),  # 残缺标记：仅匹配时清理
                (self.SINGLE_HEX_PATTERN, True),  # 单标记：总是清理
            ]
            for pattern, always_clean in patterns:
                cleaned_text, found = self._extract_with_pattern(
                    pattern, cleaned_text, valid_categories, seen, always_clean
                )
                res.extend(found)

            return res, cleaned_text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    @staticmethod
    def _extract_with_pattern(
        pattern: re.Pattern,
        text: str,
        valid_categories: set[str],
        seen: set[str],
        always_clean: bool,
    ) -> tuple[str, list[str]]:
        """用指定正则从文本中提取情绪标签并清理原文。

        Args:
            pattern: 编译好的正则
            text: 待处理文本
            valid_categories: 合法分类集合
            seen: 已出现的分类（去重用，会被原地修改）
            always_clean: True=无论是否匹配到分类都清理标记; False=仅匹配到有效分类时清理

        Returns:
            (cleaned_text, new_emotions)
        """
        found: list[str] = []
        temp = text
        for match in pattern.finditer(text):
            norm_cat = match.group(1).strip().lower()
            if norm_cat in valid_categories and norm_cat not in seen:
                seen.add(norm_cat)
                found.append(norm_cat)
                temp = temp.replace(match.group(0), "", 1)
            elif always_clean:
                temp = temp.replace(match.group(0), "", 1)
        return temp, found

    async def select_emoji(self, category: str, context_text: str = "") -> str | None:
        """选择表情包（智能或随机）。"""
        async with self._selection_lock:
            use_smart = self.plugin.smart_emoji_selection
            candidate_categories = self._get_candidate_categories(category)

            if use_smart and context_text and len(context_text.strip()) > 5:
                smart_path = await self._select_emoji_smart_impl(
                    category,
                    context_text,
                    candidate_categories=candidate_categories,
                )
                if smart_path:
                    return smart_path

            for candidate_category in candidate_categories:
                random_path = self._select_emoji_random_impl(
                    candidate_category, use_smart=use_smart
                )
                if random_path:
                    return random_path

            return None

    async def select_emoji_smart(self, category: str, context_text: str) -> str | None:
        """智能选择表情包（强制智能）。"""
        async with self._selection_lock:
            return await self._select_emoji_smart_impl(
                category,
                context_text,
                candidate_categories=self._get_candidate_categories(category),
            )

    def _select_emoji_random_impl(self, category: str, use_smart: bool) -> str | None:
        cfg = self.plugin.plugin_config
        categories_dir = cfg.categories_dir if cfg else None

        if not categories_dir:
            return None

        cat_dir = Path(categories_dir) / category
        if not cat_dir.exists():
            return None

        try:
            files = [p for p in cat_dir.iterdir() if p.is_file()]
            if not files:
                return None

            recent_usage = self._get_recent_usage(category)
            recent_set = set(recent_usage)
            candidates = [(p, self._canon_path(str(p))) for p in files]

            # 过滤最近使用
            available = [p for p, canon in candidates if canon not in recent_set]
            if not available:
                available = [p for p, _ in candidates]
                recent_usage = []
                recent_set = set()

            # 尝试选择一个存在的文件（最多重试3次）
            max_retries = min(3, len(available))
            for _ in range(max_retries):
                picked = random.choice(available)
                # 检查文件是否仍然存在
                if picked.exists():
                    picked_path = self._canon_path(str(picked))

                    if picked_path in recent_set:
                        recent_usage = [p for p in recent_usage if p != picked_path]
                    recent_usage.append(picked_path)

                    max_recent = min(
                        self.MAX_RECENT_USAGE, max(self.MIN_RECENT_USAGE, len(files) // 2)
                    )
                    if len(recent_usage) > max_recent:
                        recent_usage = recent_usage[-max_recent:]

                    self._set_recent_usage(category, recent_usage)
                    return str(picked)
                else:
                    # 文件已不存在，从候选列表中移除
                    available.remove(picked)
                    if not available:
                        break

            return None
        except Exception as e:
            logger.error(f"随机选择表情包失败: {e}")
            return None

    async def _select_emoji_smart_impl(
        self,
        category: str,
        context_text: str,
        candidate_categories: list[str] | None = None,
    ) -> str | None:
        """?????????????????"""
        try:
            cache_service = self.plugin.cache_service
            if not cache_service:
                return None

            idx = cache_service.get_index_cache_readonly()
            if not idx:
                return None

            allowed_categories = set(candidate_categories or [category])
            candidates = []
            low_score_candidates = []
            context_lower = context_text.lower()
            context_words = _extract_words(context_text)

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                entry_category = self._get_category_from_data(data)
                if entry_category not in allowed_categories:
                    continue

                tags = self._parse_tags(data.get("tags", []))
                scenes = self._parse_tags(data.get("scenes", []))
                desc, tag_words, scene_words, _, _ = self._prepare_entry_text_features(
                    entry_category, str(data.get("desc", "")), tuple(tags), tuple(scenes)
                )
                desc_score = calculate_hybrid_similarity(context_text, desc)
                if desc_score < 0.25:
                    desc_words = _extract_words(desc)
                    overlap = context_words & desc_words
                    bigram_hits = sum(1 for w in overlap if len(w) >= 2)
                    unigram_hits = len(overlap) - bigram_hits
                    boost = bigram_hits * 0.25 + unigram_hits * 0.1
                    if boost > 0:
                        desc_score = max(desc_score, min(1.0, boost))

                tag_score = 0.0
                if tags:
                    matched_tags = sum(1 for tag in tags if tag in context_lower)
                    tag_score = min(1.0, matched_tags / max(len(tags), 1))
                    if context_words & tag_words:
                        tag_score = min(1.0, tag_score + 0.3)

                scene_score = 0.0
                if scenes:
                    matched_scenes = sum(1 for scene in scenes if scene in context_lower)
                    scene_score = min(1.0, matched_scenes / max(len(scenes), 1))
                    if context_words & scene_words:
                        scene_score = min(1.0, scene_score + 0.35)

                category_bonus = 0.12 if entry_category == category else 0.04
                use_count_bonus = min(0.08, int(data.get("use_count", 0) or 0) * 0.01)
                base_score = (
                    desc_score * 0.35
                    + tag_score * 0.25
                    + scene_score * 0.2
                    + category_bonus
                    + use_count_bonus
                )

                if base_score < 0.15:
                    if desc_score > 0.1:
                        low_score_candidates.append(
                            (
                                file_path,
                                desc_score,
                                desc_score,
                                0.0,
                                0.0,
                                entry_category,
                            )
                        )
                    continue

                diversity_bonus = random.uniform(0, 0.15)
                canon_path = self._canon_path(file_path)
                history_penalty = self._calculate_recent_penalty(
                    entry_category, canon_path
                )

                final_score = max(0.0, base_score + diversity_bonus - history_penalty)
                if final_score > 0.1:
                    candidates.append(
                        (
                            file_path,
                            final_score,
                            desc_score,
                            tag_score,
                            scene_score,
                            entry_category,
                        )
                    )

            if not candidates:
                candidates = low_score_candidates

            if not candidates:
                return None

            candidates.sort(key=lambda item: item[1], reverse=True)
            top_candidates = candidates[: min(3, len(candidates))]
            if len(top_candidates) > 1:
                weights = [item[1] for item in top_candidates]
                total_weight = sum(weights)
                if total_weight > 0:
                    selected = random.choices(top_candidates, weights=weights, k=1)[0]
                    self._update_recent_usage(selected[5], selected[0])
                    return selected[0]

            result = candidates[0]
            self._update_recent_usage(result[5], result[0])
            logger.debug(
                f"[????] ??={category}, ??={len(candidates)}, ????={result[5]}, "
                f"??={result[1]:.2f} (desc={result[2]:.2f}, tag={result[3]:.2f}, scene={result[4]:.2f})"
            )
            return result[0]

        except Exception as e:
            logger.error(f"?????????: {e}")
            return None

    @staticmethod
    def _parse_tags(raw_tags: Any) -> list[str]:
        """安全解析 tags 字段，兼容字符串和列表类型。"""
        if isinstance(raw_tags, str):
            return [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        if isinstance(raw_tags, list):
            return [str(t).lower() for t in raw_tags if t]
        return []

    async def search_images(
        self, query: str, limit: int = 1, idx: dict | None = None
    ) -> list[tuple[str, str, str, str]]:
        """根据查询词搜索图片（纯评分搜索，不含关键词映射）。"""
        try:
            if idx is None:
                cache_service = self.plugin.cache_service
                if cache_service:
                    idx = cache_service.get_index_cache_readonly()

            if not idx:
                return []

            query_lower = query.lower()
            query_tokens = [t for t in query_lower.split() if len(t) > 1]
            # 使用分词提取查询词的词汇（更适合中文）
            query_words = _extract_words(query)

            MAX_STR_LENGTH = 20
            top_k = limit * 3
            top_candidates: list[tuple[str, str, str, str, int]] = []

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                tags = self._parse_tags(data.get("tags", []))
                scenes = self._parse_tags(data.get("scenes", []))
                tags_for_score = tags + scenes
                tags_str = ", ".join(tags)
                category = self._get_category_from_data(data)
                desc, tag_words, _, all_words, all_text = self._prepare_entry_text_features(
                    category, str(data.get("desc", "")), tuple(tags_for_score)
                )

                # 快速分词过滤：检查词汇重叠（比字符级更适合中文）
                if query_words:
                    # 至少有一个词匹配才继续
                    if not query_words.intersection(all_words):
                        # 兜底：字符级检查（处理单字查询等情况）
                        query_chars = set(query_lower)
                        if query_chars and not query_chars.intersection(set(all_text)):
                            continue

                score = self._score_entry(
                    query_lower,
                    query_tokens,
                    category,
                    desc,
                    tags_for_score,
                    MAX_STR_LENGTH,
                    tag_words=tag_words,
                )

                if score > 0:
                    top_candidates.append((file_path, desc, category, tags_str, score))
                    # 定期裁剪，避免候选列表过大
                    if len(top_candidates) > top_k:
                        top_candidates.sort(key=lambda x: x[4], reverse=True)
                        top_candidates = top_candidates[:top_k]

            top_candidates.sort(key=lambda x: x[4], reverse=True)
            return [
                (item[0], item[1], item[2], item[3]) for item in top_candidates[:limit]
            ]

        except Exception as e:
            logger.error(f"搜索图片失败: {e}")
            return []

    def _score_entry(
        self,
        query_lower: str,
        query_tokens: list[str],
        category: str,
        desc: str,
        tags: list[str],
        max_str_len: int,
        tag_words: frozenset[str] | None = None,
    ) -> int:
        """计算单条索引条目与查询的匹配得分。"""
        # 精确匹配分类
        if query_lower == category:
            return 20

        score = 0

        # 包含匹配分类
        if query_lower in category or category in query_lower:
            score = 10

        # 描述匹配
        if query_lower == desc:
            score = max(score, 15)
        elif query_lower in desc:
            score = max(score, 12)
        elif query_tokens:
            matched = sum(1 for t in query_tokens if t in desc)
            score = max(score, matched * 3)

        if score >= 15:
            return score

        # 标签匹配
        if tags:
            if query_lower == tags[0]:
                score = max(score, 12)
            elif query_lower in tags[0]:
                score = max(score, 8)
            if query_tokens:
                for tag in tags:
                    matched = sum(1 for t in query_tokens if t in tag)
                    score = max(score, matched * 2)
                    if score >= 15:
                        return score

        # 分词匹配（n-gram 改进：bigram 匹配权重更高，如"猫娘"中的"猫"可匹配单字）
        if score < 10:
            query_words = _extract_words(query_lower)
            if query_words:
                # 检查分词在描述中的匹配
                desc_words = _extract_words(desc)
                overlap = query_words & desc_words
                bigram_hits = sum(1 for w in overlap if len(w) >= 2)
                unigram_hits = len(overlap) - bigram_hits
                word_score = bigram_hits * 6 + unigram_hits * 3
                if word_score > 0:
                    score = max(score, word_score)

                # 检查分词在标签中的匹配
                current_tag_words = tag_words or self._collect_phrase_words(tuple(tags))
                tag_overlap = query_words & current_tag_words
                tag_bigram = sum(1 for w in tag_overlap if len(w) >= 2)
                tag_unigram = len(tag_overlap) - tag_bigram
                tag_word_score = tag_bigram * 7 + tag_unigram * 3
                if tag_word_score > 0:
                    score = max(score, tag_word_score)

        # 模糊匹配（使用多策略融合相似度）
        if score < 10:
            for target in [category, desc] + tags[:2]:
                if len(target) > 1 and len(query_lower) > 1:
                    sim = calculate_hybrid_similarity(
                        query_lower[:max_str_len], target[:max_str_len]
                    )
                    if sim >= self.SIMILARITY_THRESHOLD:
                        score = max(
                            score, int(4 + (sim - self.SIMILARITY_THRESHOLD) * 16)
                        )
                        break

        return score

    async def smart_search(
        self,
        query: str,
        limit: int = 5,
        idx: dict | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """智能搜索表情包（带多级 fallback）。

        搜索顺序：
        1) 直接用 query 调用 search_images
        2) 关键词映射（如"无语" -> dumb）
        3) 模糊匹配到分类（相似度阈值 0.4）

        Args:
            query: 搜索关键词
            limit: 返回结果数量
            idx: 索引缓存，为 None 时自动加载

        Returns:
            list[tuple[path, desc, emotion, tags]]
        """
        # 1) 直接搜索
        results = await self.search_images(query, limit=limit, idx=idx)
        if results:
            return results

        # 2) 关键词映射
        cfg = self.plugin.plugin_config
        keyword_map = cfg.get_keyword_map() if cfg else {}
        if query in keyword_map:
            mapped_category = keyword_map[query]
            results = await self.search_images(mapped_category, limit=limit, idx=idx)
            if results:
                return results

        # 3) 模糊匹配到分类
        best_match = self._find_best_category_match(query, threshold=0.4)
        if best_match:
            results = await self.search_images(best_match, limit=limit, idx=idx)

        return results

    def _find_best_category_match(self, query: str, threshold: float = 0.4) -> str | None:
        """找到与查询词最相似的分类。

        Args:
            query: 查询词
            threshold: 相似度阈值

        Returns:
            最佳匹配的分类，无则返回 None
        """
        if not query or not self.categories:
            return None

        best_match = None
        best_score = 0.0
        for category in self.categories:
            score = calculate_hybrid_similarity(query, category)
            if score > best_score and score > threshold:
                best_score = score
                best_match = category

        return best_match

    def find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的多个分类。

        Args:
            query: 查询词
            top_n: 返回数量

        Returns:
            相似分类列表
        """
        if not query or not self.categories:
            return []

        scores = []
        for category in self.categories:
            score = calculate_hybrid_similarity(query, category)
            scores.append((category, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in scores[:top_n]]

    def check_send_probability(self) -> bool:
        """检查表情包发送概率。"""
        try:
            emoji_chance = self.plugin.emoji_chance
            chance = float(emoji_chance)
            if chance <= 0:
                logger.debug("表情包自动发送概率为0，未触发图片发送")
                return False
            if chance > 1:
                chance = 1.0
            if random.random() >= chance:
                logger.debug(f"表情包自动发送概率检查未通过 ({chance}), 未触发图片发送")
                return False

            logger.debug("表情包自动发送概率检查通过")
            return True
        except Exception as e:
            logger.error(f"解析表情包自动发送概率配置失败: {e}")
            return False

    async def _encode_emoji(self, emoji_path: str) -> str | None:
        """将表情包文件编码为 base64，失败返回 None。"""
        if not emoji_path or not isinstance(emoji_path, str):
            logger.warning(f"[表情包编码] 无效的文件路径: {emoji_path!r}")
            return None
        if not os.path.exists(emoji_path):
            logger.warning(f"表情包文件不存在: {emoji_path}")
            return None
        image_processor = self.plugin.image_processor_service
        if not image_processor:
            logger.warning("[表情包编码] image_processor_service 未初始化")
            return None
        try:
            return await image_processor._file_to_gif_base64(emoji_path)
        except Exception as e:
            logger.error(f"编码表情包失败: {emoji_path}, {e}")
            return None

    async def send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ) -> None:
        """发送表情包（异步场景下直接发送新消息）。"""
        try:
            if event.get_extra("stealer_active_sent"):
                logger.debug("[Stealer] 已主动发送过表情包，跳过自动发送")
                return

            from astrbot.api.event import MessageChain
            from astrbot.api.message_components import Image as ImageComponent

            if not self._check_group_allowed(event):
                return

            b64 = await self._encode_emoji(emoji_path)
            if not b64:
                return

            await event.send(MessageChain([ImageComponent.fromBase64(b64)]))
            await self.record_emoji_usage(emoji_path, trigger="auto")
            logger.debug(f"[Stealer] 已发送表情包: {emoji_path}")

        except Exception as e:
            logger.error(f"发送表情包失败: {e}", exc_info=True)

    async def send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ) -> None:
        """发送显式指定的表情包列表和文本。"""
        from astrbot.api.message_components import Plain

        try:
            result = event.get_result()
            new_result = event.make_result().set_result_content_type(
                result.result_content_type
            )

            # 保留非 Plain 组件
            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            # 依次编码并添加图片
            sent_paths = []
            for path in emoji_paths:
                b64 = await self._encode_emoji(path)
                if b64:
                    new_result.base64_image(b64)
                    sent_paths.append(path)

            event.set_result(new_result)
            for path in sent_paths:
                await self.record_emoji_usage(path, trigger="explicit")
        except Exception as e:
            logger.error(f"发送显式表情包失败: {e}", exc_info=True)

    async def try_send_emoji(
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str
    ) -> bool:
        """尝试发送表情包，遍历 emotions 列表直到第一个匹配到的表情包。"""
        if not self._check_group_allowed(event):
            return False

        if event.get_extra("stealer_active_sent"):
            logger.debug("[Stealer] 检测到 stealer_active_sent=True，跳过自动表情发送")
            return False

        if not self.check_send_probability():
            return False

        # 遍历情绪列表，第一个能选到表情包的就发送
        for emotion in emotions:
            emoji_path = await self.select_emoji(emotion, cleaned_text)
            if emoji_path:
                await self.send_emoji_with_text(event, emoji_path, cleaned_text)
                logger.debug(f"已发送表情包 (情绪={emotion})")
                return True

        logger.debug("[Stealer] 所有情绪均未匹配到表情包")
        return False
