import asyncio
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .emoji_search_engine import EmojiSearchEngine
from .emoji_selection_strategy import EmojiSelectionStrategy

from .text_similarity import (
    _extract_words,
)


class EmojiSelector:
    """表情包选择器，负责查找、筛选和选择表情包。"""

    # 正则表达式模式常量
    HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:&&|\\&\\&)")
    INCOMPLETE_HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:[|]|\n|$)")
    SINGLE_HEX_PATTERN = re.compile(r"&([^&\s]+?)&")

    # 选择器常量
    MAX_RECENT_USAGE = 10  # 最近使用记录最大数量
    MIN_RECENT_USAGE = 3  # 最近使用记录最小数量

    # 相似度阈值常量（v2 n-gram 改进后适当降低阈值以提升召回率）
    SIMILARITY_THRESHOLD = 0.45  # 模糊匹配相似度阈值
    SMART_BM25_PREFILTER_LIMIT = 80  # 智能选择时 BM25 预召回上限
    SMART_BM25_BONUS_WEIGHT = 0.2  # BM25 分数对最终打分的加权
    SMART_FAST_PREFILTER_TOP_K = 120  # 轻量预筛后保留的候选上限
    SMART_FAST_PREFILTER_MIN_CANDIDATES = 48  # 候选少于该值时跳过预筛
    SMART_FAST_PREFILTER_FUZZY_RESERVE = 24  # 预筛时保留部分模糊候选，避免误杀

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self.categories: list[str] = getattr(plugin_instance, "categories", [])
        self._selection_lock = asyncio.Lock()

        # 子服务（职责拆分）
        from .emoji_smart_select_service import EmojiSmartSelectService

        self._search_engine = EmojiSearchEngine(plugin_instance, self)
        self._selection_strategy = EmojiSelectionStrategy(plugin_instance, self)
        self._smart_select_service = EmojiSmartSelectService(plugin_instance)
        self._smart_select_service._search_engine = self._search_engine
        self._smart_select_service._selector = self
        self._recent_usage = self._selection_strategy._recent_usage

    def __getattr__(self, name: str):
        """向后兼容：将 BM25 属性委托给 EmojiSearchEngine。"""
        if name in ("_bm25_dirty", "_bm25_doc_paths", "_bm25_signature", "_bm25_documents"):
            return getattr(self._search_engine, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        """向后兼容：将 BM25 属性设置委托给 EmojiSearchEngine。"""
        if name in ("_bm25_dirty", "_bm25_doc_paths", "_bm25_signature", "_bm25_documents"):
            if hasattr(self, "_search_engine"):
                setattr(self._search_engine, name, value)
                return
        super().__setattr__(name, value)

    def _get_index(self) -> dict[str, Any]:
        db_service = getattr(self.plugin, "db_service", None)
        if db_service:
            idx = db_service.get_index_cache_readonly()
            if idx:
                return idx
        cache_service = getattr(self.plugin, "cache_service", None)
        if cache_service:
            idx = cache_service.get_index_cache_readonly()
            if idx:
                return idx
        return {}

    @staticmethod
    def _search_signature_from_index(idx: dict[str, Any]) -> str:
        """委托给 EmojiSearchEngine。"""
        return EmojiSearchEngine._search_signature_from_index(idx)

    def _compute_bm25_signature(
        self, idx: dict[str, Any], *, prefer_db_signature: bool = True
    ) -> str:
        """委托给 EmojiSearchEngine。"""
        return self._search_engine._compute_bm25_signature(
            idx, prefer_db_signature=prefer_db_signature
        )

    async def _build_bm25_index(self, idx: dict | None = None) -> None:
        """委托给 EmojiSearchEngine。"""
        await self._search_engine._build_bm25_index(idx)

    def _invalidate_bm25_index(self) -> None:
        """委托给 EmojiSearchEngine。"""
        self._search_engine._invalidate_bm25_index()

    def _check_group_allowed(self, event: AstrMessageEvent) -> bool:
        """检查当前群组是否允许使用表情包功能。

        Args:
            event: 消息事件对象

        Returns:
            bool: True 表示允许，False 表示不允许
        """
        return self.plugin.is_send_enabled_for_event(event)

    # ===== 门面委托：EmojiScopeService =====

    def _get_event_target_entry(self, event: AstrMessageEvent | None) -> str:
        """获取事件目标条目（已迁移到 EmojiScopeService）。"""
        from .emoji_scope_service import EmojiScopeService

        return EmojiScopeService(self.plugin)._get_event_target_entry(event)

    def _is_entry_allowed_for_event(
        self, data: dict | None, event: AstrMessageEvent | None
    ) -> bool:
        """检查条目是否允许（已迁移到 EmojiScopeService）。"""
        from .emoji_scope_service import EmojiScopeService

        return EmojiScopeService(self.plugin)._is_entry_allowed_for_event(data, event)

    def is_path_allowed_for_event(self, path: str, event: AstrMessageEvent | None) -> bool:
        """检查路径是否允许（已迁移到 EmojiScopeService）。"""
        from .emoji_scope_service import EmojiScopeService

        return EmojiScopeService(self.plugin).is_path_allowed_for_event(path, event)

    def _canon_path(self, path: str) -> str:
        """规范化路径（已迁移到 EmojiScopeService）。"""
        from .emoji_scope_service import EmojiScopeService

        return EmojiScopeService(self.plugin)._canon_path(path)

    def find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的分类（委托给 EmojiSearchEngine）。"""
        return self._search_engine.find_similar_categories(query, top_n)

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
    def _parse_tags(raw_tags: Any) -> list[str]:
        """解析标签/场景为列表（委托给 EmojiSmartSelectService）。"""
        from .emoji_smart_select_service import EmojiSmartSelectService

        return EmojiSmartSelectService._parse_tags(raw_tags)

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
        """委托给 EmojiSelectionStrategy。"""
        return self._selection_strategy._get_recent_usage(category)

    def _set_recent_usage(self, category: str, recent_usage: list[str]) -> None:
        """委托给 EmojiSelectionStrategy。"""
        self._selection_strategy._set_recent_usage(category, recent_usage)

    def _update_recent_usage(self, category: str, path: str) -> None:
        """委托给 EmojiSelectionStrategy。"""
        self._selection_strategy._update_recent_usage(category, path)

    def _calculate_recent_penalty(self, category: str, path: str) -> float:
        """委托给 EmojiSelectionStrategy。"""
        return self._selection_strategy._calculate_recent_penalty(category, path)

    def _get_candidate_categories(self, category: str, limit: int = 3) -> list[str]:
        """委托给 EmojiSelectionStrategy。"""
        return self._selection_strategy._get_candidate_categories(category, limit)

    async def record_emoji_usage(self, emoji_path: str, trigger: str = "auto") -> None:
        """记录表情包使用次数。

        优先使用数据库服务进行增量更新，fallback 到 cache_service。
        """
        if not emoji_path:
            return

        db_service = getattr(self.plugin, "db_service", None)
        cache_service = getattr(self.plugin, "cache_service", None)

        if db_service:
            # 使用数据库增量更新
            target_path = self._canon_path(emoji_path)
            # 尝试直接更新
            db_service.increment_usage_sync(target_path)
        elif cache_service:
            # fallback 到旧版 cache_service
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

    async def select_emoji(
        self,
        category: str,
        context_text: str = "",
        event: AstrMessageEvent | None = None,
    ) -> str | None:
        """选择表情包（智能或随机）。"""
        async with self._selection_lock:
            use_smart = self.plugin.smart_emoji_selection
            candidate_categories = self._get_candidate_categories(category)

            if use_smart and context_text and len(context_text.strip()) > 5:
                smart_path = await self._select_emoji_smart_impl(
                    category,
                    context_text,
                    candidate_categories=candidate_categories,
                    event=event,
                )
                if smart_path:
                    return smart_path

            for candidate_category in candidate_categories:
                random_path = self._select_emoji_random_impl(candidate_category, event=event)
                if random_path:
                    return random_path

            return None

    def _select_emoji_random_impl(
        self,
        category: str,
        event: AstrMessageEvent | None = None,
    ) -> str | None:
        try:
            files: list[Path] = []
            idx = self._get_index()
            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue
                if self._get_category_from_data(data) != category:
                    continue
                if not self._is_entry_allowed_for_event(data, event):
                    continue
                path_obj = Path(file_path)
                if path_obj.is_file():
                    files.append(path_obj)

            if not files:
                # 带事件上下文时必须依赖索引元数据判断作用域，避免在索引缺项、
                # 缓存未初始化或重建中断时通过目录兜底误发 local 表情。
                if event is not None:
                    return None

                cfg = self.plugin.plugin_config
                categories_dir = cfg.categories_dir if cfg else None
                if not categories_dir:
                    return None

                cat_dir = Path(categories_dir) / category
                if not cat_dir.exists():
                    return None
                files = []
                for path_obj in cat_dir.iterdir():
                    if not path_obj.is_file():
                        continue
                    if not self.is_path_allowed_for_event(str(path_obj), event):
                        continue
                    files.append(path_obj)

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

    # ===== 门面委托：EmojiSmartSelectService =====

    async def _select_emoji_smart_impl(
        self, category: str, context_text: str, candidate_categories=None, event=None
    ):
        """智能选择表情包（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service._select_emoji_smart_impl(
            category, context_text, candidate_categories, event
        )

    async def search_images(
        self, query: str, *, limit: int = 10, idx: dict | None = None, event=None
    ):
        """搜索表情包（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.search_images(
            query, limit=limit, idx=idx, event=event
        )

    async def _search_images_fallback(
        self, query: str, *, limit: int = 10, idx: dict | None = None
    ):
        """降级搜索（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service._search_images_fallback(query, limit=limit, idx=idx)

    async def smart_search(
        self, query: str, *, limit: int = 10, idx: dict | None = None, event=None
    ):
        """智能搜索（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.smart_search(
            query, limit=limit, idx=idx, event=event
        )

    async def send_emoji_message(self, event: AstrMessageEvent, path: str):
        """发送表情包消息（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.send_emoji_message(event, path)

    async def send_emoji_with_text(self, event: AstrMessageEvent, path: str, text: str):
        """带文本发送表情包（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.send_emoji_with_text(event, path, text)

    async def send_explicit_emojis(self, event: AstrMessageEvent, paths: list[str], text: str):
        """发送指定表情包列表（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.send_explicit_emojis(event, paths, text)

    async def try_send_emoji(self, event: AstrMessageEvent, emotions: list[str], text: str) -> bool:
        """尝试发送表情包（已迁移到 EmojiSmartSelectService）。"""
        return await self._smart_select_service.try_send_emoji(event, emotions, text)
