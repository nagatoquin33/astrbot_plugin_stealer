import asyncio
import random
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..util.normalization import canonicalize_path
from .meme_search_engine import MemeSearchEngine
from .meme_selection_strategy import MemeSelectionStrategy

from ..events.event_context import unwrap_event
from .meme_scope_service import MemeScopeService
from .meme_smart_select_service import MemeSmartSelectService
from .jev_selector import JevSelector
from .search_features import entry_category, normalize_category


class MemeSelector:
    """表情包选择器，负责查找、筛选和选择表情包。"""

    # 选择器常量
    MAX_RECENT_USAGE = 10  # 最近使用记录最大数量
    MIN_RECENT_USAGE = 3  # 最近使用记录最小数量

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self._selection_lock = asyncio.Lock()

        # 子服务（职责拆分）
        self._scope_service = MemeScopeService(plugin_instance)
        self._selection_strategy = MemeSelectionStrategy(plugin_instance)
        self._search_engine = MemeSearchEngine(
            plugin_instance, self._selection_strategy, self._scope_service
        )
        self._smart_select_service = MemeSmartSelectService(
            plugin_instance, self._search_engine, self._selection_strategy, self._scope_service
        )
        self._jev_selector = JevSelector(plugin_instance)

    async def select_emoji_with_jev(
        self, event: AstrMessageEvent, text: str, *, user_message: str = "",
    ) -> str | None:
        """相关度粗筛 Top 10 后由 JEV 选图；不提前更新使用记录。"""
        if not self._check_group_allowed(event):
            return None
        history = await self._jev_selector.get_history(event)
        async with self._selection_lock:
            ranked = await self._smart_select_service._rank_emoji_candidates(
                "", f"{user_message[:2000]}\n{text[:2000]}", event=event,
                deterministic=True,
            )
            idx = self._get_index()
            candidates = []
            seen = set()
            for item in ranked:
                path = item[0]
                canon = canonicalize_path(path)
                data = idx.get(path) or idx.get(canon)
                if canon in seen or not isinstance(data, dict):
                    continue
                if not Path(path).is_file() or not self._is_entry_allowed_for_event(data, event):
                    continue
                seen.add(canon)
                candidates.append((path, data))
                if len(candidates) == JevSelector.TOP_K:
                    break
        return await self._jev_selector.select(
            candidates, history=history, user_message=user_message, reply=text,
        )

    async def send_jev_selection(self, event: AstrMessageEvent, path: str, text: str) -> bool:
        """网络请求及延迟后重新核验候选，再走既有发送/计数链路。"""
        idx = self._get_index()
        data = idx.get(path) or idx.get(canonicalize_path(path))
        if not isinstance(data, dict) or not Path(path).is_file():
            return False
        if not self._is_entry_allowed_for_event(data, event):
            return False
        sent = await self.send_emoji_with_text(event, path, text)
        if sent:
            logger.debug(f"[JEV] 已发送图片: {path}")
            self._selection_strategy._update_recent_usage(entry_category(data), canonicalize_path(path))
        return sent

    def _get_index(self) -> dict[str, Any]:
        return self._search_engine.get_index()

    def _invalidate_bm25_index(self) -> None:
        """委托给 MemeSearchEngine。"""
        self._search_engine._invalidate_bm25_index()

    def _check_group_allowed(self, event: AstrMessageEvent) -> bool:
        """检查当前群组是否允许使用表情包功能。

        Args:
            event: 消息事件对象

        Returns:
            bool: True 表示允许，False 表示不允许
        """
        return self.plugin.is_send_enabled_for_event(event)

    # ===== 门面委托：MemeScopeService =====

    def _get_event_target_entry(self, event: AstrMessageEvent | None) -> str:
        """获取事件目标条目（已迁移到 MemeScopeService）。"""
        return self._scope_service._get_event_target_entry(event)

    def _is_entry_allowed_for_event(
        self, data: dict | None, event: AstrMessageEvent | None
    ) -> bool:
        """检查条目是否允许（已迁移到 MemeScopeService）。"""
        return self._scope_service._is_entry_allowed_for_event(data, event)

    def is_path_allowed_for_event(self, path: str, event: AstrMessageEvent | None) -> bool:
        """检查路径是否允许（已迁移到 MemeScopeService）。"""
        return self._scope_service.is_path_allowed_for_event(path, event)

    def find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的分类（委托给 MemeSearchEngine）。"""
        return self._search_engine.find_similar_categories(query, top_n)

    async def record_emoji_usage(self, emoji_path: str, trigger: str = "auto") -> None:
        """记录表情包使用次数。

        优先使用数据库服务进行增量更新。
        """
        if not emoji_path:
            return

        db_service = getattr(self.plugin, "db_service", None)
        if db_service is None:
            return

        # 使用数据库增量更新
        target_path = canonicalize_path(emoji_path)
        db_service.increment_usage_sync(target_path)

    def normalize_category(self, category: str) -> str:
        """归一化当前配置中的分类名。"""
        return normalize_category(self.plugin, category)

    async def select_emoji(
        self,
        category: str,
        context_text: str = "",
        event: AstrMessageEvent | None = None,
        extra_categories: list[str] | None = None,
    ) -> str | None:
        """选择表情包（智能或随机）。"""
        async with self._selection_lock:
            use_smart = self.plugin.plugin_config.smart_meme_selection
            extra = [item for item in (extra_categories or []) if item]
            primary = self.normalize_category(category) or str(category or "").lower().strip()
            candidate_categories: list[str] = []
            if extra:
                for item in [primary, *extra]:
                    mapped = self.normalize_category(item) or str(item).lower().strip()
                    if mapped and mapped not in candidate_categories:
                        candidate_categories.append(mapped)
            elif primary:
                candidate_categories = self._selection_strategy._get_candidate_categories(primary)

            if use_smart and context_text and len(context_text.strip()) > 5:
                smart_path = await self._smart_select_service._select_emoji_smart_impl(
                    primary or (candidate_categories[0] if candidate_categories else ""),
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
            entries: list[tuple[Path, dict]] = []
            idx = self._get_index()
            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue
                if entry_category(data) != category:
                    continue
                if not self._is_entry_allowed_for_event(data, event):
                    continue
                path_obj = Path(file_path)
                if path_obj.is_file():
                    entries.append((path_obj, data))

            if not entries:
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
                entries = []
                for path_obj in cat_dir.iterdir():
                    if not path_obj.is_file():
                        continue
                    if not self.is_path_allowed_for_event(str(path_obj), event):
                        continue
                    entries.append((path_obj, {}))

            if not entries:
                return None

            recent_usage = self._selection_strategy._get_recent_usage(category)
            recent_set = set(recent_usage)
            candidates = [(p, canonicalize_path(str(p)), data) for p, data in entries]

            # 过滤最近使用
            available = [(p, data) for p, canon, data in candidates if canon not in recent_set]
            if not available:
                available = [(p, data) for p, _, data in candidates]
                recent_usage = []
                recent_set = set()

            # 加权随机选择：收藏项权重 ×3
            weights = [3.0 if data.get("is_favorite") else 1.0 for _, data in available]
            total_weight = sum(weights)

            # 尝试选择一个存在的文件（最多重试3次）
            max_retries = min(3, len(available))
            for _ in range(max_retries):
                r = random.uniform(0, total_weight)
                cumulative = 0.0
                picked = None
                picked_weight = 0.0
                for (path_obj, data), weight in zip(available, weights):
                    cumulative += weight
                    if r <= cumulative:
                        picked = path_obj
                        picked_weight = weight
                        break
                if picked is None:
                    picked, picked_weight = available[-1][0], weights[-1]

                # 检查文件是否仍然存在
                if picked.exists():
                    picked_path = canonicalize_path(str(picked))

                    if picked_path in recent_set:
                        recent_usage = [p for p in recent_usage if p != picked_path]
                    recent_usage.append(picked_path)

                    max_recent = min(
                        self.MAX_RECENT_USAGE, max(self.MIN_RECENT_USAGE, len(entries) // 2)
                    )
                    if len(recent_usage) > max_recent:
                        recent_usage = recent_usage[-max_recent:]

                    self._selection_strategy._set_recent_usage(category, recent_usage)
                    return str(picked)
                else:
                    # 文件已不存在，从候选列表中移除
                    idx_to_remove = next(
                        i for i, (p, _) in enumerate(available) if p == picked
                    )
                    available.pop(idx_to_remove)
                    weights.pop(idx_to_remove)
                    total_weight -= picked_weight
                    if not available:
                        break

            return None
        except Exception as e:
            logger.error(f"随机选择表情包失败: {e}")
            return None

    # ===== 门面委托：MemeSmartSelectService =====

    async def search_images(
        self, query: str, *, limit: int = 10, idx: dict | None = None, event=None
    ):
        """搜索表情包（已迁移到 MemeSmartSelectService）。"""
        return await self._smart_select_service.search_images(
            query, limit=limit, idx=idx, event=event
        )

    async def smart_search(
        self, query: str, *, limit: int = 10, idx: dict | None = None, event=None
    ):
        """智能搜索（已迁移到 MemeSmartSelectService）。"""
        return await self._smart_select_service.smart_search(
            query, limit=limit, idx=idx, event=event
        )

    async def send_emoji_message(self, event: AstrMessageEvent, path: str):
        """发送表情包消息（已迁移到 MemeSmartSelectService）。"""
        return await self._smart_select_service.send_emoji_message(event, path)

    async def send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ) -> bool:
        """Send one emoji message in the fastest compatible format."""
        event = unwrap_event(event)
        try:
            # active_sent means an emoji was actually sent, not merely auto-claimed.
            if self.plugin._emoji_sender_engine.emoji_turn_state(event).is_active_sent():
                logger.debug("[Stealer] 已主动发送过表情包，跳过自动发送")
                return False

            if not self._check_group_allowed(event):
                return False

            send_mode = await self.send_emoji_message(event, emoji_path)
            if not send_mode:
                return False

            try:
                await self.record_emoji_usage(emoji_path, trigger="auto")
            except Exception as e:
                logger.debug(f"[Stealer] 记录表情包使用失败: {e}")
            logger.debug(f"[Stealer] 已发送表情包 ({send_mode}): {emoji_path}")
            return True

        except Exception as e:
            logger.error(f"发送表情包失败: {e}", exc_info=True)
            return False

    async def try_send_emoji(
        self,
        event: AstrMessageEvent,
        emotions: list[str],
        cleaned_text: str,
    ) -> bool:
        """尝试发送表情包。多个情绪作为先验一次召回，不再按桶逐个试。

        注意：概率判定由 Main 在调用前通过 MemeSenderEngine._resolve_with_log 完成，
        本方法只负责选图和发图。
        """
        event = unwrap_event(event)
        if not self._check_group_allowed(event):
            return False

        if self.plugin._emoji_sender_engine.emoji_turn_state(event).is_active_sent():
            logger.debug("[Stealer] 检测到已发送，跳过表情发送")
            return False

        priors = [item for item in (emotions or []) if item]
        primary = priors[0] if priors else ""
        emoji_path = await self.select_emoji(
            primary,
            cleaned_text,
            event=event,
            extra_categories=priors,
        )
        if emoji_path:
            sent = await self.send_emoji_with_text(event, emoji_path, cleaned_text)
            if sent:
                if priors:
                    logger.debug(
                        "已发送表情包：情绪先验=["
                        + ", ".join(priors)
                        + "]，按文本/图上文字/角色/BM25 综合匹配"
                    )
                else:
                    logger.debug("已发送表情包：情绪先验=无，按文本/图上文字/角色/BM25 匹配")
                return True

        logger.debug("[Stealer] 未匹配到表情包")
        return False
