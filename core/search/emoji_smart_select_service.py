"""表情包智能选择服务。"""

import os
import random
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.core.agent.run_context import ContextWrapper

from ..events.emoji_sender_engine import _send_qq_image_as_sticker

from .text_similarity import calculate_hybrid_similarity, calculate_simple_similarity, tokenize_for_bm25, _extract_words
from .embedding_service import EmbeddingService


def _unwrap_event(event: AstrMessageEvent) -> AstrMessageEvent:
    """从 ContextWrapper 中提取 AstrMessageEvent（兼容 v4.26+ 工具调用上下文）。

    v4.26+ 中 @filter.llm_tool 处理函数接收 ContextWrapper 而非直接 AstrMessageEvent，
    通过 ContextWrapper.context.event 获取内部的 AstrMessageEvent。
    参考: astrbot.core.astr_agent_context.AstrAgentContext
    """
    if isinstance(event, ContextWrapper):
        return event.context.event
    return event


class EmojiSmartSelectService:
    """负责智能选择表情包。"""

    # 从 EmojiSelector 同步的常量（供内部方法直接使用，避免通过 _selector 委托）
    SMART_FAST_PREFILTER_MIN_CANDIDATES = 48
    SMART_FAST_PREFILTER_TOP_K = 120
    SMART_FAST_PREFILTER_FUZZY_RESERVE = 24
    SMART_BM25_BONUS_WEIGHT = 0.2

    def __init__(self, plugin_instance: Any = None) -> None:
        self.plugin = plugin_instance
        self._selector = None
        self._search_engine = None

        # ── Embedding ──
        self._embedding_service = EmbeddingService(plugin_instance) if plugin_instance else None

    def __getattr__(self, name: str):
        """将缺失的属性/方法委托给 EmojiSelector。"""
        if self._selector is not None:
            return getattr(self._selector, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    async def _select_emoji_smart_impl(
        self,
        category: str,
        context_text: str,
        candidate_categories: list[str] | None = None,
        event: AstrMessageEvent | None = None,
    ) -> str | None:
        """智能选择表情包实现（内部方法）。"""
        try:
            idx = self._get_index()
            if not idx:
                return None

            allowed_categories = set(candidate_categories or [category])
            candidates = []
            low_score_candidates = []
            context_lower = context_text.lower()
            context_words = _extract_words(context_text)
            query_tokens = tokenize_for_bm25(context_text)
            query_token_set = set(query_tokens)

            # 热路径优化：先按分类与作用域过滤，再做轻量词法预筛，
            # 避免每次自动发送都触发全量 BM25 索引加载/查询。
            scoped_entries: list[
                tuple[str, dict[str, Any], str, list[str], list[str], tuple[str, ...], float]
            ] = []

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue

                entry_category = self._get_category_from_data(data)
                if entry_category not in allowed_categories:
                    continue
                if not self._is_entry_allowed_for_event(data, event):
                    continue

                desc_raw = str(data.get("desc", "") or "")
                tags = self._parse_tags(data.get("tags", []))
                scenes = self._parse_tags(data.get("scenes", []))
                entry_text = " ".join([entry_category, desc_raw] + tags + scenes)
                entry_tokens = tokenize_for_bm25(entry_text)

                fast_score = 0.0
                if query_token_set and entry_tokens:
                    entry_token_set = set(entry_tokens)
                    overlap = query_token_set & entry_token_set
                    if overlap:
                        fast_score = len(overlap) / max(1, len(query_token_set))

                scoped_entries.append(
                    (file_path, data, entry_category, tags, scenes, entry_tokens, fast_score)
                )

            if not scoped_entries:
                return None

            # 仅在候选较多时启用轻量预筛，兼顾召回与性能。
            if len(scoped_entries) > self.SMART_FAST_PREFILTER_MIN_CANDIDATES and query_token_set:
                lexical_matches = [item for item in scoped_entries if item[6] > 0]
                fuzzy_only_matches = [item for item in scoped_entries if item[6] <= 0]

                if len(lexical_matches) > self.SMART_FAST_PREFILTER_TOP_K:
                    lexical_matches.sort(key=lambda item: item[6], reverse=True)
                    prefiltered_entries = lexical_matches[: self.SMART_FAST_PREFILTER_TOP_K]

                    if fuzzy_only_matches:
                        # 使用更轻量的相似度算法做模糊候选初筛
                        fuzzy_only_matches.sort(
                            key=lambda item: calculate_simple_similarity(
                                context_text,
                                " ".join(
                                    [
                                        item[2],
                                        str(item[1].get("desc", "") or ""),
                                        *item[3],
                                        *item[4],
                                    ]
                                ),
                            ),
                            reverse=True,
                        )
                        prefiltered_entries.extend(
                            fuzzy_only_matches[: self.SMART_FAST_PREFILTER_FUZZY_RESERVE]
                        )
                else:
                    prefiltered_entries = scoped_entries
            else:
                prefiltered_entries = scoped_entries

            # 嵌入检索增强：预计算 top-K 语义相似 path → 为后续打分提供 bonus
            embedding_paths: dict[str, float] = {}
            emb_ready = self._is_embedding_ready()
            if not emb_ready:
                logger.debug("[Embedding] auto-emoji: 嵌入未就绪，跳过语义增强")
            elif not context_text.strip():
                pass  # 空文本跳过
            else:
                try:
                    emb_results = await self._embedding_service.search(context_text, k=80)
                    for path, score in emb_results:
                        canonical = self._canon_path(path)
                        embedding_paths[canonical] = score
                    if embedding_paths:
                        logger.info(
                            f"[Embedding] 语义增强: context='{context_text[:50]}', topk={len(embedding_paths)}"
                        )
                    else:
                        logger.debug("[Embedding] auto-emoji: 嵌入搜索返回空结果（向量库可能为空）")
                except Exception as e:
                    logger.warning(f"[Embedding] auto-emoji 语义增强异常: {e}")

            # 智能选择时若已积累足够高分候选，可提前终止打分
            SMART_EARLY_STOP_COUNT = 5
            SMART_EARLY_STOP_THRESHOLD = 0.7

            for file_path, data, entry_category, tags, scenes, _, fast_score in prefiltered_entries:
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
                bm25_bonus = fast_score * self.SMART_BM25_BONUS_WEIGHT
                favorite_bonus = 0.3 if data.get("is_favorite") else 0.0
                # 嵌入语义 bonus：score 已是 [0,1] 余弦相似度
                embedding_bonus = embedding_paths.get(
                    self._canon_path(file_path), 0.0
                ) * 0.20
                base_score = (
                    desc_score * 0.35
                    + tag_score * 0.25
                    + scene_score * 0.2
                    + category_bonus
                    + use_count_bonus
                    + bm25_bonus
                    + favorite_bonus
                    + embedding_bonus
                )

                if base_score < 0.15:
                    if desc_score > 0.1:
                        history_penalty = self._calculate_recent_penalty(
                            entry_category, self._canon_path(file_path)
                        )
                        adjusted_score = max(0.0, desc_score - history_penalty)
                        if adjusted_score > 0.05:
                            low_score_candidates.append(
                                (
                                    file_path,
                                    adjusted_score,
                                    desc_score,
                                    0.0,
                                    0.0,
                                    entry_category,
                                )
                            )
                    continue

                diversity_bonus = random.uniform(0, 0.15)
                canon_path = self._canon_path(file_path)
                history_penalty = self._calculate_recent_penalty(entry_category, canon_path)

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

                # 提前终止：若已有足够高分候选，跳过剩余低相关条目
                high_quality = [c for c in candidates if c[1] >= SMART_EARLY_STOP_THRESHOLD]
                if len(high_quality) >= SMART_EARLY_STOP_COUNT:
                    break

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
                f"[智能选择] 分类={category}, 候选数={len(candidates)}, "
                f"结果={result[5]}, 分数={result[1]:.2f} (desc={result[2]:.2f}, tag={result[3]:.2f}, scene={result[4]:.2f})"
            )
            return result[0]

        except Exception as e:
            logger.error(f"智能选择失败: {e}")
            return None

    @staticmethod
    def _parse_tags(raw_tags: Any) -> list[str]:
        """安全解析 tags 字段，兼容字符串和列表类型。"""
        if isinstance(raw_tags, str):
            return [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        if isinstance(raw_tags, list):
            return [str(t).lower() for t in raw_tags if t]
        return []

    # ═══════════════════════════════════════════════════
    #  嵌入检索
    # ═══════════════════════════════════════════════════

    def _is_embedding_ready(self) -> bool:
        """嵌入检索是否就绪。"""
        if not self._embedding_service:
            return False
        return self._embedding_service.is_available()

    def _invalidate_embedding_index(self) -> None:
        """标记嵌入索引为过期（新增/删除 emoji 时调用）。"""
        if self._embedding_service:
            self._embedding_service.invalidate_cache()

    async def search_images(
        self,
        query: str,
        limit: int = 1,
        idx: dict | None = None,
        event: AstrMessageEvent | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """根据查询词搜索图片（嵌入优先 → BM25 降级）。"""
        try:
            # ── 嵌入检索路径 ──
            if self._is_embedding_ready():
                embedding_results = await self._embedding_service.search(query, limit * 5)
                if embedding_results:
                    if idx is None:
                        idx = self._get_index()

                    recently_used_paths: set[str] = set()
                    for cat_paths in self._recent_usage.values():
                        recently_used_paths.update(cat_paths)

                    results: list[tuple[str, str, str, str]] = []
                    seen_paths: set[str] = set()
                    for file_path, _cos_sim in embedding_results:
                        if file_path in seen_paths:
                            continue
                        if file_path in recently_used_paths:
                            continue
                        if not self._is_entry_allowed_for_event(idx.get(file_path) if idx else None, event):
                            continue
                        seen_paths.add(file_path)
                        data = idx.get(file_path, {}) if idx else {}
                        desc = str(data.get("desc", "") or "")
                        category = self._get_category_from_data(data)
                        tags = self._parse_tags(data.get("tags", []))
                        tags_str = ", ".join(tags)
                        results.append((file_path, desc, category, tags_str))
                        if len(results) >= limit:
                            break
                    if results:
                        logger.debug(f"[Embedding] 嵌入检索命中 {len(results)} 条, query='{query}'")
                        return results
                    # 嵌入结果都被过滤掉了，降级 BM25
                    logger.debug("[Embedding] 嵌入结果均被过滤，降级 BM25")
                else:
                    logger.debug("[Embedding] 嵌入检索无结果，降级 BM25")
            else:
                logger.debug("[Embedding] 嵌入检索不可用，走 BM25")

            # ── BM25 降级路径 ──
            # 安全网：即便写入路径遗漏了 _invalidate_bm25_index，
            # 也会在签名与当前语料不一致时强制重建，避免使用过期的 BM25 索引。
            need_rebuild = (
                self._search_engine._bm25_dirty
                or self._search_engine._bm25_index is None
            )
            if not need_rebuild:
                try:
                    current_sig = self._search_engine._compute_bm25_signature(
                        idx if idx is not None else self._selector._get_index(),
                        prefer_db_signature=True,
                    )
                    if current_sig and current_sig != self._search_engine._bm25_signature:
                        need_rebuild = True
                except Exception:
                    pass
            if need_rebuild:
                await self._search_engine._build_bm25_index(idx)

            if self._search_engine._bm25_index is None or not self._search_engine._bm25_doc_paths:
                return await self._search_images_fallback(query, limit, idx, event)

            query_tokens = tokenize_for_bm25(query)
            if not query_tokens:
                return await self._search_images_fallback(query, limit, idx, event)

            bm25_results = self._search_engine._bm25_index.get_top_k(query_tokens, k=limit * 5)
            logger.debug(
                f"[BM25] 查询='{query}', tokens={query_tokens}, top_doc_scores={bm25_results[:10]}"
            )

            if not idx:
                idx = self._get_index()

            recently_used_paths: set[str] = set()
            for cat_paths in self._recent_usage.values():
                recently_used_paths.update(cat_paths)

            results: list[tuple[str, str, str, str]] = []
            seen_paths: set[str] = set()

            for doc_idx, bm25_score in bm25_results:
                if doc_idx >= len(self._search_engine._bm25_doc_paths):
                    continue
                file_path = self._search_engine._bm25_doc_paths[doc_idx]
                if file_path in seen_paths:
                    continue
                if file_path in recently_used_paths:
                    continue
                if not self._is_entry_allowed_for_event(idx.get(file_path) if idx else None, event):
                    continue
                seen_paths.add(file_path)

                data = idx.get(file_path, {}) if idx else {}
                desc = str(data.get("desc", "") or "")
                category = self._get_category_from_data(data)
                tags = self._parse_tags(data.get("tags", []))
                tags_str = ", ".join(tags)
                results.append((file_path, desc, category, tags_str))
                if len(results) >= limit:
                    break

            if results:
                return results

            return await self._search_images_fallback(query, limit, idx, event)

        except Exception as e:
            logger.error(f"BM25 搜索图片失败: {e}")
            return await self._search_images_fallback(query, limit, idx, event)

    async def _search_images_fallback(
        self,
        query: str,
        limit: int = 1,
        idx: dict | None = None,
        event: AstrMessageEvent | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """委托给 EmojiSearchEngine。"""
        return await self._search_engine._search_images_fallback(query, limit, idx, event)

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
        """委托给 EmojiSearchEngine。"""
        return self._search_engine._score_entry(
            query_lower, query_tokens, category, desc, tags, max_str_len, tag_words
        )

    async def smart_search(
        self,
        query: str,
        limit: int = 5,
        idx: dict | None = None,
        event: AstrMessageEvent | None = None,
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
        results = await self.search_images(query, limit=limit, idx=idx, event=event)
        if results:
            return results

        # 2) 关键词映射
        cfg = self.plugin.plugin_config
        keyword_map = cfg.get_keyword_map() if cfg else {}
        if query in keyword_map:
            mapped_category = keyword_map[query]
            results = await self.search_images(mapped_category, limit=limit, idx=idx, event=event)
            if results:
                return results

        # 3) 模糊匹配到分类
        best_match = self._find_best_category_match(query, threshold=0.4)
        if best_match:
            results = await self.search_images(best_match, limit=limit, idx=idx, event=event)

        return results

    def _find_best_category_match(self, query: str, threshold: float = 0.4) -> str | None:
        """委托给 EmojiSearchEngine。"""
        return self._search_engine._find_best_category_match(query, threshold)

    def find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """委托给 EmojiSearchEngine。"""
        return self._search_engine.find_similar_categories(query, top_n)

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

    async def _try_send_telegram_sticker(self, event: AstrMessageEvent, emoji_path: str) -> bool:
        """Telegram 平台优先尝试以贴纸发送，失败返回 False 供上层回退。"""
        try:
            platform_name = str(event.get_platform_name() or "").strip().lower()
        except Exception:
            platform_name = ""

        if platform_name != "telegram":
            return False

        client = getattr(event, "client", None)
        if client is None or not hasattr(client, "send_sticker"):
            return False

        if not emoji_path or not os.path.exists(emoji_path):
            return False

        chat_id = ""
        try:
            chat_id = str(event.get_group_id() or "").strip()
        except Exception:
            chat_id = ""
        if not chat_id:
            try:
                chat_id = str(event.get_sender_id() or "").strip()
            except Exception:
                chat_id = ""
        if not chat_id:
            return False

        message_thread_id = None
        if "#" in chat_id:
            chat_id, thread_part = chat_id.split("#", 1)
            thread_part = str(thread_part or "").strip()
            if thread_part:
                try:
                    message_thread_id = int(thread_part)
                except Exception:
                    message_thread_id = thread_part

        payload = {"chat_id": chat_id, "sticker": emoji_path}
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id

        try:
            await client.send_sticker(**payload)
            logger.debug(f"[Stealer] Telegram 已按贴纸发送: {emoji_path}")
            return True
        except Exception as e:
            logger.debug(f"[Stealer] Telegram 贴纸发送失败，将回退图片发送: {e}")
            return False

    def _should_send_file_directly(self, event: AstrMessageEvent) -> bool:
        """Prefer local file sends when GIF coercion is disabled and the adapter is friendly."""
        if getattr(self.plugin, "send_meme_as_gif", True):
            return False

        try:
            platform_name = str(event.get_platform_name() or "").strip().lower()
        except Exception:
            platform_name = ""

        return platform_name != "aiocqhttp"

    async def _send_emoji_file_directly(self, event: AstrMessageEvent, emoji_path: str) -> bool:
        """Attempt the lowest-overhead file send path and let callers fall back on failure."""
        if not self._should_send_file_directly(event):
            return False
        if not emoji_path or not os.path.exists(emoji_path):
            return False

        try:
            make_result = getattr(event, "make_result", None)
            if not callable(make_result):
                return False

            result = make_result()
            if result is None or not hasattr(result, "file_image"):
                return False

            payload = result.file_image(emoji_path)
            if payload is None:
                payload = result
            if hasattr(payload, "stop_event"):
                payload = payload.stop_event()

            await event.send(payload)
            return True
        except Exception as e:
            logger.debug(f"[Stealer] file_image 发送失败，回退 base64: {e}")
            return False

    async def _append_emoji_to_result(
        self, event: AstrMessageEvent, result: Any, emoji_path: str
    ) -> bool:
        """Append an emoji to an existing result, preferring file sends when safe."""
        if self._should_send_file_directly(event) and hasattr(result, "file_image"):
            try:
                result.file_image(emoji_path)
                return True
            except Exception as e:
                logger.debug(f"[Stealer] result.file_image 失败，回退 base64: {e}")

        b64 = await self._encode_emoji(emoji_path)
        if not b64:
            return False
        result.base64_image(b64)
        return True

    async def send_emoji_message(self, event: AstrMessageEvent, emoji_path: str) -> str | None:
        """Send a single emoji using the fastest compatible path."""
        event = _unwrap_event(event)
        if await self._try_send_telegram_sticker(event, emoji_path):
            return "telegram_sticker"

        if await _send_qq_image_as_sticker(event, emoji_path):
            return "qq_sticker"

        if await self._send_emoji_file_directly(event, emoji_path):
            return "file_image"

        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Image as ImageComponent

        b64 = await self._encode_emoji(emoji_path)
        if not b64:
            return None

        await event.send(MessageChain([ImageComponent.fromBase64(b64)]))
        return "base64_image"

    async def send_emoji_with_text(
        self, event: AstrMessageEvent, emoji_path: str, cleaned_text: str
    ) -> bool:
        """Send one emoji message in the fastest compatible format."""
        event = _unwrap_event(event)
        try:
            # active_sent means an emoji was actually sent, not merely auto-claimed.
            if self.plugin._emoji_turn_state(event).is_active_sent():
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

    async def send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ) -> None:
        """Append explicitly selected emojis to the current result."""
        event = _unwrap_event(event)
        from astrbot.api.message_components import Plain

        try:
            result = event.get_result()
            new_result = event.make_result().set_result_content_type(result.result_content_type)

            for comp in result.chain:
                if not isinstance(comp, Plain):
                    new_result.chain.append(comp)

            if cleaned_text.strip():
                new_result.message(cleaned_text.strip())

            sent_paths = []
            for path_str in emoji_paths:
                if await self._try_send_telegram_sticker(event, path_str):
                    sent_paths.append(path_str)
                    continue
                if await self._append_emoji_to_result(event, new_result, path_str):
                    sent_paths.append(path_str)

            event.set_result(new_result)
            for path_str in sent_paths:
                await self.record_emoji_usage(path_str, trigger="explicit")
        except Exception as e:
            logger.error(f"发送显式表情包失败: {e}", exc_info=True)

    async def try_send_emoji(
        self,
        event: AstrMessageEvent,
        emotions: list[str],
        cleaned_text: str,
    ) -> bool:
        """尝试发送表情包，遍历 emotions 列表直到第一个匹配到的表情包。

        注意：概率判定由 Main 在调用前通过 _resolve_auto_emoji_turn_permission 完成，
        本方法只负责选图和发图。
        """
        event = _unwrap_event(event)
        if not self._check_group_allowed(event):
            return False

        # active_sent means an emoji was actually sent, not merely auto-claimed.
        if self.plugin._emoji_turn_state(event).is_active_sent():
            logger.debug("[Stealer] 检测到已发送，跳过表情发送")
            return False

        # 遍历情绪列表，第一个能选到表情包的就发送
        for emotion in emotions:
            emoji_path = await self.plugin.emoji_selector.select_emoji(emotion, cleaned_text, event=event)
            if emoji_path:
                sent = await self.send_emoji_with_text(event, emoji_path, cleaned_text)
                if not sent:
                    continue
                logger.debug(f"已发送表情包 (情绪={emotion})")
                return True

        logger.debug("[Stealer] 所有情绪均未匹配到表情包")
        return False
