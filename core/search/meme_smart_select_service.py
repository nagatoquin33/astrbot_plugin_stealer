"""表情包智能选择服务。"""

import asyncio
import os
import random
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..processing.semantic_schema import build_meme_search_text
from ..events.emoji_delivery import send_qq_image_as_sticker
from ..events.event_context import get_event_platform_name, unwrap_event

from .text_similarity import calculate_hybrid_similarity, tokenize_for_bm25, _extract_words, _has_negation_prefix
from .embedding_service import EmbeddingService
from ..util.normalization import canonicalize_path, normalize_character_key
from .search_features import entry_category, parse_tags, prepare_entry_text_features

class MemeSmartSelectService:
    """负责智能选择表情包。"""

    SMART_BM25_BONUS_WEIGHT = 0.2
    SMART_RECALL_K = 48
    SMART_OVERLAY_RECALL_LIMIT = 16

    def __init__(
        self, plugin_instance: Any, search_engine, selection_strategy, scope_service
    ) -> None:
        self.plugin = plugin_instance
        self._search_engine = search_engine
        self._selection_strategy = selection_strategy
        self._scope_service = scope_service
        self._embedding_service = EmbeddingService(plugin_instance) if plugin_instance else None

    def _text_recall_paths(self, idx, context_text, event, field, limit):
        """Shared literal matching with the same scope and phrase boundaries."""
        context = (context_text or "").lower()
        if len(context) < 2:
            return []
        hits = []
        for path, data in idx.items():
            if not isinstance(data, dict) or not self._scope_service._is_entry_allowed_for_event(data, event):
                continue
            values = [str(data.get(field) or "")] if field == "overlay_text" else parse_tags(data.get(field, []))
            for value in values:
                text = value.strip().lower()
                if len(text) < 2:
                    continue
                if text in context or any(
                    len(piece) >= 2 and piece in context
                    for piece in re.split(r"[\s,，。！？!?、~～]+", text)
                ):
                    hits.append(path)
                    break
            if len(hits) >= limit:
                break
        return hits

    def _overlay_recall_paths(self, idx, context_text, event, limit=16):
        return self._text_recall_paths(idx, context_text, event, "overlay_text", limit)

    def _scene_recall_paths(self, idx, context_text, event, limit=16):
        return self._text_recall_paths(idx, context_text, event, "scenes", limit)

    def _character_names(self, data: dict[str, Any]) -> list[str]:
        key = str(data.get("character") or "").strip()
        if not key:
            return []
        names = [key.lower()]
        cfg = getattr(self.plugin, "plugin_config", None)
        info = (getattr(cfg, "character_info", None) or {}).get(key) if cfg else None
        if isinstance(info, dict):
            name = str(info.get("name") or "").strip()
            if name:
                names.append(name.lower())
        return names

    def _character_match_score(self, query_lower: str, data: dict[str, Any]) -> float:
        if not query_lower:
            return 0.0
        for name in self._character_names(data):
            if len(name) >= 2 and name in query_lower:
                return 1.0
        return 0.0

    def _character_recall_paths(
        self,
        idx: dict[str, Any],
        context_text: str,
        event: AstrMessageEvent | None,
        limit: int = 16,
    ) -> list[str]:
        ctx = (context_text or "").lower()
        if len(ctx) < 2:
            return []
        hits: list[str] = []
        for file_path, data in idx.items():
            if not isinstance(data, dict):
                continue
            if not self._scope_service._is_entry_allowed_for_event(data, event):
                continue
            if self._character_match_score(ctx, data) >= 1.0:
                hits.append(file_path)
                if len(hits) >= limit:
                    break
        return hits

    async def _recall_candidate_paths(
        self,
        idx: dict[str, Any],
        context_text: str,
        event: AstrMessageEvent | None,
        prior_categories: set[str],
        *, require_files: bool = False,
    ) -> tuple[list[str], dict[str, float]]:
        """各路独立召回后用 reciprocal rank fusion 合并，分类只作兜底。"""
        recalled: dict[str, str] = {}
        fusion_scores: dict[str, float] = {}
        embedding_paths: dict[str, float] = {}

        def _add(paths: list[str]) -> None:
            seen: set[str] = set()
            for rank, path in enumerate(paths, 1):
                canon = canonicalize_path(path)
                if not path or canon in seen or (require_files and not os.path.isfile(path)):
                    continue
                data = idx.get(path) or idx.get(canon)
                if not isinstance(data, dict):
                    continue
                if not self._scope_service._is_entry_allowed_for_event(data, event):
                    continue
                seen.add(canon)
                recalled.setdefault(canon, path)
                fusion_scores[canon] = fusion_scores.get(canon, 0.0) + 1.0 / (60 + rank)

        _add(self._overlay_recall_paths(idx, context_text, event, self.SMART_OVERLAY_RECALL_LIMIT))
        _add(self._character_recall_paths(idx, context_text, event, self.SMART_OVERLAY_RECALL_LIMIT))
        _add(self._scene_recall_paths(idx, context_text, event, self.SMART_OVERLAY_RECALL_LIMIT))

        if self._is_embedding_ready() and context_text.strip():
            try:
                emb_results = await self._embedding_service.search(
                    context_text, k=self.SMART_RECALL_K
                )
                emb_paths: list[str] = []
                for path, score in emb_results:
                    if score <= 0:
                        continue
                    embedding_paths[canonicalize_path(path)] = score
                    emb_paths.append(path)
                _add(emb_paths)
                if embedding_paths:
                    logger.info(
                        f"[Embedding] 语义召回: context='{context_text[:50]}', topk={len(embedding_paths)}"
                    )
            except Exception as e:
                logger.warning(f"[Embedding] auto-emoji 语义召回异常: {e}")
        elif not self._is_embedding_ready():
            logger.debug("[Embedding] auto-emoji: 嵌入未就绪，改走 BM25/分类兜底")

        if context_text.strip():
            try:
                signature = getattr(self._search_engine, "_compute_bm25_signature", None)
                stale = callable(signature) and signature(idx) != self._search_engine._bm25_signature
                if stale or self._search_engine._bm25_dirty or self._search_engine._bm25_index is None:
                    engine_plugin = getattr(self._search_engine, "plugin", None)
                    await self._search_engine._build_bm25_index(
                        None if getattr(engine_plugin, "db_service", None) else idx,
                    )
                tokens = tokenize_for_bm25(context_text)
                if tokens and self._search_engine._bm25_index is not None:
                    top = self._search_engine._bm25_index.get_top_k(
                        list(tokens), k=self.SMART_RECALL_K
                    )
                    bm25_paths = []
                    for doc_idx, score in top:
                        if score > 0 and 0 <= doc_idx < len(self._search_engine._bm25_doc_paths):
                            bm25_paths.append(self._search_engine._bm25_doc_paths[doc_idx])
                    _add(bm25_paths)
            except Exception as e:
                logger.debug(f"[BM25] auto-emoji 召回失败: {e}")

        if not recalled and prior_categories:
            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue
                if entry_category(data) not in prior_categories:
                    continue
                _add([file_path])
                if len(recalled) >= self.SMART_RECALL_K:
                    break

        active_character = self._scope_service.active_character(event)
        if active_character:
            matching = [path for path, data in idx.items()
                        if isinstance(data, dict)
                        and normalize_character_key(data.get("character")) == active_character
                        and self._scope_service._is_entry_allowed_for_event(data, event)
                        and (not require_files or os.path.isfile(path))]
            query_tokens = set(tokenize_for_bm25(context_text))
            if query_tokens:
                def matching_tokens(path):
                    data = idx[path]
                    text = build_meme_search_text(data, bm25=True)
                    return len(query_tokens.intersection(tokenize_for_bm25(text)))
                matching.sort(key=matching_tokens, reverse=True)
            _add(matching[:self.SMART_RECALL_K])

        ordered = sorted(recalled, key=lambda path: -fusion_scores[path])
        if active_character:
            # 保留独立的角色候选额度，避免通用库挤掉角色候选。
            character_paths = [path for path in ordered if normalize_character_key((idx.get(recalled[path]) or {}).get("character")) == active_character]
            character_set = set(character_paths)
            general_paths = [path for path in ordered if path not in character_set]
            ordered = character_paths[:self.SMART_RECALL_K] + general_paths[:self.SMART_RECALL_K]
        else:
            ordered = ordered[:self.SMART_RECALL_K]
        return [recalled[path] for path in ordered], embedding_paths

    async def _select_emoji_smart_impl(
        self, category: str, context_text: str,
        candidate_categories: list[str] | None = None,
        event: AstrMessageEvent | None = None,
        *, emotion_weights: dict[str, float] | None = None,
    ) -> str | None:
        """按相关度选图，只有接近同分的候选参与随机选择。"""
        idx = await asyncio.to_thread(self._search_engine.get_index)
        candidates = await self._rank_emoji_candidates(
            category, context_text, candidate_categories, event,
            idx=idx, emotion_weights=emotion_weights,
            require_files=True if self._scope_service.active_character(event) else None,
        )
        candidates = self._prefer_character_candidates(candidates, idx, event)
        if not candidates:
            return None
        try:
            top_candidates = [item for item in candidates[:3] if item[1] >= candidates[0][1] - 0.05]
            if len(top_candidates) > 1:
                weights = [item[1] for item in top_candidates]
                total_weight = sum(weights)
                if total_weight > 0:
                    selected = random.choices(top_candidates, weights=weights, k=1)[0]
                    self._selection_strategy._update_recent_usage(selected[5], selected[0])
                    return selected[0]

            result = candidates[0]
            self._selection_strategy._update_recent_usage(result[5], result[0])
            logger.debug(
                f"[智能选择] 分类={category}, 候选数={len(candidates)}, "
                f"结果={result[5]}, 分数={result[1]:.2f} (desc={result[2]:.2f}, tag={result[3]:.2f}, scene={result[4]:.2f})"
            )
            return result[0]

        except Exception as e:
            logger.error(f"智能选择失败: {e}")
            return None

    async def _rank_emoji_candidates(
        self,
        category: str,
        context_text: str,
        candidate_categories: list[str] | None = None,
        event: AstrMessageEvent | None = None,
        *,
        deterministic: bool = False,
        idx: dict | None = None,
        require_files: bool | None = None,
        penalize_recent: bool = True,
        emotion_weights: dict[str, float] | None = None,
    ) -> list[tuple]:
        """召回并评分；JEV 模式过滤文件并禁用随机加分和提前停止。"""
        try:
            if idx is None:
                idx = await asyncio.to_thread(self._search_engine.get_index)
            if not idx:
                return []

            allowed_categories = {
                item for item in (candidate_categories or [category]) if item
            }
            candidates = []
            context_lower = context_text.lower()
            context_words = _extract_words(context_text)
            query_tokens = tokenize_for_bm25(context_text)
            query_token_set = set(query_tokens)

            recalled_paths, embedding_paths = await self._recall_candidate_paths(
                idx, context_text, event, allowed_categories | set(emotion_weights or {}),
                require_files=deterministic if require_files is None else require_files,
            )
            if not recalled_paths:
                return []

            for file_path in recalled_paths:
                data = idx.get(file_path) or idx.get(canonicalize_path(file_path))
                if not isinstance(data, dict):
                    continue
                item_category = entry_category(data)
                tags = parse_tags(data.get("tags", []))
                scenes = parse_tags(data.get("scenes", []))
                overlay = str(data.get("overlay_text") or "")
                if overlay and (_has_negation_prefix(context_lower, overlay.lower()) or any(
                    _has_negation_prefix(overlay.lower(), token) for token in context_lower.split()
                )):
                    continue
                entry_text = " ".join(
                    [overlay, item_category, str(data.get("desc", "") or "")] + tags + scenes
                )
                entry_tokens = tokenize_for_bm25(entry_text)
                fast_score = 0.0
                if query_token_set and entry_tokens:
                    overlap = query_token_set & set(entry_tokens)
                    if overlap:
                        fast_score = len(overlap) / max(1, len(query_token_set))
                desc, tag_words, scene_words, _, _ = prepare_entry_text_features(
                    item_category,
                    str(data.get("desc", "")),
                    tuple(tags),
                    tuple(scenes),
                    str(data.get("overlay_text") or ""),
                    str(data.get("character") or ""),
                )
                desc_score = calculate_hybrid_similarity(context_text, desc)
                if desc_score < 0.25:
                    desc_words = _extract_words(desc)
                    overlap = context_words & desc_words
                    bigram_hits = sum(1 for w in overlap if len(w) >= 2)
                    boost = bigram_hits * 0.25
                    if boost > 0:
                        desc_score = max(desc_score, min(1.0, boost))

                tag_score = 0.0
                if tags:
                    matched_tags = sum(1 for tag in tags if tag in context_lower)
                    tag_score = min(1.0, matched_tags / max(len(tags), 1))
                    if any(len(word) >= 2 for word in context_words & tag_words):
                        tag_score = min(1.0, tag_score + 0.3)

                scene_score = 0.0
                if scenes:
                    matched_scenes = sum(1 for scene in scenes if scene in context_lower)
                    scene_score = min(1.0, matched_scenes / max(len(scenes), 1))
                    scene_phrases = {word for word in scene_words if len(word) >= 2}
                    scene_hits = scene_phrases & context_words
                    if scene_hits:
                        scene_score = max(scene_score, 0.35 + 0.65 * len(scene_hits) / len(scene_phrases))

                overlay = str(data.get("overlay_text") or "").strip().lower()
                overlay_score = 0.0
                if overlay:
                    if overlay in context_lower:
                        overlay_score = 1.0
                    else:
                        pieces = [
                            piece
                            for piece in re.split(r"[\s,，。！？!?、~～]+", overlay)
                            if len(piece) >= 2
                        ]
                        hits = sum(1 for piece in pieces if piece in context_lower)
                        if hits:
                            overlay_score = min(1.0, hits / max(len(pieces), 1) + 0.35)

                entry_emotions = set(parse_tags(data.get("emotions", [])))
                if not entry_emotions and item_category:
                    entry_emotions = {item_category}
                if item_category in allowed_categories or (entry_emotions & allowed_categories):
                    category_bonus = 0.04
                else:
                    category_bonus = 0.0
                use_count_bonus = min(0.02, int(data.get("use_count", 0) or 0) * 0.002)
                bm25_bonus = fast_score * self.SMART_BM25_BONUS_WEIGHT
                favorite_bonus = 0.04 if data.get("is_favorite") else 0.0
                character_score = self._character_match_score(context_lower, data)
                embedding_bonus = embedding_paths.get(
                    canonicalize_path(file_path), 0.0
                ) * 0.25
                base_score = (
                    overlay_score * 0.28
                    + desc_score * 0.18
                    + tag_score * 0.05
                    + scene_score * 0.25
                    + character_score * 0.24
                    + bm25_bonus
                    + embedding_bonus
                )

                if base_score < 0.08:
                    continue

                punctuation_bonus = max(
                    ((emotion_weights or {}).get(emotion, 0.0)
                     for emotion in entry_emotions | {item_category}), default=0.0,
                )

                canon_path = canonicalize_path(file_path)
                history_penalty = self._selection_strategy._calculate_recent_penalty(item_category, canon_path) if penalize_recent else 0.0

                active_character = self._scope_service.active_character(event)
                persona_bonus = 0.08 if active_character and normalize_character_key(data.get("character")) == active_character else 0.0
                final_score = max(
                    0.0, base_score + category_bonus + use_count_bonus
                    + favorite_bonus + punctuation_bonus + persona_bonus - history_penalty,
                )
                if final_score > 0.08:
                    candidates.append(
                        (
                            file_path,
                            final_score,
                            desc_score,
                            tag_score,
                            scene_score,
                            item_category,
                        )
                    )

            if not candidates:
                return []

            candidates.sort(key=lambda item: item[1], reverse=True)
            return candidates
        except Exception as e:
            logger.error(f"智能候选评分失败: {e}")
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

    def _prefer_character_candidates(self, candidates, idx, event):
        """专属候选通过相关性筛选时优先使用，否则保留通用候选。"""
        character = self._scope_service.active_character(event)
        if not character:
            return candidates
        preferred = [item for item in candidates if normalize_character_key(
            (idx.get(item[0]) or idx.get(canonicalize_path(item[0])) or {}).get("character")
        ) == character]
        return preferred or candidates

    async def search_images(self, query, limit=1, idx=None, event=None):
        """Manual search uses the same multi-channel relevance ranking as auto selection."""
        query = str(query or "").strip()
        if not query or limit <= 0:
            return []
        if idx is None:
            idx = await asyncio.to_thread(self._search_engine.get_index)
        ranked = await self._rank_emoji_candidates(
            "", query, event=event, deterministic=True, idx=idx,
            require_files=False, penalize_recent=False,
        )
        ranked = self._prefer_character_candidates(ranked, idx, event)
        results = []
        for path, *_score in ranked[:limit]:
            data = idx.get(path) or idx.get(canonicalize_path(path))
            if isinstance(data, dict):
                results.append((path, str(data.get("desc") or ""), entry_category(data),
                                ", ".join(parse_tags(data.get("tags", [])))))
        return results

    async def smart_search(
        self,
        query: str,
        limit: int = 5,
        idx: dict | None = None,
        event: AstrMessageEvent | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """Expand applicable emotion aliases while preserving the full request."""
        query = str(query or "").strip()
        if not query:
            return []

        cfg = self.plugin.plugin_config
        keyword_map_getter = getattr(cfg, "get_keyword_map", None)
        keyword_map = keyword_map_getter() if callable(keyword_map_getter) else {}
        query_folded = query.casefold()
        mapped_categories: list[str] = []
        if isinstance(keyword_map, dict):
            # 长词优先；单字和英文别名只做精确匹配，避免“可爱”误命中“爱”、
            # “emotion”误命中“emo”。中文多字别名允许出现在自然短句中。
            aliases = sorted(keyword_map.items(), key=lambda item: len(str(item[0])), reverse=True)
            for raw_alias, raw_category in aliases:
                alias = str(raw_alias or "").strip().casefold()
                category = str(raw_category or "").strip()
                if not alias or not category:
                    continue
                exact_only = len(alias) <= 1 or alias.isascii()
                matched = (query_folded == alias if exact_only else alias in query_folded) and not _has_negation_prefix(query_folded, alias)
                if matched and category not in mapped_categories:
                    mapped_categories.append(category)

        # 将分类 key 加入原查询，使只有英文分类、缺少中文描述的旧索引也能被召回。
        expanded_query = " ".join([query, *mapped_categories])
        results = await self.search_images(
            expanded_query,
            limit=limit,
            idx=idx,
            event=event,
        )
        return results

    async def _encode_emoji(self, emoji_path: str) -> str | None:
        """将表情包文件编码为 base64，失败返回 None。"""
        if not emoji_path or not isinstance(emoji_path, str):
            logger.warning(f"[表情包编码] 无效的文件路径: {emoji_path!r}")
            return None
        if not os.path.exists(emoji_path):
            logger.warning(f"表情包文件不存在: {emoji_path}")
            return None
        renderer = self.plugin.image_render_service
        if not renderer:
            logger.warning("[表情包编码] image_render_service 未初始化")
            return None
        try:
            return await renderer.file_to_gif_base64(emoji_path)
        except Exception as e:
            logger.error(f"编码表情包失败: {emoji_path}, {e}")
            return None

    async def _try_send_telegram_sticker(self, event: AstrMessageEvent, emoji_path: str) -> bool:
        """Telegram 平台优先尝试以贴纸发送，失败返回 False 供上层回退。"""
        if get_event_platform_name(event) != "telegram":
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
        if self.plugin.plugin_config.send_meme_as_gif:
            return False

        return get_event_platform_name(event) != "aiocqhttp"

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
        event = unwrap_event(event)
        if await self._try_send_telegram_sticker(event, emoji_path):
            return "telegram_sticker"

        if await send_qq_image_as_sticker(event, emoji_path, plugin=self.plugin):
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
