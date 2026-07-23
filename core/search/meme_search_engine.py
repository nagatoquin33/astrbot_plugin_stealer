"""表情包搜索引擎：负责 BM25 索引构建和搜索辅助方法。"""

import hashlib
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .text_similarity import (
    BM25,
    _extract_words,
    calculate_hybrid_similarity,
    tokenize_for_bm25,
)


class MemeSearchEngine:
    """负责 BM25 索引构建、搜索签名和条目打分。"""

    SIMILARITY_THRESHOLD = 0.45

    def __init__(self, plugin_instance: Any, selector: Any) -> None:
        self.plugin = plugin_instance
        self.selector = selector
        self._bm25_index: Any | None = None
        self._bm25_doc_paths: list[str] = []
        self._bm25_dirty: bool = True
        self._bm25_signature: str = ""

    @staticmethod
    def _search_signature_from_index(idx: dict[str, Any]) -> str:
        if not idx:
            return "empty"

        hasher = hashlib.sha256()
        for file_path in sorted(idx.keys()):
            data = idx.get(file_path)
            if not isinstance(data, dict):
                continue
            tags = tuple(str(tag).strip() for tag in data.get("tags", []) if str(tag).strip())
            scenes = tuple(
                str(scene).strip() for scene in data.get("scenes", []) if str(scene).strip()
            )
            desc = str(data.get("desc", "") or "")
            category = str(data.get("category", "") or "")
            payload = "\x1f".join(
                [
                    str(file_path),
                    category,
                    desc,
                    "\x1e".join(tags),
                    "\x1e".join(scenes),
                ]
            )
            hasher.update(payload.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def _compute_bm25_signature(
        self, idx: dict[str, Any], *, prefer_db_signature: bool = True
    ) -> str:
        """计算 BM25 语料签名。"""
        db_service = getattr(self.plugin, "db_service", None)

        if prefer_db_signature and db_service and db_service.count_total() > 0:
            try:
                return db_service.get_corpus_signature()
            except Exception as e:
                logger.debug(f"[BM25] 获取数据库签名失败: {e}")

        # Fallback
        return self._search_signature_from_index(idx)

    async def _build_bm25_index(self, idx: dict | None = None) -> None:
        if BM25 is None:
            return

        # 优先使用数据库服务
        cache_service = getattr(self.plugin, "cache_service", None)
        explicit_idx = idx is not None

        if idx is None:
            idx = self.selector._get_index()

        if not idx:
            return

        signature = self._compute_bm25_signature(idx, prefer_db_signature=not explicit_idx)

        # 优先尝试从持久化缓存加载 BM25 文档语料。
        if cache_service:
            try:
                cached = cache_service.get_cache("bm25_cache")
                if isinstance(cached, dict) and cached.get("signature") == signature:
                    cached_documents = cached.get("documents")
                    cached_doc_paths = cached.get("doc_paths")
                    if isinstance(cached_documents, list) and isinstance(cached_doc_paths, list):
                        normalized_docs: list[list[str]] = []
                        for item in cached_documents:
                            if isinstance(item, (list, tuple)):
                                normalized_docs.append([str(tok) for tok in item if str(tok)])
                        if normalized_docs and len(normalized_docs) == len(cached_doc_paths):
                            self._bm25_index = BM25(normalized_docs)
                            self._bm25_doc_paths = [str(p) for p in cached_doc_paths]
                            self._bm25_signature = signature
                            self._bm25_dirty = False
                            logger.debug(f"[BM25] 从缓存恢复索引: {len(normalized_docs)} 文档")
                            return
            except Exception as e:
                logger.debug(f"[BM25] 读取缓存失败，改为重建: {e}")

        documents: list[tuple[str, ...]] = []
        doc_paths: list[str] = []

        for file_path, data in idx.items():
            if not isinstance(data, dict):
                continue
            tags = self.selector._parse_tags(data.get("tags", []))
            scenes = self.selector._parse_tags(data.get("scenes", []))
            desc = str(data.get("desc", "") or "")
            category = self.selector._get_category_from_data(data)
            text_content = " ".join([category, desc] + tags + scenes)
            tokens = tokenize_for_bm25(text_content)
            if tokens:
                documents.append(tokens)
                doc_paths.append(file_path)

        if documents:
            self._bm25_index = BM25(documents)
            self._bm25_doc_paths = doc_paths
            self._bm25_signature = signature
            self._bm25_dirty = False
            logger.debug(f"[BM25] 索引构建完成: {len(documents)} 文档, 示例: {documents[:3]}")

            if cache_service:
                try:
                    await cache_service.set_cache(
                        "bm25_cache",
                        {
                            "signature": signature,
                            "documents": [list(doc) for doc in documents],
                            "doc_paths": doc_paths,
                            "version": 1,
                        },
                        persist=True,
                    )
                except Exception as e:
                    logger.debug(f"[BM25] 持久化缓存失败: {e}")

    def _invalidate_bm25_index(self) -> None:
        self._bm25_dirty = True
        self._bm25_index = None
        self._bm25_doc_paths = []

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
                current_tag_words = tag_words or self.selector._collect_phrase_words(tuple(tags))
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
                        score = max(score, int(4 + (sim - self.SIMILARITY_THRESHOLD) * 16))
                        break

        return score

    def _find_best_category_match(self, query: str, threshold: float = 0.4) -> str | None:
        """找到与查询词最相似的分类。

        Args:
            query: 查询词
            threshold: 相似度阈值

        Returns:
            最佳匹配的分类，无则返回 None
        """
        if not query or not self.selector.categories:
            return None

        best_match = None
        best_score = 0.0
        for category in self.selector.categories:
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
        if not query or not self.selector.categories:
            return []

        scores = []
        for category in self.selector.categories:
            score = calculate_hybrid_similarity(query, category)
            scores.append((category, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [cat for cat, _ in scores[:top_n]]

    async def _search_images_fallback(
        self,
        query: str,
        limit: int = 1,
        idx: dict | None = None,
        event: AstrMessageEvent | None = None,
    ) -> list[tuple[str, str, str, str]]:
        """降级搜索：使用旧的评分算法。"""
        try:
            if idx is None:
                idx = self.selector._get_index()

            if not idx:
                return []

            recently_used_paths: set[str] = set()
            for cat_paths in self.selector._recent_usage.values():
                recently_used_paths.update(cat_paths)

            query_lower = query.lower()
            query_tokens = [t for t in query_lower.split() if len(t) > 1]
            query_words = _extract_words(query)

            MAX_STR_LENGTH = 20
            top_k = limit * 3
            top_candidates: list[tuple[str, str, str, str, int]] = []

            for file_path, data in idx.items():
                if not isinstance(data, dict):
                    continue
                if not self.selector._is_entry_allowed_for_event(data, event):
                    continue
                if file_path in recently_used_paths:
                    continue

                tags = self.selector._parse_tags(data.get("tags", []))
                scenes = self.selector._parse_tags(data.get("scenes", []))
                tags_for_score = tags + scenes
                tags_str = ", ".join(tags)
                category = self.selector._get_category_from_data(data)
                desc, tag_words, _, all_words, all_text = (
                    self.selector._prepare_entry_text_features(
                        category, str(data.get("desc", "")), tuple(tags_for_score)
                    )
                )

                if query_words:
                    if not query_words.intersection(all_words):
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
                    if len(top_candidates) > top_k:
                        top_candidates.sort(key=lambda x: x[4], reverse=True)
                        top_candidates = top_candidates[:top_k]

            top_candidates.sort(key=lambda x: x[4], reverse=True)
            return [(item[0], item[1], item[2], item[3]) for item in top_candidates[:limit]]

        except Exception as e:
            logger.error(f"降级搜索图片失败: {e}")
            return []
