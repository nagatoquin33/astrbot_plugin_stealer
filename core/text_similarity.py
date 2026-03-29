"""
文本相似度计算工具
提供多种文本相似度算法，用于表情包智能匹配

改进说明（v2）：
- 中文 n-gram 分词：提取 bigram/trigram 保留词组语义（如 "开心" 作为整体而非 "开"+"心"）
- 子串包含匹配：短文本包含在长文本中时给予高分
- 多策略融合：n-gram Jaccard + 子串匹配 + 编辑距离，加权融合
- 无额外依赖：不依赖 jieba 等第三方分词库
"""

import math
import re
from collections import Counter
from functools import lru_cache

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_PUNCT_RE = re.compile(r"[^\w\s]")
_EN_WORD_RE = re.compile(r"^[a-zA-Z]+$")

K1 = 1.5
B = 0.75
EPSILON = 0.25


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self._initialize(corpus)

    def _initialize(self, corpus: list[list[str]]) -> None:
        nd: dict[str, list[int]] = {}
        self.doc_len = []
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies: dict[str, int] = {}
            for word in document:
                word = word.lower()
                frequencies[word] = frequencies.get(word, 0) + 1
            for word, freq in frequencies.items():
                if word not in nd:
                    nd[word] = []
                nd[word].append(freq)
            self.corpus_size += 1

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 1

        for word, freq_list in nd.items():
            df = len(freq_list)
            self.doc_freqs[word] = df
            self.idf[word] = math.log(
                (self.corpus_size - df + 0.5) / (df + 0.5) + 1
            )

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            scores[i] = self._calculate_score(query, i)
        return scores

    def get_top_k(self, query: list[str], k: int = 10) -> list[tuple[int, float]]:
        scores = self.get_scores(query)
        indexed = [(i, score) for i, score in enumerate(scores)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:k]

    def _calculate_score(self, query: list[str], doc_index: int) -> float:
        score = 0.0
        doc_len = self.doc_len[doc_index]
        freq: Counter[str] = Counter()
        query_lower = [term.lower() for term in query]
        for term in query_lower:
            freq[term] += 1

        for term in query_lower:
            if term not in self.doc_freqs:
                continue
            freq_tf = freq.get(term, 0)
            if freq_tf == 0:
                continue
            idf = self.idf.get(term, 0)
            numerator = freq_tf * (self.k1 + 1)
            denominator = freq_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / (denominator + EPSILON)
        return score


@lru_cache(maxsize=8192)
def tokenize_for_bm25(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    cn_buffer: list[str] = []
    for char in cleaned:
        if _CJK_RE.match(char):
            cn_buffer.append(char)
        else:
            if cn_buffer:
                n = len(cn_buffer)
                for i in range(n):
                    tokens.append(cn_buffer[i])
                    if i + 1 < n:
                        tokens.append(cn_buffer[i] + cn_buffer[i + 1])
                cn_buffer = []
            if char.isalnum() and len(char) > 1:
                tokens.append(char)
    if cn_buffer:
        n = len(cn_buffer)
        for i in range(n):
            tokens.append(cn_buffer[i])
            if i + 1 < n:
                tokens.append(cn_buffer[i] + cn_buffer[i + 1])
    return tuple(tokens)


@lru_cache(maxsize=4096)
def _normalize_text(text: str) -> str:
    return str(text or "").lower().strip()


# ──────────────────────────────────────────────────────────────
# 公开 API（保持原有签名兼容）
# ──────────────────────────────────────────────────────────────


def calculate_simple_similarity(text1: str, text2: str) -> float:
    """计算两个文本的简单相似度（基于 n-gram 词重叠率）

    改进：中文使用 unigram + bigram，避免单字拆分丢失语义。

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    words1 = _extract_words(_normalize_text(text1))
    words2 = _extract_words(_normalize_text(text2))

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def calculate_levenshtein_similarity(text1: str, text2: str) -> float:
    """计算两个文本的编辑距离相似度

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    s1 = _normalize_text(text1)
    s2 = _normalize_text(text2)

    distance = _levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    return max(0.0, 1 - distance / max_len)


def calculate_hybrid_similarity(text1: str, text2: str) -> float:
    """混合相似度算法（多策略融合，v2）

    融合策略：
    1. n-gram Jaccard（语义词组重叠）
    2. 子串包含匹配（短文本被长文本包含时高分）
    3. 中文字符级 Jaccard（兜底召回）
    4. 编辑距离相似度（补充字形相近的情况）

    各策略取最优组合，而非简单加权。

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    t1 = _normalize_text(text1)
    t2 = _normalize_text(text2)

    if t1 == t2:
        return 1.0

    if t1 > t2:
        t1, t2 = t2, t1

    return _calculate_hybrid_similarity_cached(t1, t2)


@lru_cache(maxsize=8192)
def _calculate_hybrid_similarity_cached(t1: str, t2: str) -> float:
    if not t1 or not t2:
        return 0.0

    # ---- 策略1: n-gram Jaccard ----
    ngram_sim = calculate_simple_similarity(t1, t2)

    # ---- 策略2: 子串包含匹配 ----
    substr_sim = _substring_similarity(t1, t2)

    # ---- 策略3: 中文字符级 Jaccard（仅中文字符）----
    char_sim = _chinese_char_similarity(t1, t2)

    # ---- 策略4: 编辑距离（仅在短文本时有效）----
    edit_sim = 0.0
    if max(len(t1), len(t2)) <= 20:
        edit_sim = calculate_levenshtein_similarity(t1, t2)

    # ---- 融合 ----
    # 核心思路：取各策略的加权组合，但保证强信号不被弱信号拉低
    # 如果子串完全包含，直接给高分
    if substr_sim >= 0.7:
        return min(1.0, max(substr_sim, ngram_sim * 0.3 + substr_sim * 0.7))

    # 正常融合
    score = (
        ngram_sim * 0.40
        + substr_sim * 0.25
        + char_sim * 0.15
        + edit_sim * 0.20
    )

    # 如果任一策略给出很高分，提升最终分数（避免被其他低分策略拉低）
    best_single = max(ngram_sim, substr_sim, char_sim, edit_sim)
    if best_single > score:
        score = score * 0.6 + best_single * 0.4

    return min(1.0, score)


# ──────────────────────────────────────────────────────────────
# 分词与特征提取
# ──────────────────────────────────────────────────────────────


@lru_cache(maxsize=8192)
def _extract_words(text: str) -> frozenset[str]:
    """从文本中提取特征词汇集合（改进版）

    中文：unigram + bigram（如 "开心快乐" → {"开", "心", "快", "乐", "开心", "心快", "快乐"}）
    英文：完整单词

    Args:
        text: 输入文本（应已转小写）

    Returns:
        set[str]: 特征词汇集合
    """
    words: set[str] = set()

    # 移除标点
    cleaned = _PUNCT_RE.sub(" ", text)

    # 提取中文字符序列，生成 unigram + bigram
    cn_chars: list[str] = []
    for char in cleaned:
        if _CJK_RE.match(char):
            cn_chars.append(char)
        else:
            # 遇到非中文字符，处理之前积累的中文序列
            if cn_chars:
                _add_chinese_ngrams(cn_chars, words)
                cn_chars = []
    # 处理末尾中文
    if cn_chars:
        _add_chinese_ngrams(cn_chars, words)

    # 提取英文单词
    for token in cleaned.split():
        if _EN_WORD_RE.match(token) and len(token) > 1:
            words.add(token)

    return frozenset(words)


def _add_chinese_ngrams(chars: list[str], words: set[str]) -> None:
    """将中文字符序列转为 unigram + bigram 加入词集。

    Args:
        chars: 连续中文字符列表
        words: 输出词集（原地修改）
    """
    n = len(chars)
    for i in range(n):
        # unigram（单字）
        words.add(chars[i])
        # bigram（双字词组）
        if i + 1 < n:
            words.add(chars[i] + chars[i + 1])


# ──────────────────────────────────────────────────────────────
# 子串包含匹配
# ──────────────────────────────────────────────────────────────


def _substring_similarity(text1: str, text2: str) -> float:
    """子串包含相似度

    如果较短文本是较长文本的子串，给予高分。
    部分包含（最长公共子串）也按比例给分。

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    short, long = (text1, text2) if len(text1) <= len(text2) else (text2, text1)

    # 完全包含
    if short in long:
        # 按短文本占长文本的比例给分，但保底 0.7
        ratio = len(short) / len(long)
        return max(0.7, ratio)

    # 最长公共子串
    lcs_len = _longest_common_substring_length(short, long)
    if lcs_len == 0:
        return 0.0

    # 按 LCS 占短文本的比例给分
    return (lcs_len / len(short)) * 0.6


def _longest_common_substring_length(s1: str, s2: str) -> int:
    """计算最长公共子串长度（空间优化版）

    Args:
        s1: 较短字符串
        s2: 较长字符串

    Returns:
        int: 最长公共子串长度
    """
    if not s1 or not s2:
        return 0

    # 确保 s1 是较短的
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    max_len = 0
    prev = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        curr = [0] * (len(s2) + 1)
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                curr[j + 1] = prev[j] + 1
                if curr[j + 1] > max_len:
                    max_len = curr[j + 1]
        prev = curr

    return max_len


# ──────────────────────────────────────────────────────────────
# 中文字符级相似度
# ──────────────────────────────────────────────────────────────


def _chinese_char_similarity(text1: str, text2: str) -> float:
    """仅基于中文字符的 Jaccard 相似度（兜底策略）

    在 n-gram 和子串匹配都不理想时，用单字级别兜底。

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    chars1 = _extract_chinese_chars(_normalize_text(text1))
    chars2 = _extract_chinese_chars(_normalize_text(text2))

    if not chars1 or not chars2:
        return 0.0

    intersection = chars1 & chars2
    union = chars1 | chars2

    if not union:
        return 0.0

    return len(intersection) / len(union)


# ──────────────────────────────────────────────────────────────
# 编辑距离
# ──────────────────────────────────────────────────────────────


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein距离）

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        int: 编辑距离
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    if not s2:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


@lru_cache(maxsize=4096)
def _normalize_text(text: str) -> str:
    return str(text or "").lower().strip()


@lru_cache(maxsize=8192)
def _extract_chinese_chars(text: str) -> frozenset[str]:
    return frozenset(c for c in text if _CJK_RE.match(c))
