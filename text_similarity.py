"""
文本相似度计算工具
提供多种文本相似度算法，用于表情包智能匹配
"""
import re


def calculate_simple_similarity(text1: str, text2: str) -> float:
    """计算两个文本的简单相似度（基于词重叠率）

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # 转换为小写
    text1 = text1.lower()
    text2 = text2.lower()

    # 提取词汇（中文按字符，英文按单词）
    words1 = _extract_words(text1)
    words2 = _extract_words(text2)

    if not words1 or not words2:
        return 0.0

    # 计算交集和并集
    intersection = words1 & words2
    union = words1 | words2

    # Jaccard相似度
    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def calculate_levenshtein_similarity(text1: str, text2: str) -> float:
    """计算两个文本的编辑距离相似度（类似MaiBot）

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # 转换为小写
    text1 = text1.lower()
    text2 = text2.lower()

    # 计算编辑距离
    distance = _levenshtein_distance(text1, text2)

    # 转换为相似度
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0

    similarity = 1 - (distance / max_len)
    return max(0.0, similarity)


def calculate_hybrid_similarity(text1: str, text2: str) -> float:
    """混合相似度算法（结合词重叠和编辑距离）

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0

    # 计算两种相似度
    jaccard_sim = calculate_simple_similarity(text1, text2)
    levenshtein_sim = calculate_levenshtein_similarity(text1, text2)

    # 加权平均（词重叠权重更高，因为更适合中文）
    return jaccard_sim * 0.7 + levenshtein_sim * 0.3


def _extract_words(text: str) -> set[str]:
    """从文本中提取词汇集合

    中文按字符分割，英文按单词分割

    Args:
        text: 输入文本

    Returns:
        Set[str]: 词汇集合
    """
    words = set()

    # 移除标点符号
    text = re.sub(r"[^\w\s]", " ", text)

    # 处理中文字符（每个字符都是一个词）
    for char in text:
        if re.match(r"[\u4e00-\u9fff]", char):
            words.add(char)

    # 处理英文单词
    tokens = text.split()
    for token in tokens:
        # 英文单词（长度>1）
        if re.match(r"^[a-zA-Z]+$", token) and len(token) > 1:
            words.add(token.lower())

    return words


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离（Levenshtein距离）

    Args:
        s1: 第一个字符串
        s2: 第二个字符串

    Returns:
        int: 编辑距离
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
