import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class EmotionAnalyzerService:
    """情感分析服务类，处理文本情绪分析和映射。"""

    # 正则表达式模式常量
    HEX_PATTERN = re.compile(r"&&([^&]+?)&&")
    BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")
    PAREN_PATTERN = re.compile(r"\(([^()]+)\)")
    HTML_TAG_PATTERN = re.compile(r"<.*?>")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, plugin_instance):
        self.plugin_instance = plugin_instance
        self.categories = (
            plugin_instance.categories if hasattr(plugin_instance, "categories") else []
        )

    def normalize_category(self, category: str) -> str:
        """将任意文本归一化到预定义的情绪分类中。"""
        if not category:
            return ""

        category = category.strip().lower()

        # 检查是否直接匹配categories中的类别（LLM输出的标签已经通过提示词限制，直接匹配即可）
        for cat in self.categories:
            if cat.lower() == category:
                return cat

        # 不进行兜底分类，返回空字符串表示无法分类
        return ""

    async def extract_emotions_from_text(
        self, event: AstrMessageEvent | None, text: str
    ) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。

        针对LLM输出的单个&&emotion&&标签进行优化处理。
        """
        try:
            res: list[str] = []
            cleaned_text = str(text)
            valid_categories = set(self.categories)

            # 1. 优先处理&&情绪&&格式（LLM主要输出格式）
            hex_match = self.HEX_PATTERN.search(cleaned_text)
            if hex_match:
                emotion = hex_match.group(1).strip()
                norm_cat = self.normalize_category(emotion)
                if norm_cat and norm_cat in valid_categories:
                    res.append(norm_cat)
                # 移除标签
                cleaned_text = self.HEX_PATTERN.sub("", cleaned_text)

            # 2. 如果没找到&&格式，尝试其他格式
            if not res:
                # 处理[emotion]格式
                bracket_match = self.BRACKET_PATTERN.search(cleaned_text)
                if bracket_match:
                    emotion = bracket_match.group(1).strip()
                    norm_cat = self.normalize_category(emotion)
                    if norm_cat and norm_cat in valid_categories:
                        res.append(norm_cat)
                    # 移除标签
                    cleaned_text = self.BRACKET_PATTERN.sub("", cleaned_text)

            # 3. 最后尝试(emotion)格式，但需要验证
            if not res:
                paren_match = self.PAREN_PATTERN.search(cleaned_text)
                if paren_match:
                    emotion = paren_match.group(1).strip()
                    norm_cat = self.normalize_category(emotion)
                    if (
                        norm_cat
                        and norm_cat in valid_categories
                        and self._is_likely_emotion_markup(
                            paren_match.group(0), cleaned_text, paren_match.start()
                        )
                    ):
                        res.append(norm_cat)
                        # 移除标签
                        cleaned_text = self.PAREN_PATTERN.sub("", cleaned_text, count=1)

            # 4. 清理文本，移除HTML标签和多余空格
            cleaned_text = self.HTML_TAG_PATTERN.sub("", cleaned_text)
            cleaned_text = self.WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

            return res, cleaned_text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    def _is_likely_emotion_markup(
        self, original: str, text: str, start_pos: int
    ) -> bool:
        """
        判断是否为情绪标记，避免误判普通括号内容

        Args:
            original: 原始匹配字符串
            text: 完整文本
            start_pos: 匹配开始位置

        Returns:
            bool: 是否为情绪标记
        """
        # 检查前后是否为单词边界
        if start_pos > 0 and text[start_pos - 1].isalnum():
            return False

        end_pos = start_pos + len(original)
        if end_pos < len(text) and text[end_pos].isalnum():
            return False

        return True

    def update_config(self, categories=None):
        """更新配置参数"""
        if categories is not None:
            self.categories = categories

    def cleanup(self):
        """清理资源。"""
        pass
