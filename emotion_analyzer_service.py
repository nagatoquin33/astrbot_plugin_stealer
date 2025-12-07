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
        self.categories = plugin_instance.categories if hasattr(plugin_instance, "categories") else []

    def normalize_category(self, category: str) -> str:
        """将任意文本归一化到预定义的情绪分类中。"""
        if not category:
            return "speechless"

        category = category.strip().lower()

        # 检查是否直接匹配categories中的类别（LLM输出的标签已经通过提示词限制，直接匹配即可）
        for cat in self.categories:
            if cat.lower() == category:
                return cat

        # 兜底
        return "speechless" if "speechless" in self.categories else "其它"

    async def classify_text_emotion(self, event: AstrMessageEvent, text: str) -> str:
        """调用文本模型判断文本情绪并映射到插件分类。"""
        try:
            # LLM已经通过提示词限制了输出标签格式，不再需要本地规则匹配
            # 直接返回默认值
            return "speechless" if "speechless" in self.categories else "其它"
        except Exception as e:
            logger.error(f"文本情绪分类失败: {e}")
            return "speechless" if "speechless" in self.categories else "其它"

    async def extract_emotions_from_text(self, event: AstrMessageEvent | None, text: str) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.categories)  # 使用self.categories确保一致性

            # 1. 处理显式包裹标记：&&情绪&&
            matches = list(self.HEX_PATTERN.finditer(cleaned_text))

            # 收集所有匹配项，避免索引偏移问题
            temp_replacements = []
            for match in matches:
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories:
                    temp_replacements.append((original, norm_cat))
                else:
                    temp_replacements.append((original, ""))  # 非法或未知情绪静默移除

            # 保持原始顺序替换
            for original, emotion in temp_replacements:
                cleaned_text = cleaned_text.replace(original, "", 1)
                if emotion and emotion not in seen:
                    seen.add(emotion)
                    res.append(emotion)

            # 2. 替代标记处理（如[emotion]、(emotion)等）
            # 处理[emotion]格式
            matches = list(self.BRACKET_PATTERN.finditer(cleaned_text))
            bracket_replacements = []
            invalid_brackets = []

            for match in matches:
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories:
                    bracket_replacements.append((original, norm_cat))
                else:
                    # 记录无效标记，稍后删除
                    invalid_brackets.append(original)

            # 删除所有无效标记
            for invalid in invalid_brackets:
                cleaned_text = cleaned_text.replace(invalid, "", 1)

            for original, emotion in bracket_replacements:
                cleaned_text = cleaned_text.replace(original, "", 1)
                if emotion and emotion not in seen:
                    seen.add(emotion)
                    res.append(emotion)

            # 处理(emotion)格式
            matches = list(self.PAREN_PATTERN.finditer(cleaned_text))
            paren_replacements = []
            invalid_parens = []

            for match in matches:
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories:
                    # 需要额外验证，确保不是普通句子的一部分
                    if self._is_likely_emotion_markup(original, cleaned_text, match.start()):
                        paren_replacements.append((original, norm_cat))
                else:
                    # 记录无效标记，稍后删除
                    invalid_parens.append(original)

            # 删除所有无效标记
            for invalid in invalid_parens:
                cleaned_text = cleaned_text.replace(invalid, "", 1)

            for original, emotion in paren_replacements:
                cleaned_text = cleaned_text.replace(original, "", 1)
                if emotion and emotion not in seen:
                    seen.add(emotion)
                    res.append(emotion)

            # 3. 清理文本，移除HTML标签
            cleaned_text = self.HTML_TAG_PATTERN.sub("", cleaned_text)

            # 4. 清理多余的空格
            cleaned_text = self.WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

            return res, cleaned_text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text

    def _is_likely_emotion_markup(self, original: str, text: str, start_pos: int) -> bool:
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











