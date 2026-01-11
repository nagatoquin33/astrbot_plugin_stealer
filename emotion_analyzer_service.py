import re

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class EmotionAnalyzerService:
    """情感分析服务类，处理文本情绪分析和映射。"""

    # 正则表达式模式常量
    # 1. 标准模式：&&happy&& 或 && happy && (允许内含空格)
    # 2. 容错模式：处理可能被Markdown转义的情况，如 \&\&happy\&\&
    HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:&&|\\&\\&)")
    
    # 3. 残缺模式：处理模型输出 &&happy| 或 &&happy\n 这种忘记闭合的情况
    # 仅匹配后跟 |、换行符或字符串结束的情况，避免误伤正常文本
    INCOMPLETE_HEX_PATTERN = re.compile(r"(?:&&|\\&\\&)\s*([a-zA-Z0-9_]+)\s*(?:[|]|\n|$)")

    # 单个&的匹配太容易误伤（如 URL 参数），仅作为最后的兜底，且要求情绪词必须在列表内
    SINGLE_HEX_PATTERN = re.compile(r"&([^&\s]+?)&")

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
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.categories)  # 使用self.categories确保一致性

            # 1. 处理显式包裹标记：&&情绪&&
            temp_text = cleaned_text
            for match in self.HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    seen.add(norm_cat)
                    res.append(norm_cat)
                temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            # 2. 处理残缺标记：&&情绪| 或 &&情绪\n
            # 必须严格校验是否在 valid_categories 中，避免误判
            temp_text = cleaned_text
            for match in self.INCOMPLETE_HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    # 只有当它是合法的情绪词时，才认为是标签并移除
                    logger.debug(f"检测到残缺情绪标签: {emotion} -> {norm_cat}")
                    seen.add(norm_cat)
                    res.append(norm_cat)
                    temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            # 3. 处理单个&包裹的标记：&情绪&
            temp_text = cleaned_text
            for match in self.SINGLE_HEX_PATTERN.finditer(cleaned_text):
                original = match.group(0)
                emotion = match.group(1).strip()
                norm_cat = self.normalize_category(emotion)

                if norm_cat and norm_cat in valid_categories and norm_cat not in seen:
                    seen.add(norm_cat)
                    res.append(norm_cat)
                    temp_text = temp_text.replace(original, "", 1)
            cleaned_text = temp_text

            return res, cleaned_text
        except Exception as e:
            logger.error(f"提取文本情绪失败: {e}")
            return [], text



    def update_config(self, categories=None):
        """更新配置参数"""
        if categories is not None:
            self.categories = categories

    def cleanup(self):
        """清理资源。"""
        pass
