import logging
import re

from astrbot.api.event import AstrMessageEvent

logger = logging.getLogger(__name__)

class EmotionAnalyzerService:
    """情感分析服务类，处理文本情绪分析和映射。"""

    # 情绪映射规则
    EMOTION_MAPPING = {
        "开心": ["开心", "快乐", "高兴", "愉悦", "欢乐", "愉快", "喜悦", "兴奋", "狂喜", "欣喜", "满足", "满足感", "欣慰", "痛快", "开心果", "欢乐时光", "欢声笑语", "喜笑颜开"],
        "难过": ["难过", "悲伤", "伤心", "悲痛", "哀伤", "忧伤", "沮丧", "低落", "痛苦", "难受", "心碎", "哭泣", "泪目", "难过到极点", "心如刀割", "悲痛欲绝"],
        "愤怒": ["愤怒", "生气", "气愤", "恼火", "暴怒", "发火", "怒发冲冠", "火冒三丈", "怒火中烧", "暴跳如雷", "咬牙切齿", "愤愤不平"],
        "惊讶": ["惊讶", "吃惊", "震惊", "诧异", "惊愕", "惊叹", "意外", "没想到", "目瞪口呆", "大吃一惊", "出乎意料", "超乎想象"],
        "恐惧": ["恐惧", "害怕", "惧怕", "惊恐", "恐慌", "畏惧", "胆战心惊", "毛骨悚然", "不寒而栗", "心惊肉跳", "提心吊胆"],
        "厌恶": ["厌恶", "讨厌", "反感", "憎恶", "嫌弃", "鄙夷", "不屑", "恶心", "反感至极", "深恶痛绝", "令人作呕"],
        "尴尬": ["尴尬", "难堪", "窘迫", "难为情", "无地自容", "汗颜", "脸红", "不好意思"],
        "无语": ["无语", "无奈", "无解", "无力", "无话可说", "哭笑不得", "哑口无言", "不知所措"],
        "感动": ["感动", "感人", "暖心", "温情", "温馨", "动容", "热泪盈眶", "感人至深", "温暖人心"],
        "思念": ["思念", "想念", "怀念", "牵挂", "惦记", "依依不舍", "朝思暮想", "魂牵梦绕"],
        "期待": ["期待", "盼望", "期望", "渴望", "憧憬", "向往", "翘首以盼", "望眼欲穿"],
        "感恩": ["感恩", "感谢", "感激", "致谢", "谢意", "知恩图报", "感激不尽"],
        "愧疚": ["愧疚", "内疚", "歉意", "抱歉", "对不起", "深感歉意", "愧疚难当"],
        "羡慕": ["羡慕", "艳羡", "嫉妒", "妒忌", "眼红", "仰慕", "倾慕"],
        "骄傲": ["骄傲", "自豪", "荣耀", "光荣", "引以为豪", "骄傲自满"],
        "孤独": ["孤独", "孤单", "孤寂", "寂寞", "形单影只", "孤苦伶仃"],
        "疲惫": ["疲惫", "疲倦", "劳累", "劳累过度", "精疲力竭", "筋疲力尽", "身心俱疲"],
        "烦躁": ["烦躁", "烦躁不安", "心浮气躁", "坐立不安", "焦躁", "焦虑"],
        "平静": ["平静", "平和", "安静", "安宁", "宁静", "心如止水", "波澜不惊"],
        "淡定": ["淡定", "从容", "镇定", "冷静", "泰然自若", "处变不惊"],
        "兴奋": ["兴奋", "激动", "亢奋", "振奋", "精神抖擞", "兴高采烈", "欢呼雀跃"],
        "紧张": ["紧张", "焦虑", "忐忑", "不安", "提心吊胆", "坐立不安", "战战兢兢"],
        "沮丧": ["沮丧", "失落", "失望", "心灰意冷", "灰心丧气", "垂头丧气", "一蹶不振"],
        "贪婪": ["贪婪", "贪心", "贪得无厌", "得寸进尺", "见利忘义"],
        "嫉妒": ["嫉妒", "妒忌", "吃醋", "眼红", "妒火中烧", "嫉贤妒能"],
        "虚荣": ["虚荣", "虚伪", "爱慕虚荣", "自命不凡", "自以为是"],
        "自卑": ["自卑", "自我否定", "缺乏自信", "妄自菲薄", "自惭形秽"],
        "自信": ["自信", "信心十足", "胸有成竹", "自信满满", "从容不迫"],
        "好奇": ["好奇", "新奇", "兴趣", "求知欲", "好奇心", "打破砂锅问到底"],
        "疑惑": ["疑惑", "疑问", "困惑", "不解", "迷茫", "大惑不解", "疑惑不解"],
        "矛盾": ["矛盾", "纠结", "两难", "左右为难", "犹豫不决", "举棋不定"],
        "无奈": ["无奈", "无助", "无能为力", "无可奈何", "一筹莫展", "爱莫能助"],
        "生气": ["生气", "恼火", "恼怒", "气愤", "怒火中烧", "恼羞成怒"]
    }

    def __init__(self, plugin_instance):
        self.plugin_instance = plugin_instance
        self.categories = plugin_instance.categories if hasattr(plugin_instance, "categories") else []

    def normalize_category(self, category: str) -> str:
        """将任意文本归一化到预定义的情绪分类中。"""
        if not category:
            return "无语"

        category = category.strip().lower()

        # 检查是否直接匹配
        for cat in self.categories:
            if cat.lower() == category:
                return cat

        # 模糊匹配
        for cat, synonyms in self.EMOTION_MAPPING.items():
            if cat not in self.categories:
                continue
            if category in [s.lower() for s in synonyms]:
                return cat

        # 关键词匹配
        for cat, synonyms in self.EMOTION_MAPPING.items():
            if cat not in self.categories:
                continue
            for synonym in synonyms:
                if synonym.lower() in category:
                    return cat

        # 兜底
        return "无语" if "无语" in self.categories else "其它"

    async def classify_text_emotion(self, event: AstrMessageEvent, text: str) -> str:
        """调用文本模型判断文本情绪并映射到插件分类。"""
        try:
            # 简单的本地规则匹配
            for cat, keywords in self.EMOTION_MAPPING.items():
                for keyword in keywords:
                    if keyword in text:
                        return self.normalize_category(cat)

            # 兜底
            return "无语" if "无语" in self.categories else "其它"
        except Exception as e:
            logger.error(f"文本情绪分类失败: {e}")
            return "无语" if "无语" in self.categories else "其它"

    async def extract_emotions_from_text(self, event: AstrMessageEvent | None, text: str) -> tuple[list[str], str]:
        """从文本中提取情绪关键词。"""
        try:
            res: list[str] = []
            seen: set[str] = set()
            cleaned_text = str(text)
            valid_categories = set(self.plugin_instance.categories if hasattr(self.plugin_instance, "categories") else [])

            # 1. 处理显式包裹标记：&&情绪&&
            hex_pattern = r"&&([^&]+?)&&"
            matches = list(re.finditer(hex_pattern, cleaned_text))

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
            bracket_pattern = r"\[([^\[\]]+)\]"
            matches = list(re.finditer(bracket_pattern, cleaned_text))
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
            paren_pattern = r"\(([^()]+)\)"
            matches = list(re.finditer(paren_pattern, cleaned_text))
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
            cleaned_text = re.sub(r"<.*?>", "", cleaned_text)

            # 4. 提取情绪关键词
            for cat, keywords in self.EMOTION_MAPPING.items():
                for keyword in keywords:
                    if keyword in cleaned_text:
                        if cat not in seen and cat in valid_categories:
                            seen.add(cat)
                            res.append(cat)
                        break

            # 清理多余的空格
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

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

    def cleanup(self):
        """清理资源。"""
        pass
