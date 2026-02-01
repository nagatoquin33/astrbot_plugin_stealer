"""
自然语言情绪分析器
使用小模型对LLM回复进行语义分析，识别隐含情绪
"""

import re
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class NaturalEmotionAnalyzer:
    """自然语言情绪分析器 - 使用小模型理解LLM回复的真实情绪"""

    def __init__(self, plugin_instance):
        self.plugin = plugin_instance
        self.categories = plugin_instance.categories

        # 缓存机制
        self.analysis_cache = {}
        self.cache_max_size = 1000

        # 性能统计
        self.stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "successful_analyses": 0,
        }

        # 小模型提示词模板
        self.emotion_analysis_prompt = self._build_analysis_prompt()

    def _build_analysis_prompt(self) -> str:
        """构建情绪分析提示词（精简版）"""
        # 精简分类描述，只保留核心关键词
        categories_desc = {
            "happy": "开心、愉快",
            "sad": "难过、失落",
            "angry": "愤怒、生气",
            "surprised": "惊讶、震惊",
            "confused": "困惑、疑惑",
            "shy": "害羞、腼腆",
            "fear": "害怕、担心",
            "disgust": "厌恶、反感",
            "excitement": "兴奋、激动",
            "tired": "疲惫、困倦",
            "embarrassed": "尴尬、窘迫",
            "love": "喜爱、温柔",
            "sigh": "无奈、叹气",
            "thank": "感谢、感激",
            "dumb": "无语、傻眼",
            "troll": "调皮、搞怪",
            "cry": "哭泣、流泪",
        }

        # 构建分类说明（单行格式，更紧凑）
        categories_text = ", ".join(
            [
                f"{key}({desc})"
                for key, desc in categories_desc.items()
                if key in self.categories
            ]
        )

        # 精简提示词，移除冗余说明和大部分示例
        return f"""分析文本情绪，从以下分类选择最匹配的：{categories_text}

示例："哈哈笑死"→happy, "太离谱了"→dumb, "算了懒得说"→sigh

文本："{{text}}"
只返回英文分类名。"""

    async def analyze_emotion(self, event: AstrMessageEvent, text: str) -> str | None:
        """分析文本的自然情绪

        Args:
            event: 消息事件（用于获取LLM提供商）
            text: 要分析的文本

        Returns:
            情绪分类，如果分析失败返回None
        """
        if not text or len(text.strip()) < 3:
            return None

        # 清理文本
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return None

        # 检查缓存
        cache_key = self._get_cache_key(cleaned_text)
        if cache_key in self.analysis_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"[情绪分析] 缓存命中: {cleaned_text[:30]}...")
            return self.analysis_cache[cache_key]

        # 执行分析
        start_time = time.time()
        emotion = await self._analyze_with_llm(event, cleaned_text)
        end_time = time.time()

        # 更新统计
        self.stats["total_analyses"] += 1
        response_time = (end_time - start_time) * 1000
        self._update_stats(response_time, emotion is not None)

        # 缓存结果
        if emotion:
            self._cache_result(cache_key, emotion)
            logger.info(
                f"[情绪分析] {cleaned_text[:30]}... → {emotion} ({response_time:.0f}ms)"
            )
        else:
            logger.warning(f"[情绪分析] 分析失败: {cleaned_text[:30]}...")

        return emotion

    async def _analyze_with_llm(self, event: AstrMessageEvent, text: str) -> str | None:
        """使用小模型分析情绪"""
        try:
            # 获取文本模型提供商（优先使用配置的小模型）
            provider_id = await self._get_text_provider(event)
            if not provider_id:
                logger.warning("[情绪分析] 未找到可用的文本模型")
                return None

            # 构建提示词
            prompt = self.emotion_analysis_prompt.format(text=text)

            # 调用LLM（限制 max_tokens 提升速度）
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                max_tokens=15,  # 只需返回一个单词，大幅降低生成时间
            )

            if not response or not response.completion_text:
                return None

            # 解析结果
            result_text = response.completion_text.strip().lower()
            emotion = self._parse_emotion_result(result_text)

            return emotion

        except Exception as e:
            logger.error(f"[情绪分析] LLM调用失败: {e}")
            return None

    async def _get_text_provider(self, event: AstrMessageEvent) -> str | None:
        """获取文本模型提供商ID"""
        # 1. 优先使用插件配置的情绪分析专用模型
        if (
            hasattr(self.plugin, "emotion_analysis_provider_id")
            and self.plugin.emotion_analysis_provider_id
        ):
            return self.plugin.emotion_analysis_provider_id

        # 2. 使用插件配置的文本模型
        if hasattr(self.plugin, "text_provider_id") and self.plugin.text_provider_id:
            return self.plugin.text_provider_id

        # 3. 使用当前会话的模型
        try:
            return await self.plugin.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""

        # 移除情绪标记
        cleaned = re.sub(r"&&[^&]*&&", "", text)

        # 移除多余空白
        cleaned = re.sub(r"\s+", " ", cleaned.strip())

        # 限制长度（小模型处理能力有限）
        if len(cleaned) > 200:
            cleaned = cleaned[:200] + "..."

        return cleaned

    def _parse_emotion_result(self, result_text: str) -> str | None:
        """解析LLM返回的情绪结果"""
        if not result_text:
            return None

        # 清理结果
        result = result_text.strip().lower()

        # 直接匹配分类
        for category in self.categories:
            if category.lower() in result:
                return category

        # 尝试提取第一个有效单词
        words = re.findall(r"\w+", result)
        for word in words:
            if word in [cat.lower() for cat in self.categories]:
                # 找到对应的原始分类名
                for cat in self.categories:
                    if cat.lower() == word:
                        return cat

        return None

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        import hashlib

        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _cache_result(self, cache_key: str, emotion: str):
        """缓存分析结果"""
        # 清理过期缓存
        if len(self.analysis_cache) >= self.cache_max_size:
            # 移除最旧的一半缓存
            items = list(self.analysis_cache.items())
            self.analysis_cache = dict(items[len(items) // 2 :])

        self.analysis_cache[cache_key] = emotion

    def _update_stats(self, response_time: float, success: bool):
        """更新性能统计"""
        # 更新平均响应时间
        total = self.stats["total_analyses"]
        current_avg = self.stats["avg_response_time"]
        self.stats["avg_response_time"] = (
            current_avg * (total - 1) + response_time
        ) / total

        if success:
            self.stats["successful_analyses"] += 1

    def get_stats(self) -> dict:
        """获取性能统计"""
        total = self.stats["total_analyses"]
        if total == 0:
            return {"message": "暂无分析数据"}

        cache_hit_rate = (self.stats["cache_hits"] / total) * 100
        success_rate = (self.stats["successful_analyses"] / total) * 100

        return {
            "total_analyses": total,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{self.stats['avg_response_time']:.0f}ms",
            "cache_size": len(self.analysis_cache),
        }

    def clear_cache(self):
        """清空缓存"""
        self.analysis_cache.clear()
        logger.info("[情绪分析] 缓存已清空")


class SmartEmotionMatcher:
    """智能情绪匹配器 - 使用自然语言分析"""

    def __init__(self, plugin_instance):
        self.plugin = plugin_instance
        self.natural_analyzer = NaturalEmotionAnalyzer(plugin_instance)

    async def analyze_and_match_emotion(
        self, event: AstrMessageEvent, text: str, use_natural_analysis: bool = True
    ) -> str | None:
        """分析并匹配情绪

        Args:
            event: 消息事件
            text: 要分析的文本
            use_natural_analysis: 是否使用自然语言分析

        Returns:
            匹配的情绪分类
        """
        if not text or len(text.strip()) < 3:
            return None

        # 使用自然语言分析（主要方案）
        if use_natural_analysis and getattr(
            self.plugin, "enable_natural_emotion_analysis", True
        ):
            emotion = await self.natural_analyzer.analyze_emotion(event, text)
            if emotion:
                return emotion
            else:
                logger.warning(f"[智能匹配] 自然语言分析失败: {text[:30]}...")
                return None

        # 如果禁用了自然语言分析，返回None（被动模式依赖标签）
        logger.debug("[智能匹配] 自然语言分析已禁用")
        return None

    def get_analyzer_stats(self) -> dict:
        """获取分析器统计信息"""
        return self.natural_analyzer.get_stats()

    def clear_cache(self):
        """清空分析缓存"""
        self.natural_analyzer.clear_cache()
