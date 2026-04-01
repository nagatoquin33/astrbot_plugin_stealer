"""
自然语言情绪分析器
使用小模型对LLM回复进行语义分析，识别隐含情绪
"""

import asyncio
import hashlib
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class NaturalEmotionAnalyzer:
    """自然语言情绪分析器 - 使用小模型理解LLM回复的真实情绪"""

    # 常量定义
    CACHE_MAX_SIZE = 1000  # 缓存最大容量
    TEXT_MAX_LENGTH = 200  # 文本最大长度

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        self.categories: list[str] = plugin_instance.categories

        # 缓存机制
        self.analysis_cache: dict[str, str] = {}
        self.cache_max_size: int = self.CACHE_MAX_SIZE
        self._cache_lock = asyncio.Lock()

        # 性能统计
        self.stats: dict[str, float | int] = {
            "total_analyses": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "successful_analyses": 0,
        }

        # 小模型提示词模板
        self.emotion_analysis_prompt = self._build_analysis_prompt()
        self.emotion_analysis_qa_prompt = self._build_qa_analysis_prompt()

    def _build_analysis_prompt(self) -> str:
        """构建情绪分析提示词（仅分析回复文本，无用户问题时使用）"""
        categories_text = self._build_categories_text()

        return f"""你是对话情绪分类器。请从下列分类中选择一个最匹配的结果：{categories_text}

分类规则：
1. 只基于文本表达的语气与态度，不推测图片内容，不扩展剧情。
2. 负面情绪优先细分：愤怒→angry，无奈/叹气→sigh，震惊到无语→dumb，悲伤→sad，不解→confused。
3. `troll` 仅用于明显的阴阳怪气、挑衅、嘲讽、故意拱火、玩梗发癫语气。
4. 普通吐槽、拒绝、冷淡、抱怨，不要判为 `troll`。
5. 证据不足时选择最保守、最贴近字面语气的分类，不要为追求“有趣”而偏向 `troll`。

示例：
- "哈哈笑死" -> happy
- "太离谱了" -> dumb
- "算了懒得说" -> sigh

文本："{{text}}"

输出要求：
- 只能输出一个英文分类名
- 不能输出解释、标点、代码块或其他文字"""

    def _build_qa_analysis_prompt(self) -> str:
        """构建 QA 上下文情绪分析提示词（有用户问题时使用）

        通过提供用户问题（Q）和 LLM 回复（A）的完整上下文，
        让轻量模型更准确地理解回复的情绪语境。
        """
        categories_text = self._build_categories_text()

        return f"""你是对话情绪分类器。请根据 Q/A 语境判断 A 的情绪，并从下列分类中选择一个：{categories_text}

    任务边界：
    1. 只分类 A（回复）的情绪，不分类 Q（提问）。
    2. Q 仅用于帮助理解 A 的语气与立场。

    分类规则：
    1. 负面情绪优先细分：愤怒→angry，无奈/叹气→sigh，震惊到无语→dumb，悲伤→sad，不解→confused。
    2. `troll` 仅用于明显阴阳怪气、挑衅、嘲讽、故意拱火语气。
    3. 普通吐槽、拒绝、抱怨、冷淡，不要判为 `troll`。
    4. 证据不足时选择最保守、最贴近字面语气的分类。

    示例：
    - Q:"今天加班到几点" A:"别提了，干到十一点" -> sigh
    - Q:"这个好看吗" A:"也太好看了吧！" -> excitement
    - Q:"你怎么看" A:"太离谱了，无话可说" -> dumb

    Q:"{{user_query}}"
    A:"{{text}}"

    输出要求：
    - 只能输出一个英文分类名
    - 不能输出解释、标点、代码块或其他文字"""

    def _build_categories_text(self) -> str:
        """构建分类描述文本（供两种 prompt 共用）"""
        categories_desc = {}
        cfg = self.plugin.plugin_config
        if cfg:
            for key in self.categories:
                info = cfg.DEFAULT_CATEGORY_INFO.get(key, {})
                name = str(info.get("name", "")).strip()
                desc = str(info.get("desc", "")).strip()
                desc_text = desc or name or key
                categories_desc[key] = desc_text
        else:
            categories_desc = {key: key for key in self.categories}

        return ", ".join(
            [
                f"{key}({desc})"
                for key, desc in categories_desc.items()
                if key in self.categories
            ]
        )

    async def analyze_emotion(
        self, event: AstrMessageEvent, text: str,
        *, user_query: str = "",
    ) -> str | None:
        """分析文本的自然情绪

        Args:
            event: 消息事件（用于获取LLM提供商）
            text: LLM 回复文本
            user_query: 用户原始消息，与 text 组成 QA 上下文

        Returns:
            情绪分类，如果分析失败返回None
        """
        if not text or len(text.strip()) < 3:
            return None

        # 清理文本
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return None

        # 清理用户消息（用于缓存 key 和 prompt）
        cleaned_query = self._clean_text(user_query) if user_query else ""

        # 检查缓存（缓存 key 同时包含 Q 和 A）
        cache_key = self._get_cache_key(cleaned_query + "|||" + cleaned_text)
        async with self._cache_lock:
            if cache_key in self.analysis_cache:
                self.stats["cache_hits"] += 1
                logger.debug(f"[情绪分析] 缓存命中: {cleaned_text[:30]}...")
                return self.analysis_cache[cache_key]

        # 本地预匹配：先用分词匹配关键词映射（快速路径）
        local_match = self._local_keyword_match(cleaned_text)
        if local_match:
            logger.info(f"[情绪分析] 本地匹配: {cleaned_text[:30]}... → {local_match}")
            async with self._cache_lock:
                self._cache_result(cache_key, local_match)
            return local_match

        # 执行 LLM 分析（传入用户问题作为上下文）
        start_time = time.time()
        emotion = await self._analyze_with_llm(
            event,
            cleaned_text,
            user_query=cleaned_query,
        )
        end_time = time.time()

        # 更新统计
        self.stats["total_analyses"] += 1
        response_time = (end_time - start_time) * 1000
        self._update_stats(response_time, emotion is not None)

        # LLM 失败时降级到本地匹配
        if not emotion:
            emotion = self._local_keyword_match(cleaned_text, fallback=True)
            if emotion:
                logger.info(f"[情绪分析] LLM失败，降级匹配: {cleaned_text[:30]}... → {emotion}")

        # 缓存结果
        if emotion:
            async with self._cache_lock:
                self._cache_result(cache_key, emotion)
            logger.info(
                f"[情绪分析] {cleaned_text[:30]}... → {emotion} ({response_time:.0f}ms)"
            )
        else:
            logger.warning(f"[情绪分析] 分析失败: {cleaned_text[:30]}...")

        return emotion

    def _local_keyword_match(self, text: str, fallback: bool = False) -> str | None:
        """本地关键词匹配（快速路径/降级方案）

        Args:
            text: 要分析的文本
            fallback: 是否为降级模式（降级模式阈值更低）

        Returns:
            匹配的分类，无则返回 None
        """
        if not text:
            return None

        cfg = self.plugin.plugin_config
        if not cfg:
            return None

        # 获取关键词映射
        keyword_map = cfg.get_keyword_map() if hasattr(cfg, "get_keyword_map") else {}
        if not keyword_map:
            return None

        text_lower = text.lower()

        # 1. 精确匹配关键词
        for keyword, category in keyword_map.items():
            if keyword in text_lower and category:
                return category

        # 2. 分词匹配（降级模式下执行）
        if fallback:
            best_match = None
            best_score = 0.0
            threshold = 0.3  # 降级模式阈值更低

            for category in self.categories:
                # 与分类名比较
                score = calculate_hybrid_similarity(text, category)
                # 与分类中文描述比较
                info = cfg.DEFAULT_CATEGORY_INFO.get(category, {})
                desc = info.get("desc", "") or info.get("name", "")
                if desc:
                    score = max(score, calculate_hybrid_similarity(text, desc))

                if score > best_score and score > threshold:
                    best_score = score
                    best_match = category

            return best_match

        return None

    async def _analyze_with_llm(
        self, event: AstrMessageEvent, text: str,
        *, user_query: str = "",
    ) -> str | None:
        """使用小模型分析情绪

        Args:
            event: 消息事件
            text: LLM 回复文本（已清理）
            user_query: 用户原始消息（已清理），有值时使用 QA 模板
        """
        try:
            # 获取文本模型提供商（优先使用配置的小模型）
            provider_id = await self._get_text_provider(event)
            if not provider_id:
                logger.warning("[情绪分析] 未找到可用的文本模型")
                return None

            # 构建提示词：有用户问题时使用 QA 模板，否则使用纯文本模板
            if user_query:
                prompt = self.emotion_analysis_qa_prompt.format(
                    user_query=user_query, text=text,
                )
                logger.debug(f"[情绪分析] 使用QA模板，Q={user_query[:30]}...")
            else:
                prompt = self.emotion_analysis_prompt.format(text=text)

            # 调用LLM（限制 max_tokens 提升速度）
            logger.debug(f"[情绪分析] 调用LLM，provider_id={provider_id}")
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                max_tokens=15,  # 只需返回一个单词，大幅降低生成时间
            )

            # 安全获取响应文本
            if not response:
                logger.warning("[情绪分析] LLM返回空响应")
                return None
            result_text = response.completion_text
            if not result_text:
                logger.warning("[情绪分析] LLM返回空文本")
                return None

            # 解析结果
            result_text = result_text.strip().lower()
            emotion = self._parse_emotion_result(result_text)

            return emotion

        except Exception as e:
            error_msg = str(e)
            if "Provider" in error_msg or "提供商" in error_msg:
                logger.error(
                    f"[情绪分析] 模型提供商错误: {e}\n"
                    f"  配置的provider_id: {provider_id}\n"
                    f"  提示: 请检查插件配置中的'情绪分析专用模型'是否有效，"
                    f"  或尝试清空该配置使用默认模型"
                )
            else:
                logger.error(f"[情绪分析] LLM调用失败: {e}")
            return None

    async def _get_text_provider(self, event: AstrMessageEvent) -> str | None:
        """获取文本模型提供商ID"""
        # 1. 优先使用插件配置的情绪分析专用模型
        configured_provider = self.plugin.emotion_analysis_provider_id
        if configured_provider:
            logger.debug(f"[情绪分析] 尝试使用配置的提供商: {configured_provider}")
            return configured_provider

        # 2. 使用当前会话的模型
        try:
            current_provider = await self.plugin.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            logger.debug(f"[情绪分析] 使用当前会话模型: {current_provider}")
            return current_provider
        except Exception as e:
            logger.error(f"[情绪分析] 获取当前会话模型失败: {e}")
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
        if len(cleaned) > self.TEXT_MAX_LENGTH:
            cleaned = cleaned[: self.TEXT_MAX_LENGTH] + "..."

        return cleaned

    def _parse_emotion_result(self, result_text: str) -> str | None:
        """解析LLM返回的情绪结果

        支持从带解释的文字中提取分类名，如：
        - "happy" -> happy
        - "这个文本表达的是 happy 情绪" -> happy
        - "分类：sad" -> sad
        - "我觉得是 angry" -> angry
        """
        if not result_text:
            return None

        # 清理结果
        result = result_text.strip().lower()

        cfg = self.plugin.plugin_config

        # 尝试从文本中提取已知的分类名
        # 优先匹配完整的单词（避免部分匹配如 "sad" 匹配到 "sadness"）
        for category in self.categories:
            # 检查是否是完整单词匹配（前后是边界或标点）
            pattern = r'(?:^|[\s:：,，.。!！?？])' + re.escape(category) + r'(?:$|[\s:：,，.。!！?？])'
            if re.search(pattern, result, re.IGNORECASE):
                logger.debug(f"[情绪分析] 从文本中提取分类: '{result}' -> '{category}'")
                return category

        # 尝试严格归一化（处理直接返回分类名的情况）
        if cfg:
            try:
                normalized = cfg.normalize_category_strict(result)
                logger.debug(f"[情绪分析] 解析结果: '{result}' -> '{normalized}'")
                if normalized:
                    return normalized
            except Exception as e:
                logger.error(f"[情绪分析] 解析异常: {e}")

        # Fallback: 直接匹配分类名
        if result in self.categories:
            logger.debug(f"[情绪分析] Fallback 匹配: '{result}'")
            return result

        logger.warning(f"[情绪分析] 无法从回复中解析分类: '{result_text}'")
        return None

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

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
        cache_hits = self.stats["cache_hits"]
        grand_total = total + cache_hits  # 总请求数 = LLM调用 + 缓存命中
        if grand_total == 0:
            return {"message": "暂无分析数据"}

        cache_hit_rate = (cache_hits / grand_total) * 100
        success_rate = (
            (self.stats["successful_analyses"] / total) * 100 if total > 0 else 0.0
        )

        return {
            "total_analyses": grand_total,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "success_rate": f"{success_rate:.1f}%",
            "avg_response_time": f"{self.stats['avg_response_time']:.0f}ms",
            "cache_size": len(self.analysis_cache),
        }

    async def clear_cache(self):
        """清空缓存"""
        async with self._cache_lock:
            self.analysis_cache.clear()
        logger.info("[情绪分析] 缓存已清空")


class SmartEmotionMatcher:
    """智能情绪匹配器 - 使用自然语言分析"""

    def __init__(self, plugin_instance):
        self.plugin = plugin_instance
        self.natural_analyzer = NaturalEmotionAnalyzer(plugin_instance)

    async def analyze_and_match_emotion(
        self, event: AstrMessageEvent, text: str,
        use_natural_analysis: bool = True,
        *, user_query: str = "",
    ) -> str | None:
        """分析并匹配情绪

        Args:
            event: 消息事件
            text: LLM 回复文本
            use_natural_analysis: 是否使用自然语言分析
            user_query: 用户原始消息，与 text 组成 QA 上下文提升分析准确度

        Returns:
            匹配的情绪分类
        """
        if not text or len(text.strip()) < 3:
            return None

        # 使用自然语言分析（主要方案）
        if use_natural_analysis and self.plugin.enable_natural_emotion_analysis:
            emotion = await self.natural_analyzer.analyze_emotion(
                event, text, user_query=user_query,
            )
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

    async def clear_cache(self):
        """清空分析缓存"""
        await self.natural_analyzer.clear_cache()
