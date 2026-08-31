"""
自然语言情绪分析器
使用小模型对LLM回复进行语义分析，识别隐含情绪
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..search.text_similarity import calculate_hybrid_similarity, _has_negation_prefix

# 标记：模型明确选择不发送，用于区分"LLM 异常"和"模型说 none"
_EMOTION_ABSTAIN = object()

_EMOTION_ANALYSIS_DEFAULT_TEMPLATE = (
    "你是表情包检索词提取器。根据对话从 AI 回复中提取能搜到表情包的关键词或短句，并给出 1~3 个情绪先验。\n"
    "可选情绪分类：{emotion_list}\n"
    "\n"
    "任务：\n"
    "1. 只从回复里摘录关键词或短句，不要总结成抽象概念。\n"
    "2. 如果没有明显关键词，就用回复原文。\n"
    "3. emotions 从给定分类里选 1~3 个，按相关度排序。每个分类机会均等，不要把某一类当默认。\n"
    "\n"
    "用户消息：{user_message}\n"
    "回复：{llm_reply}\n"
    "\n"
    "输出示例：\n"
    "{\"query\": \"摸鱼 下班 辛苦了\", \"emotions\": [\"tired\", \"sigh\"]}\n"
)


@dataclass
class EmotionQuery:
    """查询改写结果：检索句 + 情绪先验。分类只作加分，不再是唯一键。"""

    should_send: bool = True
    search_query: str = ""
    emotion_priors: list[str] = field(default_factory=list)

    @property
    def primary(self) -> str | None:
        return self.emotion_priors[0] if self.emotion_priors else None

    def to_cache(self) -> dict[str, Any]:
        return {
            "s": self.should_send,
            "q": self.search_query,
            "e": list(self.emotion_priors),
        }

    @classmethod
    def from_cache(cls, value: Any) -> EmotionQuery | None:
        if isinstance(value, cls):
            return value
        if isinstance(value, str) and value:
            return cls(True, value, [value])
        if isinstance(value, dict):
            priors = [str(item) for item in (value.get("e") or []) if item]
            query = str(value.get("q") or "").strip()
            should = bool(value.get("s", True))
            if not should:
                return cls(False, "", [])
            if query or priors:
                return cls(should, query, priors)
        return None


class NaturalEmotionAnalyzer:
    """自然语言情绪分析器 - 使用小模型理解LLM回复的真实情绪"""

    # 常量定义
    CACHE_MAX_SIZE = 1000  # 缓存最大容量
    TEXT_MAX_LENGTH = 200  # 文本最大长度

    def __init__(self, plugin_instance: Any):
        self.plugin = plugin_instance
        # 标记：上一次分析是否为模型主动 abstain
        self.last_analysis_abstained: bool = False
        # v2.7.5+：配置统一通过 plugin_config 读取
        self.plugin_config = plugin_instance.plugin_config
        self.categories: list[str] = self.plugin_config.get_categories()

        # 缓存机制
        self.analysis_cache: dict[str, Any] = {}
        self.cache_max_size: int = self.CACHE_MAX_SIZE
        self._cache_lock = asyncio.Lock()

        # 性能统计
        self.stats: dict[str, float | int] = {
            "total_analyses": 0,
            "cache_hits": 0,
            "avg_response_time": 0,
            "successful_analyses": 0,
        }

        # 小模型提示词模板（从配置加载）
        self._emotion_analysis_template = self._load_emotion_analysis_template()

    def _load_emotion_analysis_template(self) -> str:
        """从配置加载情绪分析提示词；为空时回退到内置模板。"""
        custom = getattr(self.plugin_config, "emotion_analysis_prompt", "") or ""
        if custom.strip():
            return str(custom).strip()
        return _EMOTION_ANALYSIS_DEFAULT_TEMPLATE

    def _build_emotion_list_text(self) -> str:
        """构建分类描述文本（紧凑格式，供模板中 {emotion_list} 替换用）"""
        info_map = self.plugin_config.category_info or {}
        parts = []
        for key in self.categories:
            info = info_map.get(key, {})
            if isinstance(info, dict):
                name = str(info.get("name", "")).strip()
                desc = str(info.get("desc", "")).strip()
                desc_text = desc or name or key
            else:
                desc_text = key
            parts.append(f"{key}({desc_text})")
        return ", ".join(parts) if parts else ", ".join(self.categories)

    async def analyze_emotion(
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        *,
        user_message: str = "",
    ) -> EmotionQuery | None:
        """分析文本并生成表情检索查询。

        Returns:
            EmotionQuery；失败或不该发送时返回 None（abstain 通过 last_analysis_abstained 区分）
        """
        if not llm_reply or len(llm_reply.strip()) < 3:
            return None

        # 清理文本
        cleaned_reply = self._clean_text(llm_reply)
        if not cleaned_reply:
            return None

        # 清理用户消息（用于缓存 key 和 prompt）
        cleaned_msg = self._clean_text(user_message) if user_message else ""

        # 检查缓存（缓存 key 同时包含用户消息和回复）
        cache_key = self._get_cache_key(cleaned_msg + "|||" + cleaned_reply)
        async with self._cache_lock:
            if cache_key in self.analysis_cache:
                self.stats["cache_hits"] += 1
                logger.debug(f"[情绪分析] 缓存命中: {cleaned_reply[:30]}...")
                cached = EmotionQuery.from_cache(self.analysis_cache[cache_key])
                if cached:
                    return cached

        # 本地预匹配：先用分词匹配关键词映射（快速路径）
        local_match = self._local_keyword_match(cleaned_reply)
        if local_match:
            query = EmotionQuery(True, cleaned_reply, [local_match])
            logger.debug(f"[情绪分析] 本地匹配: {cleaned_reply[:30]}... → {local_match}")
            async with self._cache_lock:
                self._cache_result(cache_key, query.to_cache())
            return query

        # 执行 LLM 分析（传入用户消息作为上下文）
        start_time = time.time()
        parsed = await self._analyze_with_llm(
            event,
            cleaned_reply,
            user_message=cleaned_msg,
        )
        end_time = time.time()

        if parsed is _EMOTION_ABSTAIN:
            self.last_analysis_abstained = True
            return None
        self.last_analysis_abstained = False

        self.stats["total_analyses"] += 1
        response_time = (end_time - start_time) * 1000
        query = parsed if isinstance(parsed, EmotionQuery) else None
        if query is None and isinstance(parsed, str):
            query = EmotionQuery(True, cleaned_reply, [parsed])
        if query is None:
            fallback = self._local_keyword_match(cleaned_reply, fallback=True)
            if fallback:
                query = EmotionQuery(True, cleaned_reply, [fallback])
                logger.debug(f"[情绪分析] LLM失败，降级匹配: {cleaned_reply[:30]}... → {fallback}")

        self._update_stats(response_time, query is not None)

        if query:
            if not query.search_query:
                query.search_query = cleaned_reply
            async with self._cache_lock:
                self._cache_result(cache_key, query.to_cache())
            logger.info(
                f"[情绪分析] {cleaned_reply[:30]}... → {query.emotion_priors} "
                f"q='{query.search_query[:40]}' ({response_time:.0f}ms)"
            )
        else:
            logger.warning(f"[情绪分析] 分析失败: {cleaned_reply[:30]}...")

        return query

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

        # 1. 精确匹配关键词（带否定词检测）
        for keyword, category in keyword_map.items():
            if keyword in text_lower and category:
                # 检查关键词前是否有否定词
                if _has_negation_prefix(text_lower, keyword):
                    continue  # 跳过被否定的关键词
                return category

        # 2. 分词匹配（降级模式下执行）
        if fallback:
            # 检查文本中是否有否定词+关键词的组合（语义反转）
            has_negated_emotion = any(
                _has_negation_prefix(text_lower, keyword) for keyword in keyword_map.keys()
            )

            # 如果存在否定词反转语义，不进行降级匹配
            if has_negated_emotion:
                return None

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

    @staticmethod
    def _render_emotion_analysis_template(
        template: str,
        *,
        emotion_list: str,
        llm_reply: str,
        user_message: str,
    ) -> str:
        """仅替换受支持的占位符，保留 JSON 花括号和未知占位符原样输出。"""
        mapping = {
            "emotion_list": emotion_list,
            "llm_reply": llm_reply,
            "user_message": user_message,
        }

        # 先保护 Python format 风格的转义花括号，避免替换到 {{placeholder}}。
        template = template.replace("{{", "\x00").replace("}}", "\x01")

        def _replace(match: re.Match[str]) -> str:
            return mapping.get(match.group(1), match.group(0))

        prompt = re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", _replace, template)
        return prompt.replace("\x00", "{").replace("\x01", "}")

    async def _analyze_with_llm(
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        *,
        user_message: str = "",
    ) -> EmotionQuery | object | None:
        """使用小模型生成检索查询。兼容旧的单分类名输出。"""
        try:
            # 获取文本模型提供商（优先使用配置的小模型）
            provider_id = await self._get_text_provider(event)
            if not provider_id:
                logger.warning("[情绪分析] 未找到可用的文本模型")
                return None

            # 构建提示词：用分类列表替换 {emotion_list}，再填入具体文本
            emotion_list = self._build_emotion_list_text()
            prompt = self._render_emotion_analysis_template(
                self._emotion_analysis_template,
                emotion_list=emotion_list,
                llm_reply=llm_reply,
                user_message=user_message if user_message else "",
            )

            # 调用LLM（限制 max_tokens 提升速度）
            logger.debug(f"[情绪分析] 调用LLM，provider_id={provider_id}")
            response = await self.plugin.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                max_tokens=120,
            )

            # 安全获取响应文本
            if not response:
                logger.warning("[情绪分析] LLM返回空响应")
                return None
            result_text = response.completion_text
            if not result_text:
                logger.warning("[情绪分析] LLM返回空文本")
                return None

            return self._parse_emotion_query(result_text, fallback_query=llm_reply)

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
        configured_provider = self.plugin.plugin_config.emotion_analysis_provider_id
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

        # 仅压缩空白；插件已不再注入或解析 &&emotion&& 标签。
        cleaned = re.sub(r"\s+", " ", text.strip())

        # 限制长度（小模型处理能力有限）
        if len(cleaned) > self.TEXT_MAX_LENGTH:
            cleaned = cleaned[: self.TEXT_MAX_LENGTH] + "..."

        return cleaned

    def _parse_emotion_query(self, result_text: str, *, fallback_query: str) -> EmotionQuery | object | None:
        """解析 LLM 输出：优先 JSON 查询改写，兼容旧的单分类名。"""
        if not result_text:
            return None
        stripped = result_text.strip()
        lowered = stripped.lower()
        if re.match(r"^none[\s:：,，.。!！?？]*$", lowered):
            logger.info("[情绪分析] 模型输出 none，跳过发送")
            return _EMOTION_ABSTAIN

        data = self._extract_json_object(stripped)
        if isinstance(data, dict):
            # 旧输出可能带 should_send，但小模型不再负责“是否发表情”，这里忽略该字段。
            query = str(data.get("query") or data.get("search_query") or "").strip()
            raw_emotions = data.get("emotions") or data.get("emotion") or []
            if isinstance(raw_emotions, str):
                raw_emotions = [raw_emotions]
            priors: list[str] = []
            seen: set[str] = set()
            for item in raw_emotions:
                mapped = self._map_category(str(item))
                if mapped and mapped not in seen:
                    seen.add(mapped)
                    priors.append(mapped)
                if len(priors) >= 3:
                    break
            if not priors:
                mapped = self._parse_emotion_result(query or lowered)
                if mapped and mapped is not _EMOTION_ABSTAIN:
                    priors = [mapped]
            return EmotionQuery(True, query or fallback_query, priors)

        emotion = self._parse_emotion_result(lowered)
        if emotion is _EMOTION_ABSTAIN or emotion is None:
            return emotion
        return EmotionQuery(True, fallback_query, [emotion])

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    def _map_category(self, raw: str) -> str | None:
        text = str(raw or "").strip().lower()
        if not text:
            return None
        if text in self.categories:
            return text
        cfg = self.plugin.plugin_config
        if cfg:
            try:
                return cfg.normalize_category_strict(text)
            except Exception:
                return None
        return None

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

        # 模型选择不发送
        if re.match(r"^none[\s:：,，.。!！?？]*$", result):
            logger.info("[情绪分析] 模型输出 none，跳过发送")
            return _EMOTION_ABSTAIN  # type: ignore

        cfg = self.plugin.plugin_config

        # 尝试从文本中提取已知的分类名
        # 优先匹配完整的单词（避免部分匹配如 "sad" 匹配到 "sadness"）
        for category in self.categories:
            # 检查是否是完整单词匹配（前后是边界或标点）
            pattern = (
                r"(?:^|[\s:：,，.。!！?？])" + re.escape(category) + r"(?:$|[\s:：,，.。!！?？])"
            )
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

        logger.debug(f"[情绪分析] 无法从回复中解析分类: '{result_text}'")
        return None

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _cache_result(self, cache_key: str, emotion: Any):
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
        self.stats["avg_response_time"] = (current_avg * (total - 1) + response_time) / total

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
        success_rate = (self.stats["successful_analyses"] / total) * 100 if total > 0 else 0.0

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
        self,
        event: AstrMessageEvent,
        llm_reply: str,
        use_natural_analysis: bool = True,
        *,
        user_message: str = "",
    ) -> EmotionQuery | None:
        """分析并匹配情绪

        Args:
            event: 消息事件
            llm_reply: LLM 回复文本
            use_natural_analysis: 是否使用自然语言分析
            user_message: 用户原始消息，与 llm_reply 组成对话上下文提升分析准确度

        Returns:
            EmotionQuery 或 None
        """
        if not llm_reply or len(llm_reply.strip()) < 3:
            return None

        if use_natural_analysis and self.plugin.plugin_config.enable_natural_emotion_analysis:
            emotion = await self.natural_analyzer.analyze_emotion(
                event,
                llm_reply,
                user_message=user_message,
            )
            if emotion:
                return emotion
            if self.natural_analyzer.last_analysis_abstained:
                return None
            logger.debug(f"[智能匹配] 自然语言分析失败: {llm_reply[:30]}...")
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
