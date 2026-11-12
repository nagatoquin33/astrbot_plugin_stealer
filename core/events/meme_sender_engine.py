"""表情包发送决策引擎：负责 LLM 响应拦截、自动发送决策和表情包发送。"""

import asyncio
import hashlib
import random
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Image as MessageImage

try:
    from astrbot.api.message_components import Record as MessageRecord
except ImportError:
    MessageRecord = None  # type: ignore[assignment]

from ..processing.natural_emotion_analyzer import EmotionQuery
from ..search.jev_selector import JevSelector, JevSelectionError
from ..search.punctuation_emotion import punctuation_emotion_weights
from .event_context import get_event_session_key


class _MemeTurnState:
    """封装单次会话中的表情包发送状态。"""

    def __init__(self) -> None:
        self._active_sent = False
        self._candidates: list[dict] = []
        self._auto_decided = False
        self._auto_allowed = False
        self._auto_reason = ""
        self._auto_send_meme_claimed = False
        self._llm_reply: str | None = None

    def set_llm_reply(self, text: str) -> None:
        """记录本轮用于自动配图的 LLM 回复。"""
        self._llm_reply = text

    def get_llm_reply(self) -> str | None:
        return self._llm_reply

    def invalidate_llm_reply(self) -> None:
        self._llm_reply = None

    def mark_active_sent(self) -> None:
        """标记当前回合已主动发送过表情包。"""
        self._active_sent = True
        if hasattr(self, "_event"):
            self._event.set_extra("stealer_active_sent", True)

    def is_active_sent(self) -> bool:
        """检查当前回合是否已主动发送过表情包。"""
        return self._active_sent

    def set_candidates(self, candidates: list[dict]) -> None:
        """设置候选列表。"""
        self._candidates = candidates

    def get_candidates(self) -> list[dict]:
        """获取候选列表。"""
        return self._candidates

    def is_auto_decided(self) -> bool:
        """检查是否已做出自动决策。"""
        return self._auto_decided

    def set_auto_decision(self, allowed: bool, reason: str = "") -> None:
        """设置自动决策结果。"""
        self._auto_decided = True
        self._auto_allowed = allowed
        self._auto_reason = reason

    def get_auto_allowed(self) -> bool:
        """获取自动决策是否允许。"""
        return self._auto_allowed

    def get_auto_reason(self) -> str:
        """获取自动决策原因。"""
        return self._auto_reason

    def claim_auto_send_meme(self) -> bool:
        """尝试占用自动发送权限。"""
        if self._auto_decided and self._auto_allowed and not self._auto_send_meme_claimed:
            self._auto_send_meme_claimed = True
            return True
        return False

    def is_auto_claimed(self) -> bool:
        """检查是否已占用自动发送权限。"""
        return self._auto_send_meme_claimed

    def reset_for_new_turn(self) -> None:
        """重置回合状态，为新的一轮对话做准备。"""
        self._active_sent = False
        self._candidates = []
        self._auto_decided = False
        self._auto_allowed = False
        self._auto_reason = ""
        self._auto_send_meme_claimed = False
        self._llm_reply = None


class MemeSenderEngine:
    """负责表情包自动发送决策和响应处理。"""

    AUTO_EMOJI_COOLDOWN_SECONDS = 20  # 同一会话自动发表情的最短间隔

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance
        self._auto_emoji_cooldowns: dict[str, float] = {}
        self._auto_emoji_cooldowns_max = 1000  # 最大条目数，防止内存泄漏
        self._auto_emoji_cooldowns_lock = asyncio.Lock()
        self._pending_auto_emoji_tasks: dict[str, asyncio.Task] = {}
        self._pending_auto_emoji_sources: dict[str, tuple[asyncio.Task, Any]] = {}

    # --- 状态管理 ---

    def emoji_turn_state(self, event: AstrMessageEvent) -> _MemeTurnState:
        """获取或创建当前会话的 EmojiTurnState。"""
        key = self.get_auto_emoji_session_key(event)
        if not hasattr(event, "_emoji_turn_state"):
            event._emoji_turn_state = {}  # type: ignore[attr-defined]
        turn_states = event._emoji_turn_state  # type: ignore[attr-defined]
        if key not in turn_states:
            turn_states[key] = _MemeTurnState()
            turn_states[key]._event = event
        return turn_states[key]

    def get_auto_emoji_session_key(self, event: AstrMessageEvent) -> str:
        """获取自动表情会话键。"""
        return get_event_session_key(event)

    def reset_turn_state(self, event: AstrMessageEvent) -> None:
        """重置表情包回合状态及事件 extras，为新的一轮对话做准备。"""
        turn_state = self.emoji_turn_state(event)
        turn_state.reset_for_new_turn()
        for key in (
            "stealer_active_sent",
            "stealer_auto_emoji_turn_decided",
            "stealer_auto_emoji_turn_allowed",
            "stealer_auto_emoji_turn_claimed",
        ):
            try:
                event.set_extra(key, False)
            except Exception:
                pass

    def is_new_message_turn(self, event: AstrMessageEvent) -> bool:
        """仅将新的用户消息视为会中断待发表情的对话轮次。"""
        message = getattr(event, "message_obj", None)
        raw = getattr(message, "raw_message", None)
        if isinstance(raw, dict) and raw.get("post_type") in {"notice", "request", "meta_event", "message_sent"}:
            return False
        sender = str(getattr(getattr(message, "sender", None), "user_id", "") or "").strip()
        self_id = str(getattr(message, "self_id", "") or "").strip()
        if sender and self_id and sender == self_id:
            return False
        key = self.get_auto_emoji_session_key(event)
        source = self._pending_auto_emoji_sources.get(key)
        if source and not source[0].done():
            original = source[1]
            if event is original:
                return False
            original_message = getattr(original, "message_obj", None)
            current_id = str(getattr(message, "message_id", "") or "").strip()
            original_id = str(getattr(original_message, "message_id", "") or "").strip()
            if current_id and original_id and current_id == original_id:
                return False
        return True

    @staticmethod
    def _message_debug_id(event: Any) -> str:
        message_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "").strip()
        return hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:12] if message_id else "unknown"

    def cancel_pending_auto_emoji(self, event: AstrMessageEvent, reason: str = "new_message") -> bool:
        """取消当前会话尚未发出的自动表情任务。"""
        key = self.get_auto_emoji_session_key(event)
        task = self._pending_auto_emoji_tasks.pop(key, None)
        source = self._pending_auto_emoji_sources.pop(key, None)
        if task and not task.done():
            task.cancel()
            raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
            raw_type = raw.get("post_type") if isinstance(raw, dict) else None
            event_type = raw_type if raw_type in {"message", "message_sent", "notice", "request", "meta_event"} else "unknown"
            logger.debug(
                f"[MemeSenderEngine] 取消待发送自动表情: session={key}, reason={reason}, "
                f"event_type={event_type}, source_id={self._message_debug_id(source[1]) if source else 'unknown'}, "
                f"event_id={self._message_debug_id(event)}"
            )
            return True
        return False

    def schedule_auto_emoji_task(
        self,
        event: AstrMessageEvent,
        task: asyncio.Task | None,
    ) -> asyncio.Task | None:
        """记录当前会话的自动表情任务，并替换掉旧任务。"""
        if task is None:
            return None
        key = self.get_auto_emoji_session_key(event)
        old_task = self._pending_auto_emoji_tasks.get(key)
        if old_task and old_task is not task and not old_task.done():
            old_task.cancel()
        self._pending_auto_emoji_tasks[key] = task
        self._pending_auto_emoji_sources[key] = (task, event)

        def _clear(done_task: asyncio.Task) -> None:
            if self._pending_auto_emoji_tasks.get(key) is done_task:
                self._pending_auto_emoji_tasks.pop(key, None)
                self._pending_auto_emoji_sources.pop(key, None)

        task.add_done_callback(_clear)
        return task

    # --- 决策检查 ---

    @staticmethod
    def is_command_event(event: AstrMessageEvent) -> bool:
        """使用框架解析的命令参数映射识别已触发的插件命令。"""
        getter = getattr(event, "get_extra", None)
        params = getter("handlers_parsed_params") if callable(getter) else None
        return isinstance(params, dict) and bool(params)

    @staticmethod
    def is_llm_reply_result(result: Any) -> bool:
        """接受普通 LLM 回复及框架标记的流式回复。"""
        checker = getattr(result, "is_llm_result", None)
        if callable(checker) and checker():
            return True
        content_type = getattr(result, "result_content_type", None)
        name = getattr(content_type, "name", str(content_type).rsplit(".", 1)[-1])
        return name in {"STREAMING_RESULT", "STREAMING_FINISH"}

    def invalidate_auto_emoji_source(self, event: AstrMessageEvent) -> None:
        """取消本轮已失去 LLM 来源的配图任务。"""
        state = self.emoji_turn_state(event)
        state.invalidate_llm_reply()
        if state.is_auto_claimed():
            self.cancel_pending_auto_emoji(event, reason="reply_source_changed")

    @staticmethod
    def _is_record_component(comp: Any) -> bool:
        if MessageRecord is not None and isinstance(comp, MessageRecord):
            return True
        comp_type = getattr(comp, "type", None)
        type_name = getattr(comp_type, "name", str(comp_type or ""))
        return type_name.lower() == "record"

    def is_auto_emoji_source_current(self, event: AstrMessageEvent, text: str) -> bool:
        """在异步分析和发送边界复核本轮回复的来源。"""
        state = self.emoji_turn_state(event)
        if state.get_llm_reply() != text or self.is_command_event(event):
            return False
        result = event.get_result()
        # 框架发送完主回复后会清空结果；本轮已记录的来源继续有效。
        if result is None:
            return True
        if not self.is_llm_reply_result(result):
            state.invalidate_llm_reply()
            return False
        chain = getattr(result, "chain", None) or []
        if any(isinstance(comp, MessageImage) for comp in chain):
            return False
        content_type = getattr(result, "result_content_type", None)
        name = getattr(content_type, "name", str(content_type).rsplit(".", 1)[-1])
        return (
            bool(result.get_plain_text().strip())
            or any(self._is_record_component(comp) for comp in chain)
            or name == "STREAMING_RESULT"
        )

    def should_skip_auto_emoji_by_gate(self, text: str) -> bool:
        """根据文本内容判断是否跳过自动发送。"""
        if not text:
            return True
        if not self.plugin.plugin_config.auto_meme_intent_gate:
            return False
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= 2:
            return True
        # 如果包含明确的指令或标记，跳过自动发送
        skip_patterns = [
            r"/meme\s+\w+",
            r"^\\/",
            r"^\s*[!！/#]",
        ]
        for pattern in skip_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        # 仅拦截明确的道歉、澄清和技术故障；一般否定词交由语境选图判断。
        serious_patterns = (
            r"^(?:很|非常|实在)?(?:抱歉|对不起)[，,。.!！\s]",
            r"^(?:请提供|需要更多)(?:信息|详情|上下文|参数)",
            r"(?:发生|出现|遇到)(?:了)?(?:错误|异常)",
            r"(?:请求|连接|下载|上传|调用|执行|验证|认证)(?:发生)?(?:失败|错误|异常)",
            r"(?:无法|不能)(?:完成|执行|访问|连接|处理|读取|提供)",
            r"\b(?:traceback|exception|error\s*[:：]|http\s*[45]\d\d)\b",
        )
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in serious_patterns):
            return True
        return False

    async def is_auto_emoji_cooldown_ready(self, event: AstrMessageEvent) -> bool:
        """检查自动表情冷却是否就绪。"""
        key = self.get_auto_emoji_session_key(event)
        now = asyncio.get_event_loop().time()
        async with self._auto_emoji_cooldowns_lock:
            last = self._auto_emoji_cooldowns.get(key, 0)
            return now - last >= self.AUTO_EMOJI_COOLDOWN_SECONDS

    def normalize_auto_meme_chance(self) -> float:
        """归一化自动表情发送概率。"""
        try:
            chance = float(self.plugin.plugin_config.meme_chance)
        except (TypeError, ValueError):
            chance = 0.4
        return max(0.0, min(1.0, chance))

    async def resolve_auto_emoji_turn_permission(self, event: AstrMessageEvent) -> bool:
        """解析自动表情发送权限。"""
        turn_state = self.emoji_turn_state(event)

        if turn_state.is_auto_decided():
            return turn_state.get_auto_allowed()

        def decide(allowed: bool, reason: str) -> bool:
            event.set_extra("stealer_auto_emoji_turn_decided", True)
            event.set_extra("stealer_auto_emoji_turn_allowed", allowed)
            event.set_extra("stealer_auto_emoji_turn_reason", reason)
            turn_state.set_auto_decision(allowed, reason)
            return allowed

        if not self.plugin.plugin_config.auto_send_meme:
            return decide(False, "auto_send_meme_disabled")
        if not self.plugin.is_send_enabled_for_event(event):
            return decide(False, "send_disabled")
        if not await self.is_auto_emoji_cooldown_ready(event):
            return decide(False, "cooldown")
        chance = self.normalize_auto_meme_chance()
        if chance <= 0:
            return decide(False, "chance_zero")
        if chance >= 1:
            return decide(True, "chance_hit")
        if random.random() < chance:
            return decide(True, "chance_hit")
        return decide(False, "chance_miss")

    async def _resolve_with_log(self, event: AstrMessageEvent) -> bool:
        """带日志的权限判断包装，方便排查发送概率问题。"""
        already_decided = self.emoji_turn_state(event).is_auto_decided()
        allowed = await self.resolve_auto_emoji_turn_permission(event)
        if not already_decided:
            reason = event.get_extra("stealer_auto_emoji_turn_reason") or "unknown"
            logger.debug(f"[Stealer] 当前轮次自动发表情判定: allowed={allowed}, reason={reason}")
        return allowed

    def claim_auto_emoji_turn(self, event: AstrMessageEvent) -> bool:
        """尝试占用当前回合的表情包发送权。"""
        turn_state = self.emoji_turn_state(event)
        if event.get_extra("stealer_auto_emoji_turn_claimed"):
            return False
        if turn_state.is_active_sent():
            return False
        if not turn_state.claim_auto_send_meme():
            return False
        event.set_extra("stealer_auto_emoji_turn_claimed", True)
        return True

    def prune_auto_emoji_cooldowns(self, now: float) -> None:
        """清理过期的自动表情冷却记录。"""
        cutoff = now - self.AUTO_EMOJI_COOLDOWN_SECONDS * 2
        expired = [k for k, v in self._auto_emoji_cooldowns.items() if v < cutoff]
        for k in expired:
            del self._auto_emoji_cooldowns[k]

    async def mark_auto_emoji_sent(self, event: AstrMessageEvent) -> None:
        """标记已发送自动表情。"""
        key = self.get_auto_emoji_session_key(event)
        now = asyncio.get_event_loop().time()
        async with self._auto_emoji_cooldowns_lock:
            self.prune_auto_emoji_cooldowns(now)
            self._auto_emoji_cooldowns[key] = now

    # --- 发送 ---

    async def try_send_emoji(
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str,
        *, emotion_weights: dict[str, float] | None = None,
    ) -> bool:
        """尝试发送表情包（委托给 MemeSelector）。"""
        try:
            selector = getattr(self.plugin, "meme_selector", None)
            if selector is None:
                return False
            options = {"emotion_weights": emotion_weights} if emotion_weights else {}
            return await selector.try_send_emoji(event, emotions, cleaned_text, **options)
        except Exception as e:
            logger.warning(f"[MemeSenderEngine] 尝试发送表情包失败: {e}")
            return False

    def get_meme_send_delay(self, text: str = "", task_start: float = 0.0) -> float:
        """获取表情包发送延迟（秒）。"""
        char_delay = self.plugin.plugin_config.meme_send_char_delay
        if char_delay > 0 and text:
            desired = len(text) * char_delay
            if task_start > 0:
                elapsed = asyncio.get_event_loop().time() - task_start
                return max(0.0, desired - elapsed)
            return desired
        delay = self.plugin.plugin_config.meme_send_delay
        try:
            base = float(delay)
        except (TypeError, ValueError):
            base = 0.5
        # meme_send_delay_random (bool) 开启时在 [base, base+max] 之间随机
        use_random = bool(self.plugin.plugin_config.meme_send_delay_random)
        if use_random:
            delay_max = self.plugin.plugin_config.meme_send_delay_max
            try:
                rand_max = float(delay_max)
            except (TypeError, ValueError):
                rand_max = 8.0
            if rand_max > 0:
                return base + random.random() * rand_max
        return base

    async def async_analyze_and_send_emoji(
        self,
        event: AstrMessageEvent,
        text: str,
        emotions: list[str],
        *,
        user_message: str = "",
    ):
        """异步分析并发送表情包。

        调用方 _prepare_emoji_response 已通过 claim_auto_emoji_turn
        占用发送权（_auto_send_meme_claimed），防止重复创建任务。
        本方法成功发送后再标记 mark_active_sent 以记录实际发送时间。
        """
        try:
            if not self.is_auto_emoji_source_current(event, text):
                return
            task_start = asyncio.get_event_loop().time()
            final_emotions = list(emotions or [])

            jev_path = None
            use_jev = JevSelector.enabled(self.plugin.plugin_config)
            if use_jev:
                try:
                    jev_path = await self.plugin.meme_selector.select_emoji_with_jev(
                        event, text, user_message=user_message,
                    )
                except JevSelectionError as exc:
                    logger.warning(f"[JEV] 选图失败，回退小模型链路: {exc}")
                    use_jev = False
                else:
                    if jev_path is None:
                        return

            search_text = text
            if not use_jev and self.plugin.plugin_config.enable_natural_emotion_analysis and hasattr(
                self.plugin, "emotion_analyzer"
            ):
                analyzed = await self.plugin.emotion_analyzer.analyze_for_reply(
                    event,
                    text,
                    user_message=user_message,
                )
                if isinstance(analyzed, EmotionQuery):
                    # 从语境分析结果提取检索词和情绪先验。
                    if analyzed.emotion_priors:
                        final_emotions = analyzed.emotion_priors
                    if analyzed.search_query:
                        search_text = analyzed.search_query
                elif analyzed:
                    final_emotions = [analyzed]

            if not final_emotions and not search_text:
                return

            if not self.is_auto_emoji_source_current(event, text):
                return

            delay = self.get_meme_send_delay(text, task_start)
            if delay > 0:
                await asyncio.sleep(delay)

            if not self.is_auto_emoji_source_current(event, text):
                logger.debug("[MemeSenderEngine] 回复来源已变化，跳过自动表情")
                return

            if use_jev:
                sent = await self.plugin.meme_selector.send_jev_selection(event, jev_path, text)
            else:
                weights = punctuation_emotion_weights(text, final_emotions)
                options = {"emotion_weights": weights} if weights else {}
                sent = await self.try_send_emoji(event, final_emotions, search_text or text, **options)
            if sent:
                await self.mark_auto_emoji_sent(event)
                self.emoji_turn_state(event).mark_active_sent()
        except asyncio.CancelledError:
            logger.debug("[MemeSenderEngine] 自动表情任务已取消")
            raise
        except Exception as e:
            logger.warning(f"[MemeSenderEngine] 异步分析发送表情包失败: {e}")
