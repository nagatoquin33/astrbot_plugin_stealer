"""表情包发送决策引擎：负责 LLM 响应拦截、自动发送决策和表情包发送。"""

import asyncio
import random
import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class _EmojiTurnState:
    """封装单次会话中的表情包发送状态。"""

    def __init__(self) -> None:
        self._active_sent = False
        self._candidates: list[dict] = []
        self._auto_decided = False
        self._auto_allowed = False
        self._auto_reason = ""
        self._auto_send_claimed = False

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

    def claim_auto_send(self) -> bool:
        """尝试占用自动发送权限。"""
        if self._auto_decided and self._auto_allowed and not self._auto_send_claimed:
            self._auto_send_claimed = True
            return True
        return False

    def is_auto_claimed(self) -> bool:
        """检查是否已占用自动发送权限。"""
        return self._auto_send_claimed

    def reset_for_new_turn(self) -> None:
        """重置回合状态，为新的一轮对话做准备。"""
        self._active_sent = False
        self._candidates = []
        self._auto_decided = False
        self._auto_allowed = False
        self._auto_reason = ""
        self._auto_send_claimed = False


class EmojiSenderEngine:
    """负责表情包自动发送决策、情绪注入和响应处理。"""

    AUTO_EMOJI_COOLDOWN_SECONDS = 20  # 同一会话自动发表情的最短间隔

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance
        self._auto_emoji_cooldowns: dict[str, float] = {}
        self._auto_emoji_cooldowns_max = 1000  # 最大条目数，防止内存泄漏
        self._auto_emoji_cooldowns_lock = asyncio.Lock()

    # --- 状态管理 ---

    def emoji_turn_state(self, event: AstrMessageEvent) -> _EmojiTurnState:
        """获取或创建当前会话的 EmojiTurnState。"""
        key = self.get_auto_emoji_session_key(event)
        if not hasattr(event, "_emoji_turn_state"):
            event._emoji_turn_state = {}  # type: ignore[attr-defined]
        turn_states = event._emoji_turn_state  # type: ignore[attr-defined]
        if key not in turn_states:
            turn_states[key] = _EmojiTurnState()
            turn_states[key]._event = event
        return turn_states[key]

    def get_auto_emoji_session_key(self, event: AstrMessageEvent) -> str:
        """获取自动表情会话键。"""
        session_id = ""
        if hasattr(event, "get_session_id"):
            try:
                session_id = str(event.get_session_id())
            except Exception:
                pass
        if not session_id and hasattr(event, "unified_msg_origin"):
            try:
                session_id = str(event.unified_msg_origin)
            except Exception:
                pass
        return session_id or "global"

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

    # --- 决策检查 ---

    def should_skip_auto_emoji_by_gate(self, text: str) -> bool:
        """根据文本内容判断是否跳过自动发送。"""
        if not text:
            return True
        # 如果包含明确的指令或标记，跳过自动发送
        skip_patterns = [
            r"/meme\s+\w+",
            r"^\\/",
        ]
        for pattern in skip_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    async def is_auto_emoji_cooldown_ready(self, event: AstrMessageEvent) -> bool:
        """检查自动表情冷却是否就绪。"""
        key = self.get_auto_emoji_session_key(event)
        now = asyncio.get_event_loop().time()
        async with self._auto_emoji_cooldowns_lock:
            last = self._auto_emoji_cooldowns.get(key, 0)
            return now - last >= self.AUTO_EMOJI_COOLDOWN_SECONDS

    def normalize_auto_emoji_chance(self) -> float:
        """归一化自动表情发送概率。"""
        try:
            chance = float(getattr(self.plugin, "emoji_chance", 0.4))
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

        if not getattr(self.plugin, "auto_send", False):
            return decide(False, "auto_send_disabled")
        if not self.plugin.is_send_enabled_for_event(event):
            return decide(False, "send_disabled")
        if not await self.is_auto_emoji_cooldown_ready(event):
            return decide(False, "cooldown")
        chance = self.normalize_auto_emoji_chance()
        if chance <= 0:
            return decide(False, "chance_zero")
        if chance >= 1:
            return decide(True, "chance_hit")
        if random.random() < chance:
            return decide(True, "chance_hit")
        return decide(False, "chance_miss")

    def claim_auto_emoji_turn(self, event: AstrMessageEvent) -> bool:
        """尝试占用当前回合的表情包发送权。"""
        turn_state = self.emoji_turn_state(event)
        if event.get_extra("stealer_auto_emoji_turn_claimed"):
            return False
        if turn_state.is_active_sent():
            return False
        if not turn_state.claim_auto_send():
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
        self, event: AstrMessageEvent, emotions: list[str], cleaned_text: str
    ) -> bool:
        """尝试发送表情包。"""
        try:
            selector = getattr(self.plugin, "emoji_selector", None)
            if selector is None:
                return False
            if hasattr(selector, "try_send_emoji"):
                return await selector.try_send_emoji(event, emotions, cleaned_text)


            # 提取情绪
            emotion = emotions[0] if emotions else "default"
            emoji_path = await selector.select_emoji(emotion, cleaned_text, event)
            if not emoji_path:
                return False

            # 发送
            return await self.send_explicit_emojis(event, [emoji_path], cleaned_text)
        except Exception as e:
            logger.debug(f"[EmojiSenderEngine] 尝试发送表情包失败: {e}")
            return False

    async def send_explicit_emojis(
        self, event: AstrMessageEvent, emoji_paths: list[str], cleaned_text: str
    ) -> bool:
        """发送指定的表情包。"""
        from astrbot.api.message_components import Image

        if not emoji_paths:
            return False

        sent = False
        for path in emoji_paths:
            try:
                await event.send(Image(file=path))
                sent = True
            except Exception as e:
                logger.warning(f"[EmojiSenderEngine] 发送表情包失败: {e}")
        return sent

    def get_emoji_send_delay(self) -> float:
        """获取表情包发送延迟（秒）。"""
        delay = getattr(self.plugin, "emoji_send_delay", 0.5)
        delay_random = getattr(self.plugin, "emoji_send_delay_random", 0.0)
        try:
            base = float(delay)
        except (TypeError, ValueError):
            base = 0.5
        try:
            rand = float(delay_random)
        except (TypeError, ValueError):
            rand = 0.0
        if rand > 0:
            return base + random.random() * rand
        return base

    async def async_analyze_and_send_emoji(
        self,
        event: AstrMessageEvent,
        text: str,
        emotions: list[str],
        *,
        user_query: str = "",
    ):
        """异步分析并发送表情包。

        调用方 _prepare_emoji_response 已通过 claim_auto_emoji_turn
        占用发送权（_auto_send_claimed），防止重复创建任务。
        本方法成功发送后再标记 mark_active_sent 以记录实际发送时间。
        """
        try:
            final_emotions = list(emotions or [])

            if getattr(self.plugin, "enable_natural_emotion_analysis", False) and hasattr(
                self.plugin, "smart_emotion_matcher"
            ):
                analyzed = await self.plugin.smart_emotion_matcher.analyze_and_match_emotion(
                    event,
                    text,
                    use_natural_analysis=True,
                    user_query=user_query,
                )
                if analyzed:
                    final_emotions = [analyzed]

            if not final_emotions:
                return

            delay = self.get_emoji_send_delay()
            if delay > 0:
                await asyncio.sleep(delay)

            sent = await self.try_send_emoji(event, final_emotions, text)
            if sent:
                await self.mark_auto_emoji_sent(event)
                self.emoji_turn_state(event).mark_active_sent()
        except Exception as e:
            logger.debug(f"[EmojiSenderEngine] 异步分析发送表情包失败: {e}")

    # --- 结果处理 ---

    def validate_result(self, result) -> bool:
        """验证结果是否有效。"""
        if result is None:
            return False
        return True

    def update_result_with_cleaned_text_safe(
        self, event: AstrMessageEvent, result, cleaned_text: str
    ):
        """安全地更新结果中的清理后文本。"""
        try:
            if hasattr(result, "cleaned_text"):
                result.cleaned_text = cleaned_text
            elif hasattr(result, "result"):
                result.result = cleaned_text
        except Exception as e:
            logger.debug(f"[EmojiSenderEngine] 更新结果文本失败: {e}")
