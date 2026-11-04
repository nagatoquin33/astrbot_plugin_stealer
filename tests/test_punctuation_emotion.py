"""标点线索、相关度边界及原回复到评分链路的回归。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.events.meme_sender_engine import MemeSenderEngine
from core.processing.natural_emotion_analyzer import EmotionQuery
from core.search.meme_selector import MemeSelector
from core.search.punctuation_emotion import punctuation_emotion_weights
from tests.test_auto_emoji_flow import DummyEvent
from tests.test_matching_usability import selector_for


@pytest.mark.parametrize("text,category", [
    ("你在说什么？？", "confused"),
    ("居然成功了？！", "surprised"),
    ("终于放假了！！", "excitement"),
    ("又要加班了……", "sigh"),
    ("喜欢你呀～", "love"),
])
def test_punctuation_supplies_bounded_tone_weights(text, category):
    weights = punctuation_emotion_weights(text)
    assert 0 < weights[category] <= 0.06
    assert all(0 < value <= 0.06 for value in weights.values())


@pytest.mark.parametrize("text", [
    "看这个链接 https://example.test/path?q=!!",
    "示例：`what??`", "代码：```\nwhat!!\n```",
    "他说“怎么回事？？”，我知道了", '他说"怎么回事!!"，我知道了',
    "原文：\n> 你在说什么？？\n我知道了", "版本号 1.2.3", "???!!!~~...",
])
def test_non_speech_punctuation_does_not_supply_tone_weights(text):
    assert punctuation_emotion_weights(text) == {}


def test_emphasis_respects_existing_emotion_and_saturates():
    assert "angry" not in punctuation_emotion_weights("太好了！！")
    assert punctuation_emotion_weights("气死我了！！", ["angry"]) == {"angry": 0.05}
    assert punctuation_emotion_weights("好难过！", ["sad"]) == {}
    assert punctuation_emotion_weights("好耶！" + "!" * 100) == punctuation_emotion_weights("好耶！！")


@pytest.mark.asyncio
async def test_category_weight_reranks_relevant_candidates_and_keeps_relevance_floor(tmp_path):
    index = {
        "happy": {"category": "happy", "overlay_text": "你在说什么", "desc": "回应朋友"},
        "confused": {"category": "confused", "overlay_text": "你在说什么", "desc": "回应朋友"},
        "unrelated": {"category": "confused", "desc": "晚安睡觉"},
    }
    service = selector_for(index, tmp_path)._smart_select_service
    query = "你在说什么"
    plain = await service._rank_emoji_candidates("", query, idx=index, penalize_recent=False)
    weighted = await service._rank_emoji_candidates(
        "", query, idx=index, penalize_recent=False,
        emotion_weights=punctuation_emotion_weights(query + "？？"),
    )
    before = {item[0]: item[1] for item in plain}
    after = {item[0]: item[1] for item in weighted}
    assert weighted[0][0] == "confused"
    assert after["confused"] - before["confused"] == pytest.approx(0.06)
    assert after["happy"] == before["happy"]
    assert "unrelated" not in after
    assert await service._rank_emoji_candidates(
        "", "量子纠缠公式", idx=index, emotion_weights={"confused": 0.06},
    ) == []


@pytest.mark.asyncio
async def test_multi_emotion_candidate_receives_one_bonus(tmp_path):
    index = {"both": {"category": "confused", "emotions": ["confused", "surprised"],
                      "overlay_text": "你在说什么"}}
    service = selector_for(index, tmp_path)._smart_select_service
    plain = await service._rank_emoji_candidates("", "你在说什么", idx=index)
    weighted = await service._rank_emoji_candidates(
        "", "你在说什么", idx=index,
        emotion_weights=punctuation_emotion_weights("你在说什么？！"),
    )
    assert weighted[0][1] - plain[0][1] == pytest.approx(0.06)


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_sender_uses_original_llm_punctuation_after_query_rewrite(rewrite):
    analyzer = SimpleNamespace(analyze_for_reply=AsyncMock(
        return_value=EmotionQuery(True, "询问情况", ["confused"]),
    ))
    plugin = SimpleNamespace(
        plugin_config=SimpleNamespace(enable_jev=False, enable_natural_emotion_analysis=rewrite),
        enable_natural_emotion_analysis=rewrite, emotion_analyzer=analyzer,
    )
    engine = MemeSenderEngine(plugin)
    engine.get_meme_send_delay = lambda *args: 0
    engine.try_send_emoji = AsyncMock(return_value=False)
    event = DummyEvent("你在说什么？？")
    engine.emoji_turn_state(event).set_llm_reply("你在说什么？？")
    await engine.async_analyze_and_send_emoji(event, "你在说什么？？", [])
    args = engine.try_send_emoji.await_args
    assert args.args[2] == ("询问情况" if rewrite else "你在说什么？？")
    assert args.kwargs["emotion_weights"] == {"confused": 0.06}


@pytest.mark.asyncio
async def test_jev_recall_only_uses_assistant_punctuation(tmp_path):
    selector = MemeSelector(SimpleNamespace(plugin_config=SimpleNamespace(enable_embedding_search=False)))
    selector._check_group_allowed = lambda event: True
    selector._jev_selector.get_history = AsyncMock(return_value=[])
    selector._jev_selector.select = AsyncMock(return_value=None)
    selector._candidate_index = AsyncMock(return_value={})
    rank = selector._smart_select_service._rank_emoji_candidates = AsyncMock(return_value=[])
    await selector.select_emoji_with_jev(DummyEvent(), "你在说什么？？", user_message="好耶！！")
    assert rank.await_args.kwargs["emotion_weights"] == {"confused": 0.06}
    await selector.select_emoji_with_jev(DummyEvent(), "这是一条普通回复", user_message="好耶！！")
    assert rank.await_args.kwargs["emotion_weights"] == {}
