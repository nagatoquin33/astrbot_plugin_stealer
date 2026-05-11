# 修复 v2.6.2 概率发送失效

## 基线

- 上游：`nagatoquin33/astrbot_plugin_stealer`
- 版本：`v2.6.2`
- 提交：`aa438d3`

## Bug

`emoji_chance` 没有真正参与普通聊天发图判定。

实际表现：

- `emoji_chance=0` 时，机器人普通回复仍可能发送表情包。
- `emoji_chance=0.2` 时，表现接近每轮都发，而不是按 20% 概率发送。
- LLM 工具链里的 `search_emoji` 会提前标记 `active_sent`，导致状态语义混乱。
- cooldown 检查会在“判断阶段”写入时间戳，导致失败发送也可能影响后续回合。
- 底层发送函数没有返回真实发送结果，上层无法可靠判断是否应该写 cooldown 和 `active_sent`。

## 修复路径

统一普通聊天发图入口的 gate：

1. `resolve_auto_emoji_turn_permission()` 恢复同一回合判定缓存，避免 `on_llm_request` 和 `on_decorating_result` 同一轮重复抽概率。
2. 判定顺序固定为：
   - `auto_send`
   - 当前会话是否允许发送
   - cooldown 是否就绪
   - `emoji_chance`
3. `emoji_chance` 语义改回：
   - `0` 必定不发
   - `1` 必定允许
   - `0 < chance < 1` 使用 `random.random() < chance`
4. `is_auto_emoji_cooldown_ready()` 改成只读检查；只有真实发送成功后 `mark_auto_emoji_sent()` 写入 cooldown，并顺手清理过期记录。
5. `_prepare_emoji_response()` 的智能模式、被动模式、`[ast_emoji:...]` 都先通过同一个回合 gate。
6. `send_emoji_by_id` 作为普通 LLM 回复中的发图工具，也受同一个 gate 控制；`search_emoji` 只设置候选，不再标记真实发送。
7. `claimed` 只表示本回合已有发送流程占位；`active_sent` 只表示真实发出过图片。
8. `send_emoji_with_text()` 和 `try_send_emoji()` 改为返回真实发送结果，只有成功发送才让上层写 cooldown 和 `active_sent`。
9. 被动模式恢复 `&&happy&&` 情绪标签提取和文本清理，避免空情绪列表进入发送链路。

## 验证

```powershell
python -m unittest discover tests
python -m py_compile main.py core\events\emoji_sender_engine.py core\search\emoji_smart_select_service.py
```
