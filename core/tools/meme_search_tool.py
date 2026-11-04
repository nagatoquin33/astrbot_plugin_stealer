"""LLM meme search and send workflows used by the registered entrypoints."""

import asyncio
import os

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..events.event_context import unwrap_event
from ..events.persona_library import prepare_persona_scope
from ..util.normalization import normalize_label_list

class MemeSearchToolWorkflow:
    async def _search_meme_candidates(
        self,
        event: AstrMessageEvent,
        query: str,
        *,
        limit: int = 5,
        idx: dict | None = None,
        reply: str = "",
        character: str = "",
    ):
        """委托给 MemeSelector.smart_search。"""
        if idx is None:
            idx = (
                await asyncio.to_thread(self.db_service.get_index_cache_readonly)
                if self.db_service.count_total() > 0
                else {}
            )

        if character:
            names = self.meme_selector._smart_select_service._character_names
            idx = {path: data for path, data in idx.items()
                   if character.strip().casefold() in names(data)}
        context = " ".join(part for part in (query, reply, character) if part).strip()
        return await self.meme_selector.smart_search(context, limit=limit, idx=idx, event=event)

    async def _search_meme_impl(self, event: AstrMessageEvent, query: str, reply: str = "", character: str = ""):
        """Build the current search's numbered, sendable candidates."""
        event = unwrap_event(event)
        query = str(query or "").strip()
        logger.info(f"[Tool] LLM 搜索表情包: {query}")

        turn_state = self._emoji_sender_engine.emoji_turn_state(event)
        turn_state.set_candidates([])

        try:
            if not query:
                yield "搜索失败：缺少 query。请描述要表达的台词、语气或场景，例如：安慰刚加班的朋友。"
                return

            if not self.is_send_enabled_for_event(event):
                yield "搜索失败：当前会话已禁用表情包发送。"
                return

            if self.db_service.count_total() > 0:
                idx = await asyncio.to_thread(self.db_service.get_index_cache_readonly)
            else:
                logger.debug("索引未加载，正在加载...")
                await self.index_manager.load_index()
                idx = await asyncio.to_thread(self.db_service.get_index_cache_readonly)

            idx = await asyncio.to_thread(lambda: {
                path: data for path, data in idx.items() if os.path.isfile(path)
            })
            search_options = {"limit": self.MAX_SEARCH_RESULTS, "idx": idx}
            if reply:
                search_options["reply"] = str(reply).strip()[:500]
            if character:
                search_options["character"] = str(character).strip()
            results = await self._search_meme_candidates(event, query, **search_options)

            if not results:
                yield (f"未找到与“{query}”贴合的表情包。可补充图上原文或 reply；"
                       "指定 character 时请核对角色名。当前没有可发送候选，可直接用文字回复。")
                return

            candidates = []
            result_lines = ["可发送候选；核对台词与语气，再选择编号："]

            for path, desc, emotion, tags in results:
                if os.path.exists(path):
                    meta = idx.get(path, {}) if isinstance(idx, dict) else {}
                    raw_scenes = meta.get("scenes", None) if isinstance(meta, dict) else None
                    if not raw_scenes:
                        raw_scenes = meta.get("scene", None) if isinstance(meta, dict) else None

                    scenes_items = normalize_label_list(raw_scenes)
                    scenes_str = ", ".join(scenes_items)
                    overlay_text = str(meta.get("overlay_text", "") or "") if isinstance(meta, dict) else ""
                    character_key = str(meta.get("character", "") or "") if isinstance(meta, dict) else ""
                    character_name = character_key
                    if character_key:
                        info_map = getattr(self.plugin_config, "character_info", None) or {}
                        info = info_map.get(character_key) if isinstance(info_map, dict) else None
                        if isinstance(info, dict) and info.get("name"):
                            character_name = str(info.get("name"))
                    source = str(meta.get("source", "") or "") if isinstance(meta, dict) else ""
                    scope_mode = str(meta.get("scope_mode", "public") or "public") if isinstance(meta, dict) else "public"
                    origin_target = str(meta.get("origin_target", "") or "") if isinstance(meta, dict) else ""
                    use_count = int(meta.get("use_count", 0) or 0) if isinstance(meta, dict) else 0

                    candidate_id = len(candidates) + 1
                    candidates.append(
                        {
                            "id": candidate_id,
                            "path": path,
                            "desc": desc,
                            "emotion": emotion,
                            "tags": tags,
                            "scenes": scenes_str,
                            "overlay_text": overlay_text,
                            "character": character_key,
                            "source": source,
                            "scope_mode": scope_mode,
                            "origin_target": origin_target,
                            "use_count": use_count,
                        }
                    )
                    result_lines.append(f"\n[{candidate_id}] 分类：{emotion}")
                    if character_name:
                        result_lines.append(f"    角色：{character_name}")
                    if overlay_text:
                        result_lines.append(f"    图上文字：{overlay_text}")
                    if tags:
                        result_lines.append(f"    标签：{tags}")
                    if scenes_str:
                        result_lines.append(f"    适合回应：{scenes_str}")
                    result_lines.append(f"    描述：{desc}")

            if not candidates:
                yield "搜索失败：找到的表情包文件均已丢失"
                return

            turn_state.set_candidates(candidates)
            result_lines.append(
                f"\n\n下一步请从候选中选择一项，并调用 {self.SEND_MEME_TOOL_NAME}(emoji_id=编号) 发送。"
                "编号仅对应这次搜索。候选都不合适时可不发送，或补充 reply/character 重新搜索。"
            )

            result_text = "\n".join(result_lines)
            logger.info(f"[Tool] 搜索完成，返回 {len(candidates)} 个候选")
            yield result_text

        except Exception as e:
            logger.error(f"[Tool] 搜索表情包失败: {e}", exc_info=True)
            yield f"搜索出错：{e}"

    async def _send_meme_impl(self, event: AstrMessageEvent, emoji_id: int):
        """按本次 search_meme 返回的候选编号发送表情包。

        选择原则：核对台词与适合回应的场景，保持与你准备发给用户的回复立场一致。

        Args:
            emoji_id(number): 表情包编号（从 search_meme 返回的候选列表中选择）

        """
        event = unwrap_event(event)
        logger.info(f"[Tool] LLM 选择发送表情包编号: {emoji_id}")
        turn_state = self._emoji_sender_engine.emoji_turn_state(event)

        try:
            if not self.is_send_enabled_for_event(event):
                yield "发送失败：reason=send_disabled。当前会话已禁用表情包发送功能，请不要继续调用发送工具。"
                return

            if emoji_id is None:
                yield f"发送失败：reason=missing_id。缺少 emoji_id 参数。请先调用 {self.SEARCH_MEME_TOOL_NAME}，再传入候选编号。"
                return

            try:
                if isinstance(emoji_id, bool) or (isinstance(emoji_id, float) and not emoji_id.is_integer()):
                    raise ValueError("integer candidate id required")
                emoji_id = int(emoji_id)
            except Exception:
                yield f"发送失败：reason=invalid_id。编号 {emoji_id} 无法解析为整数，请输入有效的数字编号。"
                return

            candidates = turn_state.get_candidates()
            if not candidates:
                yield f"发送失败：reason=candidate_expired。没有可用候选列表。请先调用 {self.SEARCH_MEME_TOOL_NAME} 重新搜索。"
                return

            if emoji_id < 1 or emoji_id > len(candidates):
                yield f"发送失败：reason=invalid_id。编号 {emoji_id} 无效。可选编号范围：1-{len(candidates)}，请重新选择。"
                return

            selected = candidates[emoji_id - 1]
            path = selected["path"]
            desc = selected["desc"]
            emotion = selected["emotion"]

            if not os.path.exists(path):
                yield f"发送失败：reason=file_missing。表情包文件已丢失。\n你选择的是：编号 {emoji_id}，分类 {emotion}，描述 {desc}\n请重新搜索并选择其他表情包。"
                return

            await prepare_persona_scope(self, event)
            if not self.meme_selector.is_path_allowed_for_event(path, event):
                yield "发送失败：reason=scope_denied。该表情包被限制为仅来源会话可发送，请选择 public 表情或重新搜索。"
                return

            logger.info(f"[Tool] 发送选中的表情包: {path} (emotion={emotion})")
            send_mode = await self.meme_selector.send_emoji_message(event, path)
            if not send_mode:
                yield "发送失败：reason=send_failed。表情包编码或平台发送失败，请重新搜索或选择其他候选。"
                return
            sent_as_sticker = send_mode == "telegram_sticker"

            await self.meme_selector.record_emoji_usage(path, trigger="llm_tool")
            await self._emoji_sender_engine.mark_auto_emoji_sent(event)
            turn_state.mark_active_sent()
            turn_state.set_candidates([])

            mode_desc = "Telegram贴纸" if sent_as_sticker else "图片"
            success_msg = f"发送成功（{mode_desc}）。\n\n你发送的表情包：\n- 编号：{emoji_id}\n- 分类：{emotion}\n- 描述：{desc}"
            logger.info(f"[Tool] {success_msg}")
            yield success_msg
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield f"发送出错：{e}"
            return
