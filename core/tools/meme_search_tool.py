"""LLM meme search and send workflows used by the registered entrypoints."""

import os

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..events.event_context import unwrap_event
from ..util.normalization import normalize_label_list

class MemeSearchToolWorkflow:
    async def _search_meme_candidates(
        self,
        event: AstrMessageEvent,
        query: str,
        *,
        limit: int = 5,
        idx: dict | None = None,
    ):
        """委托给 MemeSelector.smart_search。"""
        if idx is None:
            idx = (
                self.db_service.get_index_cache_readonly()
                if self.db_service.count_total() > 0
                else {}
            )

        return await self.meme_selector.smart_search(query, limit=limit, idx=idx, event=event)

    def _find_similar_categories(self, query: str, top_n: int = 3) -> list[str]:
        """找到与查询词最相似的多个分类，委托给 MemeSelector。"""
        return self.meme_selector.find_similar_categories(query, top_n)

    async def _search_meme_impl(self, event: AstrMessageEvent, query: str):
        """从插件索引中搜索表情包候选；不要直接浏览表情库目录或发送本地文件。

        Args:
            query(string): 2-8 个检索关键词。优先写图上文字、角色名、画面或语气，例如“猫猫 震惊 怎么会这样”。

        使用建议：
        - 这是从表情库选图的唯一入口；不要使用文件、终端或通用消息工具绕过它
        - 若候选不合适，换图上原文、角色名或更具体的画面词再次调用本工具

        返回值：
        返回候选表情包列表，每个包含：
        - 编号：用于调用 send_meme
        - 分类：浏览分区，仅供参考
        - 角色 / 图上文字 / 描述：选图时优先看这些

        根据候选的图上文字、角色和描述选择最贴合的一张，然后调用 send_meme；不要猜测或传递文件路径。
        """
        event = unwrap_event(event)
        query = str(query or "").strip()
        logger.info(f"[Tool] LLM 搜索表情包: {query}")

        turn_state = self._emoji_sender_engine.emoji_turn_state(event)

        try:
            if not query:
                yield "搜索失败：缺少 query 参数。请传入你当前心情词，例如：开心、无语、尴尬、感谢。"
                return

            if not self.is_send_enabled_for_event(event):
                yield "搜索失败：当前群聊已禁用表情包功能"
                return

            if self.db_service.count_total() > 0:
                idx = self.db_service.get_index_cache_readonly()
            else:
                logger.debug("索引未加载，正在加载...")
                await self.index_manager.load_index()
                idx = self.db_service.get_index_cache_readonly()

            # smart_search 已内置关键词映射和模糊匹配（阈值0.4）
            results = await self._search_meme_candidates(
                event, query, limit=self.MAX_SEARCH_RESULTS, idx=idx
            )

            if not results:
                similar = self._find_similar_categories(query, top_n=3)
                suggestion = f"未找到与'{query}'匹配的表情包。"
                if similar:
                    suggestion += "\n\n您是否想找以下分类？\n- " + "\n- ".join(similar)
                cats = self.plugin_config.get_categories()
                suggestion += "\n\n可用分类：" + ", ".join(cats[:10])
                if len(cats) > 10:
                    suggestion += f" 等共{len(cats)}个分类"
                logger.warning(f"[Tool] 未找到匹配: {query}, 推荐: {similar}")
                yield suggestion
                return

            candidates = []
            result_lines = [f"找到 {len(results)} 个匹配的表情包：\n"]

            for i, (path, desc, emotion, tags) in enumerate(results):
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

                    candidate_id = f"emoji_{i + 1}"
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
                    result_lines.append(f"\n[{i + 1}] 分类：{emotion}")
                    if character_name:
                        result_lines.append(f"    角色：{character_name}")
                    if overlay_text:
                        result_lines.append(f"    图上文字：{overlay_text}")
                    if tags:
                        result_lines.append(f"    标签：{tags}")
                    if scenes_str:
                        result_lines.append(f"    画面短语：{scenes_str}")
                    result_lines.append(f"    作用域：{scope_mode}")
                    if use_count:
                        result_lines.append(f"    使用次数：{use_count}")
                    if source == "qq_store":
                        result_lines.append("    来源：QQ商城")
                    result_lines.append(f"    描述：{desc}")

            if not candidates:
                yield "搜索失败：找到的表情包文件均已丢失"
                return

            turn_state.set_candidates(candidates)
            result_lines.append(
                f"\n\n下一步请从候选中选择一项，并调用 {self.SEND_MEME_TOOL_NAME}(emoji_id=编号) 发送。"
                "候选不合适时请换关键词再次搜索；不要用文件、终端或通用消息工具直接发送表情库文件。"
            )

            result_text = "\n".join(result_lines)
            logger.info(f"[Tool] 搜索完成，返回 {len(candidates)} 个候选")
            yield result_text

        except Exception as e:
            logger.error(f"[Tool] 搜索表情包失败: {e}", exc_info=True)
            yield f"搜索出错：{e}"

    async def _send_meme_impl(self, event: AstrMessageEvent, emoji_id: int):
        """发送 search_meme 返回的候选表情包；不要猜测或直接传递文件路径。

        选择原则：优先发送能代表你"当前心情词"的候选项。

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

            mode_desc = "Telegram贴纸" if sent_as_sticker else "图片"
            success_msg = f"发送成功（{mode_desc}）。\n\n你发送的表情包：\n- 编号：{emoji_id}\n- 分类：{emotion}\n- 描述：{desc}"
            logger.info(f"[Tool] {success_msg}")
            yield success_msg
            return

        except Exception as e:
            logger.error(f"[Tool] 发送表情包失败: {e}", exc_info=True)
            yield f"发送出错：{e}"
            return
