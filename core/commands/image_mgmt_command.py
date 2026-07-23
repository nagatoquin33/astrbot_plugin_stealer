"""表情包管理命令：负责表情包的 list、delete、blacklist、scope 操作。"""

import time
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ..util.safe_io import safe_remove_file


class ImageManagementCommand:
    """负责表情包的列表、删除、拉黑和作用域管理。"""

    def __init__(self, plugin_instance: Any) -> None:
        self.plugin = plugin_instance

    async def _add_blacklist_hash(self, image_hash: str) -> bool:
        if not image_hash:
            return False
        try:
            db = getattr(self.plugin, "db_service", None)
            if db is not None and hasattr(db, "add_blacklist"):
                await db.add_blacklist(image_hash, int(time.time()))
                return True
            if getattr(self.plugin, "cache_service", None):
                await self.plugin.cache_service.set(
                    "blacklist_cache", image_hash, int(time.time()), persist=True
                )
                return True
        except Exception as e:
            logger.error(f"写入黑名单失败: {e}", exc_info=True)
        return False

    async def list_images(
        self,
        event: AstrMessageEvent,
        category: str = "",
        limit: str = "10",
        page: str = "1",
    ):
        """列出表情包，支持按分类筛选。

        Args:
            event: 消息事件
            category: 可选的分类筛选
            limit: 显示数量限制，默认10张
        """
        # 参数解析目标：
        # - /meme list            -> page=1, per_page=默认
        # - /meme list 2          -> page=2 (默认每页数量)
        # - /meme list happy 2    -> 分类=happy, page=2
        # - /meme list 20 2       -> per_page=20, page=2
        # - /meme list happy 20 2 -> 分类=happy, per_page=20, page=2
        category = str(category or "").strip()
        limit = str(limit or "").strip()
        page = str(page or "").strip()

        # 仅提供一个数字：视为翻页
        if category.isdigit() and (not limit or limit == "10") and (not page or page == "1"):
            page, category = category, ""
            limit = "10"
        # /meme list happy 2 -> 分类 + 页码
        elif (
            category and (not category.isdigit()) and limit.isdigit() and (not page or page == "1")
        ):
            page, limit = limit, "10"
        # /meme list 20 2 -> 每页数量 + 页码
        elif category.isdigit() and limit.isdigit() and (not page or page == "1"):
            page, limit, category = limit, category, ""

        # 解析每页数量
        try:
            per_page = int(limit)
        except Exception:
            per_page = 10
        per_page = max(1, min(100, per_page))

        # 解析页码
        try:
            page_num = int(page)
        except Exception:
            page_num = 1
        page_num = max(1, page_num)

        # 使用数据库分页查询，避免全量加载
        db_service = getattr(self.plugin, "db_service", None)
        if db_service and hasattr(db_service, "get_emojis_paginated"):
            cat_filter = category if category else None
            rows, total_filtered, category_counts = db_service.get_emojis_paginated(
                page=page_num, page_size=per_page, category=cat_filter, sort_order="newest"
            )
            total_all = db_service.count_total()

            if total_all == 0:
                yield event.plain_result("暂无表情包数据")
                return

            # 构建展示列表，标记文件是否存在
            display_images = []
            missing_count = 0
            for row in rows:
                img_path = row.get("path", "")
                file_exists = Path(img_path).exists()
                if not file_exists:
                    missing_count += 1
                display_images.append({
                    "path": img_path,
                    "name": Path(img_path).name,
                    "category": row.get("category", "未分类"),
                    "desc": str(row.get("desc", "") or ""),
                    "source": str(row.get("source", "") or ""),
                    "qq_emoji_package_id": str(row.get("qq_emoji_package_id", "") or ""),
                    "created_at": row.get("created_at", 0),
                    "file_exists": file_exists,
                })

            if not display_images and not category:
                yield event.plain_result("暂无有效的表情包文件")
                return
            if not display_images and category:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
                return

            total_pages = max(1, (total_filtered + per_page - 1) // per_page)
            if page_num > total_pages:
                page_num = total_pages

        else:
            # 兜底：旧的全量加载方式
            image_index = await self.plugin.index_manager.load_index()

            if not image_index:
                yield event.plain_result("暂无表情包数据")
                return

            all_images = []
            missing_count = 0
            for img_path, img_info in image_index.items():
                if isinstance(img_info, dict):
                    file_exists = Path(img_path).exists()
                    if not file_exists:
                        missing_count += 1
                    img_category = img_info.get("category", "未分类")
                    img_desc = img_info.get("desc", "")
                    img_source = img_info.get("source", "")
                    img_pkg = img_info.get("qq_emoji_package_id", "")
                    all_images.append({
                        "path": img_path,
                        "name": Path(img_path).name,
                        "category": img_category,
                        "desc": str(img_desc or ""),
                        "source": str(img_source or ""),
                        "qq_emoji_package_id": str(img_pkg or ""),
                        "created_at": img_info.get("created_at", 0),
                        "file_exists": file_exists,
                    })

            if not all_images:
                if category:
                    yield event.plain_result(f"分类 '{category}' 中暂无表情包")
                else:
                    yield event.plain_result("暂无有效的表情包文件")
                return

            all_images.sort(key=lambda x: x["created_at"], reverse=True)
            for i, img in enumerate(all_images, 1):
                img["index"] = i

            filtered_images = [
                img for img in all_images if (not category or img.get("category") == category)
            ]
            if not filtered_images:
                yield event.plain_result(f"分类 '{category}' 中暂无表情包")
                return

            total_filtered = len(filtered_images)
            total_all = len(all_images)
            total_pages = max(1, (total_filtered + per_page - 1) // per_page)
            if page_num > total_pages:
                page_num = total_pages

            start = (page_num - 1) * per_page
            display_images = filtered_images[start : start + per_page]

        # 渲染输出
        if getattr(self.plugin, "image_processor_service", None):
            if event.get_platform_name() == "aiocqhttp":
                url = await self.plugin.image_processor_service.render_emoji_list_page_url(
                    items=display_images,
                    page=page_num,
                    total_pages=total_pages,
                    total_filtered=total_filtered,
                    total_all=total_all,
                    category=category,
                    per_page=per_page,
                )
                if url:
                    yield event.image_result(url).stop_event()
                    return

            file_path = await self.plugin.image_processor_service.render_emoji_list_page_file(
                items=display_images,
                page=page_num,
                total_pages=total_pages,
                total_filtered=total_filtered,
                total_all=total_all,
                category=category,
                per_page=per_page,
            )
            if file_path:
                if event.get_platform_name() != "aiocqhttp":
                    yield event.make_result().file_image(file_path).stop_event()
                    return

            b64 = await self.plugin.image_processor_service.render_emoji_list_page_base64(
                items=display_images,
                page=page_num,
                total_pages=total_pages,
                total_filtered=total_filtered,
                total_all=total_all,
                category=category,
                per_page=per_page,
            )
            if b64:
                yield event.make_result().base64_image(b64).stop_event()
                return

        # 纯文本 fallback
        title = f"表情包列表 ({page_num}/{total_pages}) ({len(display_images)}/{total_filtered})"
        if total_all != total_filtered:
            title += f" (总 {total_all})"
        if category:
            title += f" - 分类: {category}"
        if missing_count:
            title += f" [文件丢失: {missing_count}]"

        result_text = title + "\n\n"
        for img in display_images:
            idx = int(img.get("index", 0) or 0)
            desc = str(img.get("desc", "") or "").strip()
            if not desc:
                desc = str(img.get("name", "") or "")
            if len(desc) > 28:
                desc = desc[:25] + "..."
            marker = "" if img.get("file_exists", True) else "⚠"
            result_text += f"{idx:4d}. {marker}{desc}\n"

        if total_pages > 1:
            next_page_hint = f"\n下一页: /meme list {page_num + 1}"
            if category:
                next_page_hint = f"\n下一页: /meme list {category} {page_num + 1}"
            result_text += next_page_hint
        result_text += "\n用法: /meme list [分类] [每页数量] [页码]"
        yield event.plain_result(result_text).stop_event()

    async def delete_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定的表情包。

        Args:
            event: 消息事件
            identifier: 图片标识符，可以是序号、文件名或路径
        """
        if not identifier:
            yield event.plain_result(
                "用法: /meme delete <序号|文件名>\n先使用 /meme list 查看图片列表获取序号"
            )
            return

        image_index = await self.plugin.index_manager.load_index()

        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        target_image = self._find_target_image(image_index, identifier)

        if not target_image:
            yield event.plain_result(
                f"未找到图片: {identifier}\n请使用 /meme list 查看可用的图片列表"
            )
            return

        # 执行删除操作
        success = await self._delete_image_files(target_image["path"])

        if success:
            # 从索引中移除
            target_hash = str(target_image.get("hash", "") or "").strip()
            for path, meta in list(image_index.items()):
                if path == target_image["path"] or (
                    target_hash and isinstance(meta, dict) and meta.get("hash") == target_hash
                ):
                    del image_index[path]
            await self.plugin.index_manager.save_index(image_index)

            yield event.plain_result(
                f"✅ 已删除表情包:\n文件: {target_image['name']}\n分类: {target_image['category']}"
            )
        else:
            yield event.plain_result(f"❌ 删除失败: {target_image['name']}")

    async def blacklist_image(self, event: AstrMessageEvent, identifier: str = ""):
        """删除指定表情包并加入黑名单。"""
        if not identifier:
            yield event.plain_result(
                "用法: /meme blacklist <序号|文件名>\n先使用 /meme list 查看图片列表获取序号"
            )
            return

        image_index = await self.plugin.index_manager.load_index()

        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        target_image = self._find_target_image(image_index, identifier)
        if not target_image:
            yield event.plain_result(
                f"未找到图片: {identifier}\n请使用 /meme list 查看可用的图片列表"
            )
            return

        target_hash = str(target_image.get("hash", "") or "").strip()
        if not target_hash:
            yield event.plain_result(f"❌ 拉黑失败: {target_image['name']} 缺少 hash")
            return

        if not await self._add_blacklist_hash(target_hash):
            yield event.plain_result(f"❌ 拉黑失败: {target_image['name']} 无法写入黑名单")
            return

        success = await self._delete_image_files(target_image["path"])
        if not success:
            yield event.plain_result(f"❌ 拉黑失败: {target_image['name']}")
            return

        for path, meta in list(image_index.items()):
            if path == target_image["path"] or (
                target_hash and isinstance(meta, dict) and meta.get("hash") == target_hash
            ):
                del image_index[path]
        await self.plugin.index_manager.save_index(image_index)

        logger.info(f"已加入黑名单: {target_hash}")

        yield event.plain_result(
            f"✅ 已拉黑表情包:\n"
            f"文件: {target_image['name']}\n"
            f"分类: {target_image['category']}\n"
            f"状态: 已删除并加入黑名单"
        )

    async def set_image_scope(
        self, event: AstrMessageEvent, identifier: str = "", scope_mode: str = ""
    ):
        """设置表情包作用域。"""
        if not identifier or not scope_mode:
            yield event.plain_result(
                "用法: /meme scope <序号|文件名> <public|local>\n"
                "public=公开表情包，所有群可发送\n"
                "local=仅来源群可发送"
            )
            return

        image_index = await self.plugin.index_manager.load_index()
        if not image_index:
            yield event.plain_result("暂无表情包数据")
            return

        target_image = self._find_target_image(image_index, identifier)
        if not target_image:
            yield event.plain_result(
                f"未找到图片: {identifier}\n请使用 /meme list 查看可用的图片列表"
            )
            return

        normalized_mode = str(scope_mode or "").strip().lower()
        if normalized_mode in {"public", "global", "all"}:
            normalized_mode = "public"
        elif normalized_mode in {"local", "private", "scoped"}:
            normalized_mode = "local"
        else:
            yield event.plain_result("作用域无效，请使用 public 或 local")
            return

        meta = image_index.get(target_image["path"])
        if not isinstance(meta, dict):
            yield event.plain_result("该表情包索引数据异常，无法设置作用域")
            return

        if normalized_mode == "local" and not str(meta.get("origin_target", "") or "").strip():
            yield event.plain_result("该表情包缺少来源群信息，暂时无法限制为仅来源群发送")
            return

        meta["scope_mode"] = normalized_mode
        await self.plugin.index_manager.save_index(image_index)

        origin_target = str(meta.get("origin_target", "") or "").strip() or "未知"
        scope_text = "公开" if normalized_mode == "public" else "仅来源群"
        yield event.plain_result(
            f"✅ 已更新表情包作用域:\n"
            f"文件: {target_image['name']}\n"
            f"分类: {target_image['category']}\n"
            f"作用域: {scope_text}\n"
            f"来源: {origin_target}"
        )

    def _collect_valid_images(self, image_index: dict) -> list[dict[str, Any]]:
        """收集有效图片，并按 list/delete 一致的顺序返回。"""
        valid_images: list[dict[str, Any]] = []
        for img_path, img_info in image_index.items():
            if isinstance(img_info, dict) and Path(img_path).exists():
                valid_images.append(
                    {
                        "path": img_path,
                        "name": Path(img_path).name,
                        "category": img_info.get("category", "未分类"),
                        "created_at": img_info.get("created_at", 0),
                        "hash": img_info.get("hash", ""),
                        "origin_target": img_info.get("origin_target", ""),
                        "scope_mode": img_info.get("scope_mode", "public"),
                    }
                )

        valid_images.sort(key=lambda x: x["created_at"], reverse=True)
        return valid_images

    def _find_target_image(self, image_index: dict, identifier: str) -> dict[str, Any] | None:
        """按 /meme list 的全局序号或文件名定位图片。"""
        valid_images = self._collect_valid_images(image_index)

        try:
            index = int(identifier) - 1
            if 0 <= index < len(valid_images):
                return valid_images[index]
        except ValueError:
            pass

        for img in valid_images:
            if img["name"] == identifier or img["name"].startswith(identifier):
                return img

        return None

    async def _delete_image_files(self, img_path: str) -> bool:
        """删除图片文件（raw目录和categories目录）。

        Args:
            img_path: 图片路径

        Returns:
            bool: 是否删除成功
        """
        try:
            deleted_files = []

            # 删除主文件（通常在raw目录）
            if Path(img_path).exists():
                if await safe_remove_file(img_path):
                    deleted_files.append(img_path)
                logger.info(f"已删除主文件: {img_path}")

            # 查找并删除categories目录中的对应文件
            if hasattr(self.plugin, "categories_dir") and self.plugin.plugin_config.categories_dir:
                img_name = Path(img_path).name

                # 遍历所有分类目录
                for category_dir in self.plugin.plugin_config.categories_dir.iterdir():
                    if category_dir.is_dir():
                        category_file = category_dir / img_name
                        if category_file.exists():
                            if await safe_remove_file(str(category_file)):
                                deleted_files.append(str(category_file))
                            logger.info(f"已删除分类文件: {category_file}")

            logger.info(f"删除操作完成，共删除 {len(deleted_files)} 个文件")
            return len(deleted_files) > 0

        except Exception as e:
            logger.error(f"删除图片文件失败: {e}")
            return False
