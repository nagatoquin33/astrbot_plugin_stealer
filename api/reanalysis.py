"""批量重新分析的 HTTP 边界，业务状态与写回由分析服务负责。"""

import asyncio
from functools import wraps

from quart import jsonify, request
from astrbot.api import logger

from ..core.processing.library_analysis import (
    ReanalysisInputError, ReanalysisNotFoundError, ReanalysisConflictError,
)


def reanalysis_response(key="task"):
    """统一重分析接口的响应结构和业务错误映射。"""
    def decorate(handler):
        @wraps(handler)
        async def wrapped(self):
            try:
                return jsonify({"success": True, key: await handler(self)})
            except (ReanalysisInputError, ReanalysisNotFoundError, ReanalysisConflictError) as exc:
                status = 404 if isinstance(exc, ReanalysisNotFoundError) else (
                    409 if isinstance(exc, ReanalysisConflictError) else 400
                )
                return jsonify({"success": False, "error": str(exc),
                                "task": self.plugin.batch_reanalysis.snapshot()}), status
            except Exception as exc:
                logger.error(f"重新分析请求失败: {exc}", exc_info=True)
                return jsonify({"success": False, "error": str(exc),
                                "task": self.plugin.batch_reanalysis.snapshot()}), 500
        return wrapped
    return decorate


async def reanalysis_payload():
    data = await request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ReanalysisInputError("请求内容必须是 JSON 对象")
    return data


class ReanalysisRoutes:
    @reanalysis_response("result")
    async def handle_batch_reanalyze_result(self):
        return self.plugin.batch_reanalysis.result_detail(
            request.args.get("task_id", ""), request.args.get("hash", ""),
        )

    @reanalysis_response()
    async def handle_batch_reanalyze_review(self):
        data = await reanalysis_payload()
        if data.get("all"):
            return await self.plugin.batch_reanalysis.review_all(
                data.get("task_id"), data.get("decision"),
            )
        return await self.plugin.batch_reanalysis.review(
            data.get("task_id"), data.get("hash"), data.get("decision"),
        )

    @reanalysis_response("missing")
    async def handle_batch_reanalyze_missing(self):
        max_images = getattr(self.plugin.batch_reanalysis, "MAX_IMAGES", 500)
        library = str(request.args.get("library", "") or "").strip()
        category = str(request.args.get("category", "") or "").strip() or None
        character = str(request.args.get("character", "") or "").strip() or None
        favorite_only = str(request.args.get("favorite_only", "false")).lower() == "true"
        hashes, total_missing = await asyncio.to_thread(
            self.plugin.db_service.list_hashes_missing_semantics,
            max_images,
            library=library,
            category=category,
            character=character,
            favorite_only=favorite_only,
        )
        return {
            "hashes": hashes,
            "total_missing": total_missing,
            "max_images": max_images,
            "scope": {
                "library": library,
                "category": category or "",
                "character": character or "",
                "favorite_only": favorite_only,
            },
        }

    @reanalysis_response()
    async def handle_batch_reanalyze_retry(self):
        data = await reanalysis_payload()
        return self.plugin.batch_reanalysis.retry_failed(data.get("task_id"))

    @reanalysis_response()
    async def handle_batch_reanalyze(self):
        data = await reanalysis_payload()
        if data.get("fill_empty_only"):
            return self.plugin.batch_reanalysis.start(
                data.get("hashes"), fill_empty_only=True,
            )
        return self.plugin.batch_reanalysis.start(data.get("hashes"))

    @reanalysis_response()
    async def handle_batch_reanalyze_status(self):
        return self.plugin.batch_reanalysis.snapshot()

    @reanalysis_response()
    async def handle_batch_reanalyze_cancel(self):
        data = await reanalysis_payload()
        return self.plugin.batch_reanalysis.cancel(data.get("task_id"))
