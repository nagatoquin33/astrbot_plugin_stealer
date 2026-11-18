"""图库重分析：复用视觉模型调用，独立管理后台并发批次和受保护的写回。"""

import asyncio
from collections import deque
from copy import deepcopy
import time
import uuid
from pathlib import Path

from astrbot.api import logger

from ..util.model_requests import model_request_context
from .semantic_schema import validate_emotion_labels
from .library_analysis import (
    ANALYSIS_FIELDS, analyze_library_image, merge_empty_fields, ReanalysisInputError,
    ReanalysisNotFoundError, ReanalysisConflictError,
)


_merge_empty_fields = merge_empty_fields


class BatchReanalysisService:
    """每个实例只运行一个批次；模型限速与并发上限由请求门控协同控制。"""

    TASK_NAME = "library_reanalysis"
    MAX_IMAGES = 500
    MAX_CONCURRENCY = 8

    def __init__(self, db, processor, config, scheduler, on_updated, on_batch_updated=None):
        self.db = db
        self.processor = processor
        self.config = config
        self.scheduler = scheduler
        self.on_updated = on_updated
        self.on_batch_updated = on_batch_updated
        self._job = None
        self._results = {}
        self._review_lock = asyncio.Lock()
        self._closed = False

    def _concurrency(self) -> int:
        raw = getattr(self.config, "reanalysis_concurrency", 1)
        try:
            return max(1, min(self.MAX_CONCURRENCY, int(raw)))
        except (TypeError, ValueError):
            return 1

    @property
    def busy(self) -> bool:
        return self._job is not None and self._job["status"] in {"running", "stopping"}

    def snapshot(self) -> dict | None:
        if self._job is None:
            return None
        return {
            **self._job, "errors": [dict(item) for item in self._job["errors"]],
            "results": [{key: item[key] for key in ("hash", "name", "status")}
                        for item in self._results.values()],
        }

    def start(self, hashes: list[str], *, fill_empty_only: bool = False) -> dict:
        if self._closed:
            raise ReanalysisConflictError("插件正在关闭")
        if self.busy:
            raise ReanalysisConflictError("已有批量重新分析任务正在运行")
        if self._job and self._job["pending"]:
            raise ReanalysisConflictError("请先保存或放弃当前批次的待确认结果")
        if not isinstance(hashes, list) or not hashes or len(hashes) > self.MAX_IMAGES:
            raise ReanalysisInputError(f"请选择 1 至 {self.MAX_IMAGES} 张图片")
        if any(not isinstance(value, str) or not value.strip() for value in hashes):
            raise ReanalysisInputError("图片 hash 无效")
        hashes = list(dict.fromkeys(value.strip() for value in hashes))
        job = {
            "id": uuid.uuid4().hex, "status": "running", "total": len(hashes),
            "processed": 0, "succeeded": 0, "failed": 0, "skipped": 0,
            "pending": 0, "saved": 0, "discarded": 0,
            "concurrency": self._concurrency(),
            "fill_empty_only": bool(fill_empty_only),
            "current": "", "errors": [], "created_at": time.time(),
        }
        self._job = job
        self._results = {}
        return self._schedule(hashes)

    def _schedule(self, hashes: list[str]) -> dict:
        job = self._job
        job["status"] = "running"
        job["concurrency"] = self._concurrency()
        job.pop("completed_at", None)
        operation = self._run(job, hashes)
        if self.scheduler.create_task(self.TASK_NAME, operation) is None:
            operation.close()
            job["status"] = "failed"
            raise ReanalysisConflictError("无法创建后台分析任务")
        return self.snapshot()

    def cancel(self, task_id: str) -> dict:
        self._check_job(task_id)
        if self.busy:
            # 当前在途图片完成分析后停止，已经产生的待确认结果继续保留。
            self._job["status"] = "stopping"
        return self.snapshot()

    async def close(self):
        self._closed = True
        await self.scheduler.cancel_task(self.TASK_NAME)
        async with self._review_lock:
            pass

    async def _process_one(
        self, image_hash: str, job: dict, active_names: dict[str, str], concurrency: int,
    ) -> tuple[str, str]:
        found = await asyncio.to_thread(self.db.get_emoji_by_hash, image_hash)
        if found is None:
            return "skipped", "图片已删除或不在正式图库中"
        path, before = found
        name = Path(path).name
        active_names[image_hash] = name
        job["current"] = "、".join(active_names.values())
        try:
            if not await asyncio.to_thread(Path(path).is_file):
                return "skipped", "原图文件已丢失"
            with model_request_context(concurrency=concurrency, priority=1):
                result = await analyze_library_image(self.processor, path, list(self.config.categories))
            fresh_after = {key: result[key] for key in ANALYSIS_FIELDS}
            after = _merge_empty_fields(before, fresh_after) if job.get("fill_empty_only") else fresh_after
            self._results[image_hash] = {
                "hash": image_hash, "path": path, "name": name, "status": "ready",
                "before": {key: before.get(key) for key in (*ANALYSIS_FIELDS, "hash", "category")},
                "after": after,
            }
            job["pending"] += 1
            return "succeeded", ""
        finally:
            active_names.pop(image_hash, None)
            job["current"] = "、".join(active_names.values())

    def _check_job(self, task_id: str):
        if self._job is None or self._job["id"] != task_id:
            raise ReanalysisNotFoundError("任务不存在或已被新批次替换")

    def result_detail(self, task_id: str, image_hash: str) -> dict:
        self._check_job(task_id)
        item = self._results.get(image_hash)
        if item is None:
            raise ReanalysisNotFoundError("这张图片没有可确认的分析结果")
        return deepcopy({
            "hash": item["hash"], "name": item["name"], "status": item["status"],
            "before": {key: item["before"].get(key) for key in ANALYSIS_FIELDS},
            "after": item["after"],
        })

    def retry_failed(self, task_id: str) -> dict:
        self._check_job(task_id)
        if self._closed or self.busy:
            raise ReanalysisConflictError("任务仍在运行或插件正在关闭")
        hashes = [item["hash"] for item in self._job["errors"] if item["status"] == "failed"]
        if not hashes:
            raise ReanalysisInputError("没有需要重试的失败图片")
        self._job["errors"] = [item for item in self._job["errors"] if item["status"] != "failed"]
        self._job["processed"] -= len(hashes)
        self._job["failed"] = 0
        return self._schedule(hashes)

    def _resolve_review(self, item: dict, status: str) -> None:
        """集中结算一次确认，调用前已检查结果仍处于待确认状态。"""
        item["status"] = status
        self._job["pending"] -= 1
        if status in {"saved", "discarded"}:
            self._job[status] += 1

    async def review(self, task_id: str, image_hash: str, decision: str) -> dict:
        async with self._review_lock:
            if self._closed:
                raise ReanalysisConflictError("插件正在关闭")
            self._check_job(task_id)
            item = self._results.get(image_hash)
            if item is None or decision not in {"save", "discard"}:
                raise ReanalysisInputError("图片结果或确认操作无效")
            target_status = "saved" if decision == "save" else "discarded"
            if item["status"] == target_status:
                return self.snapshot()
            if item["status"] != "ready":
                raise ReanalysisInputError("这张图片的分析结果已经处理过")
            if decision == "discard":
                self._resolve_review(item, "discarded")
                return self.snapshot()

            async def commit():
                async with self.db.file_lock:
                    try:
                        validate_emotion_labels(item["after"].get("emotions", []), self.config.categories)
                    except ValueError as exc:
                        self._resolve_review(item, "conflict")
                        error = f"{exc}，请重新分析"
                        self._job["errors"].append({"hash": image_hash, "status": "conflict", "error": error})
                        raise ReanalysisConflictError(error) from exc
                    if not await self.db.update_path(item["path"], item["after"], expected=item["before"]):
                        self._resolve_review(item, "conflict")
                        error = "图片已被修改或删除，未保存此分析结果，请重新分析"
                        self._job["errors"].append({"hash": image_hash, "status": "conflict", "error": error})
                        raise ReanalysisConflictError(error)
                    self._resolve_review(item, "saved")
                await self.on_updated(item["path"])

            # 请求断开或插件卸载时，等待已经获准的原子写回完成。
            task = asyncio.create_task(commit())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            return self.snapshot()

    async def review_all(self, task_id: str, decision: str) -> dict:
        """一键全部保存或全部放弃当前所有待确认结果，保存时合并刷新检索索引。"""
        async with self._review_lock:
            if self._closed:
                raise ReanalysisConflictError("插件正在关闭")
            self._check_job(task_id)
            if decision not in {"save", "discard"}:
                raise ReanalysisInputError("确认操作无效")
            ready_items = [item for item in self._results.values() if item["status"] == "ready"]
            if not ready_items:
                return self.snapshot()
            if decision == "discard":
                for item in ready_items:
                    self._resolve_review(item, "discarded")
                return self.snapshot()

            async def commit_all():
                saved_paths: list[str] = []
                async with self.db.file_lock:
                    for item in ready_items:
                        if item["status"] != "ready":
                            continue
                        try:
                            validate_emotion_labels(item["after"].get("emotions", []), self.config.categories)
                        except ValueError as exc:
                            self._resolve_review(item, "conflict")
                            self._job["errors"].append({
                                "hash": item["hash"], "status": "conflict", "error": f"{exc}，请重新分析",
                            })
                            continue
                        if not await self.db.update_path(item["path"], item["after"], expected=item["before"]):
                            self._resolve_review(item, "conflict")
                            self._job["errors"].append({
                                "hash": item["hash"], "status": "conflict",
                                "error": "图片已被修改或删除，未保存此分析结果，请重新分析",
                            })
                            continue
                        self._resolve_review(item, "saved")
                        saved_paths.append(item["path"])
                if saved_paths:
                    if self.on_batch_updated is not None:
                        await self.on_batch_updated(saved_paths)
                    else:
                        for path in saved_paths:
                            await self.on_updated(path)

            task = asyncio.create_task(commit_all())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            return self.snapshot()

    async def _run(self, job: dict, hashes: list[str]):
        queue = deque(hashes)
        active_names: dict[str, str] = {}
        concurrency = max(1, int(job.get("concurrency") or self._concurrency()))
        worker_count = min(concurrency, len(hashes))

        async def worker():
            while queue and job["status"] != "stopping":
                image_hash = queue.popleft()
                try:
                    outcome, error = await self._process_one(image_hash, job, active_names, concurrency)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    outcome, error = "failed", str(exc)
                    logger.warning(f"图库重新分析失败 hash={image_hash}: {exc}")
                job[outcome] += 1
                job["processed"] += 1
                if error:
                    job["errors"].append({"hash": image_hash, "status": outcome, "error": error})

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        try:
            await asyncio.gather(*workers)
            job["status"] = "cancelled" if job["status"] == "stopping" else "completed"
        except asyncio.CancelledError:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            job["status"] = "cancelled"
            raise
        finally:
            job["current"] = ""
            job["completed_at"] = time.time()
