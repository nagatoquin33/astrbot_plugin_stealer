"""图库单图分析及业务错误定义，与批次调度和 HTTP 无关。"""


class ReanalysisInputError(ValueError):
    """请求参数不符合分析或确认要求。"""


class ReanalysisNotFoundError(ValueError):
    """请求的批次或图片结果不存在。"""


class ReanalysisConflictError(RuntimeError):
    """当前任务或图库状态不允许执行请求。"""


ANALYSIS_FIELDS = ("desc", "tags", "scenes", "overlay_text", "emotions")


async def analyze_library_image(processor, file_path: str, categories: list[str]) -> dict:
    category, tags, description, emotion, scenes, text, emotions = await processor.classify_image(
        event=None, file_path=file_path, categories=categories, content_filtration=False,
    )
    rejected = {processor.CATEGORY_FILTERED, processor.CATEGORY_NOT_EMOJI}
    if category in rejected or emotion in rejected:
        raise ValueError("图片内容审核不通过或不适合收录")
    if not category:
        raise ValueError("无法识别图片分类")
    return {
        "category": category, "tags": tags or [], "desc": description or "",
        "scenes": scenes or [], "overlay_text": text or "", "emotions": emotions or [],
    }
