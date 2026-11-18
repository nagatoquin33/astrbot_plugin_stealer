"""图库单图分析及业务错误定义，与批次调度和 HTTP 无关。"""


class ReanalysisInputError(ValueError):
    """请求参数不符合分析或确认要求。"""


class ReanalysisNotFoundError(ValueError):
    """请求的批次或图片结果不存在。"""


class ReanalysisConflictError(RuntimeError):
    """当前任务或图库状态不允许执行请求。"""


ANALYSIS_FIELDS = ("desc", "tags", "scenes", "overlay_text", "emotions")


def is_missing_semantics(entry: dict) -> bool:
    """判断条目是否缺少完整语义元数据（缺少描述，或同时无标签与适用场景）。"""
    desc = str(entry.get("desc") or "").strip()
    tags = entry.get("tags") or []
    scenes = entry.get("scenes") or []
    return not desc or (not tags and not scenes)


def merge_empty_fields(before: dict, fresh: dict) -> dict:
    """仅填补空白语义字段，保留已存在的描述、标签或场景。"""
    merged = {}
    for key in ANALYSIS_FIELDS:
        old_val = before.get(key)
        if key in ("desc", "overlay_text"):
            merged[key] = old_val if isinstance(old_val, str) and old_val.strip() else fresh.get(key, "")
        elif key in ("tags", "scenes"):
            merged[key] = list(old_val) if isinstance(old_val, list) and old_val else list(fresh.get(key) or [])
        elif key == "emotions":
            old_list = [x for x in (old_val or []) if x] if isinstance(old_val, list) else []
            fresh_list = [x for x in (fresh.get("emotions") or []) if x]
            primary = before.get("category")
            if len(old_list) > 1 or (old_list and old_list != [primary]):
                merged[key] = list(dict.fromkeys([*old_list, *fresh_list]))[:5]
            else:
                merged[key] = fresh_list or old_list
        else:
            merged[key] = fresh.get(key)
    return merged


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
