"""图片渲染服务：负责缩略图生成、列表页渲染、GIF 转换和 base64 编码。"""

import asyncio
import base64
import os
import time
from io import BytesIO
from typing import Any

from astrbot.api import logger

try:
    from PIL import Image as PILImage
    from PIL import ImageDraw as PILImageDraw
    from PIL import ImageFont as PILImageFont

    try:
        LANCZOS = PILImage.Resampling.LANCZOS
    except AttributeError:
        LANCZOS = PILImage.LANCZOS
except Exception:
    PILImage = None
    PILImageDraw = None
    PILImageFont = None
    LANCZOS = None


class ImageRenderService:
    """负责图片渲染、缩略图生成、列表页绘制和 base64 编码。"""

    GIF_CACHE_MAX_SIZE = 50
    GIF_CACHE_MAX_SIZE_BYTES = 10 * 1024 * 1024
    CACHE_EXPIRE_TIME = 3600

    def __init__(self, plugin_instance: Any = None) -> None:
        self.plugin = plugin_instance
        self._gif_base64_cache: dict[str, tuple[float, str]] = {}
        self._gif_base64_cache_max_size = self.GIF_CACHE_MAX_SIZE
        self._gif_base64_cache_expire_time = self.CACHE_EXPIRE_TIME

    # ── base64 编码 ──────────────────────────────────────────

    async def file_to_base64(self, file_path: str) -> str:
        """将文件转换为 base64 编码。"""

        def _sync_read_and_encode(fp: str) -> str:
            with open(fp, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        try:
            return await asyncio.to_thread(_sync_read_and_encode, file_path)
        except Exception as e:
            logger.error(f"文件转换为base64失败: {e}")
            return ""

    async def file_to_gif_base64(self, file_path: str) -> str:
        """将文件转换为 GIF 格式的 base64 编码（用于发送侧强制 GIF）。"""
        if not getattr(self.plugin, "send_emoji_as_gif", True):
            return await self.file_to_base64(file_path)

        try:
            stat_mtime = os.path.getmtime(file_path)
        except Exception:
            stat_mtime = None

        cache_key = f"{file_path}:{stat_mtime}"
        now = time.time()
        cached = self._gif_base64_cache.get(cache_key)
        if cached is not None:
            cached_at, cached_b64 = cached
            if now - cached_at <= self._gif_base64_cache_expire_time and cached_b64:
                return cached_b64

        try:
            if PILImage is None:
                return await self.file_to_base64(file_path)

            # GIF 转换限制常量
            MAX_FRAMES = 30
            MAX_DIMENSION = 2048

            def _sync_convert_to_gif(fp: str) -> str:
                with PILImage.open(fp) as im:
                    buf = BytesIO()
                    is_animated = bool(getattr(im, "is_animated", False))
                    n_frames = int(getattr(im, "n_frames", 1) or 1)

                    # 检查图片尺寸
                    width, height = im.size
                    scale = 1.0
                    if width > MAX_DIMENSION or height > MAX_DIMENSION:
                        scale = min(MAX_DIMENSION / width, MAX_DIMENSION / height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        logger.debug(
                            f"GIF 图片尺寸过大 ({width}x{height}), 缩放至 {new_width}x{new_height}"
                        )
                    else:
                        new_width, new_height = width, height

                    if is_animated and n_frames > 1:
                        actual_frames = min(n_frames, MAX_FRAMES)
                        frames = []
                        durations = []
                        frame_step = max(1, n_frames // actual_frames)
                        for frame_idx in range(0, n_frames, frame_step):
                            if len(frames) >= MAX_FRAMES:
                                break
                            im.seek(frame_idx)
                            frame = im.convert("RGBA")
                            if scale != 1.0:
                                frame = frame.resize((new_width, new_height), PILImage.LANCZOS)
                            else:
                                frame = frame.resize((new_width, new_height), PILImage.BILINEAR)
                            frames.append(frame)
                            durations.append(im.info.get("duration", 100))

                        if frames:
                            frames[0].save(
                                buf,
                                format="GIF",
                                save_all=True,
                                append_images=frames[1:],
                                duration=durations,
                                loop=0,
                                optimize=True,
                                disposal=2,
                            )
                    else:
                        if scale != 1.0:
                            im = im.resize((new_width, new_height), PILImage.LANCZOS)
                        im.save(buf, format="GIF", optimize=True)

                    result = base64.b64encode(buf.getvalue()).decode("utf-8")
                    self._gif_base64_cache[cache_key] = (time.time(), result)
                    self._evict_gif_base64_cache()
                    return result

            return await asyncio.to_thread(_sync_convert_to_gif, file_path)
        except Exception as e:
            logger.error(f"转换为 GIF base64 失败: {e}")
            return await self.file_to_base64(file_path)

    def _evict_gif_base64_cache(self) -> None:
        """淘汰 _gif_base64_cache 中最旧的条目。"""
        total_bytes = sum(len(v[1]) for v in self._gif_base64_cache.values())
        if (
            len(self._gif_base64_cache) <= self._gif_base64_cache_max_size
            and total_bytes <= self.GIF_CACHE_MAX_SIZE_BYTES
        ):
            return

        sorted_items = sorted(self._gif_base64_cache.items(), key=lambda kv: kv[1][0])
        target_count = max(1, self._gif_base64_cache_max_size // 2)
        target_bytes = max(1024 * 1024, self.GIF_CACHE_MAX_SIZE_BYTES // 2)

        keep_items = []
        current_bytes = 0
        for key, value in reversed(sorted_items):
            if len(keep_items) >= target_count or current_bytes >= target_bytes:
                break
            keep_items.append((key, value))
            current_bytes += len(value[1])

        self._gif_base64_cache.clear()
        self._gif_base64_cache.update(keep_items)
        logger.debug(
            f"_gif_base64_cache 淘汰完成，当前 {len(self._gif_base64_cache)} 条，"
            f"总大小 {current_bytes / 1024 / 1024:.2f}MB"
        )

    # ── 缩略图 ───────────────────────────────────────────────

    @staticmethod
    def generate_thumb_uri(pth: str, size: int = 84) -> str:
        if PILImage is None or not os.path.exists(pth):
            return ""
        try:
            with PILImage.open(pth) as im:
                try:
                    if getattr(im, "is_animated", False):
                        im.seek(0)
                except Exception:
                    pass
                im = im.convert("RGBA")
                im.thumbnail((size, size), LANCZOS or PILImage.BICUBIC)
                canvas = PILImage.new("RGBA", (size, size), (255, 255, 255, 0))
                x = (size - im.size[0]) // 2
                y = (size - im.size[1]) // 2
                canvas.paste(im, (x, y), im)
                buf = BytesIO()
                canvas.save(buf, format="PNG", optimize=True)
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return ""

    @staticmethod
    def render_items_for_list(items: list[dict], thumb_size: int = 84) -> list[dict]:
        return [
            {
                "index": int(item.get("index", 0) or 0),
                "desc": str(item.get("desc", "") or "").strip(),
                "category": str(item.get("category", "") or "").strip(),
                "tags": ", ".join(t.strip() for t in (item.get("tags") or []) if t and t.strip()),
                "scenes": ", ".join(
                    s.strip() for s in (item.get("scenes") or []) if s and s.strip()
                ),
                "scope_mode": str(item.get("scope_mode", "public") or "public").strip(),
                "origin_target": str(item.get("origin_target", "") or "").strip(),
                "use_count": int(item.get("use_count", 0) or 0),
                "thumb_data_uri": ImageRenderService.generate_thumb_uri(
                    str(item.get("path", "") or ""), thumb_size
                ),
            }
            for item in items
        ]

    # ── 列表页渲染（html-to-pic）─────────────────────────────

    async def render_emoji_list_page_file(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """使用 AstrBot 内置 html-to-pic 渲染列表，返回本地图片文件路径。"""
        plugin = getattr(self, "plugin", None)
        if not plugin or not hasattr(plugin, "html_render"):
            return ""

        rendered_items = []
        for item in items:
            pth = str(item.get("path", "") or "")
            thumb_uri = self.generate_thumb_uri(pth)
            rendered_items.append(
                {
                    "index": int(item.get("index", 0) or 0),
                    "desc": str(item.get("desc", "") or "").strip()
                    or str(item.get("name", "") or ""),
                    "category": str(item.get("category", "") or ""),
                    "source": str(item.get("source", "") or ""),
                    "thumb": thumb_uri,
                }
            )

        tmpl = _LIST_PAGE_FILE_TEMPLATE
        data = {
            "items": rendered_items,
            "page": int(page),
            "total_pages": int(total_pages),
            "total_filtered": int(total_filtered),
            "total_all": int(total_all),
            "category": str(category or ""),
            "per_page": int(per_page),
        }

        try:
            return await plugin.html_render(
                tmpl,
                data,
                return_url=False,
                options={"full_page": True, "type": "png"},
            )
        except Exception as e:
            logger.debug(f"html-to-pic(PNG) 渲染失败，尝试 JPEG: {e}")
            try:
                return await plugin.html_render(
                    tmpl,
                    data,
                    return_url=False,
                    options={"full_page": True, "type": "jpeg", "quality": 70},
                )
            except Exception as e2:
                logger.debug(f"html-to-pic(JPEG) 渲染失败，回退到本地渲染: {e2}")
                return ""

    async def render_emoji_list_page_url(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """使用 AstrBot 内置 html-to-pic 渲染列表，返回可公网访问的图片 URL。"""
        plugin = getattr(self, "plugin", None)
        if not plugin or not hasattr(plugin, "html_render"):
            return ""

        rendered_items = []
        for item in items:
            pth = str(item.get("path", "") or "")
            thumb_uri = self.generate_thumb_uri(pth)
            rendered_items.append(
                {
                    "index": int(item.get("index", 0) or 0),
                    "desc": str(item.get("desc", "") or "").strip()
                    or str(item.get("name", "") or ""),
                    "category": str(item.get("category", "") or ""),
                    "source": str(item.get("source", "") or ""),
                    "thumb": thumb_uri,
                }
            )

        tmpl = _LIST_PAGE_URL_TEMPLATE
        data = {
            "items": rendered_items,
            "page": int(page),
            "total_pages": int(total_pages),
            "total_filtered": int(total_filtered),
            "total_all": int(total_all),
            "category": str(category or ""),
            "per_page": int(per_page),
        }

        try:
            return await plugin.html_render(
                tmpl,
                data,
                return_url=True,
                options={"full_page": True, "type": "jpeg", "quality": 70},
            )
        except Exception as e:
            logger.debug(f"html-to-pic(URL) 渲染失败: {e}")
            return ""

    # ── 列表页渲染（本地 PIL）────────────────────────────────

    async def render_emoji_list_page_base64(
        self,
        *,
        items: list[dict],
        page: int,
        total_pages: int,
        total_filtered: int,
        total_all: int,
        category: str,
        per_page: int,
    ) -> str:
        """把表情包列表渲染成一张 PNG，并返回 base64（不带 data:image/png 前缀）。"""
        if PILImage is None or PILImageDraw is None or PILImageFont is None:
            return ""

        def _pick_font(size: int):
            candidates = [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyh.ttf",
                "C:/Windows/Fonts/simhei.ttf",
                "C:/Windows/Fonts/simsun.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
                "/System/Library/Fonts/PingFang.ttc",
            ]
            for fp in candidates:
                try:
                    if os.path.exists(fp):
                        return PILImageFont.truetype(fp, size=size)
                except Exception:
                    continue
            try:
                return PILImageFont.load_default()
            except Exception:
                return None

        def _wrap_text(draw, text: str, font, max_width: int) -> list[str]:
            text = (text or "").strip()
            if not text:
                return [""]
            lines = []
            buf = ""
            for ch in text:
                nxt = buf + ch
                w = (
                    draw.textlength(nxt, font=font)
                    if hasattr(draw, "textlength")
                    else draw.textbbox((0, 0), nxt, font=font)[2]
                )
                if w <= max_width or not buf:
                    buf = nxt
                    continue
                lines.append(buf)
                buf = ch
                if len(lines) >= 2:
                    break
            if buf and len(lines) < 2:
                lines.append(buf)
            if len(lines) >= 2 and (len(text) > sum(len(x) for x in lines)):
                ell = "..."
                while lines[-1] and (
                    (
                        draw.textlength(lines[-1] + ell, font=font)
                        if hasattr(draw, "textlength")
                        else draw.textbbox((0, 0), lines[-1] + ell, font=font)[2]
                    )
                    > max_width
                ):
                    lines[-1] = lines[-1][:-1]
                lines[-1] = (lines[-1] + ell) if lines[-1] else ell
            return lines

        def _load_thumb(path: str, size: int):
            try:
                with PILImage.open(path) as im:
                    try:
                        if getattr(im, "is_animated", False):
                            im.seek(0)
                    except Exception:
                        pass
                    im = im.convert("RGBA")
                    im.thumbnail((size, size), LANCZOS or PILImage.BICUBIC)
                    canvas = PILImage.new("RGBA", (size, size), (255, 255, 255, 0))
                    x = (size - im.size[0]) // 2
                    y = (size - im.size[1]) // 2
                    canvas.paste(im, (x, y), im)
                    return canvas
            except Exception:
                return None

        def _sync_render() -> str:
            width = 980
            pad = 24
            title_h = 84
            footer_h = 56
            row_h = 118
            thumb = 88

            height = title_h + footer_h + row_h * max(1, len(items))
            bg = PILImage.new("RGB", (width, height), (250, 250, 252))
            draw = PILImageDraw.Draw(bg)

            title_font = _pick_font(30)
            body_font = _pick_font(22)
            small_font = _pick_font(18)
            if title_font is None or body_font is None or small_font is None:
                return ""

            header = f"表情包列表  第 {page}/{total_pages} 页  显示 {total_filtered} 张(全部 {total_all} 张)"
            if category:
                header += f"  分类: {category}"
            draw.text((pad, 22), header, fill=(20, 22, 30), font=title_font)
            draw.line(
                (pad, title_h - 10, width - pad, title_h - 10),
                fill=(220, 220, 230),
                width=2,
            )

            y0 = title_h
            text_x = pad + thumb + 18
            text_w = width - pad - text_x
            for idx, item in enumerate(items):
                y = y0 + idx * row_h
                if idx % 2 == 1:
                    draw.rectangle((0, y, width, y + row_h), fill=(246, 246, 250))

                n = int(item.get("index", 0) or 0)
                desc = str(item.get("desc", "") or "").strip()
                if not desc:
                    desc = str(item.get("name", "") or "")
                cat = str(item.get("category", "") or "")
                pth = str(item.get("path", "") or "")

                thumb_im = _load_thumb(pth, thumb)
                if thumb_im is not None:
                    bg.paste(thumb_im, (pad, y + (row_h - thumb) // 2), thumb_im)
                else:
                    draw.rectangle(
                        (pad, y + (row_h - thumb) // 2, pad + thumb, y + (row_h + thumb) // 2),
                        outline=(210, 210, 225),
                        width=2,
                    )
                    draw.text(
                        (pad + 16, y + (row_h - 18) // 2),
                        "N/A",
                        fill=(140, 140, 155),
                        font=small_font,
                    )

                prefix = f"{n:04d}."
                draw.text((text_x, y + 22), prefix, fill=(60, 70, 90), font=body_font)
                lines = _wrap_text(draw, desc, body_font, max_width=text_w - 96)
                draw.text((text_x + 78, y + 22), lines[0], fill=(25, 28, 35), font=body_font)
                if len(lines) > 1:
                    draw.text((text_x + 78, y + 52), lines[1], fill=(25, 28, 35), font=body_font)
                if cat:
                    draw.text(
                        (text_x, y + 84), f"分类: {cat}", fill=(110, 115, 130), font=small_font
                    )
                if str(item.get("source", "") or "") == "qq_store":
                    draw.text(
                        (text_x + 200, y + 84),
                        "QQ商城",
                        fill=(110, 115, 130),
                        font=small_font,
                    )

                draw.line(
                    (pad, y + row_h - 1, width - pad, y + row_h - 1),
                    fill=(230, 230, 238),
                    width=1,
                )

            foot = "翻页: /meme list 2 或 /meme list happy 2 或 /meme list 20 2    删除: /meme delete <序号>"
            draw.text((pad, height - footer_h + 16), foot, fill=(90, 95, 110), font=small_font)

            buf = BytesIO()
            bg.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            return await asyncio.to_thread(_sync_render)
        except Exception as e:
            logger.debug(f"渲染表情列表失败: {e}")
            return ""

    def cleanup(self) -> None:
        """清理渲染缓存。"""
        self._gif_base64_cache.clear()
        logger.debug("ImageRenderService 缓存已清理")


# ── HTML 模板（文件渲染）────────────────────────────────────
_LIST_PAGE_FILE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #fafafe;
      --panel: #ffffff;
      --line: #e7e7f0;
      --text: #171925;
      --muted: #6b7184;
      --chip: #f3f4f8;
    }
    html, body {
      width: 100%;
    }
    body {
      margin: 0;
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
    }
    .wrap {
      width: 100%;
      padding: 24px 24px 16px;
      box-sizing: border-box;
    }
    .title {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: .2px;
      margin: 0 0 10px 0;
    }
    .subtitle {
      font-size: 14px;
      color: var(--muted);
      margin: 0 0 18px 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(15, 18, 35, .06);
    }
    .row {
      display: flex;
      gap: 16px;
      align-items: center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .row:last-child { border-bottom: none; }
    .thumb {
      width: 84px;
      height: 84px;
      border-radius: 12px;
      background: #f6f7fb;
      border: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      flex: 0 0 auto;
    }
    .thumb img { width: 84px; height: 84px; object-fit: contain; }
    .meta { flex: 1 1 auto; min-width: 0; }
    .line1 {
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin: 0 0 6px 0;
    }
    .idx {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-weight: 700;
      color: #3b3f53;
      flex: 0 0 auto;
    }
    .desc {
      font-size: 18px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chip {
      display: inline-block;
      background: var(--chip);
      color: var(--muted);
      border: 1px solid var(--line);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
    }
    .footer {
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">表情包列表</h1>
    <p class="subtitle">第 {{ page }}/{{ total_pages }} 页，显示 {{ total_filtered }} 张（全部 {{ total_all }} 张）{% if category %}，分类: {{ category }}{% endif %}，每页 {{ per_page }} 张</p>
    <div class="panel">
      {% for it in items %}
      <div class="row">
        <div class="thumb">
          {% if it.thumb %}
            <img src="{{ it.thumb }}" />
          {% else %}
            <span class="chip">No Preview</span>
          {% endif %}
        </div>
        <div class="meta">
          <div class="line1">
            <div class="idx">{{ "%04d"|format(it.index) }}.</div>
            <div class="desc">{{ it.desc }}</div>
          </div>
          {% if it.category %}
            <span class="chip">分类: {{ it.category }}</span>
          {% endif %}
          {% if it.source == "qq_store" %}
            <span class="chip">QQ商城</span>
          {% endif %}
        </div>
      </div>
      {% endfor %}
    </div>
    <div class="footer">
      <div>翻页: /meme list 2 或 /meme list happy 2 或 /meme list 20 2</div>
      <div>删除: /meme delete &lt;序号&gt;</div>
    </div>
  </div>
</body>
</html>"""

# ── HTML 模板（URL 渲染）────────────────────────────────────
_LIST_PAGE_URL_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #fafafe;
      --panel: #ffffff;
      --line: #e7e7f0;
      --text: #171925;
      --muted: #6b7184;
      --chip: #f3f4f8;
    }
    html, body {
      width: 100%;
      height: 100%;
    }
    body {
      margin: 0;
      width: 100vw;
      background: var(--bg);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
    }
    .wrap {
      width: 100%;
      max-width: none;
      padding: 24px 24px 16px;
      box-sizing: border-box;
    }
    .title {
      font-size: 28px;
      font-weight: 700;
      letter-spacing: .2px;
      margin: 0 0 10px 0;
    }
    .subtitle {
      font-size: 14px;
      color: var(--muted);
      margin: 0 0 18px 0;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(15, 18, 35, .06);
      width: 100%;
      box-sizing: border-box;
      padding: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .card {
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 14px;
      align-items: center;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
    }
    .thumb {
      width: 128px;
      height: 128px;
      border-radius: 14px;
      background: #f6f7fb;
      border: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .thumb img { width: 128px; height: 128px; object-fit: contain; }
    .meta { min-width: 0; }
    .line1 {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 6px 0;
    }
    .idx {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-weight: 800;
      font-size: 14px;
      color: #222;
      background: #eef0f6;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 3px 8px;
      flex: 0 0 auto;
    }
    .desc {
      font-size: 17px;
      font-weight: 600;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .chip {
      display: inline-block;
      background: var(--chip);
      color: var(--muted);
      border: 1px solid var(--line);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
    }
    .footer {
      margin-top: 14px;
      font-size: 13px;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      gap: 12px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">表情包列表</h1>
    <p class="subtitle">第 {{ page }}/{{ total_pages }} 页，显示 {{ total_filtered }} 张（全部 {{ total_all }} 张）{% if category %}，分类: {{ category }}{% endif %}，每页 {{ per_page }} 张</p>
    <div class="panel">
      <div class="grid">
        {% for it in items %}
        <div class="card">
          <div class="thumb">
            {% if it.thumb %}
              <img src="{{ it.thumb }}" />
            {% else %}
              <span class="chip">No Preview</span>
            {% endif %}
          </div>
          <div class="meta">
            <div class="line1">
              <div class="idx">{{ "%04d"|format(it.index) }}</div>
              <div class="desc">{{ it.desc }}</div>
            </div>
            {% if it.category %}
              <span class="chip">分类: {{ it.category }}</span>
            {% endif %}
            {% if it.source == "qq_store" %}
              <span class="chip">QQ商城</span>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
    <div class="footer">
      <div>翻页: /meme list 2 或 /meme list happy 2 或 /meme list 20 2</div>
      <div>删除: /meme delete &lt;序号&gt;</div>
    </div>
  </div>
</body>
</html>"""
