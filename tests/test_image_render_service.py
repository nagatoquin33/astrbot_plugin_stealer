import base64
import io
import types
from pathlib import Path

import pytest
from PIL import Image

from astrbot_plugin_stealer.core.processing.image_render_service import ImageRenderService


def _save_animation(path: Path, frames: list[Image.Image], durations: list[int], *, loop: int = 0) -> bytes:
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF" if path.suffix.lower() == ".gif" else "PNG",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
    )
    path.write_bytes(buffer.getvalue())
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_existing_gif_is_sent_byte_for_byte(tmp_path):
    source = tmp_path / "original.gif"
    frames = [
        Image.new("RGBA", (12, 12), (220, 40, 40, 255)),
        Image.new("RGBA", (12, 12), (40, 40, 220, 255)),
    ]
    original = _save_animation(source, frames, [40, 120], loop=3)
    service = ImageRenderService(types.SimpleNamespace(send_meme_as_gif=True))

    encoded = await service.file_to_gif_base64(str(source))

    assert base64.b64decode(encoded) == original


@pytest.mark.asyncio
async def test_non_gif_animation_keeps_total_duration_and_loop(tmp_path):
    source = tmp_path / "animated.png"
    frames = [
        Image.new("RGBA", (12, 12), (220, 40, 40, 255)),
        Image.new("RGBA", (12, 12), (40, 220, 40, 255)),
        Image.new("RGBA", (12, 12), (40, 40, 220, 255)),
    ]
    _save_animation(source, frames, [40, 80, 120], loop=2)
    service = ImageRenderService(types.SimpleNamespace(send_meme_as_gif=True))

    encoded = await service.file_to_gif_base64(str(source))

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as converted:
        durations = []
        for index in range(converted.n_frames):
            converted.seek(index)
            durations.append(int(converted.info.get("duration", 0) or 0))
        assert converted.n_frames == 3
        assert sum(durations) == 240
        assert converted.info.get("loop") == 2


@pytest.mark.asyncio
async def test_sparse_non_gif_animation_keeps_a_changed_frame(tmp_path):
    source = tmp_path / "sparse.png"
    red = Image.new("RGBA", (12, 12), (220, 40, 40, 255))
    blue = Image.new("RGBA", (12, 12), (40, 40, 220, 255))
    frames = [red.copy() for _ in range(60)]
    frames[31] = blue
    _save_animation(source, frames, [40] * len(frames))
    service = ImageRenderService(types.SimpleNamespace(send_meme_as_gif=True))

    encoded = await service.file_to_gif_base64(str(source))

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as converted:
        pixels = []
        for index in range(converted.n_frames):
            converted.seek(index)
            pixels.append(converted.convert("RGB").getpixel((0, 0)))
        assert len(set(pixels)) >= 2
