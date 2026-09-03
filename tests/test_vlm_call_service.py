"""GIF VLM 预处理：均匀九帧、九宫格布局和动图提示词。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

Image = pytest.importorskip("PIL.Image")

from core.processing.vlm_call_service import VLMCallService


def _service() -> VLMCallService:
    plugin = SimpleNamespace(plugin_config=SimpleNamespace(vision_provider_id="vision"))
    return VLMCallService(plugin)


def _write_color_gif(path: Path, colors: list[tuple[int, int, int]], size=(40, 24)) -> None:
    frames = [Image.new("RGB", size, color) for color in colors]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        optimize=False,
        disposal=2,
    )


def test_uniform_indices_cover_first_last_and_fill_nine_slots():
    assert VLMCallService._uniform_frame_indices(17) == [0, 2, 4, 6, 8, 10, 12, 14, 16]
    assert VLMCallService._uniform_frame_indices(5) == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert VLMCallService._uniform_frame_indices(2) == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert VLMCallService._uniform_frame_indices(0) == []


def test_storyboard_size_preserves_ratio_and_stays_within_vlm_limit():
    service = _service()
    frame_width, frame_height = service._storyboard_frame_size(5000, 3000)
    grid_width = frame_width * 3 + service.STORYBOARD_GAP * 2
    grid_height = (
        (frame_height + service.STORYBOARD_LABEL_HEIGHT) * 3
        + service.STORYBOARD_GAP * 2
    )

    assert grid_width <= service.MAX_VLM_DIMENSION
    assert grid_height <= service.MAX_VLM_DIMENSION
    assert frame_width / frame_height == pytest.approx(5 / 3, rel=0.01)


@pytest.mark.asyncio
async def test_animated_gif_becomes_ordered_three_by_three_storyboard(tmp_path: Path):
    colors = [(index * 12, 40 + index, 220 - index * 8) for index in range(17)]
    gif_path = tmp_path / "timeline.gif"
    _write_color_gif(gif_path, colors)

    output_path, is_animated = await _service()._prepare_image_for_vlm(str(gif_path))
    output = Path(output_path)
    try:
        assert is_animated is True
        assert output != gif_path
        assert output.suffix == ".png"

        expected_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        with Image.open(output) as storyboard:
            frame_width, frame_height = 40, 24
            cell_height = frame_height + VLMCallService.STORYBOARD_LABEL_HEIGHT
            assert storyboard.size == (
                frame_width * 3 + VLMCallService.STORYBOARD_GAP * 2,
                cell_height * 3 + VLMCallService.STORYBOARD_GAP * 2,
            )

            for sample_index, source_index in enumerate(expected_indices):
                row, column = divmod(sample_index, 3)
                x = column * (frame_width + VLMCallService.STORYBOARD_GAP) + frame_width // 2
                y = (
                    row * (cell_height + VLMCallService.STORYBOARD_GAP)
                    + VLMCallService.STORYBOARD_LABEL_HEIGHT
                    + frame_height // 2
                )
                actual = storyboard.convert("RGB").getpixel((x, y))
                expected = colors[source_index]
                assert max(abs(actual[i] - expected[i]) for i in range(3)) <= 3
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_static_gif_is_left_unchanged(tmp_path: Path):
    gif_path = tmp_path / "still.gif"
    Image.new("RGB", (24, 24), (10, 20, 30)).save(gif_path, "GIF")

    output_path, is_animated = await _service()._prepare_image_for_vlm(str(gif_path))

    assert output_path == str(gif_path)
    assert is_animated is False


@pytest.mark.asyncio
async def test_animated_prompt_explains_storyboard_and_temp_file_is_removed(tmp_path: Path):
    source = tmp_path / "source.gif"
    source.write_bytes(b"gif-placeholder")
    prepared = tmp_path / "storyboard.png"
    prepared.write_bytes(b"png-placeholder")

    service = _service()
    service._resolve_vision_provider = AsyncMock(return_value="vision")
    service._prepare_image_for_vlm = AsyncMock(return_value=(str(prepared), True))
    service._do_vlm_call = AsyncMock(return_value="{}")

    assert await service._call_vision_model(None, str(source), "BASE PROMPT") == "{}"
    actual_prompt = service._do_vlm_call.await_args.args[1]
    assert actual_prompt.startswith("[GIF 九宫格时间序列说明]")
    assert "第一行 1→2→3" in actual_prompt
    assert "同一句文字跨格重复时只记录一次" in actual_prompt
    assert actual_prompt.endswith("BASE PROMPT")
    assert not prepared.exists()
