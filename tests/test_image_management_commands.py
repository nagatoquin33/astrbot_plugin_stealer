"""Command list numbering must resolve to the same files as numeric actions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.commands.image_mgmt_command import ImageManagementCommand
from core.db.database_service import DatabaseService
from core.db.index_manager import IndexManager
from core.processing.image_render_service import ImageRenderService
from core.commands import image_mgmt_command as image_management_module


class _Result:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def stop_event(self):
        return self


class _Event:
    def get_platform_name(self) -> str:
        return "test"

    def make_result(self):
        return self

    def file_image(self, _path: str):
        return _Result()

    def plain_result(self, text: str):
        return _Result(text)


@pytest.mark.asyncio
async def test_list_numbers_match_numeric_actions_and_clamp_past_last_page(tmp_path: Path):
    db = DatabaseService(tmp_path / "emojis.db")
    paths = {name: tmp_path / f"{name}.png" for name in ("a", "b", "c", "missing")}
    for name in ("a", "b", "c"):
        paths[name].write_bytes(b"image")
    await db.insert_batch(
        [
            {"path": str(paths["a"]), "hash": "hash-a", "category": "happy", "desc": "a", "created_at": 10},
            {"path": str(paths["b"]), "hash": "hash-b", "category": "sad", "desc": "b", "created_at": 20},
            {"path": str(paths["c"]), "hash": "hash-c", "category": "happy", "desc": "c", "created_at": 20},
            {"path": str(paths["missing"]), "hash": "hash-m", "category": "happy", "desc": "missing", "created_at": 30},
        ]
    )

    rendered = {}

    async def html_render(_template, data, **_kwargs):
        rendered.update(data)
        return str(tmp_path / "list.png")

    renderer = ImageRenderService()
    plugin = SimpleNamespace(db_service=db, image_render_service=renderer, html_render=html_render)
    renderer.plugin = plugin
    command = ImageManagementCommand(plugin)

    images = [
        message async for message in command.list_images(_Event(), "happy", "10", "99")
    ]
    assert len(images) == 1
    assert rendered["page"] == 1
    assert [item["index_label"] for item in rendered["items"]] == [
        "----", "0001", "0003"
    ]

    index = db.get_index_cache_readonly()
    assert command._find_target_image(index, "1")["path"] == str(paths["c"])
    assert command._find_target_image(index, "3")["path"] == str(paths["a"])
    assert command._find_target_image(index, str(paths["a"]))["path"] == str(paths["a"])

    plugin.image_render_service = None
    text = [
        message async for message in command.list_images(_Event(), "happy", "10", "99")
    ][0].text
    assert "(1/1)" in text
    assert "----. ⚠missing" in text
    assert "   1. c" in text
    assert "   3. a" in text

    page_two = [
        message async for message in command.list_images(_Event(), "happy", "1", "2")
    ][0].text
    assert "下一页: /meme list happy 1 3" in page_two
    last_page = [
        message async for message in command.list_images(_Event(), "happy", "1", "3")
    ][0].text
    assert "下一页:" not in last_page


def test_ambiguous_filename_prefix_does_not_select_an_arbitrary_image(tmp_path: Path):
    first = tmp_path / "ext_alpha.png"
    second = tmp_path / "ext_beta.png"
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    index = {
        str(first): {"category": "happy", "hash": "hash-a", "created_at": 1},
        str(second): {"category": "sad", "hash": "hash-b", "created_at": 2},
    }
    command = ImageManagementCommand(SimpleNamespace())

    assert command._find_target_image(index, "ext_") is None
    assert command._find_target_image(index, "") is None
    assert command._find_target_image(index, first.name)["path"] == str(first)


@pytest.mark.asyncio
async def test_delete_removes_only_selected_record_and_file(tmp_path: Path):
    categories = tmp_path / "categories"
    target = categories / "happy" / "shared.png"
    unrelated = categories / "sad" / "shared.png"
    same_hash = categories / "angry" / "duplicate.png"
    for path in (target, unrelated, same_hash):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    db = DatabaseService(tmp_path / "emojis.db")
    await db.insert_batch([
        {"path": str(target), "hash": "same", "category": "happy", "created_at": 30},
        {"path": str(same_hash), "hash": "same", "category": "angry", "created_at": 20},
        {"path": str(unrelated), "hash": "other", "category": "sad", "created_at": 10},
    ])
    plugin = SimpleNamespace(
        db_service=db,
        categories_dir=categories,
        plugin_config=SimpleNamespace(categories_dir=categories),
    )
    plugin.index_manager = IndexManager(plugin)

    messages = [
        message async for message in ImageManagementCommand(plugin).delete_image(_Event(), "1")
    ]
    assert messages[0].text.startswith("✅")
    assert not target.exists()
    assert unrelated.exists() and same_hash.exists()
    assert db.get_emoji(str(target)) is None
    assert db.get_emoji(str(unrelated)) is not None
    assert db.get_emoji(str(same_hash)) is not None


@pytest.mark.asyncio
async def test_blacklist_deletes_every_file_with_same_hash(tmp_path: Path):
    categories = tmp_path / "categories"
    first = categories / "happy" / "first.png"
    second = categories / "sad" / "second.png"
    unrelated = categories / "sad" / "third.png"
    for path in (first, second, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")

    db = DatabaseService(tmp_path / "emojis.db")
    await db.insert_batch([
        {"path": str(first), "hash": "same", "category": "happy", "created_at": 30},
        {"path": str(second), "hash": "same", "category": "sad", "created_at": 20},
        {"path": str(unrelated), "hash": "other", "category": "sad", "created_at": 10},
    ])
    plugin = SimpleNamespace(
        db_service=db,
        categories_dir=categories,
        plugin_config=SimpleNamespace(categories_dir=categories),
    )
    plugin.index_manager = IndexManager(plugin)

    messages = [
        message async for message in ImageManagementCommand(plugin).blacklist_image(_Event(), "1")
    ]
    assert messages[0].text.startswith("✅")
    assert not first.exists() and not second.exists()
    assert unrelated.exists()
    assert db.get_emoji(str(first)) is None
    assert db.get_emoji(str(second)) is None
    assert db.get_emoji(str(unrelated)) is not None
    assert "same" in db.blacklisted_hashes()


@pytest.mark.asyncio
async def test_delete_failure_keeps_record_and_blacklist_reports_partial_cleanup(
    tmp_path: Path, monkeypatch
):
    image = tmp_path / "stuck.png"
    image.write_bytes(b"image")
    db = DatabaseService(tmp_path / "emojis.db")
    await db.insert_batch([
        {"path": str(image), "hash": "stuck-hash", "category": "happy", "created_at": 1}
    ])
    plugin = SimpleNamespace(db_service=db)
    plugin.index_manager = IndexManager(plugin)

    async def refuse_remove(_path):
        return False

    monkeypatch.setattr(image_management_module, "safe_remove_file", refuse_remove)
    command = ImageManagementCommand(plugin)

    deleted = [message async for message in command.delete_image(_Event(), "1")]
    assert deleted[0].text.startswith("❌")
    assert image.exists() and db.get_emoji(str(image)) is not None

    blacklisted = [message async for message in command.blacklist_image(_Event(), "1")]
    assert blacklisted[0].text.startswith("⚠ 已加入黑名单")
    assert image.exists() and db.get_emoji(str(image)) is not None
    assert "stuck-hash" in db.blacklisted_hashes()
