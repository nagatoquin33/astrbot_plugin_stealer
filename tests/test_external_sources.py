"""v3 external source protocol, safety, provenance, and import tests."""

from __future__ import annotations

import io
import hashlib
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from core.db.database_service import DatabaseService
from core.sources.http_source import HTTPSource, validate_source_url
from core.sources.github_source import GitHubSource, _parse_repository
from core.sources.models import ExternalSourceError, ExternalSourceSecurityError
from core.sources.pack_source import PackSource, safe_member_path
from core.sources.source_service import SourceService


def _png_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


class _Config:
    external_sources_enabled = True
    external_source_allow_http = False
    external_source_default_review = False
    external_source_max_items = 100
    external_source_max_image_bytes = 1024 * 1024
    external_source_max_archive_bytes = 10 * 1024 * 1024
    external_source_max_uncompressed_bytes = 20 * 1024 * 1024
    external_source_max_pixels = 1_000_000
    max_reg_num = 100
    characters = []
    character_info = {}

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.categories_dir = data_dir / "categories"
        self.pending_dir = data_dir / "pending"

    def get_categories(self):
        return ["happy", "sad", "confused"]

    def normalize_category_strict(self, value):
        return value if value in self.get_categories() else None

    def closest_category(self, value):
        return "happy" if value in {"开心", "joy"} else "confused"

    def ensure_category_dir(self, category):
        target = self.categories_dir / category
        target.mkdir(parents=True, exist_ok=True)
        return target

    def save_characters(self):
        return None

    def save_character_info(self):
        return None


def _service(tmp_path: Path) -> tuple[SourceService, DatabaseService]:
    data_dir = tmp_path / "stealer-data"
    data_dir.mkdir()
    db = DatabaseService(data_dir / "emoji.db")
    config = _Config(data_dir)
    plugin = SimpleNamespace(
        base_dir=data_dir,
        db_service=db,
        plugin_config=config,
        meme_selector=None,
    )
    return SourceService(plugin), db


def _pack(tmp_path: Path) -> Path:
    root = tmp_path / "manager-pack"
    image_dir = root / "memes" / "happy"
    image_dir.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "id": "demo-pack",
                "name": "Demo Pack",
                "version": "1.0.0",
                "categories": ["happy"],
                "license": "CC0-1.0",
            }
        ),
        encoding="utf-8",
    )
    (root / "memes_data.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "smile-1",
                        "relative_path": "memes/happy/smile.png",
                        "category": "happy",
                        "caption": "角色开心挥手",
                        "visible_text": "你好",
                        "tags": ["挥手", "问候", "category:happy"],
                        "scenes": ["打招呼"],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (image_dir / "smile.png").write_bytes(_png_bytes())
    return root


def test_pack_member_rejects_path_traversal():
    with pytest.raises(ExternalSourceSecurityError):
        safe_member_path("../../outside.png")
    with pytest.raises(ExternalSourceSecurityError):
        safe_member_path("C:\\outside.png")
    assert safe_member_path("memes/happy/a.png") == "memes/happy/a.png"


def test_http_source_blocks_local_and_non_http_urls():
    with pytest.raises(ExternalSourceSecurityError):
        validate_source_url("http://127.0.0.1/catalog", allow_http=True)
    with pytest.raises(ExternalSourceSecurityError):
        validate_source_url("file:///tmp/catalog.json")
    with pytest.raises(ExternalSourceSecurityError):
        validate_source_url("https://example.com:99999/catalog.json", resolve_dns=False)
    assert validate_source_url("https://example.com/catalog.json").startswith("https://")


@pytest.mark.asyncio
async def test_http_catalog_follows_cursor_and_normalizes_items(monkeypatch):
    source = HTTPSource("https://8.8.8.8/catalog.json", max_items=3)

    async def fake_json(url):
        if "cursor=page-2" in url:
            return {
                "items": [
                    {
                        "id": "two",
                        "url": "/two.png",
                        "category": "sad",
                        "caption": "second",
                    }
                ]
            }
        return {
            "id": "remote-demo",
            "name": "Remote Demo",
            "items": [
                {
                    "id": "one",
                    "image_url": "/one.png",
                    "emotion": "happy",
                    "tags": ["hello"],
                }
            ],
            "next_cursor": "page-2",
        }

    monkeypatch.setattr(source, "_request_json", fake_json)
    inspection = await source.inspect()
    assert inspection.source_id == "http:remote-demo"
    assert [item.external_id for item in inspection.items] == ["one", "two"]
    assert inspection.items[0].source_url == "https://8.8.8.8/one.png"
    assert inspection.categories == ["happy", "sad"]


class _FakeResponse:
    def __init__(self, status: int, *, headers=None, chunks=None):
        self.status = status
        self.headers = headers or {}
        self.content = self
        self._chunks = list(chunks or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def iter_chunked(self, _size):
        for chunk in self._chunks:
            yield chunk


class _FakeSession:
    closed = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, headers, allow_redirects):
        self.calls.append((url, dict(headers), allow_redirects))
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_http_source_drops_credentials_after_cross_origin_redirect():
    source = HTTPSource(
        "https://api.example/catalog.json",
        headers={"Authorization": "Bearer secret", "X-API-Key": "key", "User-Agent": "test"},
    )
    session = _FakeSession(
        [
            _FakeResponse(302, headers={"Location": "https://cdn.example/meme.png"}),
            _FakeResponse(200, chunks=[b"image"]),
        ]
    )
    source._session = session

    assert await source._request_bytes(source.endpoint, max_bytes=1024) == b"image"
    assert "Authorization" in session.calls[0][1]
    assert "X-API-Key" in session.calls[0][1]
    assert "Authorization" not in session.calls[1][1]
    assert "X-API-Key" not in session.calls[1][1]


def test_github_source_drops_credentials_after_cross_origin_redirect(tmp_path):
    source = GitHubSource(
        {"repository": "https://github.com/example/demo-pack", "ref": "v1"},
        cache_dir=tmp_path / "cache",
        headers={"Authorization": "Bearer secret", "X-API-Key": "key"},
    )
    initial = source._headers_for_url(
        "https://api.github.com/repos/example/demo-pack",
        initial_origin=("https", "api.github.com", 443),
        accept_json=True,
    )
    redirected = source._headers_for_url(
        "https://downloads.example/archive.zip",
        initial_origin=("https", "api.github.com", 443),
    )
    assert "Authorization" in initial
    assert "X-API-Key" in initial
    assert "Authorization" not in redirected
    assert "X-API-Key" not in redirected


def test_pack_inspection_reads_public_protocol_metadata(tmp_path):
    inspection = PackSource(_pack(tmp_path), max_items=10).inspect()
    assert inspection.ok
    assert inspection.source_id == "pack:demo-pack"
    assert inspection.name == "Demo Pack"
    assert len(inspection.items) == 1
    item = inspection.items[0]
    assert item.external_id == "smile-1"
    assert item.text_description == "角色开心挥手"
    assert item.visible_text == "你好"
    assert item.license == "CC0-1.0"


def test_pack_inspection_reads_semantic_metadata_v2(tmp_path):
    root = _pack(tmp_path)
    (root / "semantic_metadata.json").write_text(
        json.dumps(
            {
                "pack_id": "demo-pack",
                "schema_version": "2.0",
                "images": {
                    "entry-1": {
                        "relative_path": "memes/happy/smile.png",
                        "entry_id": "semantic-entry-1",
                        "content_sha256": "remote-sha",
                        "caption": "来自语义包的描述",
                        "tags": ["category:happy", "语义标签"],
                        "visible_text": "语义文字",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    item = PackSource(root).inspect().items[0]
    assert item.external_id == "semantic-entry-1"
    assert item.text_description == "来自语义包的描述"
    assert item.tags == ["category:happy", "语义标签"]
    assert item.visible_text == "语义文字"
    assert item.metadata["content_sha256"] == "remote-sha"


def test_pack_export_descriptor_supplies_nested_pack_identity(tmp_path):
    root = tmp_path / "pack"
    (root / "memes" / "happy").mkdir(parents=True)
    (root / "memes" / "happy" / "one.png").write_bytes(_png_bytes())
    (root / "meme_pack_export.json").write_text(
        json.dumps({"format": "astrbot-meme-pack", "pack": {"id": "nested-id"}}),
        encoding="utf-8",
    )
    inspection = PackSource(root).inspect()
    assert inspection.source_id == "pack:nested-id"


@pytest.mark.asyncio
async def test_github_repository_archive_strips_archive_root(tmp_path, monkeypatch):
    source_pack = _pack(tmp_path)
    archive_path = tmp_path / "github.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_pack.rglob("*"):
            if path.is_file():
                archive.write(path, Path("demo-pack-v1") / path.relative_to(source_pack))
    source = GitHubSource(
        {"repository": "https://github.com/example/demo-pack", "ref": "v1"},
        cache_dir=tmp_path / "cache",
    )

    async def fake_download():
        return archive_path

    monkeypatch.setattr(source, "_download_archive", fake_download)
    inspection = await source.inspect()
    assert inspection.source_type == "github"
    assert inspection.source_id == "github:example/demo-pack@v1"
    assert inspection.items[0].relative_path == "memes/happy/smile.png"
    assert inspection.items[0].text_description == "角色开心挥手"
    assert inspection.manifest["source"]["repo"] == "example/demo-pack"
    assert source.read_item(inspection.items[0]) == (source_pack / "memes" / "happy" / "smile.png").read_bytes()
    await source.close()


@pytest.mark.asyncio
async def test_service_infers_github_from_url_without_explicit_type(tmp_path, monkeypatch):
    service, _db = _service(tmp_path)

    class _Reader:
        async def inspect(self):
            return PackSource(_pack(tmp_path)).inspect()

        async def close(self):
            return None

    monkeypatch.setattr(service, "_github_reader", lambda _spec: _Reader())
    inspection, reader = await service.inspect(
        {"url": "https://github.com/example/demo-pack"}
    )
    assert inspection.source_type == "meme_pack"
    assert reader is not None


def test_pack_rejects_invalid_or_empty_archive(tmp_path):
    invalid = tmp_path / "invalid.zip"
    invalid.write_bytes(b"not a zip")
    with pytest.raises(ExternalSourceError, match="invalid or unsupported"):
        PackSource(invalid).inspect()
    empty = tmp_path / "empty"
    empty.mkdir()
    inspection = PackSource(empty).inspect()
    assert not inspection.ok
    assert "no supported images" in inspection.errors[0]


def test_github_repository_descriptor_supports_tree_and_query_ref():
    owner, repo, ref, subpath = _parse_repository(
        {
            "repository": "https://github.com/Example/Meme-Pack/tree/feature%2Fv3/memes",
        }
    )
    assert (owner, repo, ref, subpath) == ("Example", "Meme-Pack", "feature/v3", "memes")

    owner, repo, ref, subpath = _parse_repository(
        {
            "repository": "owner/repo?ref=release%2F2026&subpath=packs/main",
        }
    )
    assert (owner, repo, ref, subpath) == ("owner", "repo", "release/2026", "packs/main")


def test_github_repository_rejects_non_github_hosts():
    with pytest.raises(ExternalSourceSecurityError):
        _parse_repository({"repository": "https://evil.example/owner/repo"})


@pytest.mark.asyncio
async def test_pack_import_copies_tracks_and_deduplicates(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    await service.initialize()

    first = await service.import_now(
        {"source_type": "meme_pack", "path": str(pack), "scope_mode": "public"}
    )
    assert first["status"] == "completed"
    assert first["imported"] == 1
    assert first["failed"] == 0

    rows = db.get_index_cache_readonly()
    assert len(rows) == 1
    imported_path, metadata = next(iter(rows.items()))
    assert Path(imported_path).is_file()
    assert Path(imported_path) != pack / "memes" / "happy" / "smile.png"
    assert metadata["retention_class"] == "external"
    assert metadata["desc"] == "角色开心挥手"
    assert metadata["overlay_text"] == "你好"
    assert metadata["tags"] == ["挥手", "问候"]

    links = db.get_source_items("pack:demo-pack")
    assert len(links) == 1
    assert links[0]["path"] == imported_path
    assert links[0]["license"] == "CC0-1.0"

    second = await service.import_now({"source_type": "meme_pack", "path": str(pack)})
    assert second["status"] == "completed"
    assert second["duplicates"] == 1
    assert db.count_total() == 1
    assert (pack / "memes" / "happy" / "smile.png").is_file()
    await service.close()


@pytest.mark.asyncio
async def test_direct_import_deduplicates_against_pending_pool(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    raw = (pack / "memes" / "happy" / "smile.png").read_bytes()
    pending_path = tmp_path / "pending-copy.png"
    pending_path.write_bytes(raw)
    await db.insert_pending(
        {
            "path": str(pending_path),
            "hash": hashlib.sha256(raw).hexdigest(),
            "category": "happy",
        }
    )
    await service.initialize()
    result = await service.import_now(
        {"source_type": "meme_pack", "path": str(pack), "review": False}
    )
    assert result["duplicates"] == 1
    assert result["imported"] == 0
    assert db.count_total() == 0
    assert db.count_pending() == 1
    await service.close()


@pytest.mark.asyncio
async def test_character_assignment_updates_an_existing_pending_duplicate(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    raw = (pack / "memes" / "happy" / "smile.png").read_bytes()
    await db.insert_pending(
        {
            "path": str(tmp_path / "pending.png"),
            "hash": hashlib.sha256(raw).hexdigest(),
            "category": "happy",
        }
    )
    pending_path = tmp_path / "pending.png"
    pending_path.write_bytes(raw)
    await service.initialize()
    result = await service.import_now(
        {
            "source_type": "meme_pack",
            "path": str(pack),
            "character": "series-one",
            "create_character": True,
        }
    )
    assert result["duplicates"] == 1
    assert db.get_pending_by_hash(hashlib.sha256(raw).hexdigest())["character"] == "series-one"
    await service.close()


@pytest.mark.asyncio
async def test_import_can_create_and_assign_character_to_series(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    await service.initialize()
    result = await service.import_now(
        {
            "source_type": "meme_pack",
            "path": str(pack),
            "character": "Neuro-Sama",
            "create_character": True,
        }
    )
    assert result["imported"] == 1
    assert "neuro-sama" in service.plugin.plugin_config.characters
    rows = db.get_index_cache_readonly()
    assert next(iter(rows.values()))["character"] == "neuro-sama"
    await service.close()


@pytest.mark.asyncio
async def test_rejected_scope_does_not_create_character_or_source(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    await service.initialize()
    before = list(service.plugin.plugin_config.characters)

    result = await service.import_now(
        {
            "source_type": "meme_pack",
            "path": str(pack),
            "scope_mode": "local",
            "character": "rejected-series",
            "create_character": True,
        }
    )

    assert result["status"] == "failed"
    assert service.plugin.plugin_config.characters == before
    assert db.get_sources() == []
    await service.close()


@pytest.mark.asyncio
async def test_staged_upload_cleanup_preserves_referenced_archives(tmp_path):
    service, db = _service(tmp_path)
    await service.initialize()
    upload_dir = service.import_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    old_path = upload_dir / "old.zip"
    recent_path = upload_dir / "recent.zip"
    referenced_path = upload_dir / "referenced.zip"
    for path in (old_path, recent_path, referenced_path):
        path.write_bytes(b"archive")
    old_time = time.time() - service.STAGED_UPLOAD_TTL_SECONDS - 60
    os.utime(old_path, (old_time, old_time))
    await db.upsert_source(
        {
            "source_id": "pack:referenced",
            "source_type": "meme_pack",
            "name": "Referenced",
            "endpoint": str(referenced_path),
            "config": {"path": str(referenced_path)},
        }
    )

    removed = await service.cleanup_staged_uploads()

    assert removed == 1
    assert not old_path.exists()
    assert recent_path.exists()
    assert referenced_path.exists()
    assert await service.delete_source("pack:referenced")
    assert not referenced_path.exists()
    await service.close()


def test_public_source_redacts_credentials():
    source = SourceService._public_source(
        {
            "source_id": "http:test",
            "source_type": "http_json",
            "endpoint": "https://example.com/catalog?token=secret&page=1",
            "config": {
                "endpoint": "https://example.com/catalog?api_key=secret",
                "repository": "https://github.com/example/repo?token=secret&ref=v1",
                "headers": {"Authorization": "Bearer secret"},
            },
        }
    )
    assert "secret" not in source["endpoint"]
    assert "secret" not in source["config"]["endpoint"]
    assert "secret" not in source["config"]["repository"]
    assert source["config"]["headers"]["Authorization"] == "********"


def test_item_provenance_redacts_credentials_in_image_url():
    from core.sources.models import SourceItem

    item = SourceItem(
        external_id="one",
        source_url="https://cdn.example/meme.png?token=secret&size=small",
        metadata={"url": "https://cdn.example/meme.png?api_key=secret", "id": "one"},
    )
    public = SourceService._public_item_metadata(item)
    assert "secret" not in str(public)
    assert "size=small" in public["source_url"]


def test_inspection_manifest_redacts_nested_credentials(tmp_path):
    service, _db = _service(tmp_path)
    safe = service._json_safe(
        {
            "name": "Public pack",
            "license": "CC-BY-4.0",
            "source": {
                "repository": "https://github.com/example/repo?token=secret&ref=v1",
                "api_key": "secret",
            },
        }
    )
    assert safe["name"] == "Public pack"
    assert safe["license"] == "CC-BY-4.0"
    assert "secret" not in str(safe)
    assert "ref=v1" in safe["source"]["repository"]


def test_discovers_same_instance_meme_manager_pack(tmp_path):
    service, _db = _service(tmp_path)
    pack = tmp_path / "astrbot_plugin_meme_manager" / "packs" / "community"
    pack.mkdir(parents=True)
    (pack / "manifest.json").write_text(
        json.dumps({"id": "community", "name": "Community Pack", "version": "4"}),
        encoding="utf-8",
    )
    sources = service.discover_meme_manager_packs()
    assert sources == [
        {
            "source_id": "pack:community",
            "source_type": "meme_pack",
            "name": "Community Pack",
            "endpoint": str(pack.resolve()),
            "enabled": True,
            "status": "discovered",
            "discovered": True,
            "version": "4",
            "config": {
                "path": str(pack.resolve()),
                "provider": "meme_manager",
            },
        }
    ]


@pytest.mark.asyncio
async def test_invalid_pack_image_is_reported_without_library_write(tmp_path):
    service, db = _service(tmp_path)
    pack = _pack(tmp_path)
    (pack / "memes" / "happy" / "smile.png").write_bytes(b"not an image")
    await service.initialize()
    result = await service.import_now({"source_type": "meme_pack", "path": str(pack)})
    assert result["status"] == "completed"
    assert result["failed"] == 1
    assert db.count_total() == 0
    await service.close()


def test_external_schema_is_additive_and_tracks_sources(tmp_path):
    db_path = tmp_path / "legacy.db"
    DatabaseService(db_path)
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(emoji)").fetchall()
        }
        external_version = connection.execute(
            "SELECT value FROM meta WHERE key='external_schema_version'"
        ).fetchone()[0]
    assert {"meme_source", "meme_source_item"}.issubset(tables)
    assert "retention_class" in columns
    assert external_version == "1"


@pytest.mark.asyncio
async def test_source_registry_marks_missing_items_stale(tmp_path):
    db = DatabaseService(tmp_path / "sources.db")
    await db.upsert_source(
        {
            "source_id": "pack:test",
            "source_type": "meme_pack",
            "name": "Test",
            "endpoint": "C:/packs/test",
            "config": {"path": "C:/packs/test"},
        }
    )
    await db.insert_batch(
        [
            {
                "path": "C:/library/a.png",
                "hash": "hash-a",
                "category": "happy",
                "retention_class": "external",
            }
        ]
    )
    await db.link_source_item(
        {
            "source_id": "pack:test",
            "external_id": "a",
            "path": "C:/library/a.png",
            "remote_hash": "hash-a",
        }
    )
    assert db.count_stale_source_items("pack:test") == 0
    assert await db.mark_source_items_stale("pack:test") == 1
    assert db.count_stale_source_items("pack:test") == 1
    await db.link_source_item(
        {
            "source_id": "pack:test",
            "external_id": "a",
            "path": "C:/library/a.png",
            "remote_hash": "hash-a",
        }
    )
    assert db.count_stale_source_items("pack:test") == 0
    assert db.get_sources()[0]["config"]["path"] == "C:/packs/test"


@pytest.mark.asyncio
async def test_source_registry_reconciles_only_after_successful_catalog(tmp_path):
    db = DatabaseService(tmp_path / "reconcile.db")
    await db.upsert_source(
        {
            "source_id": "pack:test",
            "source_type": "meme_pack",
            "name": "Test",
        }
    )
    for external_id in ("present", "missing"):
        await db.link_source_item(
            {
                "source_id": "pack:test",
                "external_id": external_id,
                "remote_hash": f"hash-{external_id}",
            }
        )
    changed = await db.reconcile_source_items("pack:test", ["present"])
    assert changed == 1
    rows = {row["external_id"]: row for row in db.get_source_items("pack:test")}
    assert rows["present"]["stale"] == 0
    assert rows["missing"]["stale"] == 1
