"""Read-only GitHub repository source for Meme Pack repositories."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, quote, unquote, urljoin, urlsplit

from .http_source import (
    _SENSITIVE_HEADERS,
    _ValidatingResolver,
    _url_origin,
    validate_source_url,
)
from .models import ExternalSourceError, ExternalSourceSecurityError, PackInspection
from .pack_source import PackSource, safe_member_path


_GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]{0,254}$")
_HEADER_NAMES = frozenset({"accept", "authorization", "user-agent", "x-api-key"})


def _safe_ref(value: Any) -> str:
    ref = str(value or "").strip()
    if (
        not ref
        or len(ref) > 255
        or not _REF_RE.fullmatch(ref)
        or ref.startswith("/")
        or ref.endswith("/")
        or ".." in ref
        or "@{" in ref
    ):
        raise ExternalSourceSecurityError("GitHub ref is invalid")
    return ref


def _safe_repo_part(value: Any, label: str) -> str:
    part = str(value or "").strip()
    if not part or len(part) > 100 or not re.fullmatch(r"[A-Za-z0-9_.-]+", part):
        raise ExternalSourceSecurityError(f"GitHub {label} is invalid")
    return part


def _parse_repository(spec: Mapping[str, Any]) -> tuple[str, str, str, str]:
    raw = str(
        spec.get("repository")
        or spec.get("repo")
        or spec.get("url")
        or spec.get("endpoint")
        or ""
    ).strip()
    if not raw:
        raise ExternalSourceError("GitHub source requires a repository URL or owner/repo")
    if "://" not in raw:
        raw = f"https://github.com/{raw.lstrip('/')}"
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or (parsed.hostname or "").lower() not in _GITHUB_HOSTS:
        raise ExternalSourceSecurityError("GitHub source must use an HTTPS github.com URL")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ExternalSourceError("GitHub source URL must include owner and repository")
    owner = _safe_repo_part(parts[0], "owner")
    repo = _safe_repo_part(parts[1].removesuffix(".git"), "repository")
    ref = str(spec.get("ref") or "").strip()
    subpath = str(spec.get("subpath") or "").strip()
    if len(parts) >= 4 and parts[2].lower() == "tree":
        ref = ref or parts[3]
        if len(parts) > 4 and not subpath:
            subpath = "/".join(parts[4:])
    query_values = parse_qs(parsed.query, keep_blank_values=True)
    query = {key: values[-1] for key, values in query_values.items() if values}
    ref = ref or query.get("ref", "")
    subpath = subpath or query.get("subpath", "")
    if subpath:
        subpath = safe_member_path(subpath)
    return owner, repo, _safe_ref(ref) if ref else "", subpath


class GitHubSource:
    """Download a GitHub repository archive and read it as a Meme Pack.

    The implementation uses GitHub's archive endpoint instead of cloning or
    executing a repository.  A local cache is retained for repeat syncs, but
    each preflight refreshes the selected ref.
    """

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        cache_dir: str | Path,
        max_items: int = 2000,
        max_archive_bytes: int = 1024 * 1024 * 1024,
        max_uncompressed_bytes: int = 4 * 1024 * 1024 * 1024,
        max_file_bytes: int = 32 * 1024 * 1024,
        max_members: int = 20_000,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.owner, self.repo, self.ref, self.subpath = _parse_repository(spec)
        self.max_items = max(1, int(max_items))
        self.max_archive_bytes = max(1, int(max_archive_bytes))
        self.max_uncompressed_bytes = max(1, int(max_uncompressed_bytes))
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.max_members = max(1, int(max_members))
        raw_headers = headers if isinstance(headers, Mapping) else spec.get("headers")
        raw_headers = raw_headers if isinstance(raw_headers, Mapping) else {}
        self.headers: dict[str, str] = {}
        for key, value in list(raw_headers.items())[:16]:
            if str(key).lower() in _HEADER_NAMES:
                self.headers[str(key)[:80]] = str(value)[:4096]
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_path: Path | None = None
        self._pack: PackSource | None = None

    @property
    def source_id(self) -> str:
        ref = self.ref or "default"
        return f"github:{self.owner}/{self.repo}@{ref}"[:190]

    @property
    def display_url(self) -> str:
        value = f"https://github.com/{self.owner}/{self.repo}"
        if self.ref:
            value += f"/tree/{quote(self.ref, safe='')}"
        if self.subpath:
            value += f"/{self.subpath}"
        return value

    async def inspect(self) -> PackInspection:
        if not self.ref:
            self.ref = await self._default_branch()
        archive_path = await self._download_archive()
        root_prefix = await asyncio.to_thread(
            PackSource.detect_archive_root,
            archive_path,
            max_members=self.max_members,
        )
        member_prefix = root_prefix
        if self.subpath:
            member_prefix = f"{root_prefix}/{self.subpath}" if root_prefix else self.subpath
        reader = PackSource(
            archive_path,
            max_items=self.max_items,
            max_archive_bytes=self.max_archive_bytes,
            max_uncompressed_bytes=self.max_uncompressed_bytes,
            max_file_bytes=self.max_file_bytes,
            max_members=self.max_members,
            member_prefix=member_prefix,
        )
        inspection = await asyncio.to_thread(reader.inspect)
        manifest = dict(inspection.manifest)
        manifest["source"] = {
            "type": "github",
            "repo": f"{self.owner}/{self.repo}",
            "ref": self.ref,
            "subpath": self.subpath,
        }
        inspection.source_type = "github"
        inspection.source_id = self.source_id
        inspection.endpoint = self.display_url
        inspection.name = str(manifest.get("name") or f"{self.owner}/{self.repo}")[:160]
        inspection.manifest = manifest
        inspection.source_path = archive_path
        inspection.is_archive = True
        self._pack = reader
        return inspection

    def read_item(self, item):  # noqa: ANN001
        if self._pack is None:
            raise ExternalSourceError("GitHub source has not been inspected")
        return self._pack.read_item(item)

    async def close(self) -> None:
        self._pack = None

    async def _default_branch(self) -> str:
        payload = await self._request_json(
            f"https://api.github.com/repos/{self.owner}/{self.repo}"
        )
        branch = payload.get("default_branch") if isinstance(payload, dict) else None
        return _safe_ref(branch or "main")

    async def _download_archive(self) -> Path:
        await asyncio.to_thread(self.cache_dir.mkdir, parents=True, exist_ok=True)
        key = hashlib.sha256(
            f"{self.owner}/{self.repo}@{self.ref}#{self.subpath}".encode("utf-8")
        ).hexdigest()[:24]
        target = self.cache_dir / f"github_{key}.zip"
        temporary = self.cache_dir / f".{target.name}.{uuid.uuid4().hex}.part"
        archive_url = validate_source_url(
            f"https://codeload.github.com/{self.owner}/{self.repo}/zip/{quote(self.ref, safe='')}",
            allow_http=False,
        )
        try:
            await self._stream_to_file(archive_url, temporary)
            await asyncio.to_thread(os.replace, temporary, target)
        finally:
            if temporary.exists():
                try:
                    await asyncio.to_thread(temporary.unlink)
                except OSError:
                    pass
        self.cache_path = target
        return target

    async def _request_json(self, url: str) -> Any:
        raw = await self._request_bytes(url, max_bytes=2 * 1024 * 1024, accept_json=True)
        try:
            value = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalSourceError("GitHub API returned invalid JSON") from exc
        return value

    async def _request_bytes(
        self,
        url: str,
        *,
        max_bytes: int,
        accept_json: bool = False,
    ) -> bytes:
        current = validate_source_url(url, allow_http=False)
        initial_origin = _url_origin(current)
        try:
            import aiohttp  # type: ignore
        except ImportError as exc:
            raise ExternalSourceError(
                "aiohttp is required for safe GitHub source requests"
            ) from exc
        timeout = aiohttp.ClientTimeout(total=90, connect=15, sock_read=60)
        resolver = _ValidatingResolver(aiohttp.resolver.ThreadedResolver())
        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                raise_for_status=False,
                connector=aiohttp.TCPConnector(resolver=resolver),
            ) as session:
                for _ in range(4):
                    headers = self._headers_for_url(
                        current,
                        initial_origin=initial_origin,
                        accept_json=accept_json,
                    )
                    async with session.get(current, headers=headers, allow_redirects=False) as response:
                        if 300 <= response.status < 400:
                            location = response.headers.get("Location")
                            if not location:
                                raise ExternalSourceError(
                                    f"GitHub redirect has no Location header ({response.status})"
                                )
                            current = validate_source_url(
                                urljoin(current, location), allow_http=False
                            )
                            continue
                        if response.status < 200 or response.status >= 300:
                            raise ExternalSourceError(
                                f"GitHub source returned status {response.status}"
                            )
                        declared = response.headers.get("Content-Length")
                        if declared and int(declared) > max_bytes:
                            raise ExternalSourceSecurityError(
                                "GitHub response exceeds the byte limit"
                            )
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in response.content.iter_chunked(64 * 1024):
                            total += len(chunk)
                            if total > max_bytes:
                                raise ExternalSourceSecurityError(
                                    "GitHub response exceeds the byte limit"
                                )
                            chunks.append(chunk)
                        return b"".join(chunks)
                raise ExternalSourceError("too many GitHub redirects")
        except ExternalSourceError:
            raise
        except Exception as exc:
            raise ExternalSourceError(f"GitHub source request failed: {exc}") from exc

    async def _stream_to_file(self, url: str, target: Path) -> int:
        try:
            import aiohttp  # type: ignore
        except ImportError as exc:
            raise ExternalSourceError(
                "aiohttp is required for safe GitHub source requests"
            ) from exc
        timeout = aiohttp.ClientTimeout(total=180, connect=15, sock_read=90)
        resolver = _ValidatingResolver(aiohttp.resolver.ThreadedResolver())
        current = validate_source_url(url, allow_http=False)
        initial_origin = _url_origin(current)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout,
                raise_for_status=False,
                connector=aiohttp.TCPConnector(resolver=resolver),
            ) as session:
                for _ in range(4):
                    headers = self._headers_for_url(
                        current,
                        initial_origin=initial_origin,
                    )
                    async with session.get(current, headers=headers, allow_redirects=False) as response:
                        if 300 <= response.status < 400:
                            location = response.headers.get("Location")
                            if not location:
                                raise ExternalSourceError(
                                    f"GitHub redirect has no Location header ({response.status})"
                                )
                            current = validate_source_url(
                                urljoin(current, location), allow_http=False
                            )
                            continue
                        if response.status < 200 or response.status >= 300:
                            raise ExternalSourceError(
                                f"GitHub archive returned status {response.status}"
                            )
                        declared = response.headers.get("Content-Length")
                        if declared and int(declared) > self.max_archive_bytes:
                            raise ExternalSourceSecurityError(
                                "GitHub archive exceeds the compressed byte limit"
                            )
                        total = 0
                        with target.open("wb") as output:
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                total += len(chunk)
                                if total > self.max_archive_bytes:
                                    raise ExternalSourceSecurityError(
                                        "GitHub archive exceeds the compressed byte limit"
                                    )
                                output.write(chunk)
                        return total
                raise ExternalSourceError("too many GitHub redirects")
        except ExternalSourceError:
            raise
        except Exception as exc:
            raise ExternalSourceError(f"GitHub archive download failed: {exc}") from exc

    def _headers_for_url(
        self,
        url: str,
        *,
        initial_origin: tuple[str, str, int],
        accept_json: bool = False,
    ) -> dict[str, str]:
        """Keep GitHub credentials on the original API/archive origin only."""

        headers = dict(self.headers)
        headers.setdefault("User-Agent", "astrbot-plugin-stealer/3")
        if accept_json:
            headers.setdefault("Accept", "application/vnd.github+json")
        if _url_origin(url) != initial_origin:
            headers = {
                key: value
                for key, value in headers.items()
                if key.lower() not in _SENSITIVE_HEADERS
            }
        return headers


__all__ = ["GitHubSource"]
