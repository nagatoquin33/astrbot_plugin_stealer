"""Conservative HTTP JSON catalog reader used by the external-source API."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import socket
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import quote_plus, urljoin, urlsplit

from .models import ExternalSourceError, ExternalSourceSecurityError, SourceInspection, SourceItem


_TUN_FAKE_IP_RANGE = ipaddress.ip_network("198.18.0.0/15")
_SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key"})


def _url_origin(value: str) -> tuple[str, str, int]:
    """Return a normalized origin suitable for credential forwarding checks."""

    parsed = urlsplit(str(value or ""))
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host, int(port)


def _unsafe_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, dns_answer: bool) -> bool:
    # Clash/mihomo TUN commonly returns RFC 2544 benchmark addresses for public
    # hostnames.  Permit that DNS answer while still rejecting a literal
    # 198.18/15 URL and all actual local/private ranges.
    if dns_answer and isinstance(address, ipaddress.IPv4Address) and address in _TUN_FAKE_IP_RANGE:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def validate_source_url(
    url: str,
    *,
    allow_http: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Validate a source URL before any network request is attempted."""

    value = str(url or "").strip()
    if len(value) > 4096 or any(ord(character) < 32 for character in value):
        raise ExternalSourceSecurityError("external source URL is malformed or too long")
    parsed = urlsplit(value)
    allowed = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme.lower() not in allowed:
        raise ExternalSourceSecurityError("external source URL must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ExternalSourceSecurityError("external source URL has no safe hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ExternalSourceSecurityError("external source URL has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ExternalSourceSecurityError("external source URL has an invalid port")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain", "ip6-localhost"} or host.endswith(".localhost"):
        raise ExternalSourceSecurityError("loopback host is not allowed")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and _unsafe_ip(address, dns_answer=False):
        raise ExternalSourceSecurityError("private or loopback host is not allowed")
    # Reject a hostname that resolves to a local address.  DNS failures are
    # left to the HTTP client so test/dev hosts and offline inspection still
    # produce a useful network error instead of a false SSRF positive.
    if resolve_dns:
        try:
            addresses = socket.getaddrinfo(
                host,
                port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except (OSError, ValueError):
            addresses = []
        for address_info in addresses:
            resolved = address_info[4][0]
            try:
                resolved_ip = ipaddress.ip_address(resolved)
            except ValueError:
                continue
            if _unsafe_ip(resolved_ip, dns_answer=True):
                raise ExternalSourceSecurityError("source hostname resolves to a private address")
    # Keep fragments out of a persisted source descriptor.
    return value.split("#", 1)[0]


class _ValidatingResolver:
    """Validate the addresses aiohttp will actually connect to.

    The URL preflight catches ordinary private hosts.  Re-checking the
    resolver result at connection time closes the DNS-rebinding gap between
    that preflight and the socket connection.
    """

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: int = socket.AF_INET,
    ) -> list[dict[str, Any]]:
        results = await self._delegate.resolve(host, port, family)
        for result in results:
            try:
                address = ipaddress.ip_address(str(result.get("host") or ""))
            except ValueError:
                raise OSError("HTTP resolver returned an invalid address") from None
            if _unsafe_ip(address, dns_answer=True):
                raise OSError("HTTP resolver returned a private address")
        return results

    async def close(self) -> None:
        await self._delegate.close()


def _list_value(value: Any, limit: int = 16) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,，;；\n\t]+", value)
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= limit:
            break
    return result


class HTTPSource:
    """Read a JSON catalog and download its image items on demand.

    Supported response shapes are ``{"items": [...]}``, ``{"memes": [...]}``,
    ``{"data": [...]}``, or a bare list.  Each item needs an ``id`` (or a
    stable fallback) and an absolute/relative ``url``/``image_url``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        allow_http: bool = False,
        max_items: int = 2000,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_file_bytes: int = 32 * 1024 * 1024,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.endpoint = validate_source_url(endpoint, allow_http=allow_http)
        self.allow_http = allow_http
        self.max_items = max(1, int(max_items))
        self.max_response_bytes = max(1024, int(max_response_bytes))
        self.max_file_bytes = max(1024, int(max_file_bytes))
        self.headers: dict[str, str] = {}
        for key, value in list((headers or {}).items())[:16]:
            if str(key).lower() in {"accept", "authorization", "user-agent", "x-api-key"}:
                self.headers[str(key)[:80]] = str(value)[:4096]
        self._credential_origin = _url_origin(self.endpoint)
        self._catalog: SourceInspection | None = None
        self._session: Any = None

    async def inspect(self, *, cursor: str | None = None) -> SourceInspection:
        items: list[SourceItem] = []
        warnings: list[str] = []
        raw_seen = 0
        first_payload: Any = None
        page_cursor = str(cursor or "").strip() or None
        next_cursor: str | None = None
        seen_cursors: set[str] = set()
        truncated = False
        for _page in range(100):
            endpoint = self.endpoint
            if page_cursor:
                separator = "&" if "?" in endpoint else "?"
                endpoint = f"{endpoint}{separator}cursor={quote_plus(page_cursor)}"
            payload = await self._request_json(endpoint)
            if first_payload is None:
                first_payload = payload
            raw_items, next_cursor = self._extract_items(payload)
            remaining = self.max_items - raw_seen
            page_items = raw_items[:remaining]
            for offset, raw in enumerate(page_items):
                index = raw_seen + offset
                if not isinstance(raw, dict):
                    warnings.append(f"catalog item {index + 1} is not an object")
                    continue
                try:
                    item = self._normalize_item(raw, index)
                except ExternalSourceError as exc:
                    warnings.append(str(exc))
                    continue
                items.append(item)
            raw_seen += len(page_items)
            if len(raw_items) > remaining or (raw_seen >= self.max_items and next_cursor):
                truncated = True
                warnings.append(
                    f"catalog contains more than {self.max_items} items; import is truncated"
                )
                break
            if not next_cursor:
                break
            if next_cursor in seen_cursors:
                warnings.append("catalog pagination returned a repeated cursor")
                break
            seen_cursors.add(next_cursor)
            page_cursor = next_cursor
        else:
            truncated = True
            warnings.append("catalog pagination exceeded 100 pages")
        manifest: dict[str, Any] = {}
        if isinstance(first_payload, dict):
            for key in (
                "id",
                "name",
                "version",
                "license",
                "attribution",
                "author",
                "source",
            ):
                if first_payload.get(key) is not None:
                    manifest[key] = first_payload[key]
        default_license = str(manifest.get("license") or "").strip()
        default_attribution = str(
            manifest.get("attribution") or manifest.get("author") or ""
        ).strip()
        for item in items:
            item.license = item.license or default_license
            item.attribution = item.attribution or default_attribution
        if next_cursor:
            manifest["next_cursor"] = next_cursor
        source_name = str(manifest.get("name") or manifest.get("id") or urlsplit(self.endpoint).hostname or "HTTP source")
        source_id = str(manifest.get("id") or "").strip()
        if not source_id:
            source_id = hashlib.sha256(self.endpoint.encode("utf-8")).hexdigest()[:16]
        inspection = SourceInspection(
            source_type="http_json",
            source_id=f"http:{source_id}"[:190],
            name=source_name[:160],
            endpoint=self.endpoint,
            manifest=manifest,
            items=items,
            warnings=warnings,
            total_bytes=sum(item.size for item in items),
            truncated=truncated,
        )
        self._catalog = inspection
        return inspection

    async def read_item(self, item: SourceItem) -> bytes:
        if not item.source_url:
            raise ExternalSourceError(f"catalog item has no URL: {item.external_id}")
        return await self._request_bytes(item.source_url, max_bytes=self.max_file_bytes)

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

    @staticmethod
    def _extract_items(payload: Any) -> tuple[list[Any], str | None]:
        if isinstance(payload, list):
            return payload, None
        if not isinstance(payload, dict):
            raise ExternalSourceError("HTTP source response must be a JSON object or array")
        for key in ("items", "memes", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value, str(payload.get("next_cursor") or payload.get("next") or "").strip() or None
        raise ExternalSourceError("HTTP source response has no items array")

    def _normalize_item(self, raw: dict[str, Any], index: int) -> SourceItem:
        raw_url = raw.get("url") or raw.get("image_url") or raw.get("source_url") or raw.get("src")
        if not raw_url:
            raise ExternalSourceError(f"catalog item {index + 1} has no image URL")
        item_url = validate_source_url(
            urljoin(self.endpoint, str(raw_url)),
            allow_http=self.allow_http,
            resolve_dns=False,
        )
        fallback = PurePosixPath(urlsplit(item_url).path).name or f"item-{index + 1}"
        external_id = str(raw.get("id") or raw.get("external_id") or raw.get("key") or fallback).strip()
        try:
            item_size = max(0, int(raw.get("bytes") or raw.get("size") or 0))
        except (TypeError, ValueError):
            item_size = 0
        return SourceItem(
            external_id=external_id[:180],
            relative_path=str(raw.get("path") or raw.get("relative_path") or "").strip(),
            category=str(raw.get("category") or raw.get("emotion") or "").strip(),
            description=str(raw.get("description") or raw.get("desc") or "").strip(),
            caption=str(raw.get("caption") or "").strip(),
            visible_text=str(raw.get("visible_text") or raw.get("overlay_text") or "").strip(),
            tags=_list_value(raw.get("tags")),
            scenes=_list_value(raw.get("scenes") or raw.get("scene")),
            emotions=_list_value(raw.get("emotions")),
            filename=str(raw.get("filename") or fallback).strip()[:180],
            source_url=item_url,
            license=str(raw.get("license") or "").strip(),
            attribution=str(raw.get("attribution") or raw.get("author") or "").strip(),
            metadata=dict(raw),
            size=item_size,
        )

    async def _request_json(self, url: str) -> Any:
        raw = await self._request_bytes(url, max_bytes=self.max_response_bytes, accept_json=True)
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalSourceError("HTTP source returned invalid JSON") from exc

    def _headers_for_url(self, url: str, *, accept_json: bool = False) -> dict[str, str]:
        """Build request headers without forwarding credentials across origins."""

        headers = dict(self.headers)
        if accept_json:
            headers.setdefault("Accept", "application/json")
        if _url_origin(url) != self._credential_origin:
            headers = {
                key: value
                for key, value in headers.items()
                if key.lower() not in _SENSITIVE_HEADERS
            }
        return headers

    async def _request_bytes(self, url: str, *, max_bytes: int, accept_json: bool = False) -> bytes:
        current = validate_source_url(url, allow_http=self.allow_http)
        for _ in range(4):
            try:
                import aiohttp  # type: ignore
            except ImportError as exc:
                raise ExternalSourceError(
                    "aiohttp is required for safe external source requests"
                ) from exc
            timeout = aiohttp.ClientTimeout(total=45, connect=10, sock_read=30)
            headers = self._headers_for_url(current, accept_json=accept_json)
            try:
                if self._session is None or self._session.closed:
                    resolver = _ValidatingResolver(aiohttp.resolver.ThreadedResolver())
                    self._session = aiohttp.ClientSession(
                        timeout=timeout,
                        raise_for_status=False,
                        connector=aiohttp.TCPConnector(resolver=resolver),
                    )
                async with self._session.get(
                    current,
                    headers=headers,
                    allow_redirects=False,
                ) as response:
                    if 300 <= response.status < 400:
                        location = response.headers.get("Location")
                        if not location:
                            raise ExternalSourceError(
                                f"HTTP redirect has no Location header ({response.status})"
                            )
                        current = validate_source_url(
                            urljoin(current, location),
                            allow_http=self.allow_http,
                        )
                        continue
                    if response.status < 200 or response.status >= 300:
                        raise ExternalSourceError(
                            f"HTTP source returned status {response.status}"
                        )
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > max_bytes:
                        raise ExternalSourceSecurityError(
                            "HTTP response exceeds the byte limit"
                        )
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise ExternalSourceSecurityError(
                                "HTTP response exceeds the byte limit"
                            )
                        chunks.append(chunk)
                    return b"".join(chunks)
            except ExternalSourceError:
                raise
            except Exception as exc:
                raise ExternalSourceError(f"HTTP source request failed: {exc}") from exc
        raise ExternalSourceError("too many HTTP redirects")
