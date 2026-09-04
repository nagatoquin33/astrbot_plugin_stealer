# External meme sources

Version 3.0 adds a read-only source boundary for importing and synchronizing meme libraries. Images are copied into `astrbot_plugin_stealer` after validation; the importer never executes source code or edits the source library.

## Supported sources

### AstrBot Meme Pack / Meme Manager

The importer accepts the following layouts:

- A Meme Manager v4 pack directory containing `manifest.json`.
- An AstrBot Meme Pack ZIP or `.meme-pack` export.
- A generic ZIP/directory containing supported image files (`png`, `jpg`/`jpeg`, `gif`, `webp`, or `bmp`).
- Legacy metadata files named `meme_pack_export.json`, `memes_data.json`, `semantic_index.json`, or `semantic_metadata.json`.

The preferred shareable layout is an AstrBot Meme Pack v2 export:

```text
my-pack/
├─ manifest.json                 # pack id/name/categories (recommended)
├─ meme_pack_export.json         # optional v2 export descriptor
├─ memes/<category>/*            # required image files
├─ semantic_metadata.json        # optional per-image descriptions/tags/OCR
└─ previews/                     # optional previews; ignored by the importer
```

`manifest.json` and `semantic_metadata.json` are optional for a plain image ZIP. In that case the category is inferred from the first directory below `memes/` (or from the image's parent directory), and descriptions/tags remain empty until edited or analyzed in the WebUI. Files under `previews/`, `thumbnails/`, and similar preview directories are skipped.

Same-instance Meme Manager packs under its `packs/<pack-id>/` data directory are discovered automatically. Browser uploads are staged under this plugin's data directory. Local paths sent through the Web API must remain inside the current AstrBot plugin-data root.

The reader follows the public [AstrBot Meme Pack protocol](https://github.com/anka-afk/astrbot-meme-pack-index/blob/main/ASTRBOT_MEME_PACK_PROTOCOL_ZH.md). Unknown metadata fields are ignored safely.

### GitHub meme repository

GitHub repositories are read through the HTTPS archive endpoint and then passed through the same pack reader. The WebUI accepts either `owner/repository` or an HTTPS URL such as:

```text
https://github.com/DDZS987/astrbot-meme-pack-semantic-01
https://github.com/DDZS987/astrbot-meme-pack-semantic-01/tree/v1.1.0
```

An optional `/tree/<ref>/<subpath>` selects a branch/tag and a directory within the repository. `?ref=<ref>&subpath=<path>` can be used when a branch name contains `/`. If no ref is supplied, the default branch is resolved through the GitHub API. The repository is downloaded as a bounded ZIP into `external_sources/github_cache`; no `git clone`, hook, or repository code is executed. A repository should expose the pack files at its root or below the selected subpath. The [DDZS987 semantic pack](https://github.com/DDZS987/astrbot-meme-pack-semantic-01) is an example of the preferred layout: it provides `manifest.json`, `meme_pack_export.json`, `memes/`, and `semantic_metadata.json`.

### HTTPS JSON catalog

The endpoint may return a top-level array or an object whose item array is named `items`, `memes`, `data`, or `results`.

```json
{
  "id": "community-reactions",
  "name": "Community Reactions",
  "version": "2026.09",
  "license": "CC-BY-4.0",
  "attribution": "Example Community",
  "items": [
    {
      "id": "happy-001",
      "url": "https://cdn.example.org/memes/happy-001.webp",
      "category": "happy",
      "description": "A delighted reaction",
      "visible_text": "好耶",
      "tags": ["celebration", "yes"],
      "scenes": ["回应好消息"],
      "emotions": ["happy", "excited"],
      "filename": "happy-001.webp",
      "license": "CC-BY-4.0",
      "attribution": "Artist Name"
    }
  ],
  "next_cursor": "page-2"
}
```

Item aliases are supported for common fields:

| Canonical value | Accepted fields |
|:---|:---|
| Stable item ID | `id`, `external_id`, `key` |
| Image URL | `url`, `image_url`, `source_url`, `src` |
| Category | `category`, `emotion` |
| Description | `description`, `desc`, `caption` |
| Visible text | `visible_text`, `overlay_text` |
| Scenes | `scenes`, `scene` |
| Attribution | `attribution`, `author` |

Image URLs may be absolute or relative to the catalog endpoint. For pagination, return `next_cursor` or `next`; the importer requests the same endpoint again with `?cursor=<value>`. A repeated cursor stops pagination safely. The configured per-sync item limit applies to raw catalog entries, including malformed entries.

API clients may include `Accept`, `Authorization`, `User-Agent`, or `X-API-Key` headers when registering a source. Header values are stored in the plugin database so later syncs can authenticate, but source-list responses redact every value and item provenance excludes credential-like fields. Protect the AstrBot plugin-data directory accordingly.

## Import behavior

1. Preflight reads manifests and catalog metadata without adding images.
2. Source categories are mapped explicitly or matched to an existing local category.
3. Each image is downloaded/read within byte limits, decoded with Pillow, format-checked, pixel-limited, and hashed with SHA-256.
4. New images are atomically copied into the selected category or Pending Review.
5. Existing formal or pending hashes are linked to the source without creating a second copy.
6. A successful sync marks seen source items current. Previously known items missing from the new catalog remain in the library and become `stale` in provenance.

Imported metadata includes description, visible text, tags, scenes, emotions, source URL, original filename, dimensions, format, license, and attribution. Enabling the plugin's content filtration always forces review for external imports.

The import panel has an optional character assignment. Leave it unchecked to keep the series unassigned, choose an existing character key to apply it to every imported image, or enter a new key and allow the importer to create that character in the plugin configuration. The assignment is retained for Pending Review items and for later source synchronization.

## Safety limits

- HTTPS is required unless `external_source_allow_http` is explicitly enabled.
- Literal and DNS-resolved loopback, private, link-local, multicast, unspecified, and reserved destinations are rejected before requests and redirects.
- Clash/mihomo TUN fake-IP DNS answers in `198.18.0.0/15` are permitted only for public hostnames; literal URLs in that range remain blocked.
- ZIP absolute paths, `..` traversal, symlink entries, excessive members, excessive compressed/uncompressed size, oversized files, and empty image packs are rejected.
- Catalog bodies, individual images, item counts, pixel counts, pagination, redirects, and concurrent background jobs are bounded.
- Forgetting a source removes its registry and provenance rows while preserving imported image copies.

The WebUI is the recommended control surface. The backing routes are `api/sources`, `api/sources/inspect`, `api/sources/import`, `api/sources/sync`, `api/sources/jobs`, `api/sources/jobs/cancel`, `api/sources/delete`, and `api/sources/upload`.
