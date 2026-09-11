"""Validate and package an AstrBot plugin release.

The release workflow calls this script so the same version, Changelog and
archive checks can be run locally before a tag is pushed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
VERSION_RE = re.compile(r"^version:\s*([^\s#]+)\s*$", re.MULTILINE)
REQUIRED_FILES = (
    "metadata.yaml",
    "CHANGELOG.md",
    "README.md",
    "LICENSE",
    "main.py",
    "_conf_schema.json",
    "prompts.json",
    "requirements.txt",
    "pages/dashboard/app.js",
    "pages/dashboard/template.js",
)
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".github",
        "CLAUDE.md",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "scripts",
        "tests",
    }
)


class ReleaseError(RuntimeError):
    """Raised when a release invariant is violated."""


def read_version(root: Path = ROOT) -> str:
    metadata_path = root / "metadata.yaml"
    text = metadata_path.read_text(encoding="utf-8")
    matches = VERSION_RE.findall(text)
    if len(matches) != 1:
        raise ReleaseError(f"expected one version entry in {metadata_path}")
    version = matches[0]
    if not SEMVER_RE.fullmatch(version):
        raise ReleaseError(f"metadata version is not stable SemVer: {version}")
    return version


def extract_changelog(version: str, root: Path = ROOT) -> str:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n(?P<body>.*?)(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match or not match.group("body").strip():
        raise ReleaseError(f"CHANGELOG.md has no non-empty section for {version}")
    return match.group("body").strip()


def _version_tuple(version: str) -> tuple[int, int, int]:
    if not SEMVER_RE.fullmatch(version):
        raise ReleaseError(f"version is not stable SemVer: {version}")
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def validate(
    version: str | None = None,
    tag: str | None = None,
    previous_version: str | None = None,
    root: Path = ROOT,
) -> str:
    actual_version = read_version(root)
    if version and version != actual_version:
        raise ReleaseError(
            f"requested version {version} does not match metadata.yaml {actual_version}"
        )
    version = actual_version

    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file():
            raise ReleaseError(f"required release file is missing: {relative}")

    extract_changelog(version, root)
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ReleaseError("_conf_schema.json must contain an object")
    for locale_path in sorted((root / ".astrbot-plugin" / "i18n").glob("*.json")):
        json.loads(locale_path.read_text(encoding="utf-8"))

    if tag:
        expected_tag = f"v{version}"
        if tag != expected_tag:
            raise ReleaseError(f"tag {tag} does not match metadata version {expected_tag}")

    if previous_version and _version_tuple(version) <= _version_tuple(previous_version):
        raise ReleaseError(
            f"metadata version {version} must be greater than {previous_version}"
        )

    print(
        "release validation passed: "
        f"version={version}, tag={tag or '(none)'}, "
        f"previous={previous_version or '(none)'}"
    )
    return version


def write_notes(version: str, output: Path, root: Path = ROOT) -> None:
    body = extract_changelog(version, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body + "\n", encoding="utf-8")
    print(f"release notes written: {output}")


def _archive_entries(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
    if not names:
        raise ReleaseError("release archive is empty")
    return names


def build_package(version: str, output_dir: Path, root: Path = ROOT) -> tuple[Path, Path]:
    validate(version, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"astrbot_plugin_stealer-v{version}.zip"
    checksum_path = archive_path.with_name(archive_path.name + ".sha256")
    archive_path.unlink(missing_ok=True)
    checksum_path.unlink(missing_ok=True)

    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            "--prefix=astrbot_plugin_stealer/",
            f"--output={archive_path}",
            "HEAD",
        ],
        cwd=root,
        check=True,
    )

    names = _archive_entries(archive_path)
    prefix = "astrbot_plugin_stealer/"
    for name in names:
        relative_parts = tuple(part for part in name.split("/") if part)
        if not name.startswith(prefix) or EXCLUDED_PARTS.intersection(relative_parts):
            raise ReleaseError(f"forbidden path in release archive: {name}")
    expected = {f"{prefix}{relative}" for relative in REQUIRED_FILES}
    missing = sorted(expected.difference(names))
    if missing:
        raise ReleaseError(f"release archive misses required files: {', '.join(missing)}")

    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    size = archive_path.stat().st_size
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "checksum": str(checksum_path),
                "files": len(names),
                "compressed_bytes": size,
                "sha256": digest,
            },
            ensure_ascii=False,
        )
    )
    return archive_path, checksum_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--version")
    validate_parser.add_argument("--tag")
    validate_parser.add_argument("--previous-version")

    notes_parser = subparsers.add_parser("notes")
    notes_parser.add_argument("--version", required=True)
    notes_parser.add_argument("--output", type=Path, required=True)

    package_parser = subparsers.add_parser("package")
    package_parser.add_argument("--version", required=True)
    package_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            validate(args.version, args.tag, args.previous_version)
        elif args.command == "notes":
            validate(args.version)
            write_notes(args.version, args.output)
        elif args.command == "package":
            build_package(args.version, args.output_dir)
        return 0
    except (OSError, ReleaseError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"release check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
