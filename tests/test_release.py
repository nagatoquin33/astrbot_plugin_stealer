import pytest

from scripts.release import ReleaseError, extract_changelog, read_version, validate


def test_release_metadata_and_changelog_are_current():
    version = read_version()
    assert "### added" in extract_changelog(version)
    assert validate(version, tag=f"v{version}") == version
    assert validate(version, previous_version="0.0.0") == version


def test_validate_rejects_tag_for_another_version():
    version = read_version()
    wrong_tag = "v0.0.0" if version != "0.0.0" else "v1.0.0"
    with pytest.raises(ReleaseError, match="does not match"):
        validate(version, tag=wrong_tag)


def test_validate_requires_metadata_version_to_increase():
    version = read_version()
    with pytest.raises(ReleaseError, match="must be greater"):
        validate(version, previous_version=version)
