"""External meme source integration for v3.

The source package deliberately depends on a small, documented interchange
surface.  It can read Meme Manager pack exports and a conservative HTTP JSON
catalog without importing another plugin's private runtime objects.
"""

from .models import (
    ExternalSourceError,
    ExternalSourceSecurityError,
    PackInspection,
    SourceItem,
    SourceInspection,
)
from .github_source import GitHubSource
from .source_service import SourceService

__all__ = [
    "ExternalSourceError",
    "ExternalSourceSecurityError",
    "PackInspection",
    "SourceItem",
    "SourceInspection",
    "GitHubSource",
    "SourceService",
]
