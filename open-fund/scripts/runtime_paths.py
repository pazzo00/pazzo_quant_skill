"""Resolve data paths without depending on where the skill is installed."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the user data root from the environment or current directory."""
    configured = os.environ.get("FACTOR_PREMIUM_PROJECT_ROOT")
    return Path(configured or Path.cwd()).expanduser().resolve()
