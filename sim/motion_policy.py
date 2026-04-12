"""Compatibility shim for the moved Fisher motion policy.

The authoritative implementation now lives in
`omnimap.gaussian.renderer.nbv.motion_policy`.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
omnimap_root = repo_root / "omnimap"
for path in (repo_root, omnimap_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from omnimap.gaussian.renderer.nbv.motion_policy import (  # noqa: F401
    FisherMotionPolicy,
    MotionPolicyResult,
)

__all__ = ["FisherMotionPolicy", "MotionPolicyResult"]
