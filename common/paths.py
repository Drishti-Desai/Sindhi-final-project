"""Central path resolution.

Every script calls `load_paths()` and reads locations from configs/paths.yaml, so moving a
folder means editing one YAML file instead of hunting through code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(node: Any) -> Any:
    """Turn every string leaf into an absolute Path under the repo root."""
    if isinstance(node, dict):
        return {k: _resolve(v) for k, v in node.items()}
    if isinstance(node, str):
        return (REPO_ROOT / node).resolve()
    return node


def load_paths() -> dict:
    """Return configs/paths.yaml with all values as absolute `Path` objects."""
    return _resolve(load_yaml(PATHS_FILE))


def load_config(name: str) -> dict:
    """Load a config listed under `configs:` in paths.yaml, e.g. load_config("step1")."""
    return load_yaml(load_paths()["configs"][name])
