from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "upstream" / "openpi"


def add_openpi_to_path() -> None:
    src = OPENPI / "src"
    if not src.exists():
        raise SystemExit(f"OpenPI checkout not found: {OPENPI}")
    sys.path.insert(0, str(src))


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def missing_dependency(error: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"Missing dependency: {error.name}\n"
        "Install OpenPI dependencies first:\n"
        "  cd upstream/openpi\n"
        "  GIT_LFS_SKIP_SMUDGE=1 uv sync\n"
        "  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ."
    )


def run_openpi(args: list[str]) -> None:
    if not OPENPI.exists():
        raise SystemExit(f"OpenPI checkout not found: {OPENPI}")
    print("$", " ".join(args))
    raise SystemExit(subprocess.run(args, cwd=OPENPI, check=False).returncode)


def brief(value, depth: int = 0) -> str:
    if isinstance(value, (bytes, bytearray)):
        return f"bytes(len={len(value)})"
    if hasattr(value, "shape"):
        dtype = getattr(value, "dtype", "?")
        return f"shape={tuple(value.shape)} dtype={dtype}"
    if isinstance(value, dict):
        if depth >= 2:
            return f"dict(keys={list(value)[:8]})"
        items = list(value.items())[:8]
        return "{ " + ", ".join(f"{k}: {brief(v, depth + 1)}" for k, v in items) + " }"
    if isinstance(value, (list, tuple)):
        head = brief(value[0], depth + 1) if value else "empty"
        return f"{type(value).__name__}(len={len(value)}, first={head})"
    return f"{type(value).__name__}({value!r})"


def tree_brief(title: str, value) -> None:
    print(f"\n## {title}")
    print(brief(value))


def torch_tree_to(value, device):
    try:
        import torch
    except ImportError:
        return value

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: torch_tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [torch_tree_to(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(torch_tree_to(v, device) for v in value)
    return value
