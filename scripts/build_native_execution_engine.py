#!/usr/bin/env python3
"""Build the ABI-pinned native execution extension in place."""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native_execution"
MLX_NATIVE_ABI_VERSION = "0.32.2"
NANOBIND_VERSION = "2.15.0"


def main() -> int:
    installed_mlx = importlib.metadata.version("mlx")
    if installed_mlx != MLX_NATIVE_ABI_VERSION:
        raise RuntimeError(
            "native execution binding ABI is qualified only for "
            f"mlx=={MLX_NATIVE_ABI_VERSION}, found {installed_mlx}"
        )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to create the isolated native build env")
    command = [
        uv,
        "run",
        "--project",
        str(ROOT),
        "--with",
        "setuptools",
        "--with",
        "wheel",
        "--with",
        f"nanobind=={NANOBIND_VERSION}",
        "python",
        "setup.py",
        "build_ext",
        "--inplace",
    ]
    print(
        json.dumps(
            {
                "phase": "build-native-extension",
                "mlx_version": installed_mlx,
                "nanobind_version": NANOBIND_VERSION,
                "command": command,
            }
        )
    )
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(ROOT / "build" / "uv-cache"))
    subprocess.run(command, cwd=SOURCE, check=True, env=environment)
    outputs = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*.so", "*.dylib", "*.metallib")
        for path in (SOURCE / "glm53_native_execution").glob(pattern)
    )
    if not any(path.endswith(".so") for path in outputs):
        raise RuntimeError("native execution build did not produce a Python extension")
    if not any(path.endswith(".metallib") for path in outputs):
        raise RuntimeError("native execution build did not produce a Metal library")
    print(json.dumps({"phase": "complete", "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
