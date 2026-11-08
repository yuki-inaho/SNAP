#!/usr/bin/env python3
"""Report the Pixi / GPU facts a point-cloud GUI test run depends on.

Default mode is metadata-only: it does not import torch or build the model. Pass
``--probe`` to run the repository's Pixi interpreter and report the CUDA/extension
facts. The output is one JSON object so it can be attached to a test receipt.

Private paths are not written to the skill; callers choose the receipt location.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROBE = r"""
import json
import platform
import sys

out = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_runtime"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        out["compute_capability"] = float(f"{major}.{minor}")
        out["memory_total_mb"] = int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        )
except Exception as exc:
    out["torch_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out, sort_keys=True))
"""


def _run(argv: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-2000:],
    }


def _gpu_metadata() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        Path.cwd(),
        20,
    )
    if not result["ok"]:
        return {"available": False, "reason": result.get("error") or result["stderr"]}
    rows = []
    for line in result["stdout"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            rows.append(
                {"name": parts[0], "memory_total_mb": parts[1], "compute_cap": parts[2]}
            )
    return {"available": bool(rows), "devices": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--probe", action="store_true", help="run the Pixi GPU probe")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    repo = args.repo.resolve()
    report: dict[str, Any] = {
        "repo": str(repo),
        "pixi": None,
        "gpu": _gpu_metadata(),
        "probe": None,
    }
    errors: list[str] = []

    pixi = shutil.which("pixi")
    if not pixi:
        errors.append("pixi not found on PATH")
    else:
        version = _run([pixi, "--version"], repo, 20)
        report["pixi"] = {
            "path": pixi,
            "version": version["stdout"] or version.get("error"),
            "ok": version["ok"],
        }
        if not version["ok"]:
            errors.append("pixi --version failed")

    if args.probe:
        if not pixi:
            errors.append("probe skipped: pixi not found")
        else:
            probe = _run([pixi, "run", "python", "-c", PROBE], repo, args.timeout)
            parsed: Any = None
            if probe["ok"]:
                try:
                    parsed = json.loads(probe["stdout"].splitlines()[-1])
                except (IndexError, json.JSONDecodeError) as exc:
                    errors.append(f"probe output was not JSON: {type(exc).__name__}")
            else:
                errors.append("pixi probe failed")
            report["probe"] = {"result": parsed, "receipt": probe}

    report["errors"] = errors
    report["ok"] = not errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] or not args.probe else 1


if __name__ == "__main__":
    sys.exit(main())
