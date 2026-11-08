"""Shared helpers: seeding, I/O, environment capture."""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = ["xgboost", "mlp", "cnn", "bigru", "transformer"]
MODEL_DISPLAY = {"xgboost": "XGBoost", "mlp": "MLP", "cnn": "1D-CNN",
                 "bigru": "BiGRU", "transformer": "Transformer"}


def md_table(df, nd: int = 3) -> str:
    """Render a data frame as a compact Markdown table."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "—" if np.isnan(value) else f"{value:.{nd}f}"
        if isinstance(value, (bool, np.bool_)):
            return "**YES**" if value else "no"
        return str(value)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit(root: Path = PROJECT_ROOT) -> str:
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _read_first(path: str, key: str) -> str | None:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith(key):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def environment_info() -> dict:
    import importlib.metadata as md

    info = {
        "os": platform.platform(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "cpu": _read_first("/proc/cpuinfo", "model name"),
        "cpu_logical_cores": __import__("os").cpu_count(),
        "ram_total": _read_first("/proc/meminfo", "MemTotal"),
        "git_commit": git_commit(),
    }
    pkgs = {}
    for p in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "xgboost",
              "matplotlib", "pyarrow", "PyYAML", "joblib"]:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = None
    info["packages"] = pkgs
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info["nvidia_driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        info["nvidia_driver"] = None
    return info
