from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib


def build_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def csv_signature(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def ensure_cache_dir(cache_dir: str | Path) -> Path:
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_joblib(cache_path: str | Path) -> Any | None:
    path = Path(cache_path)
    if not path.exists():
        return None
    return joblib.load(path)


def save_joblib(cache_path: str | Path, payload: Any) -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path
