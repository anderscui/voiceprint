# coding=utf-8
from pathlib import Path


def get_absolute_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()
