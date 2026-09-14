# coding=utf-8


from pathlib import Path

from .schemas import Transcript


def transcribe(file_path: Path) -> Transcript:
    raise NotImplementedError
