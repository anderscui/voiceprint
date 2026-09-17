# coding=utf-8
from pathlib import Path

from voiceprint.cleaner import TranscriptCleaner
from voiceprint.commons.files import get_absolute_path
from voiceprint.stt import QwenASR


class STTPipeline:
    def __init__(
        self,
        stt: QwenASR | None = None,
        cleaner: TranscriptCleaner | None = None,
    ):
        self.stt = stt or QwenASR()
        self.cleaner = cleaner or TranscriptCleaner()

    def run(
        self,
        audio_path: str | Path,
        output_dir: str | Path | None = None,
        language: str | None = None,
        clean: bool = True,
    ) -> tuple[Path, Path | None]:
        audio_path = get_absolute_path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        output_dir = get_absolute_path(output_dir) if output_dir else audio_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Speech-to-text
        raw_text = self.stt.transcribe(
            audio_path,
            language=language,
        )

        # 2. Save raw transcript first
        raw_path = output_dir / f"{audio_path.stem}.raw.txt"
        raw_path.write_text(raw_text, encoding="utf-8")

        # 3. Optionally clean transcript
        if not clean:
            return raw_path, None

        clean_text = self.cleaner.clean(raw_text)

        clean_path = output_dir / f"{audio_path.stem}.txt"
        clean_path.write_text(clean_text, encoding="utf-8")

        return raw_path, clean_path


if __name__ == '__main__':
    pipeline = STTPipeline()
    raw_path, clean_path = pipeline.run(audio_path="~/Downloads/backups/Chongjiao Road 4.m4a")

