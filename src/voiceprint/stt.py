# coding=utf-8
from pathlib import Path

from mlx_audio.stt import load

from voiceprint.commons.files import get_absolute_path

DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"


class QwenASR:
    def __init__(self, model_name: str = DEFAULT_ASR_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load(self.model_name)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> str:
        audio_path = get_absolute_path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(audio_path)

        kwargs = {}
        if language:
            kwargs["language"] = language

        result = self.model.generate(
            str(audio_path),
            **kwargs,
        )

        return result.text.strip()


if __name__ == '__main__':
    import time

    asr = QwenASR()
    start = time.time()
    print(asr.transcribe('~/Downloads/backups/Kangfeng Road 17.m4a'))
    print(time.time() - start)
