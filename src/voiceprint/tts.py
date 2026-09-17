# coding=utf-8
from pathlib import Path
import re

import mlx.core as mx
import numpy as np

from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model

from voiceprint.commons.files import get_absolute_path


DEFAULT_CUSTOM_VOICE_MODEL = (
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
)

DEFAULT_CLONE_MODEL = (
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
)

DEFAULT_MAX_CHARS = 150
DEFAULT_MAX_TOKENS = 2048


def split_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[str]:
    """
    Split long text into short TTS-friendly chunks.

    Priority:
    1. paragraph boundaries
    2. sentence-ending punctuation
    3. comma-like punctuation
    4. hard character limit
    """
    text = text.strip()

    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text)

    chunks: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        chunks.extend(_split_paragraph(paragraph, max_chars))

    return chunks


def _split_paragraph(
    text: str,
    max_chars: int,
) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    # Keep punctuation attached to the preceding sentence.
    sentences = re.split(
        r"(?<=[。！？!?；;])",
        text,
    )

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # One sentence itself is too long.
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(
                _split_long_sentence(sentence, max_chars)
            )
            continue

        if not current:
            current = sentence

        elif len(current) + len(sentence) <= max_chars:
            current += sentence

        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def _split_long_sentence(
    text: str,
    max_chars: int,
) -> list[str]:
    """
    Split an unusually long sentence on softer punctuation.
    Fall back to a hard character split only when necessary.
    """
    parts = re.split(
        r"(?<=[，,、：:])",
        text,
    )

    chunks: list[str] = []
    current = ""

    for part in parts:
        part = part.strip()

        if not part:
            continue

        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for i in range(0, len(part), max_chars):
                chunks.append(part[i:i + max_chars])

            continue

        if not current:
            current = part

        elif len(current) + len(part) <= max_chars:
            current += part

        else:
            chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    return chunks


def _join_audio(
    audio_chunks: list[np.ndarray],
    sample_rate: int,
    pause_ms: int = 150,
) -> np.ndarray:
    """
    Concatenate generated audio chunks with a short silence
    between chunks.
    """
    if not audio_chunks:
        raise RuntimeError("No audio chunks generated.")

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    silence_samples = int(sample_rate * pause_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []

    for i, audio in enumerate(audio_chunks):
        parts.append(audio)

        if i < len(audio_chunks) - 1:
            parts.append(silence)

    return np.concatenate(parts)


class QwenTTS:
    def __init__(
        self,
        custom_voice_model: str = DEFAULT_CUSTOM_VOICE_MODEL,
        clone_model: str = DEFAULT_CLONE_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.custom_voice_model_name = custom_voice_model
        self.clone_model_name = clone_model

        self.max_chars = max_chars
        self.max_tokens = max_tokens

        self._custom_voice_model = None
        self._clone_model = None

    @property
    def custom_voice_model(self):
        if self._custom_voice_model is None:
            self._custom_voice_model = load_model(
                self.custom_voice_model_name
            )

        return self._custom_voice_model

    @property
    def clone_model(self):
        if self._clone_model is None:
            self._clone_model = load_model(
                self.clone_model_name
            )

        return self._clone_model

    def speak(
        self,
        text: str,
        output_path: str | Path,
        speaker: str = "Vivian",
        language: str = "Chinese",
        max_chars: int | None = None,
        pause_ms: int = 150,
    ) -> Path:
        """
        Generate speech using a built-in CustomVoice speaker.
        """
        model = self.custom_voice_model

        chunks = split_text(
            text,
            max_chars=max_chars or self.max_chars,
        )

        if not chunks:
            raise ValueError("Text is empty.")

        print(f"TTS: {len(chunks)} chunk(s)")

        audio_chunks: list[np.ndarray] = []
        sample_rate = model.sample_rate

        for i, chunk in enumerate(chunks, start=1):
            print(
                f"[{i}/{len(chunks)}] "
                f"{len(chunk)} chars: {chunk[:50]!r}"
            )

            results = list(
                model.generate_custom_voice(
                    text=chunk,
                    speaker=speaker,
                    language=language,
                    max_tokens=self.max_tokens,
                )
            )

            if not results:
                raise RuntimeError(
                    f"No audio generated for chunk {i}."
                )

            for result in results:
                audio_chunks.append(
                    np.asarray(result.audio, dtype=np.float32)
                )

            # Release temporary MLX buffers between chunks.
            mx.clear_cache()

        audio = _join_audio(
            audio_chunks,
            sample_rate,
            pause_ms=pause_ms,
        )

        output_path = get_absolute_path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audio_write(
            str(output_path),
            audio,
            sample_rate,
        )

        return output_path

    def clone(
        self,
        text: str,
        ref_audio: str | Path,
        ref_text: str,
        output_path: str | Path,
        max_chars: int | None = None,
        pause_ms: int = 150,
    ) -> Path:
        """
        Generate speech using a cloned reference voice.
        """
        ref_audio = get_absolute_path(ref_audio)

        if not ref_audio.exists():
            raise FileNotFoundError(ref_audio)

        if not ref_text.strip():
            raise ValueError("ref_text is empty.")

        model = self.clone_model

        chunks = split_text(
            text,
            max_chars=max_chars or self.max_chars,
        )

        if not chunks:
            raise ValueError("Text is empty.")

        print(f"Voice clone: {len(chunks)} chunk(s)")

        audio_chunks: list[np.ndarray] = []
        sample_rate = model.sample_rate

        for i, chunk in enumerate(chunks, start=1):
            print(
                f"[{i}/{len(chunks)}] "
                f"{len(chunk)} chars: {chunk[:50]!r}"
            )

            results = list(
                model.generate(
                    text=chunk,
                    ref_audio=str(ref_audio),
                    ref_text=ref_text,
                    max_tokens=self.max_tokens,
                )
            )

            if not results:
                raise RuntimeError(
                    f"No audio generated for chunk {i}."
                )

            for result in results:
                audio_chunks.append(
                    np.asarray(result.audio, dtype=np.float32)
                )

            mx.clear_cache()

        audio = _join_audio(
            audio_chunks,
            sample_rate,
            pause_ms=pause_ms,
        )

        output_path = get_absolute_path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audio_write(
            str(output_path),
            audio,
            sample_rate,
        )

        return output_path


if __name__ == '__main__':
    tts = QwenTTS()
    # text = "你好，这是使用参考声音生成的一段测试语音。"
    text = get_absolute_path('~/Downloads/backups/你好哇.txt').read_text()
    tts.clone(
        text=text,
        ref_audio='~/Downloads/backups/无花果是甜的.m4a',
        ref_text=get_absolute_path('~/Downloads/backups/无花果是甜的.txt').read_text(),
        output_path='~/Downloads/backups/你好哇_me_clone.wav',
    )

    # tts.speak(
    #     text=text,
    #     speaker="Vivian",
    #     language='Chinese',
    #     output_path='~/Downloads/backups/橱窗语_vivian.wav'
    # )
