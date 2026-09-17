# coding=utf-8
from pathlib import Path

import typer

from voiceprint.commons.files import get_absolute_path
from voiceprint.pipeline import STTPipeline
from voiceprint.tts import QwenTTS


app = typer.Typer(
    no_args_is_help=True,
    help="Voiceprint — speech-to-text and text-to-speech tools.",
)


@app.command()
def stt(
    audio: Path = typer.Argument(
        ...,
        exists=False,
        dir_okay=False,
        readable=True,
        help="Audio file to transcribe.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for transcript files.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Audio language.",
    ),
    raw_only: bool = typer.Option(
        False,
        "--raw-only",
        help="Skip transcript cleanup.",
    ),
):
    """Transcribe an audio file."""

    pipeline = STTPipeline()

    raw_path, clean_path = pipeline.run(
        audio_path=audio,
        output_dir=output_dir,
        language=language,
        clean=not raw_only,
    )

    typer.echo(f"Raw: {raw_path}")

    if clean_path:
        typer.echo(f"Clean: {clean_path}")


@app.command()
def tts(
    text: str = typer.Argument(
        ...,
        help="Text to synthesize.",
    ),
    output: Path = typer.Option(
        Path("output.wav"),
        "--output",
        "-o",
        help="Output WAV file.",
    ),
    speaker: str = typer.Option(
        "Vivian",
        "--speaker",
        "-s",
        help="Built-in speaker name.",
    ),
    language: str | None = typer.Option(
        "Chinese",
        "--language",
        "-l",
        help="Text language.",
    ),
):
    """Generate speech using a built-in voice."""

    engine = QwenTTS()

    engine.speak(
        text=text,
        output_path=output,
        speaker=speaker,
        language=language,
    )

    typer.echo(f"Audio: {output}")


@app.command()
def clone(
    text: str = typer.Argument(
        ...,
        help="Text to synthesize.",
    ),
    ref_audio: Path = typer.Option(
        ...,
        "--ref-audio",
        "-r",
        exists=False,
        dir_okay=False,
        readable=True,
        help="Reference voice audio file.",
    ),
    ref_text: Path = typer.Option(
        ...,
        "--ref-text-file",
        exists=False,
        dir_okay=False,
        readable=True,
        help="Transcript file of the reference audio.",
    ),
    output: Path = typer.Option(
        Path("output.wav"),
        "--output",
        "-o",
        help="Output WAV file.",
    ),
):
    """Generate speech by cloning a reference voice."""

    engine = QwenTTS()

    engine.clone(
        text=text,
        ref_audio=ref_audio,
        ref_text=get_absolute_path(ref_text).read_text(),
        output_path=output,
    )

    typer.echo(f"Audio: {output}")


def main():
    app()


if __name__ == "__main__":
    # uv run voiceprint stt "~/Downloads/backups/input.m4a"
    # uv run voiceprint tts --speaker Vivian --output test.wav "这是一段测试音频的文字。"
    # uv run voiceprint clone --output test.wav --ref-audio ~/Downloads/backups/无花果是甜的.m4a --ref-text-file ~/Downloads/backups/无花果是甜的.txt "这是一段测试音频的文字。"
    main()
