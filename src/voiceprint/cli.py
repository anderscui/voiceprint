# coding=utf-8


from pathlib import Path
import typer
from .transcribe import transcribe


app = typer.Typer()


@app.command()
def transcribe_file(file: Path):
    transcript = transcribe(file)
    typer.echo(transcript.text)


def main():
    app()
