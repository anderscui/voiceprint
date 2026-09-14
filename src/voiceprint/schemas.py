# coding=utf-8


from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Transcript(BaseModel):
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptSegment]

    @property
    def text(self) -> str:
        return "\n".join(
            segment.text.strip()
            for segment in self.segments
            if segment.text.strip()
        )
