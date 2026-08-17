import json
from typing import Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL

class ActionItem(BaseModel):
    owner: str
    task: str
    due: Optional[str] = None

class Minutes(BaseModel):
    summary: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)

def make_minutes(speaker_transcript, slides=None):
    if not GEMINI_API_KEY:
        raise RuntimeError("Set GEMINI_API_KEY in .env")

    transcript_text = "\n".join(
        f'[{s["start"]:.1f}-{s["end"]:.1f}] {s["speaker"]}: {s["text"]}'
        for s in speaker_transcript
    )
    slide_text = "\n".join(
        f'[{s["time"]:.1f}] SLIDE: {s["text"]}'
        for s in (slides or [])
    )

    prompt = f"""You are MeetMind, a meeting-intelligence assistant.
Create concise meeting minutes from the speaker-labeled transcript.
Only claim decisions or action items that are actually supported.
For action items, identify the owner exactly when the transcript supports it;
otherwise use "Unassigned". Do not invent due dates.

TRANSCRIPT:
{transcript_text}

SLIDES:
{slide_text}
"""

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Minutes,
        ),
    )
    return Minutes.model_validate_json(response.text)

if __name__ == "__main__":
    from pathlib import Path
    transcript = json.loads(Path("outputs/speaker_transcript.json").read_text())
    slides = []
    if Path("outputs/slides.json").exists():
        slides = json.loads(Path("outputs/slides.json").read_text())
    minutes = make_minutes(transcript, slides)
    Path("outputs/minutes.json").write_text(minutes.model_dump_json(indent=2))
    print("Saved outputs/minutes.json")
