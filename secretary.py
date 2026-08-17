import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


load_dotenv()


class ActionItem(BaseModel):
    owner: str
    task: str
    due: str | None = None


class Minutes(BaseModel):
    summary: list[str]
    decisions: list[str]
    action_items: list[ActionItem]


def load_transcript(path: str = "outputs/speaker_transcript.json") -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    lines = []
    for item in data:
        speaker = item.get("speaker", "UNKNOWN")
        text = item.get("text", "").strip()

        if text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines)


def make_minutes(transcript: str) -> Minutes:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are MeetMind, an AI meeting secretary.

Analyze the speaker-labeled meeting transcript below.

Return:
1. A concise bullet-point summary.
2. The important decisions actually made.
3. Action items that were actually assigned.

For every action item:
- owner = the speaker/person responsible, if identifiable
- task = the concrete task
- due = deadline if explicitly mentioned, otherwise null

Do NOT invent decisions, owners, tasks, or deadlines.
If something is unclear, leave it out.

MEETING TRANSCRIPT:
{transcript}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": Minutes.model_json_schema(),
        },
    )

    return Minutes.model_validate_json(response.text)


def save_minutes(minutes: Minutes, path: str = "outputs/minutes.json"):
    output = minutes.model_dump_json(indent=2)
    Path(path).write_text(output, encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    transcript = load_transcript()
    print(f"Loaded {len(transcript)} characters of speaker transcript.")

    minutes = make_minutes(transcript)
    save_minutes(minutes)
