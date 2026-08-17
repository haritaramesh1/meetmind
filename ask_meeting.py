import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from memory_search import smart_search


load_dotenv()


def load_minutes():
    path = Path("outputs/minutes.json")

    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def ask_meeting(question: str, k: int = 6) -> str:
    results = smart_search(question, k=k)
    minutes = load_minutes()

    evidence = "\n\n".join(
        f"[Transcript | {source} | relevance={score:.3f}]\n{text}"
        for text, source, score in results
    )

    structured_minutes = json.dumps(
        minutes,
        indent=2,
        ensure_ascii=False,
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are MeetMind, an AI meeting intelligence assistant.

Answer the user's question using ONLY the meeting information below.

There are two sources:

1. STRUCTURED MEETING MINUTES
   - summary
   - confirmed decisions
   - assigned action items

2. TRANSCRIPT EVIDENCE
   - actual passages from the meeting

IMPORTANT RULES:
- Do not invent information.
- Treat action_items in the structured minutes as assigned work.
- Treat decisions in the structured minutes as confirmed decisions.
- Use transcript evidence to provide additional context.
- If the transcript and structured minutes do not support an answer, say so.
- Clearly distinguish discussion from decisions.
- When possible, name the responsible person.
- Keep the answer concise and useful.
- For "what should happen next?" prioritize action items and decisions.

USER QUESTION:
{question}

STRUCTURED MEETING MINUTES:
{structured_minutes}

TRANSCRIPT EVIDENCE:
{evidence}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


if __name__ == "__main__":
    question = input("Ask about the meeting: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    print("\nMeetMind:")
    print(ask_meeting(question))