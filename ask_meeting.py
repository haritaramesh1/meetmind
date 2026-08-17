import os

from dotenv import load_dotenv
from google import genai

from memory_search import smart_search


load_dotenv()


def ask_meeting(question: str, k: int = 5) -> str:
    results = smart_search(question, k=k)

    if not results:
        return "I couldn't find relevant information in the meeting."

    evidence = "\n\n".join(
        f"[{source} | relevance={score:.3f}]\n{text}"
        for text, source, score in results
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
You are MeetMind, a meeting intelligence assistant.

Answer the user's question using ONLY the meeting evidence below.

Rules:
- Do not invent facts.
- If the evidence does not answer the question, say that clearly.
- Keep the answer concise.
- When possible, mention the speaker who said the relevant information.
- Distinguish between something discussed and an actual decision.
- Do not treat a suggestion as a confirmed decision.

USER QUESTION:
{question}

MEETING EVIDENCE:
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