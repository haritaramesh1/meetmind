import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "outputs/meetings.index"
CHUNKS_PATH = "outputs/meeting_chunks.json"


model = SentenceTransformer(MODEL_NAME)


def load_chunks():
    transcript_path = Path("outputs/speaker_transcript.json")

    data = json.loads(transcript_path.read_text(encoding="utf-8"))

    chunks = []

    for item in data:
        text = item.get("text", "").strip()

        if not text:
            continue

        chunks.append({
            "text": text,
            "speaker": item.get("speaker", "UNKNOWN"),
            "start": item.get("start", 0),
            "end": item.get("end", 0),
            "source": "meeting1",
        })

    return chunks


def build_index():
    chunks = load_chunks()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    Path("outputs").mkdir(exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    Path(CHUNKS_PATH).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Indexed {len(chunks)} meeting chunks.")
    print(f"Saved {INDEX_PATH}")
    print(f"Saved {CHUNKS_PATH}")


def smart_search(question, k=5):
    index = faiss.read_index(INDEX_PATH)

    chunks = json.loads(
        Path(CHUNKS_PATH).read_text(encoding="utf-8")
    )

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append(
            (
                chunks[idx]["text"],
                chunks[idx]["source"],
                float(score),
            )
        )

    return results


if __name__ == "__main__":
    build_index()

    print("\nTest search:")
    results = smart_search("What did they decide about Floating Farm?")

    for text, source, score in results:
        print(f"\n[{source} | score={score:.3f}]")
        print(text)