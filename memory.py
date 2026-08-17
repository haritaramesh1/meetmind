import json
from pathlib import Path
import numpy as np
import faiss

INDEX_PATH = Path("outputs/meetings.faiss")
DOCS_PATH = Path("outputs/meeting_chunks.json")

def chunk_documents(speaker_transcript, slides=None, max_chars=700):
    docs = []
    current = []
    size = 0
    for item in speaker_transcript:
        line = f'[{item["start"]:.1f}] {item["speaker"]}: {item["text"]}'
        if current and size + len(line) > max_chars:
            docs.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        docs.append("\n".join(current))
    for slide in slides or []:
        docs.append(f'[{slide["time"]:.1f}] SLIDE: {slide["text"]}')
    return docs

def _embed(texts):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True)
    return np.asarray(X, dtype="float32")

def build_index(texts):
    X = _embed(texts)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))
    DOCS_PATH.write_text(json.dumps(texts, indent=2), encoding="utf-8")
    return index

def smart_search(question, k=3):
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        return []
    texts = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_PATH))
    q = _embed([question])
    scores, ids = index.search(q, min(k, len(texts)))
    return [(texts[i], "meeting", float(scores[0][j])) for j, i in enumerate(ids[0]) if i >= 0]
