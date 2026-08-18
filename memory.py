import json
import sys
import threading
from pathlib import Path

import numpy as np
import faiss


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

INDEX_PATH = OUTPUTS_DIR / "meetings.faiss"
DOCS_PATH = OUTPUTS_DIR / "meeting_chunks.json"


# ============================================================
# EMBEDDING MODEL
# ============================================================

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_READY = threading.Event()
_MODEL_ERROR = None


def _log(message):
    """
    MCP uses stdout for its protocol.
    ALL debug messages must therefore go to stderr.
    """
    print(
        message,
        file=sys.stderr,
        flush=True,
    )


def _load_model():
    """
    Load the embedding model.

    This runs in a background thread so that the MCP server
    can connect to Claude immediately.
    """
    global _MODEL
    global _MODEL_ERROR

    try:
        _log("MeetMind: loading embedding model in background...")

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        with _MODEL_LOCK:
            _MODEL = model

        _log("MeetMind: embedding model ready.")

    except Exception as exc:
        _MODEL_ERROR = exc

        _log(
            "MeetMind: embedding model failed to load: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        _MODEL_READY.set()


def start_model_loading():
    """
    Start loading the embedding model without blocking MCP startup.

    Safe to call multiple times.
    """

    if _MODEL is not None:
        return

    if _MODEL_READY.is_set():
        return

    thread = threading.Thread(
        target=_load_model,
        name="meetmind-model-loader",
        daemon=True,
    )

    thread.start()


def _get_model():
    """
    Return the loaded embedding model.

    If background loading has not started yet, start it.

    This function waits for the model only when a real search
    actually needs it.
    """

    global _MODEL_ERROR

    if _MODEL is not None:
        return _MODEL

    start_model_loading()

    _log(
        "MeetMind: waiting for embedding model..."
    )

    # Give the model time to finish if a real search is
    # requested before background loading completes.
    _MODEL_READY.wait()

    if _MODEL is not None:
        return _MODEL

    if _MODEL_ERROR is not None:
        raise RuntimeError(
            "MeetMind embedding model failed to load: "
            f"{_MODEL_ERROR}"
        )

    raise RuntimeError(
        "MeetMind embedding model is unavailable."
    )


# ============================================================
# TEXT CONVERSION
# ============================================================

def _text_for_embedding(item):

    if isinstance(item, str):
        return item

    if isinstance(item, dict):

        text = item.get(
            "text",
            "",
        )

        speaker = item.get(
            "speaker",
            "",
        )

        source = item.get(
            "source",
            "",
        )

        start = item.get(
            "start"
        )

        end = item.get(
            "end"
        )

        parts = []

        if source:
            parts.append(
                f"Meeting: {source}"
            )

        if speaker:
            parts.append(
                f"Speaker: {speaker}"
            )

        if start is not None and end is not None:
            parts.append(
                f"Time: "
                f"{float(start):.1f}s-"
                f"{float(end):.1f}s"
            )

        if text:
            parts.append(
                f"Text: {text}"
            )

        return " | ".join(parts)

    return str(item)


# ============================================================
# EMBEDDING
# ============================================================

def _embed(texts):

    model = _get_model()

    clean_texts = [
        _text_for_embedding(item)
        for item in texts
    ]

    X = model.encode(
        clean_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        X,
        dtype="float32",
    )


# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(texts):

    _log(
        f"Embedding {len(texts)} meeting chunks..."
    )

    X = _embed(texts)

    index = faiss.IndexFlatIP(
        X.shape[1]
    )

    index.add(X)

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    DOCS_PATH.write_text(
        json.dumps(
            texts,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _log(
        f"Saved FAISS index: {INDEX_PATH}"
    )

    return index


# ============================================================
# SEARCH
# ============================================================

def smart_search(
    question,
    k=5,
):

    if not INDEX_PATH.exists():

        _log(
            f"Meeting index not found: "
            f"{INDEX_PATH}"
        )

        return []

    if not DOCS_PATH.exists():

        _log(
            f"Meeting chunks not found: "
            f"{DOCS_PATH}"
        )

        return []

    texts = json.loads(
        DOCS_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not texts:
        return []

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    query_embedding = _embed(
        [question]
    )

    scores, ids = index.search(
        query_embedding,
        min(k, len(texts)),
    )

    results = []

    for j, idx in enumerate(
        ids[0]
    ):

        if idx < 0:
            continue

        results.append(
            (
                texts[idx],
                "meeting",
                float(
                    scores[0][j]
                ),
            )
        )

    return results


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("MeetMind Meeting Memory")
    print("=" * 50)

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Chunks: {DOCS_PATH}"
    )

    print(
        f"Index exists: "
        f"{INDEX_PATH.exists()}"
    )

    print(
        f"Chunks exist: "
        f"{DOCS_PATH.exists()}"
    )

    print()

    # For direct testing only, load the model.
    _get_model()

    print()
    print("Test search:")
    print()

    results = smart_search(
        "What did they discuss about Floating Farm?",
        k=5,
    )

    if not results:

        print("No results found.")

    else:

        for text, source, score in results:

            print(
                f"[{source} | score={score:.3f}]"
            )

            print(text)
            print()