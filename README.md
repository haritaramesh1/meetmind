# MeetMind — Chapter 2

A compressed one-day implementation of the Chapter 2 pipeline:

1. Ears — faster-whisper timestamped transcription
2. Who-spoke-when — pyannote speaker diarization
3. Eyes — DeepFace/OpenCV face sightings
4. Fusion — temporal voice/face co-occurrence mapping
5. Reading glasses — PaddleOCR slide extraction
6. Secretary — Gemini + Pydantic structured minutes
7. Memory — FAISS semantic search
8. Plug — MCP `search_meetings` tool
9. Demo — Gradio UI

## Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Put your keys in `.env`:

```env
HF_TOKEN=...
GEMINI_API_KEY=...
```

For pyannote, accept the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` before running diarization.

Put a short test meeting at `meetings/meeting1.mp4`.

## Run the complete pipeline

```bash
python pipeline.py meetings/meeting1.mp4
```

Then launch the demo:

```bash
python app.py
```

## MCP

Run:

```bash
python mcp_server.py
```

The server exposes one tool: `search_meetings(question)`.

Configure your MCP host/Claude Desktop with the absolute path to your Python executable and `mcp_server.py`.

## Important

The first run can be slow because model weights download and CPU inference is expensive. For a same-day demo, use a 2–5 minute clip first, then process the 10–15 minute showcase clip. Cache the JSON outputs before deploying.
