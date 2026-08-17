import json
from pathlib import Path
from faster_whisper import WhisperModel

_model = None

def get_model():
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model

def transcribe(audio_path, out_path=None):
    model = get_model()
    segments, info = model.transcribe(str(audio_path))
    result = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in segments
    ]
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "meetings/meeting1.mp4"
    transcribe(path, "outputs/transcript.json")
    print("Saved outputs/transcript.json")
