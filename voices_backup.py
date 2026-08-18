import json
import os

os.add_dll_directory(
    r"C:\Users\Harita\Downloads\ffmpeg-9.0.1-full_build-shared\ffmpeg-9.0.1-full_build-shared\bin"
)

# your existing imports below this
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        token = os.environ["HF_TOKEN"]
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )
    return _pipeline

def who_spoke_when(audio_path, out_path=None):
    diarization = get_pipeline()(str(audio_path))
    turns = []
    for segment, speaker in diarization.speaker_diarization:
        turns.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker": str(speaker),
        })
    if out_path:
        Path(out_path).write_text(json.dumps(turns, indent=2), encoding="utf-8")
    return turns

def merge_transcript_with_speakers(segments, turns):
    merged = []
    for seg in segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        containing = [t for t in turns if t["start"] <= midpoint <= t["end"]]
        if containing:
            speaker = containing[0]["speaker"]
        else:
            speaker = min(
                turns,
                key=lambda t: min(abs(midpoint-t["start"]), abs(midpoint-t["end"]))
            )["speaker"] if turns else "UNKNOWN"
        merged.append({**seg, "speaker": speaker})
    return merged

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "meetings/meeting1.mp4"
    turns = who_spoke_when(path, "outputs/diarization.json")
    transcript = json.loads(Path("outputs/transcript.json").read_text())
    merged = merge_transcript_with_speakers(transcript, turns)
    Path("outputs/speaker_transcript.json").write_text(json.dumps(merged, indent=2))
    print("Saved outputs/speaker_transcript.json")
