import argparse, json, time
from pathlib import Path
from ears import transcribe
from voices import who_spoke_when, merge_transcript_with_speakers
from eyes import run as detect_faces
from fusion import fuse
from slides import extract_slides
from secretary import make_minutes
from memory import chunk_documents, build_index

def main(video):
    video = Path(video)
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    t0 = time.time()
    transcript_path = out / "transcript.json"
    diar_path = out / "diarization.json"
    faces_path = out / "faces.json"
    speaker_path = out / "speaker_transcript.json"
    slides_path = out / "slides.json"
    minutes_path = out / "minutes.json"
    fusion_path = out / "fusion.json"

    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
    else:
        transcript = transcribe(video, transcript_path)

    if diar_path.exists():
        turns = json.loads(diar_path.read_text())
    else:
        turns = who_spoke_when(video, diar_path)

    speaker_transcript = merge_transcript_with_speakers(transcript, turns)
    speaker_path.write_text(json.dumps(speaker_transcript, indent=2))

    if faces_path.exists():
        faces = json.loads(faces_path.read_text())
    else:
        faces = detect_faces(video, faces_path)

    mapping, fused = fuse(turns, faces)
    fusion_path.write_text(json.dumps({"mapping": mapping, "turns": fused}, indent=2))

    if slides_path.exists():
        slides = json.loads(slides_path.read_text())
    else:
        slides = extract_slides(video, out_path=slides_path)

    if minutes_path.exists():
        minutes = json.loads(minutes_path.read_text())
    else:
        minutes_model = make_minutes(speaker_transcript, slides)
        minutes = minutes_model.model_dump()
        minutes_path.write_text(json.dumps(minutes, indent=2))

    chunks = chunk_documents(speaker_transcript, slides)
    build_index(chunks)

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")
    print("Outputs:")
    for p in [transcript_path, diar_path, faces_path, speaker_path, slides_path, minutes_path, fusion_path]:
        print(" -", p)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to meeting video")
    args = parser.parse_args()
    main(args.video)
