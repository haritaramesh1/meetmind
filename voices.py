import json
import os

os.add_dll_directory(
    r"C:\Users\Harita\Downloads\ffmpeg-9.0.1-full_build-shared\ffmpeg-9.0.1-full_build-shared\bin"
)

from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline


load_dotenv()

_pipeline = None


# ============================================================
# PYANNOTE PIPELINE
# ============================================================

def get_pipeline():

    global _pipeline

    if _pipeline is None:

        token = os.environ["HF_TOKEN"]

        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=token,
        )

    return _pipeline


# ============================================================
# SPEAKER DIARIZATION
# ============================================================

def who_spoke_when(
    audio_path,
    out_path=None,
):

    diarization = get_pipeline()(
        str(audio_path)
    )

    turns = []

    for segment, speaker in (
        diarization.speaker_diarization
    ):

        turns.append(
            {
                "start": float(
                    segment.start
                ),
                "end": float(
                    segment.end
                ),
                "speaker": str(
                    speaker
                ),
            }
        )

    if out_path:

        Path(out_path).write_text(
            json.dumps(
                turns,
                indent=2
            ),
            encoding="utf-8",
        )

    return turns


# ============================================================
# LOAD FINAL FACE ATTRIBUTION
# ============================================================

def load_face_attribution(
    path="outputs/final_attribution.json",
):

    attribution_path = Path(path)

    if not attribution_path.exists():

        print(
            "Warning: "
            f"{path} not found."
        )

        print(
            "Speaker transcript will contain "
            "speaker IDs only."
        )

        return {}

    data = json.loads(
        attribution_path.read_text(
            encoding="utf-8"
        )
    )

    return data.get(
        "mapping",
        {}
    )


# ============================================================
# SPEAKER → PERSON
# ============================================================

def person_for_speaker(
    speaker,
    attribution,
):

    result = attribution.get(
        speaker
    )

    if not result:
        return None

    person = result.get(
        "person"
    )

    if not person:
        return None

    return person


# ============================================================
# MERGE TRANSCRIPT + SPEAKER + FACE
# ============================================================

def merge_transcript_with_speakers(
    segments,
    turns,
    attribution=None,
):

    if attribution is None:
        attribution = {}

    merged = []

    for seg in segments:

        midpoint = (
            seg["start"]
            + seg["end"]
        ) / 2

        # ----------------------------------------------------
        # Find speaker containing transcript midpoint
        # ----------------------------------------------------

        containing = [
            turn
            for turn in turns
            if (
                turn["start"]
                <= midpoint
                <= turn["end"]
            )
        ]

        if containing:

            speaker = containing[0][
                "speaker"
            ]

        elif turns:

            speaker = min(
                turns,
                key=lambda turn: min(
                    abs(
                        midpoint
                        - turn["start"]
                    ),
                    abs(
                        midpoint
                        - turn["end"]
                    ),
                ),
            )["speaker"]

        else:

            speaker = "UNKNOWN"

        # ----------------------------------------------------
        # Map speaker → visual person
        # ----------------------------------------------------

        person = person_for_speaker(
            speaker,
            attribution,
        )

        # ----------------------------------------------------
        # Add fields without destroying
        # existing transcript data.
        # ----------------------------------------------------

        item = {
            **seg,
            "speaker": speaker,
        }

        if person:

            item["person"] = person

            item["speaker_identity"] = (
                f"{speaker} / {person}"
            )

        else:

            item["person"] = None

            item["speaker_identity"] = (
                speaker
            )

        merged.append(item)

    return merged


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import sys

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "meetings/meeting1.mp4"
    )

    # --------------------------------------------------------
    # 1. Run pyannote
    # --------------------------------------------------------

    turns = who_spoke_when(
        path,
        "outputs/diarization.json",
    )

    # --------------------------------------------------------
    # 2. Load transcript
    # --------------------------------------------------------

    transcript_path = Path(
        "outputs/transcript.json"
    )

    transcript = json.loads(
        transcript_path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # 3. Load visual speaker attribution
    # --------------------------------------------------------

    attribution = load_face_attribution(
        "outputs/final_attribution.json"
    )

    # --------------------------------------------------------
    # 4. Merge everything
    # --------------------------------------------------------

    merged = (
        merge_transcript_with_speakers(
            transcript,
            turns,
            attribution,
        )
    )

    # --------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------

    output_path = Path(
        "outputs/speaker_transcript.json"
    )

    output_path.write_text(
        json.dumps(
            merged,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "Saved outputs/speaker_transcript.json"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    visual_count = sum(
        1
        for item in merged
        if item.get("person")
    )

    print(
        f"Transcript segments: "
        f"{len(merged)}"
    )

    print(
        f"Segments with visual "
        f"attribution: {visual_count}"
    )