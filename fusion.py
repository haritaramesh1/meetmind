import json
from collections import defaultdict
from pathlib import Path


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# POINT-IN-SEGMENT CHECK
# ============================================================

def timestamp_in_segment(
    timestamp,
    start,
    end,
):
    return start <= timestamp <= end


# ============================================================
# TEMPORAL CO-OCCURRENCE VOTING
# ============================================================

def fuse_speakers_and_faces(
    diarization,
    face_data,
):
    """
    Fuse pyannote speaker diarization with
    DeepFace/DBSCAN face sightings.

    IMPORTANT:
    We use the ACTUAL face sighting timestamps,
    rather than treating an entire face track as
    continuously visible.

    Each face sighting that occurs while a speaker
    is active contributes one temporal vote.
    """

    # --------------------------------------------------------
    # speaker -> person -> number of simultaneous sightings
    # --------------------------------------------------------

    votes = defaultdict(
        lambda: defaultdict(int)
    )

    # Keep detailed evidence for debugging.
    evidence = []

    # --------------------------------------------------------
    # Iterate through speaker segments
    # --------------------------------------------------------

    for speaker_segment in diarization:

        speaker = speaker_segment.get(
            "speaker"
        )

        if not speaker:
            continue

        speaker_start = float(
            speaker_segment["start"]
        )

        speaker_end = float(
            speaker_segment["end"]
        )

        # ----------------------------------------------------
        # Look at ACTUAL face sightings
        # ----------------------------------------------------

        for face_track in face_data:

            person = face_track.get(
                "person"
            )

            track_id = face_track.get(
                "track_id"
            )

            sightings = face_track.get(
                "sightings",
                []
            )

            for sighting in sightings:

                timestamp = float(
                    sighting["time"]
                )

                if timestamp_in_segment(
                    timestamp,
                    speaker_start,
                    speaker_end,
                ):

                    votes[speaker][person] += 1

                    evidence.append(
                        {
                            "speaker": speaker,
                            "person": person,
                            "track_id": track_id,
                            "timestamp": timestamp,
                            "speaker_start": speaker_start,
                            "speaker_end": speaker_end,
                        }
                    )

    # ========================================================
    # DETERMINE ATTRIBUTION
    # ========================================================

    mapping = {}

    for speaker, person_votes in votes.items():

        if not person_votes:
            continue

        total_votes = sum(
            person_votes.values()
        )

        ranked = sorted(
            person_votes.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        winner_person = ranked[0][0]
        winner_votes = ranked[0][1]

        confidence = (
            winner_votes / total_votes
            if total_votes
            else 0.0
        )

        # ----------------------------------------------------
        # Ambiguity detection
        # ----------------------------------------------------

        if len(ranked) > 1:

            second_person = ranked[1][0]
            second_votes = ranked[1][1]

        else:

            second_person = None
            second_votes = 0

        # If the top two are too close, don't pretend
        # that the attribution is certain.
        ambiguous = (
            second_votes > 0
            and (
                winner_votes
                - second_votes
            )
            <= 1
        )

        if ambiguous:
            attributed_person = None
        else:
            attributed_person = winner_person

        mapping[speaker] = {
            "person": attributed_person,
            "confidence": round(
                confidence,
                3
            ),
            "total_votes": total_votes,
            "winner_votes": winner_votes,
            "second_person": second_person,
            "second_votes": second_votes,
            "ambiguous": ambiguous,
            "votes": {
                person: count
                for person, count
                in ranked
            },
        }

    return mapping, evidence


# ============================================================
# SUMMARY
# ============================================================

def print_mapping(mapping):

    print()
    print(
        "Speaker → Face attribution"
    )

    print(
        "----------------------------------------"
    )

    if not mapping:

        print(
            "No speaker/face temporal overlap found."
        )

        return

    for speaker in sorted(mapping):

        result = mapping[speaker]

        person = result["person"]

        confidence = result["confidence"]

        votes = result["votes"]

        if result["ambiguous"]:

            print(
                f"{speaker} → AMBIGUOUS "
                f"(confidence={confidence:.2f})"
            )

        else:

            print(
                f"{speaker} → {person} "
                f"(confidence={confidence:.2f})"
            )

        print(
            f"    votes: {votes}"
        )


# ============================================================
# MAIN
# ============================================================

def run(
    diarization_path="outputs/diarization.json",
    faces_path="outputs/faces.json",
    output_path="outputs/fusion.json",
):

    print()
    print(
        "========================================"
    )
    print(
        "MeetMind Voice ↔ Face Fusion"
    )
    print(
        "Temporal Co-occurrence Voting"
    )
    print(
        "========================================"
    )
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    diarization = load_json(
        diarization_path
    )

    face_data = load_json(
        faces_path
    )

    print(
        f"Loaded {len(diarization)} "
        "speaker segments."
    )

    print(
        f"Loaded {len(face_data)} "
        "face tracks."
    )

    total_face_sightings = sum(
        len(
            track.get(
                "sightings",
                []
            )
        )
        for track in face_data
    )

    print(
        f"Loaded {total_face_sightings} "
        "actual face sightings."
    )

    # --------------------------------------------------------
    # Fuse
    # --------------------------------------------------------

    mapping, evidence = (
        fuse_speakers_and_faces(
            diarization,
            face_data,
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_mapping(
        mapping
    )

    print()

    print(
        f"Generated {len(evidence)} "
        "temporal co-occurrence votes."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "method": (
            "temporal_cooccurrence_voting"
        ),
        "mapping": mapping,
        "evidence": evidence,
    }

    output_file = Path(
        output_path
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {output_file}"
    )

    return output


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    run()