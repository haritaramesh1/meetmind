import json
from pathlib import Path


ACTIVE_FILE = "outputs/active_speaker.json"
TEMPORAL_FILE = "outputs/fusion.json"
OUTPUT_FILE = "outputs/final_attribution.json"


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
# NORMALIZE TEMPORAL SCORES
# ============================================================

def temporal_score(result):
    """
    Convert temporal co-occurrence voting into a
    0..1 score.

    The corrected fusion.py already stores confidence
    based on actual face sightings.
    """

    if not result:
        return 0.0

    return float(
        result.get(
            "confidence",
            0.0
        )
    )


# ============================================================
# NORMALIZE ACTIVE-SPEAKER SCORES
# ============================================================

def active_score(result):
    """
    Visual speaking confidence from active_speaker.py.
    """

    if not result:
        return 0.0

    return float(
        result.get(
            "confidence",
            0.0
        )
    )


# ============================================================
# FINAL FUSION
# ============================================================

def combine_scores(
    temporal_result,
    active_result,
):
    """
    Combine two independent evidence sources.

    Temporal evidence:
        45%

    Visual active-speaking evidence:
        55%

    We give the visual cue slightly more weight because
    simple temporal presence cannot distinguish between
    multiple people visible at the same time.
    """

    temporal_person = None

    if temporal_result:
        temporal_person = (
            temporal_result.get(
                "person"
            )
        )

    active_person = None

    if active_result:
        active_person = (
            active_result.get(
                "person"
            )
        )

    candidates = set()

    if temporal_person:
        candidates.add(
            temporal_person
        )

    if active_person:
        candidates.add(
            active_person
        )

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not candidates:

        return {
            "person": None,
            "confidence": 0.0,
            "status": "NO_EVIDENCE",
            "temporal_person": None,
            "active_person": None,
        }

    # --------------------------------------------------------
    # If both methods agree
    # --------------------------------------------------------

    if (
        temporal_person
        and active_person
        and temporal_person
        == active_person
    ):

        t = temporal_score(
            temporal_result
        )

        a = active_score(
            active_result
        )

        combined = (
            0.45 * t
            + 0.55 * a
        )

        # Agreement bonus.
        combined = min(
            1.0,
            combined + 0.10
        )

        return {
            "person": temporal_person,
            "confidence": round(
                combined,
                3
            ),
            "status": "AGREEMENT",
            "temporal_person":
                temporal_person,
            "active_person":
                active_person,
            "temporal_confidence":
                round(t, 3),
            "active_confidence":
                round(a, 3),
        }

    # --------------------------------------------------------
    # Methods disagree
    # --------------------------------------------------------

    if (
        temporal_person
        and active_person
    ):

        t = temporal_score(
            temporal_result
        )

        a = active_score(
            active_result
        )

        temporal_weighted = (
            0.45 * t
        )

        active_weighted = (
            0.55 * a
        )

        if active_weighted > temporal_weighted:

            winner = active_person

            confidence = active_weighted

        else:

            winner = temporal_person

            confidence = temporal_weighted

        return {
            "person": winner,
            "confidence": round(
                confidence,
                3
            ),
            "status": "DISAGREEMENT",
            "temporal_person":
                temporal_person,
            "active_person":
                active_person,
            "temporal_confidence":
                round(t, 3),
            "active_confidence":
                round(a, 3),
        }

    # --------------------------------------------------------
    # Only one evidence source exists
    # --------------------------------------------------------

    if active_person:

        return {
            "person": active_person,
            "confidence": round(
                0.55
                * active_score(
                    active_result
                ),
                3
            ),
            "status":
                "ACTIVE_ONLY",
            "temporal_person":
                None,
            "active_person":
                active_person,
        }

    return {
        "person": temporal_person,
        "confidence": round(
            0.45
            * temporal_score(
                temporal_result
            ),
            3
        ),
        "status":
            "TEMPORAL_ONLY",
        "temporal_person":
            temporal_person,
        "active_person":
            None,
    }


# ============================================================
# RUN
# ============================================================

def run(
    active_path=ACTIVE_FILE,
    temporal_path=TEMPORAL_FILE,
    output_path=OUTPUT_FILE,
):

    print()
    print(
        "========================================"
    )
    print(
        "MeetMind Final Speaker Attribution"
    )
    print(
        "========================================"
    )
    print()

    active_data = load_json(
        active_path
    )

    temporal_data = load_json(
        temporal_path
    )

    active_mapping = active_data.get(
        "mapping",
        {}
    )

    temporal_mapping = temporal_data.get(
        "mapping",
        {}
    )

    speakers = sorted(
        set(
            active_mapping.keys()
        )
        |
        set(
            temporal_mapping.keys()
        )
    )

    final_mapping = {}

    for speaker in speakers:

        temporal_result = (
            temporal_mapping.get(
                speaker
            )
        )

        active_result = (
            active_mapping.get(
                speaker
            )
        )

        final_mapping[speaker] = (
            combine_scores(
                temporal_result,
                active_result,
            )
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "Final speaker → face attribution"
    )

    print(
        "----------------------------------------"
    )

    for speaker in sorted(
        final_mapping
    ):

        result = final_mapping[
            speaker
        ]

        print(
            f"{speaker} → "
            f"{result['person']} "
            f"(confidence="
            f"{result['confidence']:.2f}, "
            f"status="
            f"{result['status']})"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "method": (
            "DBSCAN_face_identity + "
            "temporal_cooccurrence + "
            "visual_active_speaker"
        ),
        "mapping": final_mapping,
    }

    Path(output_path).write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {output_path}"
    )

    return output


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    run()