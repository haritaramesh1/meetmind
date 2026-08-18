import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

FACE_FILE = "outputs/faces.json"

DIARIZATION_FILE = "outputs/diarization.json"

OUTPUT_FILE = "outputs/active_speaker.json"

MOTION_THRESHOLD = 12.0

MAX_TIME_GAP = 2.0


# ============================================================
# LOAD
# ============================================================

def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# FACE CROP
# ============================================================

def get_lower_face_crop(
    frame,
    box,
):
    """
    Approximate the lower half of the face.

    We deliberately use the lower-face region because
    mouth movement is the visual speaking cue.
    """

    x = int(box.get("x", 0))
    y = int(box.get("y", 0))
    w = int(box.get("w", 0))
    h = int(box.get("h", 0))

    if w <= 0 or h <= 0:
        return None

    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return None

    # Lower 45% of face.
    height = face.shape[0]

    lower_start = int(
        height * 0.55
    )

    mouth_region = face[
        lower_start:,
        :
    ]

    if mouth_region.size == 0:
        return None

    return mouth_region


# ============================================================
# NORMALIZE CROP
# ============================================================

def normalize_crop(crop):

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (64, 48)
    )

    # Normalize illumination so simple lighting changes
    # contribute less to the motion score.
    gray = cv2.equalizeHist(
        gray
    )

    return gray.astype(
        np.float32
    )


# ============================================================
# MOTION SCORE
# ============================================================

def motion_score(
    previous,
    current,
):

    if previous is None or current is None:
        return 0.0

    difference = cv2.absdiff(
        previous,
        current
    )

    return float(
        np.mean(difference)
    )


# ============================================================
# ANALYZE FACE MOTION
# ============================================================

def analyze_face_motion(
    video_path,
    face_data,
):

    print()
    print(
        "Analyzing visual speaking cues..."
    )

    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = (
        video.get(
            cv2.CAP_PROP_FPS
        )
        or 25.0
    )

    # --------------------------------------------------------
    # Collect all sightings
    # --------------------------------------------------------

    observations = []

    for track in face_data:

        person = track.get(
            "person"
        )

        track_id = track.get(
            "track_id"
        )

        for sighting in track.get(
            "sightings",
            []
        ):

            observations.append(
                {
                    "person": person,
                    "track_id": track_id,
                    "time": float(
                        sighting["time"]
                    ),
                    "box": sighting["box"],
                }
            )

    observations.sort(
        key=lambda x: x["time"]
    )

    print(
        f"Analyzing {len(observations)} "
        "face sightings."
    )

    # --------------------------------------------------------
    # Cache frames by frame number
    # --------------------------------------------------------

    frame_cache = {}

    def read_frame(timestamp):

        frame_number = int(
            round(timestamp * fps)
        )

        if frame_number in frame_cache:
            return frame_cache[
                frame_number
            ]

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ok, frame = video.read()

        if not ok:
            return None

        frame_cache[
            frame_number
        ] = frame

        return frame

    # --------------------------------------------------------
    # Calculate motion for each observation
    # --------------------------------------------------------

    results = []

    previous_by_track = {}

    for index, observation in enumerate(
        observations
    ):

        person = observation[
            "person"
        ]

        track_id = observation[
            "track_id"
        ]

        timestamp = observation[
            "time"
        ]

        frame = read_frame(
            timestamp
        )

        if frame is None:
            continue

        crop = get_lower_face_crop(
            frame,
            observation["box"]
        )

        normalized = None

        if crop is not None:

            normalized = normalize_crop(
                crop
            )

        previous = previous_by_track.get(
            track_id
        )

        score = 0.0

        if previous is not None:

            previous_time = previous[
                "time"
            ]

            gap = (
                timestamp
                - previous_time
            )

            if (
                gap > 0
                and gap <= MAX_TIME_GAP
            ):

                score = motion_score(
                    previous["crop"],
                    normalized
                )

        previous_by_track[
            track_id
        ] = {
            "time": timestamp,
            "crop": normalized,
        }

        results.append(
            {
                "person": person,
                "track_id": track_id,
                "time": timestamp,
                "motion_score": round(
                    score,
                    4
                ),
                "active": (
                    score
                    >= MOTION_THRESHOLD
                ),
            }
        )

    video.release()

    print(
        f"Generated {len(results)} "
        "visual motion observations."
    )

    return results


# ============================================================
# FUSE WITH PYANNOTE
# ============================================================

def fuse_active_speaker(
    diarization,
    motion_data,
):

    scores = {}

    for segment in diarization:

        speaker = segment[
            "speaker"
        ]

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        if speaker not in scores:

            scores[speaker] = {}

        for observation in motion_data:

            timestamp = observation[
                "time"
            ]

            if not (
                start
                <= timestamp
                <= end
            ):
                continue

            person = observation[
                "person"
            ]

            motion = observation[
                "motion_score"
            ]

            if person not in scores[
                speaker
            ]:

                scores[
                    speaker
                ][person] = {
                    "observations": 0,
                    "active_observations": 0,
                    "motion": 0.0,
                }

            scores[
                speaker
            ][person]["observations"] += 1

            scores[
                speaker
            ][person]["motion"] += motion

            if observation[
                "active"
            ]:

                scores[
                    speaker
                ][person][
                    "active_observations"
                ] += 1

    # --------------------------------------------------------
    # Determine winner
    # --------------------------------------------------------

    mapping = {}

    for speaker, people in scores.items():

        if not people:
            continue

        ranked = sorted(
            people.items(),
            key=lambda item: (
                item[1][
                    "active_observations"
                ],
                item[1]["motion"],
            ),
            reverse=True,
        )

        winner_person = ranked[0][0]
        winner_stats = ranked[0][1]

        total_active = sum(
            data[
                "active_observations"
            ]
            for data in people.values()
        )

        confidence = (
            winner_stats[
                "active_observations"
            ]
            / total_active
            if total_active > 0
            else 0.0
        )

        mapping[speaker] = {
            "person": winner_person,
            "confidence": round(
                confidence,
                3
            ),
            "active_observations": (
                winner_stats[
                    "active_observations"
                ]
            ),
            "total_motion": round(
                winner_stats["motion"],
                3
            ),
            "candidates": {
                person: {
                    "active_observations":
                        data[
                            "active_observations"
                        ],
                    "observations":
                        data[
                            "observations"
                        ],
                    "motion":
                        round(
                            data["motion"],
                            3
                        ),
                }
                for person, data
                in ranked
            },
        }

    return mapping


# ============================================================
# MAIN
# ============================================================

def run(
    video_path,
    faces_path=FACE_FILE,
    diarization_path=DIARIZATION_FILE,
    output_path=OUTPUT_FILE,
):

    print()
    print(
        "========================================"
    )
    print(
        "MeetMind Active Speaker Analysis"
    )
    print(
        "========================================"
    )

    faces = load_json(
        faces_path
    )

    diarization = load_json(
        diarization_path
    )

    motion = analyze_face_motion(
        video_path,
        faces
    )

    mapping = fuse_active_speaker(
        diarization,
        motion
    )

    output = {
        "method":
            "visual_mouth_motion_plus_pyannote",
        "mapping":
            mapping,
        "observations":
            motion,
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
        "Active speaker attribution"
    )
    print(
        "----------------------------------------"
    )

    for speaker, result in sorted(
        mapping.items()
    ):

        print(
            f"{speaker} → "
            f"{result['person']} "
            f"(confidence="
            f"{result['confidence']:.2f}, "
            f"active observations="
            f"{result['active_observations']})"
        )

    print()
    print(
        f"Saved {output_path}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python active_speaker.py "
            "meetings/meeting5min.mp4"
        )

        raise SystemExit(1)

    run(
        sys.argv[1]
    )