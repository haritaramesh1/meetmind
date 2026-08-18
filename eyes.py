import json
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Facenet"

EVERY_SEC = 1.0

TRACK_MAX_GAP = 3.0

TRACK_DISTANCE_THRESHOLD = 0.55

DBSCAN_EPS = 0.30


# ============================================================
# FACE DETECTION + EMBEDDINGS
# ============================================================

def face_sightings(video_path, every_sec=EVERY_SEC):

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = video.get(
        cv2.CAP_PROP_FPS
    ) or 25.0

    sightings = []

    frame_no = 0

    step = max(
        1,
        int(fps * every_sec)
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        f"Sampling every {every_sec:.1f} seconds"
    )

    while True:

        ok, frame = video.read()

        if not ok:
            break

        if frame_no % step == 0:

            timestamp = (
                frame_no / fps
            )

            try:

                faces = DeepFace.represent(
                    img_path=frame,
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                )

                for face in faces:

                    box = face.get(
                        "facial_area",
                        {}
                    )

                    embedding = np.asarray(
                        face["embedding"],
                        dtype=np.float32,
                    )

                    sightings.append(
                        {
                            "time": float(
                                timestamp
                            ),
                            "box": {
                                "x": int(
                                    box.get(
                                        "x",
                                        0
                                    )
                                ),
                                "y": int(
                                    box.get(
                                        "y",
                                        0
                                    )
                                ),
                                "w": int(
                                    box.get(
                                        "w",
                                        0
                                    )
                                ),
                                "h": int(
                                    box.get(
                                        "h",
                                        0
                                    )
                                ),
                            },
                            "signature": (
                                embedding.tolist()
                            ),
                        }
                    )

            except Exception:
                # No face detected / detector issue.
                pass

        frame_no += 1

    video.release()

    print(
        f"Detected {len(sightings)} face sightings."
    )

    return sightings


# ============================================================
# COSINE DISTANCE
# ============================================================

def cosine_distance(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 1.0

    similarity = np.dot(
        a,
        b
    ) / (
        a_norm * b_norm
    )

    return float(
        1.0 - similarity
    )


# ============================================================
# SIMPLE TEMPORAL FACE TRACKING
# ============================================================

def track_faces(
    sightings,
    max_gap=TRACK_MAX_GAP,
    distance_threshold=TRACK_DISTANCE_THRESHOLD,
):

    if not sightings:
        return []

    sightings = sorted(
        sightings,
        key=lambda x: x["time"]
    )

    tracks = []

    for sighting in sightings:

        best_track = None
        best_distance = float("inf")

        current_time = sighting["time"]

        current_embedding = (
            sighting["signature"]
        )

        for track in tracks:

            time_gap = (
                current_time
                - track["last_time"]
            )

            if time_gap > max_gap:
                continue

            distance = cosine_distance(
                current_embedding,
                track["last_embedding"]
            )

            if (
                distance < best_distance
                and distance <= distance_threshold
            ):
                best_distance = distance
                best_track = track

        if best_track is None:

            track_id = len(tracks)

            track = {
                "track_id": track_id,
                "start": current_time,
                "end": current_time,
                "last_time": current_time,
                "last_embedding": current_embedding,
                "sightings": [sighting],
            }

            tracks.append(track)

        else:

            best_track["end"] = current_time

            best_track["last_time"] = (
                current_time
            )

            best_track["last_embedding"] = (
                current_embedding
            )

            best_track["sightings"].append(
                sighting
            )

    print(
        f"Built {len(tracks)} temporal face tracks."
    )

    return tracks


# ============================================================
# TRACK EMBEDDINGS
# ============================================================

def track_embedding(track):

    embeddings = np.asarray(
        [
            s["signature"]
            for s in track["sightings"]
        ],
        dtype=np.float32,
    )

    # Average embedding across the track.
    mean_embedding = np.mean(
        embeddings,
        axis=0
    )

    norm = np.linalg.norm(
        mean_embedding
    )

    if norm > 0:
        mean_embedding /= norm

    return mean_embedding


# ============================================================
# DBSCAN ON FACE TRACKS
# ============================================================

def cluster_people(
    tracks,
    eps=DBSCAN_EPS,
):

    if not tracks:
        return []

    X = np.asarray(
        [
            track_embedding(track)
            for track in tracks
        ],
        dtype=np.float32,
    )

    print(
        "Running DBSCAN on "
        f"{len(tracks)} face tracks..."
    )

    labels = DBSCAN(
        eps=eps,
        min_samples=1,
        metric="cosine",
    ).fit_predict(X)

    clustered = []

    for track, label in zip(
        tracks,
        labels
    ):

        item = {
            "track_id": track["track_id"],
            "person": (
                f"PERSON_{int(label):02d}"
            ),
            "start": track["start"],
            "end": track["end"],
            "sightings": [],
        }

        for sighting in track[
            "sightings"
        ]:

            item["sightings"].append(
                {
                    "time": sighting["time"],
                    "box": sighting["box"],
                }
            )

        clustered.append(item)

    person_ids = sorted(
        {
            item["person"]
            for item in clustered
        }
    )

    print(
        "DBSCAN produced "
        f"{len(person_ids)} face clusters:"
    )

    for person in person_ids:

        count = sum(
            1
            for item in clustered
            if item["person"] == person
        )

        print(
            f"  {person}: "
            f"{count} track(s)"
        )

    return clustered


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(
    video_path,
    out_path="outputs/faces.json",
):

    print()
    print(
        "========================================"
    )
    print(
        "MeetMind Visual Speaker Attribution"
    )
    print(
        "========================================"
    )

    print()

    # --------------------------------------------------------
    # 1. Detect faces + embeddings
    # --------------------------------------------------------

    sightings = face_sightings(
        video_path
    )

    if not sightings:

        print(
            "No faces were detected."
        )

        return []

    # --------------------------------------------------------
    # 2. Build temporal face tracks
    # --------------------------------------------------------

    tracks = track_faces(
        sightings
    )

    # --------------------------------------------------------
    # 3. Cluster FACE TRACKS with DBSCAN
    # --------------------------------------------------------

    clustered = cluster_people(
        tracks
    )

    # --------------------------------------------------------
    # 4. Save
    # --------------------------------------------------------

    output = Path(out_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(
        json.dumps(
            clustered,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        f"Saved {output}"
    )

    print()

    return clustered


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    import sys

    video_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "meetings/meeting5min.mp4"
    )

    run(
        video_path,
        "outputs/faces.json"
    )