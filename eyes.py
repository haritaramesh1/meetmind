import json
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.cluster import DBSCAN

def face_sightings(video_path, every_sec=2):
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    sightings = []
    frame_no = 0
    step = max(1, int(fps * every_sec))

    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_no % step == 0:
            try:
                faces = DeepFace.represent(
                    frame,
                    model_name="Facenet",
                    enforce_detection=True,
                )
                for f in faces:
                    sightings.append({
                        "time": float(frame_no / fps),
                        "box": f["facial_area"],
                        "signature": f["embedding"],
                    })
            except (ValueError, RuntimeError):
                pass
        frame_no += 1

    video.release()
    return sightings

def cluster_people(sightings, eps=0.30):
    if not sightings:
        return sightings
    X = np.asarray([s["signature"] for s in sightings], dtype=np.float32)
    labels = DBSCAN(
        eps=eps, min_samples=1, metric="cosine"
    ).fit_predict(X)
    out = []
    for s, label in zip(sightings, labels):
        item = dict(s)
        item.pop("signature", None)
        item["person"] = f"PERSON_{int(label):02d}"
        out.append(item)
    return out

def run(video_path, out_path=None):
    sightings = face_sightings(video_path)
    clustered = cluster_people(sightings)
    if out_path:
        Path(out_path).write_text(json.dumps(clustered, indent=2), encoding="utf-8")
    return clustered

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "meetings/meeting1.mp4"
    run(path, "outputs/faces.json")
    print("Saved outputs/faces.json")
