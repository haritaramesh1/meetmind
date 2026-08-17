import json, re
from pathlib import Path
import cv2

def _extract_words(result):
    words = []
    try:
        # PaddleOCR v2/v3 compatibility.
        for page in result:
            if isinstance(page, dict):
                text = page.get("rec_texts", [])
                words.extend(text)
            elif isinstance(page, list):
                for line in page:
                    if isinstance(line, list) and len(line) >= 2:
                        words.append(str(line[1][0]))
    except Exception:
        pass
    return [w.strip() for w in words if str(w).strip()]

def extract_slides(video_path, every_sec=2, min_words=20, out_path=None):
    try:
        from paddleocr import PaddleOCR
        try:
            ocr = PaddleOCR(lang="en")
        except TypeError:
            ocr = PaddleOCR(use_angle_cls=True, lang="en")
    except Exception as e:
        raise RuntimeError("Install PaddleOCR/PaddlePaddle first.") from e

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * every_sec))
    frame_no = 0
    slides = []
    last_text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no % step == 0:
            try:
                try:
                    result = ocr.predict(frame)
                except Exception:
                    result = ocr.ocr(frame, cls=True)
                words = _extract_words(result)
                text = " ".join(words)
                normalized = re.sub(r"\s+", " ", text).strip()
                if len(words) >= min_words and normalized != last_text:
                    slides.append({
                        "time": float(frame_no / fps),
                        "text": normalized,
                    })
                    last_text = normalized
            except Exception:
                pass
        frame_no += 1

    cap.release()
    if out_path:
        Path(out_path).write_text(json.dumps(slides, indent=2), encoding="utf-8")
    return slides
