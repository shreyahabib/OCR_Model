
import os
import sys

# Configure stdout and stderr for UTF-8 to display Hindi/Devanagari characters without crashing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Disable oneDNN / MKLDNN on CPU to prevent PIR attribute conversion errors in PaddlePaddle 3.x
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
import json
import time
from pathlib import Path
from paddleocr import PaddleOCR

VIDEO_PATH = Path("ocr_test_video.mp4")
OUTPUT_PATH = Path("video_ocr_results.json")
FRAME_INTERVAL = 1.0  # seconds

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

ocr = PaddleOCR(
    lang="hi",
    ocr_version="PP-OCRv5",
    device="gpu",
)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the video.")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    cap.release()
    raise RuntimeError("Could not determine video frame rate.")

results_list = []
seen = set()
frame_index = 0
next_sample_time = 0.0
start = time.perf_counter()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_index / fps

        if timestamp >= next_sample_time:
            print(f"Processing frame at {timestamp:.2f}s")

            for result in ocr.predict(frame):
                data = result.json

                if isinstance(data, dict) and "res" in data:
                    data = data["res"]

                if not isinstance(data, dict):
                    continue

                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])

                for i, text in enumerate(texts):
                    text = str(text).strip()
                    if not text:
                        continue

                    # Suppress repeated text across adjacent frames.
                    key = text.casefold()
                    if key in seen:
                        continue
                    seen.add(key)

                    confidence = (
                        float(scores[i]) if i < len(scores) else None
                    )

                    results_list.append({
                        "timestamp_seconds": round(timestamp, 2),
                        "text": text,
                        "confidence": confidence,
                    })

            next_sample_time += FRAME_INTERVAL

        frame_index += 1

finally:
    cap.release()

output = {
    "video": VIDEO_PATH.name,
    "frame_interval_seconds": FRAME_INTERVAL,
    "processing_time_seconds": round(time.perf_counter() - start, 2),
    "extracted_text": results_list,
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n--- VIDEO OCR RESULTS ---")
for item in results_list:
    print(f"[{item['timestamp_seconds']}s] {item['text']}")

print(f"\nSaved results to: {OUTPUT_PATH}")
