
import os
import sys

# Configure stdout and stderr for UTF-8 to display Hindi/Marathi/Devanagari text on Windows
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

FRAME_INTERVAL = 1.0  # For videos: process one frame per second


def extract_text(result):
    """Extract recognized text and confidence scores from PaddleOCR output."""
    data = None
    if hasattr(result, "json"):
        data = result.json
        if isinstance(data, dict) and "res" in data:
            data = data["res"]
    elif isinstance(result, dict):
        data = result.get("res", result)

    if not isinstance(data, dict):
        return []

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])

    extracted = []
    for i, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue

        score = float(scores[i]) if i < len(scores) else None
        extracted.append({
            "text": text,
            "confidence": score
        })

    return extracted



def main():
    print("=" * 55)
    print("       MULTILINGUAL SOCIAL MEDIA OCR")
    print("       English | Hindi | Marathi")
    print("=" * 55)

    raw_path = input(
        "\nEnter the full path of an image or video: "
    ).strip().strip('"').strip("'")

    file_path = Path(raw_path)

    if not file_path.is_file():
        print(f"\nERROR: File not found: {file_path}")
        return

    image_extensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"
    }
    video_extensions = {
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"
    }

    extension = file_path.suffix.lower()

    if extension in image_extensions:
        file_type = "image"
    elif extension in video_extensions:
        file_type = "video"
    else:
        print("\nERROR: Unsupported file type.")
        print("Supported inputs: JPG, PNG, WEBP, MP4, AVI, MOV, MKV")
        return

    total_start = time.perf_counter()

    try:
        print("\nLoading OCR model...")
        model_start = time.perf_counter()

        ocr = PaddleOCR(
            lang="hi",
            ocr_version="PP-OCRv5",
            device="gpu"
        )

        model_load_time = time.perf_counter() - model_start
        processing_start = time.perf_counter()

        all_results = []

        if file_type == "image":
            print("Processing image...")
            for result in ocr.predict(str(file_path)):
                all_results.extend(extract_text(result))

        else:
            print("Processing video...")
            cap = cv2.VideoCapture(str(file_path))

            if not cap.isOpened():
                raise RuntimeError("Could not open the video file.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                cap.release()
                raise RuntimeError("Could not determine video frame rate.")

            frame_number = 0
            next_sample_time = 0.0

            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    timestamp = frame_number / fps

                    if timestamp >= next_sample_time:
                        print(f"Processing frame at {timestamp:.2f}s")

                        frame_results = []
                        for result in ocr.predict(frame):
                            frame_results.extend(extract_text(result))

                        for item in frame_results:
                            all_results.append({
                                **item,
                                "timestamp_seconds": round(timestamp, 2)
                            })

                        next_sample_time += FRAME_INTERVAL

                    frame_number += 1
            finally:
                cap.release()

        processing_time = time.perf_counter() - processing_start

        output = {
            "input_file": file_path.name,
            "input_type": file_type,
            "languages": ["English", "Hindi", "Marathi"],
            "model_load_time_seconds": round(model_load_time, 3),
            "processing_time_seconds": round(processing_time, 3),
            "total_time_seconds": round(
                time.perf_counter() - total_start, 3
            ),
            "extracted_text": all_results
        }

        output_path = file_path.with_name(
            file_path.stem + "_ocr_results.json"
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 55)
        print("                 OCR RESULTS")
        print("=" * 55)

        if not all_results:
            print("No text detected in the input file.")
        else:
            for i, item in enumerate(all_results, start=1):
                if "timestamp_seconds" in item:
                    print(
                        f"\n{i}. Timestamp: "
                        f"{item['timestamp_seconds']} seconds"
                    )

                print(f"   Text: {item['text']}")

                if item["confidence"] is not None:
                    print(
                        f"   Confidence: "
                        f"{item['confidence']:.2%}"
                    )

        print("\n" + "-" * 55)
        print(f"Model loading time : {model_load_time:.3f} seconds")
        print(f"OCR processing time: {processing_time:.3f} seconds")
        print(
            f"Total time         : "
            f"{time.perf_counter() - total_start:.3f} seconds"
        )
        print(f"JSON output saved  : {output_path.resolve()}")
        print("-" * 55)

    except Exception as error:
        print(f"\nERROR: {error}")
        print("Check the input path, file format, and OCR installation.")


if __name__ == "__main__":
    main()
