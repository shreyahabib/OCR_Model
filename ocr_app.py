import os
import sys
import json
import time
from pathlib import Path

# Ensure proper UTF-8 output for English, Hindi, and Marathi text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Disable MKLDNN optimizations that may cause compatibility issues.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

import cv2
from paddleocr import PaddleOCR


# ============================================================
# VIDEO OPTIMIZATION SETTINGS
# ============================================================

FRAME_INTERVAL = 1.0
FRAME_DIFF_THRESHOLD = 5.0
MAX_OCR_INTERVAL = 2.0

COMPARE_WIDTH = 160
COMPARE_HEIGHT = 90

# Current tested batch size.
OCR_BATCH_SIZE = 1

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".tif", ".tiff"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".webm", ".m4v"
}


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(result):
    """Extract recognized text and confidence values from PaddleOCR output."""

    data = None

    if hasattr(result, "json"):
        data = result.json

        if callable(data):
            try:
                data = data()
            except Exception:
                data = None

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = None

    if isinstance(data, dict) and "res" in data:
        data = data["res"]
    elif isinstance(result, dict):
        data = result.get("res", result)

    if not isinstance(data, dict):
        return []

    texts = data.get("rec_texts", []) or []
    scores = data.get("rec_scores", []) or []

    extracted = []

    for i, recognized_text in enumerate(texts):
        recognized_text = str(recognized_text).strip()

        if not recognized_text:
            continue

        try:
            score = (
                float(scores[i])
                if i < len(scores)
                else None
            )
        except (TypeError, ValueError):
            score = None

        extracted.append({
            "text": recognized_text,
            "confidence": score
        })

    return extracted


# ============================================================
# OCR PREDICTION HELPER
# ============================================================

def predict_frames(ocr, frames):
    """Predict OCR for frames, trying batching and falling back safely."""

    if not frames:
        return [], "none"

    # Process a single frame directly.
    if len(frames) == 1:
        frame_text = []

        for result in ocr.predict(frames[0]):
            frame_text.extend(extract_text(result))

        return [frame_text], "single-frame"

    # Try batch prediction.
    try:
        batch_results = list(ocr.predict(frames))

        if len(batch_results) == len(frames):
            return [
                extract_text(result)
                for result in batch_results
            ], "batch"

        print(
            f"Batch returned {len(batch_results)} results for "
            f"{len(frames)} frames. Using individual-frame fallback."
        )

    except Exception as error:
        print("Batch prediction failed; using individual-frame fallback.")
        print(f"Details: {type(error).__name__}: {error}")

    # Compatibility fallback: process frames individually.
    extracted = []

    for frame in frames:
        frame_text = []

        for result in ocr.predict(frame):
            frame_text.extend(extract_text(result))

        extracted.append(frame_text)

    return extracted, "single-frame fallback"


# ============================================================
# IMAGE OCR
# ============================================================

def process_image(ocr, file_path):
    """Run OCR on a single image."""

    print("\nProcessing image...")

    all_results = []

    for result in ocr.predict(str(file_path)):
        all_results.extend(extract_text(result))

    return all_results


# ============================================================
# VIDEO OCR
# ============================================================

def process_video(ocr, file_path):
    """
    Sample video frames, skip visually similar frames,
    and run OCR on selected frames.
    """

    print("\nProcessing video...")

    cap = cv2.VideoCapture(str(file_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {file_path}")

    all_results = []

    sampled_frames = 0
    skipped_frames = 0
    processed_frames = 0

    batch_calls = 0
    single_frame_calls = 0
    batch_fallbacks = 0

    pending_frames = []
    pending_timestamps = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or total_frames <= 0:
            raise RuntimeError(
                "Could not determine video FPS or frame count."
            )

        duration = total_frames / fps

        print(f"Video FPS: {fps:.2f}")
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Sampling interval: {FRAME_INTERVAL:.2f} seconds")
        print(f"Frame difference threshold: {FRAME_DIFF_THRESHOLD:.2f}")
        print(f"Maximum OCR interval: {MAX_OCR_INTERVAL:.2f} seconds")
        print(f"OCR batch size: {OCR_BATCH_SIZE}")

        previous_gray = None
        last_ocr_timestamp = None
        timestamp = 0.0

        processing_start = time.perf_counter()

        def flush_pending():
            nonlocal processed_frames
            nonlocal batch_calls
            nonlocal single_frame_calls
            nonlocal batch_fallbacks

            if not pending_frames:
                return

            frames = list(pending_frames)
            timestamps = list(pending_timestamps)

            results_by_frame, method = predict_frames(ocr, frames)

            if method == "batch":
                batch_calls += 1
                print(f"Batch processed {len(frames)} frames.")

            elif method == "single-frame fallback":
                batch_fallbacks += 1
                single_frame_calls += len(frames)
                print("Queued frames processed individually.")

            else:
                single_frame_calls += len(frames)

            while len(results_by_frame) < len(timestamps):
                results_by_frame.append([])

            for frame_time, items in zip(
                timestamps, results_by_frame
            ):
                for item in items:
                    all_results.append({
                        **item,
                        "timestamp_seconds": round(frame_time, 2)
                    })

            processed_frames += len(frames)

            pending_frames.clear()
            pending_timestamps.clear()

        # Sample frames at the configured interval.
        while timestamp < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = cap.read()

            if not ok or frame is None:
                print(f"Could not read frame at {timestamp:.2f}s.")
                break

            sampled_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray_small = cv2.resize(
                gray,
                (COMPARE_WIDTH, COMPARE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            should_run_ocr = False
            difference_score = None
            reason = ""

            if previous_gray is None:
                should_run_ocr = True
                reason = "first frame"

            else:
                difference = cv2.absdiff(
                    gray_small, previous_gray
                )

                difference_score = float(difference.mean())

                if difference_score >= FRAME_DIFF_THRESHOLD:
                    should_run_ocr = True
                    reason = "significant visual change"

                # Periodically run OCR even when the frame difference
                # is small, so persistent text is not missed indefinitely.
                if (
                    last_ocr_timestamp is not None
                    and timestamp - last_ocr_timestamp
                    >= MAX_OCR_INTERVAL
                ):
                    should_run_ocr = True
                    reason = "maximum OCR interval reached"

            if should_run_ocr:
                message = f"Queueing frame at {timestamp:.2f}s"

                if difference_score is not None:
                    message += f" | difference: {difference_score:.2f}"

                print(f"{message} | reason: {reason}")

                pending_frames.append(frame)
                pending_timestamps.append(timestamp)

                last_ocr_timestamp = timestamp

                if len(pending_frames) >= OCR_BATCH_SIZE:
                    flush_pending()

            else:
                print(
                    f"Skipping OCR at {timestamp:.2f}s "
                    f"| difference: {difference_score:.2f}"
                )

                skipped_frames += 1

            # Compare the next sampled frame with this one.
            previous_gray = gray_small
            timestamp += FRAME_INTERVAL

        # Process any remaining frames.
        flush_pending()

        processing_time = time.perf_counter() - processing_start

        stats = {
            "video_duration_seconds": round(duration, 3),
            "video_fps": round(fps, 3),
            "sampled_frames": sampled_frames,
            "ocr_processed_frames": processed_frames,
            "skipped_frames": skipped_frames,
            "frame_difference_threshold": FRAME_DIFF_THRESHOLD,
            "max_ocr_interval_seconds": MAX_OCR_INTERVAL,
            "video_frame_interval_seconds": FRAME_INTERVAL,
            "ocr_batch_size": OCR_BATCH_SIZE,
            "batch_calls": batch_calls,
            "single_frame_calls": single_frame_calls,
            "batch_fallbacks": batch_fallbacks,
            "processing_time_seconds": round(processing_time, 3)
        }

    finally:
        cap.release()

    print("\n========== VIDEO PROCESSING SUMMARY ==========")
    print(f"Video duration      : {stats['video_duration_seconds']:.2f} seconds")
    print(f"Sampled frames      : {stats['sampled_frames']}")
    print(f"Frames sent to OCR  : {stats['ocr_processed_frames']}")
    print(f"Frames skipped      : {stats['skipped_frames']}")
    print(f"Batch calls         : {stats['batch_calls']}")
    print(f"Single-frame calls  : {stats['single_frame_calls']}")
    print(f"Batch fallbacks     : {stats['batch_fallbacks']}")
    print(f"Processing time     : {stats['processing_time_seconds']:.3f} seconds")
    print("==============================================")

    return all_results, stats


# ============================================================
# FILE TYPE DETECTION
# ============================================================

def detect_file_type(file_path):
    """Return image/video type, or None for unsupported extensions."""

    extension = file_path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in VIDEO_EXTENSIONS:
        return "video"

    return None


# ============================================================
# PROCESS ONE FILE AND DISPLAY COMPLETE EXTRACTED TEXT
# ============================================================

def process_one_file(ocr, file_path, file_type, model_load_time):
    """Process one file, display all recognized text, and save JSON."""

    file_start = time.perf_counter()
    processing_start = time.perf_counter()

    video_stats = None

    if file_type == "image":
        all_results = process_image(ocr, file_path)
    else:
        all_results, video_stats = process_video(ocr, file_path)

    processing_time = time.perf_counter() - processing_start

    output_path = file_path.with_name(
        file_path.stem + "_ocr_results.json"
    )

    output = {
        "input_file": file_path.name,
        "input_type": file_type,
        "languages": ["English", "Hindi", "Marathi"],
        "model_load_time_seconds": round(model_load_time, 3),
        "model_reused": True,
        "processing_time_seconds": round(processing_time, 3),
        "total_time_seconds": None,
        "video_statistics": video_stats,
        "extracted_text": all_results
    }

    # Display individual OCR results.
    print("\n" + "=" * 60)
    print("                     OCR RESULTS")
    print("=" * 60)

    if not all_results:
        print("No text detected in the input file.")

    else:
        for i, item in enumerate(all_results, start=1):

            if "timestamp_seconds" in item:
                print(
                    f"\n{i}. Timestamp: "
                    f"{item['timestamp_seconds']} seconds"
                )
            else:
                print(f"\n{i}.")

            print(f"   Text: {item['text']}")

            if item.get("confidence") is not None:
                print(f"   Confidence: {item['confidence']:.2%}")

    # NEW FEATURE:
    # Combine all recognized text into one complete text section.
    complete_text = "\n".join(
        item["text"]
        for item in all_results
        if item.get("text", "").strip()
    )

    print("\n" + "=" * 60)
    print("                COMPLETE EXTRACTED TEXT")
    print("=" * 60)

    if complete_text:
        print(complete_text)
    else:
        print("No text detected.")

    print("=" * 60)

    # Save all results, including the complete extracted text, to JSON.
    output["complete_extracted_text"] = complete_text

    # Include JSON-writing time in total_time_seconds.
    output["total_time_seconds"] = round(
        time.perf_counter() - file_start, 3
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    # Display performance information.
    print("\n" + "-" * 60)
    print(
        f"Model loading time (startup, once): "
        f"{model_load_time:.3f} seconds"
    )
    print(f"OCR processing time for this file: {processing_time:.3f} seconds")
    print(
        f"Per-file total time              : "
        f"{output['total_time_seconds']:.3f} seconds"
    )

    if video_stats is not None:
        print(f"Sampled frames                   : {video_stats['sampled_frames']}")
        print(f"Frames sent to OCR               : {video_stats['ocr_processed_frames']}")
        print(f"Frames skipped                   : {video_stats['skipped_frames']}")
        print(f"Batch calls                      : {video_stats['batch_calls']}")
        print(f"Single-frame calls               : {video_stats['single_frame_calls']}")
        print(f"Batch fallbacks                  : {video_stats['batch_fallbacks']}")

    print(f"JSON output saved                : {output_path.resolve()}")
    print("-" * 60)


# ============================================================
# MAIN APPLICATION - PERSISTENT MODEL MODE
# ============================================================

def main():
    print("=" * 60)
    print("          MULTILINGUAL SOCIAL MEDIA OCR")
    print("          English | Hindi | Marathi")
    print("          Persistent Model Mode")
    print("=" * 60)

    print("The OCR model will load once and be reused for each file.")
    print("Type 'exit', 'quit', or 'q' to close the application.\n")

    print("Loading OCR model...")

    model_start = time.perf_counter()

    try:
        ocr = PaddleOCR(
            lang="hi",
            ocr_version="PP-OCRv5",
            device="gpu",
            use_doc_unwarping=False,
            use_doc_orientation_classify=False
        )

    except Exception as error:
        print(
            f"\nERROR while loading OCR model: "
            f"{type(error).__name__}: {error}"
        )
        print("Check your PaddleOCR installation and device configuration.")
        return

    model_load_time = time.perf_counter() - model_start

    print(f"Model loaded in {model_load_time:.3f} seconds.\n")

    files_processed = 0

    while True:
        try:
            raw_path = input(
                "Enter the full path of an image or video (or 'exit'): "
            )

        except (EOFError, KeyboardInterrupt):
            print("\nClosing OCR application.")
            break

        raw_path = raw_path.strip().strip('"').strip("'")

        if raw_path.lower() in {"exit", "quit", "q"}:
            print(f"\nProcessed {files_processed} file(s) in this session.")
            print("Closing OCR application.")
            break

        if not raw_path:
            print("Please enter a file path.\n")
            continue

        file_path = Path(raw_path).expanduser()

        if not file_path.is_file():
            print(f"ERROR: File not found: {file_path}\n")
            continue

        file_type = detect_file_type(file_path)

        if file_type is None:
            print("ERROR: Unsupported file type.")
            print("Supported images: JPG, PNG, BMP, WEBP, TIF, TIFF")
            print("Supported videos: MP4, AVI, MOV, MKV, WEBM, M4V\n")
            continue

        try:
            process_one_file(
                ocr,
                file_path,
                file_type,
                model_load_time
            )

            files_processed += 1
            print("\nModel remains loaded; ready for the next file.\n")

        except Exception as error:
            print(
                f"\nERROR processing {file_path}: "
                f"{type(error).__name__}: {error}"
            )
            print("You can enter another file path, or type 'exit'.\n")


if __name__ == "__main__":
    main()