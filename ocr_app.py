import os
import sys
import json
import time
import uuid
import shutil
import argparse
from pathlib import Path

# Set before importing PaddleOCR.
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"

import cv2
import mariadb
import paddle
from paddleocr import PaddleOCR


# ============================================================
# CONFIGURATION
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_INTERVAL = 1.0
FRAME_DIFF_THRESHOLD = 5.0
MAX_OCR_INTERVAL = 2.0

COMPARE_WIDTH = 160
COMPARE_HEIGHT = 90

OCR_BATCH_SIZE = 2
CPU_THREADS = 4

MAX_UPLOAD_BYTES = int(
    os.getenv("OCR_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024))
)

WORKER_POLL_SECONDS = float(
    os.getenv("OCR_WORKER_POLL_SECONDS", "2")
)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".tif", ".tiff"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov",
    ".mkv", ".webm", ".m4v"
}

MARIADB_CONFIG = {
    "host": os.getenv("OCR_DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("OCR_DB_PORT", "3307")),
    "user": os.getenv("OCR_DB_USER", "root"),
    "password": os.getenv("OCR_DB_PASSWORD", "Shreya@123_"),
    "database": os.getenv("OCR_DB_NAME", "ocr_database"),
    "connect_timeout": 10,
}


# ============================================================
# LOGGING
# ============================================================

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ============================================================
# CPU / GPU SELECTION
# ============================================================

def choose_ocr_device():
    while True:
        print("\n========== SELECT OCR DEVICE ==========")
        print("1. CPU")
        print("2. GPU")
        print("=======================================")

        choice = input("Choose CPU or GPU: ").strip().lower()

        if choice in ("1", "cpu"):
            log("Selected OCR device: CPU")
            return "cpu"

        if choice in ("2", "gpu"):
            try:
                if not paddle.is_compiled_with_cuda():
                    log("GPU unavailable: PaddlePaddle has no CUDA support.")
                    continue

                gpu_count = paddle.device.cuda.device_count()

                if gpu_count < 1:
                    log("No CUDA GPU detected.")
                    continue

                log(f"Selected GPU 0 of {gpu_count} detected GPU(s).")
                return "gpu:0"

            except Exception as exc:
                log(f"GPU verification failed: {exc}")
                continue

        print("Invalid choice. Enter CPU, GPU, 1 or 2.")


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    return mariadb.connect(**MARIADB_CONFIG)


def initialize_database():
    started = time.perf_counter()
    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_jobs (
                id CHAR(36) PRIMARY KEY,
                original_filename VARCHAR(255) NOT NULL,
                stored_filename VARCHAR(255) NOT NULL,

                status ENUM(
                    'queued',
                    'processing',
                    'completed',
                    'failed'
                ) NOT NULL DEFAULT 'queued',

                result_json LONGTEXT NULL,
                error_text TEXT NULL,

                created_at TIMESTAMP NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                updated_at TIMESTAMP NOT NULL
                    DEFAULT CURRENT_TIMESTAMP
                    ON UPDATE CURRENT_TIMESTAMP,

                INDEX idx_status_created (status, created_at)
            ) ENGINE=InnoDB
              DEFAULT CHARSET=utf8mb4
              COLLATE=utf8mb4_unicode_ci
        """)

        connection.commit()
        cursor.close()

        elapsed = time.perf_counter() - started
        log(f"Database initialization completed in {elapsed:.3f}s")

    finally:
        connection.close()


# ============================================================
# OCR MODEL
# ============================================================

def load_ocr_model(device):
    log(f"Loading PaddleOCR on {device.upper()}...")

    started = time.perf_counter()

    options = {
        "lang": "hi",
        "ocr_version": "PP-OCRv5",
        "device": device,
        "use_doc_unwarping": False,
        "use_doc_orientation_classify": False,
    }

    if device == "cpu":
        options["cpu_threads"] = CPU_THREADS
        options["enable_mkldnn"] = False

    ocr = PaddleOCR(**options)

    elapsed = time.perf_counter() - started

    log(f"OCR model loaded in {elapsed:.3f}s")
    log(f"OCR device: {device.upper()}")

    return ocr, elapsed


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(result):
    data = getattr(result, "json", None)

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

    for index, recognized_text in enumerate(texts):
        recognized_text = str(recognized_text).strip()

        if not recognized_text:
            continue

        try:
            confidence = (
                float(scores[index])
                if index < len(scores)
                else None
            )
        except (TypeError, ValueError):
            confidence = None

        extracted.append({
            "text": recognized_text,
            "confidence": confidence
        })

    return extracted


# ============================================================
# OCR BATCH PREDICTION WITH FALLBACK
# ============================================================

def predict_frames(ocr, frames):
    if not frames:
        return [], "none"

    if len(frames) == 1:
        started = time.perf_counter()
        extracted = []

        for result in ocr.predict(frames[0]):
            extracted.extend(extract_text(result))

        log(
            f"OCR inference: 1 frame | "
            f"{time.perf_counter() - started:.3f}s"
        )

        return [extracted], "single-frame"

    batch_started = time.perf_counter()

    try:
        batch_results = list(ocr.predict(frames))
        elapsed = time.perf_counter() - batch_started

        log(
            f"OCR batch inference: {len(frames)} frames | "
            f"{elapsed:.3f}s"
        )

        if len(batch_results) == len(frames):
            return [
                extract_text(result)
                for result in batch_results
            ], "batch"

        log("Batch result count mismatch. Using fallback.")

    except Exception as exc:
        elapsed = time.perf_counter() - batch_started
        log(f"Batch OCR failed after {elapsed:.3f}s: {exc}")
        log("Using individual-frame fallback.")

    extracted_frames = []

    for index, frame in enumerate(frames, start=1):
        started = time.perf_counter()
        extracted = []

        for result in ocr.predict(frame):
            extracted.extend(extract_text(result))

        log(
            f"Fallback frame {index}/{len(frames)}: "
            f"{time.perf_counter() - started:.3f}s"
        )

        extracted_frames.append(extracted)

    return extracted_frames, "single-frame fallback"

# ============================================================
# IMAGE OCR
# ============================================================

def process_image(ocr, file_path):
    log(f"Processing image: {file_path.name}")

    started = time.perf_counter()
    all_results = []

    for result in ocr.predict(str(file_path)):
        all_results.extend(extract_text(result))

    elapsed = time.perf_counter() - started
    log(f"Image OCR inference time: {elapsed:.3f}s")

    return all_results, {
        "processing_time_seconds": round(elapsed, 3)
    }


# ============================================================
# VIDEO OCR WITH FRAME COMPARISON AND BATCHING
# ============================================================

def process_video(ocr, file_path):
    log(f"Processing video: {file_path.name}")

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

    seek_time = 0.0
    decode_time = 0.0
    preprocess_time = 0.0
    comparison_time = 0.0
    ocr_total_time = 0.0
    result_handling_time = 0.0

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

        log(f"Video FPS: {fps:.2f}")
        log(f"Video duration: {duration:.2f}s")
        log(f"Sampling interval: {FRAME_INTERVAL:.2f}s")
        log(f"Frame difference threshold: {FRAME_DIFF_THRESHOLD}")
        log(f"Maximum OCR interval: {MAX_OCR_INTERVAL:.2f}s")
        log(f"OCR batch size: {OCR_BATCH_SIZE}")

        previous_gray = None
        last_ocr_timestamp = None
        timestamp = 0.0

        processing_started = time.perf_counter()

        def flush_pending():
            nonlocal processed_frames
            nonlocal batch_calls
            nonlocal single_frame_calls
            nonlocal batch_fallbacks
            nonlocal ocr_total_time
            nonlocal result_handling_time

            if not pending_frames:
                return

            frames = list(pending_frames)
            timestamps = list(pending_timestamps)

            started = time.perf_counter()
            results_by_frame, method = predict_frames(ocr, frames)
            ocr_total_time += time.perf_counter() - started

            result_started = time.perf_counter()

            if method == "batch":
                batch_calls += 1

            elif method == "single-frame fallback":
                batch_fallbacks += 1
                single_frame_calls += len(frames)

            else:
                single_frame_calls += len(frames)

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

            result_handling_time += (
                time.perf_counter() - result_started
            )

        while timestamp < duration:

            # Directly seek to the required timestamp.
            started = time.perf_counter()

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp * 1000
            )

            seek_time += time.perf_counter() - started

            # Decode the selected frame.
            started = time.perf_counter()

            ok, frame = cap.read()

            decode_time += time.perf_counter() - started

            if not ok or frame is None:
                break

            sampled_frames += 1

            # Resize grayscale frame for faster comparison.
            started = time.perf_counter()

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            gray_small = cv2.resize(
                gray,
                (COMPARE_WIDTH, COMPARE_HEIGHT),
                interpolation=cv2.INTER_AREA
            )

            preprocess_time += time.perf_counter() - started

            # Compare with the previous sampled frame.
            started = time.perf_counter()

            should_run_ocr = False
            reason = ""

            if previous_gray is None:
                should_run_ocr = True
                reason = "first frame"

            else:
                difference_score = float(
                    cv2.absdiff(
                        previous_gray,
                        gray_small
                    ).mean()
                )

                if difference_score >= FRAME_DIFF_THRESHOLD:
                    should_run_ocr = True
                    reason = "visual difference detected"

                elif (
                    last_ocr_timestamp is not None
                    and timestamp - last_ocr_timestamp
                    >= MAX_OCR_INTERVAL
                ):
                    should_run_ocr = True
                    reason = "maximum OCR interval reached"

            comparison_time += time.perf_counter() - started

            if should_run_ocr:
                log(
                    f"Queueing frame at {timestamp:.2f}s "
                    f"| reason: {reason}"
                )

                pending_frames.append(frame)
                pending_timestamps.append(timestamp)

                last_ocr_timestamp = timestamp

                if len(pending_frames) >= OCR_BATCH_SIZE:
                    flush_pending()

            else:
                skipped_frames += 1

                log(
                    f"Skipping OCR at {timestamp:.2f}s "
                    f"| difference: {difference_score:.2f}"
                )

            previous_gray = gray_small
            timestamp += FRAME_INTERVAL

        # Process remaining frames.
        flush_pending()

        processing_time = time.perf_counter() - processing_started

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
            "processing_time_seconds": round(processing_time, 3),
            "detailed_timing_seconds": {
                "video_seeking": round(seek_time, 3),
                "frame_decoding": round(decode_time, 3),
                "preprocessing": round(preprocess_time, 3),
                "frame_comparison": round(comparison_time, 3),
                "ocr_inference": round(ocr_total_time, 3),
                "result_handling": round(result_handling_time, 3)
            }
        }

        log("========== VIDEO PROCESSING SUMMARY ==========")
        log(f"Sampled frames: {sampled_frames}")
        log(f"OCR processed frames: {processed_frames}")
        log(f"Skipped frames: {skipped_frames}")
        log(f"Batch calls: {batch_calls}")
        log(f"Batch fallbacks: {batch_fallbacks}")
        log(f"Total processing time: {processing_time:.3f}s")
        log(f"OCR inference time: {ocr_total_time:.3f}s")
        log("===============================================")

        return all_results, stats

    finally:
        cap.release()


# ============================================================
# FILE TYPE DETECTION
# ============================================================

def detect_file_type(file_path):
    extension = Path(file_path).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in VIDEO_EXTENSIONS:
        return "video"

    return None


# ============================================================
# PROCESS ONE QUEUED FILE
# ============================================================

def process_one_file(ocr, file_path, file_type, model_load_time):
    file_path = Path(file_path)
    started = time.perf_counter()

    log(f"Starting OCR job for: {file_path.name}")

    if file_type == "image":
        extracted, statistics = process_image(
            ocr,
            file_path
        )
        video_stats = None

    elif file_type == "video":
        extracted, video_stats = process_video(
            ocr,
            file_path
        )
        statistics = video_stats

    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    complete_text = "\n".join(
        item["text"]
        for item in extracted
        if item.get("text", "").strip()
    )

    processing_time = time.perf_counter() - started

    output = {
        "input_file": file_path.name,
        "input_type": file_type,
        "languages": ["English", "Hindi", "Marathi"],
        "model_load_time_seconds": round(model_load_time, 3),
        "model_reused": True,
        "processing_time_seconds": round(processing_time, 3),
        "video_statistics": video_stats,
        "statistics": statistics,
        "extracted_text": extracted,
        "complete_extracted_text": complete_text
    }

    log("========== EXTRACTED TEXT ==========")
    print(complete_text or "No text detected.", flush=True)
    log("====================================")

    log(f"OCR processing completed in {processing_time:.3f}s")

    return output

# ============================================================
# SUBMIT JOB TO MARIADB
# ============================================================

def submit_job(source_path):
    started = time.perf_counter()

    source_path = Path(source_path).expanduser()

    if not source_path.is_file():
        log(f"File not found: {source_path}")
        return

    file_type = detect_file_type(source_path)

    if file_type is None:
        log("Unsupported file type.")
        log("Supported images: PNG, JPG, JPEG, BMP, WEBP, TIF, TIFF")
        log("Supported videos: MP4, AVI, MOV, MKV, WEBM, M4V")
        return

    file_size = source_path.stat().st_size

    if file_size > MAX_UPLOAD_BYTES:
        log(
            f"File is too large. Maximum allowed size: "
            f"{MAX_UPLOAD_BYTES / (1024 * 1024):.0f} MB"
        )
        return

    job_id = str(uuid.uuid4())

    # Give the stored file a unique name to avoid filename conflicts.
    stored_filename = (
        f"{job_id}{source_path.suffix.lower()}"
    )

    destination = UPLOAD_DIR / stored_filename

    copy_started = time.perf_counter()

    shutil.copy2(source_path, destination)

    copy_time = time.perf_counter() - copy_started

    log(f"File copied to uploads: {destination}")
    log(f"File copy time: {copy_time:.3f}s")

    connection = None

    try:
        db_started = time.perf_counter()

        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO ocr_jobs (
                id,
                original_filename,
                stored_filename,
                status
            )
            VALUES (?, ?, ?, 'queued')
            """,
            (
                job_id,
                source_path.name,
                stored_filename
            )
        )

        connection.commit()
        cursor.close()

        db_time = time.perf_counter() - db_started
        total_time = time.perf_counter() - started

        log("========== JOB SUBMITTED ==========")
        log(f"Job ID: {job_id}")
        log(f"Filename: {source_path.name}")
        log(f"File type: {file_type}")
        log("Status: QUEUED")
        log(f"Database insertion time: {db_time:.3f}s")
        log(f"Total submission time: {total_time:.3f}s")
        log("===================================")

    except Exception:
        # If database insertion fails, remove the copied file.
        if destination.exists():
            destination.unlink()

        raise

    finally:
        if connection is not None:
            connection.close()


# ============================================================
# CHECK JOB STATUS
# ============================================================

def show_job_status(job_id):
    started = time.perf_counter()

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                original_filename,
                status,
                error_text,
                created_at,
                updated_at
            FROM ocr_jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        row = cursor.fetchone()
        cursor.close()

        if row is None:
            log(f"No job found with ID: {job_id}")
            return

        print("\n========== JOB STATUS ==========")
        print(f"Job ID: {row[0]}")
        print(f"Filename: {row[1]}")
        print(f"Status: {row[2]}")
        print(f"Error: {row[3] or 'None'}")
        print(f"Created at: {row[4]}")
        print(f"Updated at: {row[5]}")
        print("================================")

        log(
            f"Status lookup completed in "
            f"{time.perf_counter() - started:.3f}s"
        )

    finally:
        connection.close()


# ============================================================
# CLAIM NEXT QUEUED JOB
# ============================================================

def claim_next_job():
    """
    Atomically claim the oldest queued job.

    The row lock prevents two workers from claiming
    the same job at the same time.
    """

    started = time.perf_counter()
    connection = get_db_connection()

    try:
        connection.begin()
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                original_filename,
                stored_filename
            FROM ocr_jobs
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            FOR UPDATE
            """
        )

        row = cursor.fetchone()

        if row is None:
            connection.commit()
            cursor.close()
            return None

        job = {
            "id": row[0],
            "original_filename": row[1],
            "stored_filename": row[2]
        }

        cursor.execute(
            """
            UPDATE ocr_jobs
            SET status = 'processing',
                error_text = NULL
            WHERE id = ?
              AND status = 'queued'
            """,
            (job["id"],)
        )

        if cursor.rowcount != 1:
            connection.rollback()
            cursor.close()
            return None

        connection.commit()
        cursor.close()

        elapsed = time.perf_counter() - started

        log(
            f"Job claimed: {job['id']} "
            f"| claim time: {elapsed:.3f}s"
        )

        return job

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# UPDATE JOB IN MARIADB
# ============================================================

def update_job(job_id, status, result_json=None, error_text=None):
    started = time.perf_counter()

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE ocr_jobs
            SET status = ?,
                result_json = ?,
                error_text = ?
            WHERE id = ?
            """,
            (
                status,
                result_json,
                error_text,
                job_id
            )
        )

        connection.commit()
        cursor.close()

        elapsed = time.perf_counter() - started

        log(
            f"Database job update: {status.upper()} "
            f"| {elapsed:.3f}s"
        )

    finally:
        connection.close()


# ============================================================
# PROCESS ONE QUEUED JOB
# ============================================================

def process_queued_job(ocr, model_load_time, job):
    job_id = job["id"]

    started = time.perf_counter()

    stored_path = UPLOAD_DIR / job["stored_filename"]

    log("\n========================================")
    log(f"WORKER PROCESSING JOB: {job_id}")
    log(f"Original file: {job['original_filename']}")
    log(f"Stored file: {stored_path}")
    log("Status: PROCESSING")
    log("========================================")

    try:
        if not stored_path.is_file():
            raise FileNotFoundError(
                f"Uploaded file not found: {stored_path}"
            )

        file_type = detect_file_type(stored_path)

        if file_type is None:
            raise ValueError("Unsupported file extension.")

        # Perform OCR using the already-loaded model.
        output = process_one_file(
            ocr,
            stored_path,
            file_type,
            model_load_time
        )

        # Add job information to the output.
        output["job_id"] = job_id
        output["original_filename"] = job["original_filename"]

        # Save the JSON result using the job ID.
        result_path = RESULT_DIR / f"{job_id}_ocr_results.json"

        json_started = time.perf_counter()

        serialized_result = json.dumps(
            output,
            ensure_ascii=False,
            indent=2
        )

        result_path.write_text(
            serialized_result,
            encoding="utf-8"
        )

        json_time = time.perf_counter() - json_started

        log(f"JSON result saved: {result_path}")
        log(f"JSON serialization and write time: {json_time:.3f}s")

        # Store the result in MariaDB as well.
        update_job(
            job_id=job_id,
            status="completed",
            result_json=serialized_result,
            error_text=None
        )

        total_time = time.perf_counter() - started

        log("========== JOB COMPLETED ==========")
        log(f"Job ID: {job_id}")
        log(f"Result file: {result_path}")
        log(f"Total worker processing time: {total_time:.3f}s")
        log("Status: COMPLETED")
        log("===================================")

        # The uploaded input is intentionally retained.
        # Remove it manually after confirming the result if desired.

    except Exception as exc:
        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

        log(f"JOB FAILED: {error_message}")

        try:
            update_job(
                job_id=job_id,
                status="failed",
                result_json=None,
                error_text=error_message
            )

        except Exception as db_exc:
            log(
                f"Could not update failed job in database: {db_exc}"
            )

        log(f"Total time before failure: {time.perf_counter() - started:.3f}s")


# ============================================================
# CONTINUOUS WORKER LOOP
# ============================================================

def worker_loop():
    print("\n" + "=" * 60)
    print("MULTILINGUAL SOCIAL MEDIA OCR - WORKER")
    print("=" * 60)

    log("Initializing worker process...")

    initialize_database()

    # Select device and load model only once per worker process.
    device = choose_ocr_device()

    ocr, model_load_time = load_ocr_model(device)

    log("Worker is ready.")
    log(f"Polling interval: {WORKER_POLL_SECONDS:.1f}s")
    log("Waiting for queued jobs...")
    log("Press CTRL+C to stop the worker.")

    jobs_processed = 0

    try:
        while True:
            poll_started = time.perf_counter()

            try:
                job = claim_next_job()

                if job is None:
                    log("No queued jobs. Waiting...")
                    time.sleep(WORKER_POLL_SECONDS)
                    continue

                process_queued_job(
                    ocr,
                    model_load_time,
                    job
                )

                jobs_processed += 1

                log(f"Jobs handled in this session: {jobs_processed}")

            except mariadb.Error as exc:
                log(f"MariaDB worker error: {exc}")
                log(
                    f"Retrying after {WORKER_POLL_SECONDS:.1f}s..."
                )
                time.sleep(WORKER_POLL_SECONDS)

            except Exception as exc:
                log(
                    f"Unexpected worker error: "
                    f"{type(exc).__name__}: {exc}"
                )
                time.sleep(WORKER_POLL_SECONDS)

            poll_time = time.perf_counter() - poll_started

            log(f"Worker loop cycle time: {poll_time:.3f}s")

    except KeyboardInterrupt:
        log("Worker shutdown requested.")

    finally:
        log(f"Total jobs handled in this session: {jobs_processed}")
        log("Worker stopped.")


# ============================================================
# TERMINAL 1 - JOB SUBMISSION MODE
# ============================================================

def submitter_loop():
    print("\n" + "=" * 60)
    print("MULTILINGUAL SOCIAL MEDIA OCR")
    print("JOB SUBMISSION MODE")
    print("=" * 60)

    initialize_database()

    log("Submitter is ready.")
    log("Enter an image/video path to add a job to MariaDB.")
    log("Enter 'status' to check a job.")
    log("Enter 'exit' to close the submitter.")

    while True:
        try:
            raw_input_value = input(
                "\nEnter image/video path, status, or exit: "
            )

        except (EOFError, KeyboardInterrupt):
            print("\nClosing submitter.")
            break

        value = raw_input_value.strip().strip('"').strip("'")

        if not value:
            continue

        if value.lower() in ("exit", "quit", "q"):
            log("Submitter closed.")
            break

        if value.lower() == "status":
            job_id = input("Enter job ID: ").strip()

            if job_id:
                try:
                    show_job_status(job_id)
                except Exception as exc:
                    log(f"Status lookup failed: {exc}")

            continue

        try:
            submit_job(value)

        except mariadb.Error as exc:
            log(f"Database error while submitting job: {exc}")

        except Exception as exc:
            log(
                f"Submission failed: "
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Social Media OCR"
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help="Start the separate OCR worker process"
    )

    args = parser.parse_args()

    if args.worker:
        worker_loop()
    else:
        submitter_loop()


if __name__ == "__main__":
    main()

