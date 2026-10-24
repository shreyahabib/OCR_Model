# Multilingual Social Media OCR (English, Hindi, Marathi)

An Optical Character Recognition (OCR) pipeline and CLI tool designed to extract multilingual text (English, Hindi, and Marathi / Devanagari script) from social media images and videos using PaddleOCR (`PP-OCRv5`).

## Features

- **Multilingual Support**: High accuracy text recognition for Hindi, Marathi, and English.
- **Image OCR**: Extracts text, confidence scores, and bounding boxes from social media posts and images (`.jpg`, `.png`, `.webp`, `.tiff`, etc.).
- **Video OCR**: Samples video frames at configurable intervals (e.g., 1 frame per second), deduplicates repeated text across consecutive frames, and associates timestamps with recognized text.
- **Unified CLI Application**: An interactive prompt (`ocr_app.py`) capable of processing both image and video files, generating structured JSON output.
- **GPU & CPU Compatible**: Automatically supports GPU acceleration (NVIDIA CUDA) with safe CPU fallback.

---

## Project Structure

```text
social-media-ocr/
│
├── ocr_app.py              # Main interactive CLI application (supports images & videos)
├── video_ocr.py            # Video frame-by-frame OCR processing script
├── test_ocr.py             # Quick test script for single image OCR
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules (virtual environment, outputs, logs)
└── README.md               # Project documentation
```

---

## Installation

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd social-media-ocr
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for GPU Acceleration (NVIDIA CUDA):**  
> If using an NVIDIA GPU on Windows, install `paddlepaddle-gpu` matching your CUDA version:
> ```bash
> pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
> ```

---

## Usage

### 1. Interactive CLI Tool (`ocr_app.py`)
Run the unified OCR application:
```bash
python ocr_app.py
```
When prompted, provide the path to your image or video:
```text
Enter the full path of an image or video: Hindi.jpg
```
The application will process the input, display detected text and confidence scores in the terminal, and save the structured results to `<filename>_ocr_results.json`.

### 2. Video OCR (`video_ocr.py`)
Extracts text frame-by-frame from video files:
```bash
python video_ocr.py
```
Results are saved to `video_ocr_results.json` with frame timestamps:
```json
{
  "video": "ocr_test_video.mp4",
  "frame_interval_seconds": 1.0,
  "extracted_text": [
    {
      "timestamp_seconds": 0.0,
      "text": "नमस्ते",
      "confidence": 0.883
    }
  ]
}
```

### 3. Image OCR Test (`test_ocr.py`)
Quick test script for validating OCR on a single image:
```bash
python test_ocr.py
```

---

## License
MIT License
