import os
import sys
from pathlib import Path

# Configure stdout and stderr to handle UTF-8 (Devanagari, Marathi, Hindi text) on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Disable oneDNN / MKLDNN on CPU to avoid PIR attribute conversion errors in PaddlePaddle 3.x
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

image_path = Path("Hindi.jpg")

if not image_path.exists():
    raise FileNotFoundError(f"Image file not found: {image_path.resolve()}")

ocr = PaddleOCR(
    lang="hi",
    ocr_version="PP-OCRv5",
    device="gpu",
)

print(f"Running OCR on: {image_path.name}...")
results = ocr.predict(str(image_path))

output_lines = []
for result in results:
    if isinstance(result, dict) and "rec_texts" in result:
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        output_lines.append("\n--- OCR Results ---")
        for text, score in zip(texts, scores):
            output_lines.append(f"Text: {text} | Confidence: {score:.2%}")
    else:
        output_lines.append(str(result))

full_output = "\n".join(output_lines)
print(full_output)

# Save result to a UTF-8 text file as well
output_file = Path("output.txt")
output_file.write_text(full_output, encoding="utf-8")
print(f"\n[Saved full results to {output_file.resolve()}]")


