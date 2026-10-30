# Multilingual Social Media OCR: System Setup & Deployment Guide

This document provides a comprehensive operational guide for the **Multilingual Social Media OCR system** (supporting English, Hindi, and Marathi via PaddleOCR `PP-OCRv5`). It details project architecture, Windows installation, MariaDB configuration, CPU/GPU hardware tuning, multi-worker scaling, cPanel deployment analysis, troubleshooting, and a pre-flight verification checklist.

---

## 1. Project Files & Directory Structure

The repository is organized around a self-contained, decoupled producer-consumer architecture implemented in [`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py):

```text
social-media-ocr/
│
├── ocr_app.py              # Primary application engine (Submitter CLI, MariaDB queue, Worker loop, OCR engine)
├── requirements.txt        # Python package dependencies
├── setup.md                # System setup, MariaDB configuration, and deployment guide
├── README.md               # High-level project summary and quick-start reference
├── DATA_FLOW.md            # Detailed end-to-end data flow documentation and Mermaid diagrams
├── time_optimization.md    # Performance profiling, latency benchmarks, and optimization guide
├── .gitignore              # Git ignore rules for virtual environments, outputs, and uploads
│
├── test_ocr.py             # Standalone test script for single image inference
├── video_ocr.py            # Standalone reference script for sequential video frame processing
│
├── uploads/                # Temporary file staging directory for submitted media
├── results/                # Output directory for serialized <job_id>_ocr_results.json files
│
├── Hindi.jpg               # Sample Hindi image with Devanagari script for validation
├── marathi.jpg             # Sample Marathi image for validation
├── sample.png              # Benchmark sample image
├── sample1.png             # High-resolution benchmark sample image
├── ocr_test_video.mp4      # Sample video clip for video OCR testing
│
├── Hindi_ocr_results.json  # Baseline OCR output JSON for Hindi.jpg
├── video_ocr_results.json  # Baseline OCR output JSON for ocr_test_video.mp4
└── output.txt              # Sample OCR text extraction output
```

### File Purposes

| File / Folder | Role & Description |
| :--- | :--- |
| [`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) | **Primary Application Core**. Runs as either the Job Submitter (`python ocr_app.py`) or the Background OCR Worker (`python ocr_app.py --worker`). Manages MariaDB transactions, model caching, timestamp seeking, downsampled frame comparison, batch prediction, and JSON serialization. |
| [`requirements.txt`](file:///c:/Users/HP/social-media-ocr/requirements.txt) | **Dependency Manifest**. Lists core runtime libraries (`paddleocr`, `opencv-python`, `mariadb`, `pillow`) with clear separation between CPU and GPU (`paddlepaddle` vs. `paddlepaddle-gpu`) requirements. |
| [`setup.md`](file:///c:/Users/HP/social-media-ocr/setup.md) | **Setup & Deployment Manual**. Complete setup instructions for Windows and server environments, database initialization, multi-worker scaling, and troubleshooting. |
| [`DATA_FLOW.md`](file:///c:/Users/HP/social-media-ocr/DATA_FLOW.md) | **Architecture & Flow Documentation**. Contains ASCII and Mermaid flowcharts, state diagrams, database locking mechanics, and step-by-step pipeline explanations. |
| [`time_optimization.md`](file:///c:/Users/HP/social-media-ocr/time_optimization.md) | **Optimization Analysis**. Documents the 85%–92% latency reduction achieved by bypassing `UVDoc` book unwarping, seeking keyframe timestamps, and thumbnail-based frame skipping. |
| `uploads/` | **Staging Storage**. Stores uniquely renamed media files (`<job_id><ext>`) copied by the submitter before worker ingestion. |
| `results/` | **Output Storage**. Persists final structured `<job_id>_ocr_results.json` files containing recognized text, confidence scores, timestamps, and execution metrics. |

---

## 2. Complete Windows Setup & Installation

### 2.1. System Prerequisites

Before starting, ensure your Windows machine meets these requirements:

1. **Operating System**: Windows 10 or Windows 11 (64-bit).
2. **Python**: Python 3.10, 3.11, or 3.12 (64-bit).
   - Ensure **"Add Python to PATH"** was checked during installation.
   - Verify via PowerShell:
     ```powershell
     python --version
     ```
3. **Microsoft Visual C++ Redistributable (x64)**:
   - Required by PaddlePaddle, OpenCV, and the MariaDB Connector.
   - Download and install the latest `vc_redist.x64.exe` from [Microsoft's official site](https://aka.ms/vs/17/release/vc_redist.x64.exe).
4. **Git for Windows**:
   - Install from [git-scm.com](https://git-scm.com/).
5. **MariaDB Server (v10.6+)**:
   - See [Section 3](#3-mariadb-configuration--database-setup) for setup instructions.

---

### 2.2. Clone Repository & Setup Virtual Environment

1. Open PowerShell and navigate to your preferred workspace:
   ```powershell
   cd C:\Users\<YourUsername>
   git clone <YOUR_REPOSITORY_URL> social-media-ocr
   cd social-media-ocr
   ```

2. Create an isolated Python virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Configure execution policy (if PowerShell scripts are restricted) and activate:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\.venv\Scripts\Activate.ps1
   ```
   *(Your prompt will now show `(.venv)`)*.

4. Upgrade `pip`, `setuptools`, and `wheel`:
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```

---

### 2.3. Install Dependencies: CPU vs. GPU

#### Option A: CPU-Only Installation
If your system does not have an NVIDIA GPU:
```powershell
pip install -r requirements.txt
```
This automatically installs:
- `paddleocr>=3.7.0`
- `opencv-python>=4.10.0`
- `pillow>=10.0.0`
- `mariadb>=1.1.10`
- `paddlepaddle>=3.3.0` (CPU build)

#### Option B: NVIDIA GPU Acceleration (CUDA)
If your system has an NVIDIA GeForce / RTX / Quadro GPU:

1. **Verify NVIDIA Driver**:
   ```powershell
   nvidia-smi
   ```
   Check the CUDA Version supported by your driver (e.g., CUDA 12.x).

2. **Install Core Requirements (excluding CPU paddlepaddle)**:
   ```powershell
   pip install paddleocr opencv-python pillow mariadb
   ```

3. **Install `paddlepaddle-gpu` matching your CUDA version**:
   For CUDA 12.x on Windows:
   ```powershell
   pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
   ```
   For CUDA 11.8 on Windows:
   ```powershell
   pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   ```

4. **Verify GPU Recognition**:
   ```powershell
   python -c "import paddle; print('Paddle Version:', paddle.__version__, '| CUDA Compiled:', paddle.is_compiled_with_cuda(), '| GPU Count:', paddle.device.cuda.device_count())"
   ```
   Expected output:
   ```text
   Paddle Version: 3.3.1 | CUDA Compiled: True | GPU Count: 1
   ```

---

## 3. MariaDB Configuration & Database Setup

[`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) uses MariaDB to maintain an atomic, persistent job queue with row-level locking (`SELECT ... FOR UPDATE`).

### 3.1. Install MariaDB on Windows

1. Download the MSI installer for **MariaDB Community Server** (v10.6, 10.11 LTS, or 11.x) from [mariadb.org](https://mariadb.org/download/).
2. Run the installer:
   - Set a secure root password during installation.
   - Note the port number (default is `3306`; if MySQL is already installed, MariaDB typically uses port `3307`).
   - Leave "Install as service" checked so MariaDB starts automatically on boot.

---

### 3.2. Database & User Provisioning

Open the MariaDB Command Line Client or run `mariadb -u root -p` in PowerShell:

```sql
-- 1. Create the dedicated database with full UTF-8 Unicode support
CREATE DATABASE IF NOT EXISTS ocr_database
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 2. Create a dedicated application user (replace placeholder credentials)
CREATE USER IF NOT EXISTS 'ocr_user'@'localhost' IDENTIFIED BY 'YOUR_SECURE_PASSWORD';

-- 3. Grant required privileges on the OCR database
GRANT ALL PRIVILEGES ON ocr_database.* TO 'ocr_user'@'localhost';

-- 4. Apply privilege changes
FLUSH PRIVILEGES;
```

---

### 3.3. Environment Configuration

[`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) reads database credentials from environment variables with safe defaults. To avoid hardcoding passwords, configure your shell or Windows environment:

#### Temporary Session Variables (PowerShell):
```powershell
$env:OCR_DB_HOST="127.0.0.1"
$env:OCR_DB_PORT="3307"          # Use 3306 or 3307 based on your MariaDB setup
$env:OCR_DB_USER="ocr_user"
$env:OCR_DB_PASSWORD="YOUR_SECURE_PASSWORD"
$env:OCR_DB_NAME="ocr_database"
```

#### Persistent System Environment Variables (Windows):
You can set persistent system variables using PowerShell (Run as Administrator):
```powershell
[System.Environment]::SetEnvironmentVariable('OCR_DB_HOST', '127.0.0.1', 'User')
[System.Environment]::SetEnvironmentVariable('OCR_DB_PORT', '3307', 'User')
[System.Environment]::SetEnvironmentVariable('OCR_DB_USER', 'ocr_user', 'User')
[System.Environment]::SetEnvironmentVariable('OCR_DB_PASSWORD', 'YOUR_SECURE_PASSWORD', 'User')
[System.Environment]::SetEnvironmentVariable('OCR_DB_NAME', 'ocr_database', 'User')
```

---

### 3.4. Automatic Schema Generation

You **do not** need to manually create tables. When [`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) launches (in either Submitter or Worker mode), [`initialize_database()`](file:///c:/Users/HP/social-media-ocr/ocr_app.py#L136-L190) executes the following idempotent DDL:

```sql
CREATE TABLE IF NOT EXISTS ocr_jobs (
    id CHAR(36) PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL,
    status ENUM('queued', 'processing', 'completed', 'failed') NOT NULL DEFAULT 'queued',
    result_json LONGTEXT NULL,
    error_text TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status_created (status, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### Schema Fields Explained:
- `id`: UUID v4 primary key identifying the job.
- `original_filename`: Name of the uploaded media file as submitted.
- `stored_filename`: Sanitized unique filename (`<job_id><ext>`) stored in `uploads/`.
- `status`: Lifecycle state (`queued` $\to$ `processing` $\to$ `completed` or `failed`).
- `result_json`: Full structured output JSON (extracted text, timestamps, confidences, duration).
- `error_text`: Full error description and exception traceback if a job fails.
- `idx_status_created`: Composite B-Tree index ensuring $O(1)$ lookups for `WHERE status = 'queued' ORDER BY created_at ASC`.

---

## 4. CPU/GPU Hardware Configuration & Monitoring

### 4.1. Hardware Device Selection

When the worker process starts, [`choose_ocr_device()`](file:///c:/Users/HP/social-media-ocr/ocr_app.py#L87-L124) prompts:
```text
========== SELECT OCR DEVICE ==========
1. CPU
2. GPU
=======================================
Choose CPU or GPU:
```

- **Selection 1 (CPU)**:
  - Configures `device="cpu"`.
  - Sets `cpu_threads=4` to leverage multi-core CPU execution.
  - Automatically disables MKLDNN (`PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT="False"` and `enable_mkldnn=False`) to avoid oneDNN PIR attribute conversion bugs.
- **Selection 2 (GPU)**:
  - Verifies CUDA availability.
  - Sets `device="gpu:0"`.
  - Leverages Tensor Cores and FP32/FP16 acceleration.

---

### 4.2. Memory & Performance Footprint

| Hardware | Target Media | Model Startup Latency | Processing Latency | Peak VRAM / RAM |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX GPU (CUDA)** | 1080p Image | ~0.55 seconds | **0.85s – 1.20s** | ~2.1 GB VRAM |
| **NVIDIA RTX GPU (CUDA)** | 5s Video (150 frames) | ~0.55 seconds | **2.80s – 3.80s** | ~2.6 GB VRAM |
| **Modern Multi-Core CPU** | 1080p Image | ~0.45 seconds | **4.50s – 7.20s** | ~1.4 GB RAM |
| **Modern Multi-Core CPU** | 5s Video (150 frames) | ~0.45 seconds | **8.50s – 12.0s** | ~1.8 GB RAM |

---

### 4.3. Real-Time Performance Monitoring

#### Monitoring GPU Utilization:
Run in an open PowerShell terminal:
```powershell
nvidia-smi -l 1
```
This prints GPU temperature, utilization percentage, and VRAM consumption every second.

#### Monitoring Database Queue State:
Open a MySQL/MariaDB client and query queue depth:
```sql
SELECT status, COUNT(*) AS count
FROM ocr_database.ocr_jobs
GROUP BY status;
```

#### Application Telemetry:
[`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) outputs real-time timestamps with sub-millisecond precision:
```text
[2026-10-30 15:10:02] Starting OCR job for: sample_video.mp4
[2026-10-30 15:10:02] Video FPS: 30.00
[2026-10-30 15:10:02] Video duration: 7.00s
[2026-10-30 15:10:02] Queueing frame at 0.00s | reason: first frame
[2026-10-30 15:10:03] OCR batch inference: 2 frames | 0.812s
[2026-10-30 15:10:05] ========== VIDEO PROCESSING SUMMARY ==========
[2026-10-30 15:10:05] Sampled frames: 8
[2026-10-30 15:10:05] OCR processed frames: 3
[2026-10-30 15:10:05] Skipped frames: 5
[2026-10-30 15:10:05] Batch calls: 1
[2026-10-30 15:10:05] Total processing time: 3.124s
[2026-10-30 15:10:05] ===============================================
```

---

## 5. Scaling Architecture: 1 Submitter + 2 OCR Workers

[`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py) is architecturally designed to scale horizontally across multiple worker processes without job collisions or race conditions.

```
                      ┌──────────────────────────────────────────────┐
                      │              MariaDB Queue                   │
                      │         (ocr_jobs in ocr_database)           │
                      └──────────────┬────────────────┬──────────────┘
                                     │                │
                        Atomic Claim │                │ Atomic Claim
             (SELECT ... FOR UPDATE) │                │ (SELECT ... FOR UPDATE)
                                     ▼                ▼
                       ┌───────────────────┐    ┌───────────────────┐
                       │   OCR Worker 1    │    │   OCR Worker 2    │
                       │   (GPU Worker)    │    │   (CPU Worker)    │
                       └───────────────────┘    └───────────────────┘
```

### Why Concurrent Multi-Worker Execution is Safe:
1. **Row-Level Transaction Locking**:
   In [`claim_next_job()`](file:///c:/Users/HP/social-media-ocr/ocr_app.py#L822-L897), the worker begins a transaction and executes:
   ```sql
   SELECT id, original_filename, stored_filename
   FROM ocr_jobs
   WHERE status = 'queued'
   ORDER BY created_at ASC
   LIMIT 1
   FOR UPDATE;
   ```
2. The `FOR UPDATE` clause places an exclusive row-level lock on the claimed record.
3. If Worker 1 claims Job A, MariaDB blocks or skips Job A for Worker 2.
4. Worker 1 immediately updates status to `processing` and commits the transaction, releasing the table lock while retaining the job.
5. Worker 2 instantly picks up Job B without idle wait time.

---

### 5.1. Execution Commands (Open 3 Terminals)

#### Terminal 1: Launch OCR Worker 1 (GPU Worker)
```powershell
cd C:\Users\<YourUsername>\social-media-ocr
.\.venv\Scripts\Activate.ps1
python ocr_app.py --worker
```
- When prompted, choose `2` (GPU).
- Worker 1 will initialize database tables, load PaddleOCR onto GPU 0, and enter its polling loop:
  `[TIMESTAMP] Waiting for queued jobs...`

#### Terminal 2: Launch OCR Worker 2 (CPU Worker or GPU Worker 2)
```powershell
cd C:\Users\<YourUsername>\social-media-ocr
.\.venv\Scripts\Activate.ps1
python ocr_app.py --worker
```
- When prompted, choose `1` (CPU) or select GPU.
- Worker 2 will independently load PaddleOCR into memory and begin polling MariaDB.

#### Terminal 3: Launch Job Submitter CLI
```powershell
cd C:\Users\<YourUsername>\social-media-ocr
.\.venv\Scripts\Activate.ps1
python ocr_app.py
```
- Enter an image or video path:
  ```text
  Enter image/video path, status, or exit: Hindi.jpg
  ```
  The Submitter copies the file to `uploads/` and inserts it into MariaDB. Worker 1 or Worker 2 will instantly claim and process it.
- Submit a second file immediately:
  ```text
  Enter image/video path, status, or exit: ocr_test_video.mp4
  ```
  The other worker will claim this job simultaneously, processing both workloads in parallel.
- Check live status anytime:
  ```text
  Enter image/video path, status, or exit: status
  Enter job ID: <job_id_here>
  ```

---

## 6. cPanel Deployment Analysis & Steps

Deploying a deep learning OCR pipeline on cPanel requires a clear understanding of hosting tiers, system constraints, and process lifecycle management.

### 6.1. Hosting Environments: Shared cPanel vs. CloudLinux vs. VPS

| Feature / Capability | Shared cPanel | CloudLinux cPanel | cPanel on VPS / Dedicated Server |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU Access (CUDA)** | ❌ **Impossible** (No PCIe passthrough) | ❌ **Impossible** | ✅ **Possible** (with GPU passthrough) |
| **Persistent Background Daemons** | ❌ **Strictly Terminated** (Killed after 30–60s) | ⚠️ **Throttled / Killed** (LVE cgroups limits) | ✅ **Full Support** (`systemd`, `supervisord`) |
| **RAM Quotas** | Low (Typically 512MB – 1GB) | Capped by LVE (1GB – 2GB) | Unlimited (Entire VPS RAM, e.g. 8GB+) |
| **CPU Execution Time Limits** | Max 30–120s per request | Throttled upon CPU spike | Unlimited |
| **MariaDB Connector/C Compiler** | Often lacks `gcc` / dev headers | Often lacks `gcc` / dev headers | ✅ Full root package manager (`dnf` / `apt`) |

---

### 6.2. Key Constraints on Shared cPanel

> [!CAUTION]
> **Shared cPanel accounts cannot run persistent OCR workers.**
> 1. **No GPU Acceleration**: Shared web hosting servers do not provide NVIDIA CUDA hardware or drivers. PaddleOCR must run strictly on CPU.
> 2. **Process Murderers / LVE Killers**: Shared hosting runs process sweepers that automatically kill any background process running a continuous loop (`while True`) or consuming high CPU for more than 60 seconds.
> 3. **Memory OOM**: Loading PaddleOCR PP-OCRv5 requires ~1.4 GB of system RAM. Shared hosting accounts with 512MB or 1GB limits will crash with `MemoryError` or `Killed`.

---

### 6.3. Recommended Server Architecture for cPanel Users

If you are using cPanel, the recommended production setup is:
- **cPanel / WHM running on a VPS or Dedicated Server** (CentOS Stream, AlmaLinux 9, or Ubuntu).
- **CPU Build**: Use `paddlepaddle` (CPU) with `cpu_threads=2` or `4`.
- **Process Manager**: Use `systemd` or `supervisord` to supervise the worker daemon instead of running manual terminal windows.

---

### 6.4. Production Deployment Steps (VPS / Dedicated Server with cPanel)

#### Step 1: Install System Dependencies via SSH (Root)
```bash
# On AlmaLinux / RHEL / CentOS:
dnf install -y python311 python311-devel gcc gcc-c++ make mariadb-devel mesa-libGL

# On Ubuntu / Debian:
apt-get update
apt-get install -y python3.11 python3.11-dev build-essential libmariadb-dev libgl1-mesa-glx
```

#### Step 2: Deploy Project Code
```bash
cd /opt
git clone <YOUR_REPOSITORY_URL> social-media-ocr
cd social-media-ocr

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Configure Environment Variables
Create an environment file `/opt/social-media-ocr/.env`:
```ini
OCR_DB_HOST=127.0.0.1
OCR_DB_PORT=3306
OCR_DB_USER=ocr_user
OCR_DB_PASSWORD=YOUR_STRONG_PASSWORD
OCR_DB_NAME=ocr_database
OCR_MAX_UPLOAD_BYTES=209715200
OCR_WORKER_POLL_SECONDS=2.0
```

#### Step 4: Configure `systemd` for Persistent Worker Management
Create `/etc/systemd/system/ocr-worker@.service` to manage multiple workers:
```ini
[Unit]
Description=Multilingual Social Media OCR Worker %I
After=network.target mariadb.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/social-media-ocr
EnvironmentFile=/opt/social-media-ocr/.env
ExecStart=/opt/social-media-ocr/.venv/bin/python ocr_app.py --worker
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Step 5: Enable & Start Two Worker Daemons
```bash
systemctl daemon-reload

# Start Worker 1
systemctl enable --now ocr-worker@1.service

# Start Worker 2
systemctl enable --now ocr-worker@2.service

# Check status
systemctl status ocr-worker@1.service
```

---

## 7. Common Errors & Troubleshooting

### 7.1. MariaDB Connection Refused (`Can't connect to server`)
- **Symptom**: `mariadb.OperationalError: Can't connect to server on '127.0.0.1' (10061)`.
- **Cause**: MariaDB service is stopped, or port mismatch (e.g. MariaDB installed on port `3307` while code defaults to `3306`).
- **Fix**:
  1. Open Windows Services (`services.msc`) and verify `MariaDB` is in `Running` state.
  2. Confirm the active port in MariaDB:
     ```sql
     SHOW VARIABLES LIKE 'port';
     ```
  3. Export the correct port environment variable:
     ```powershell
     $env:OCR_DB_PORT="3307"
     ```

---

### 7.2. GPU Selected but CUDA Not Detected
- **Symptom**: `GPU unavailable: PaddlePaddle has no CUDA support.`
- **Cause**: Standard CPU `paddlepaddle` was installed instead of `paddlepaddle-gpu`, or CUDA drivers are missing.
- **Fix**:
  1. Verify GPU driver: `nvidia-smi`.
  2. Uninstall CPU Paddle: `pip uninstall -y paddlepaddle paddlepaddle-gpu`.
  3. Reinstall with matching CUDA version:
     ```powershell
     pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
     ```

---

### 7.3. oneDNN / PIR Conversion Errors on CPU
- **Symptom**: `RuntimeError: (PreconditionNotMet) ... PIR attribute conversion error`.
- **Cause**: Intel oneDNN (MKLDNN) is enabled by default in PaddlePaddle 3.x and conflicts with certain Windows CPU instruction sets.
- **Fix**:
  - [`ocr_app.py`](file:///c:/Users/HP/social-media-ocr/ocr_app.py#L11) already enforces:
    ```python
    os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
    ```
    and sets `enable_mkldnn=False` during model loading.
  - If calling custom scripts, ensure this environment flag is set before importing `paddle` or `paddleocr`.

---

### 7.4. Video Decoding Failure (`Could not open video`)
- **Symptom**: `RuntimeError: Could not open video: <file_path>`.
- **Cause**: Corrupt container format or missing OpenCV video codecs.
- **Fix**:
  - Reinstall OpenCV with full codec support:
    ```powershell
    pip uninstall -y opencv-python opencv-python-headless
    pip install opencv-python>=4.10.0
    ```

---

### 7.5. File Size Exceeded (`File is too large`)
- **Symptom**: Submitter reports: `File is too large. Maximum allowed size: 200 MB`.
- **Fix**:
  - Increase the upload threshold via environment variable:
    ```powershell
    $env:OCR_MAX_UPLOAD_BYTES="524288000"  # 500 MB
    ```

---

## 8. Final Setup Verification Checklist

Use this checklist to confirm your environment is production-ready:

| Step | Verification Item | Command / Action | Expected Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Python Environment | `python --version` | Python 3.10.x, 3.11.x, or 3.12.x | 🔲 |
| **2** | Dependencies | `pip list` | `paddleocr`, `mariadb`, `opencv-python`, `paddlepaddle` installed | 🔲 |
| **3** | MariaDB Service | `services.msc` or `Get-Service MariaDB` | Status: `Running` | 🔲 |
| **4** | Database Schema | Check database in MariaDB CLI | `ocr_jobs` table exists with index `idx_status_created` | 🔲 |
| **5** | Storage Directories | Check project root | `uploads/` and `results/` folders exist | 🔲 |
| **6** | Worker 1 Execution | Run `python ocr_app.py --worker` in Terminal 1 | Worker loads model and logs: `Waiting for queued jobs...` | 🔲 |
| **7** | Worker 2 Execution | Run `python ocr_app.py --worker` in Terminal 2 | Worker loads model and logs: `Waiting for queued jobs...` | 🔲 |
| **8** | Submitter Execution | Run `python ocr_app.py` in Terminal 3 | Submitter loop prompt appears | 🔲 |
| **9** | End-to-End Test | Submit `Hindi.jpg` in Terminal 3 | One worker claims job, logs OCR, saves JSON to `results/` | 🔲 |
| **10**| Status Lookup | Type `status` in Terminal 3 and enter `job_id` | Displays `Status: completed` with timestamps | 🔲 |
