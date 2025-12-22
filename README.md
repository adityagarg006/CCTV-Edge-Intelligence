# Enhanced CCTV Person Detection & Tracking System

A high-performance surveillance analytics solution that integrates **YOLOv8** for real-time detection, **DeepSORT** for multi-object tracking, and **ResNet50** for advanced person re-identification. This system is engineered for complexity, featuring automated entry/exit logging and comprehensive database-driven analytics.

---

## Key Features

* **Intelligent Person Detection**: Utilizes **YOLOv8** for state-of-the-art, real-time person detection.
* **Comprehensive Tracking**: Employs **DeepSORT** with a **MobileNet** embedder to maintain identity across frames.
* **Advanced Re-identification (Re-ID)**: Uses a **ResNet50** backbone to extract 256-dimensional feature vectors, ensuring matching accuracy through L2 normalization and a weighted similarity score (70% average + 30% maximum).
* **Smart Event Tracking**: Automatically detects entry/exit events with a 3-second occlusion threshold and computes visit durations[cite: 87, 90, 92].
* **Performance Engineering**: Implements batch processing (8-frame batches), multi-threading (4 workers), and dynamic memory management with CUDA optimization.
* **Forensic Database**: Powered by **SQLite** with indexed queries for real-time data insertion and frame-level history storage.

---

## System Pipeline



1.  **Video Input**: The system accepts video files for processing (e.g., `.mp4`).
2.  **Detection**: YOLOv8 identifies persons in extracted frames with a default confidence threshold of 0.5.
3.  **Tracking**: DeepSORT assigns unique IDs and maintains trajectories[cite: 130].
4.  **Re-ID**: Feature extraction and similarity matching (0.85 threshold) are used for robust identity recovery and registration.
5.  **Analytics**: Entry/exit events are logged automatically based on appearance and disappearance logic.
6.  **Storage & Reports**: Data is stored in an indexed SQLite database, and dual CSV reports (Summary + Detailed) are generated.

---

## Performance Metrics

Tested on a **Tesla T4 GPU** using the `palace.mp4` test video (1280x720, 30 FPS, 329 frames):
* **Processing Speed**: 3.32 FPS achieved during technical validation.
* **Accuracy**: 15 unique individuals successfully identified and registered.
* **Volume**: Processed 4,459 total person detection records.
* **Re-ID Consistency**: High accuracy with similarity scores ranging from 0.858 to 0.904.

---

## Hardware & Software Requirements

### Hardware Requirements
* **GPU**: CUDA-compatible (NVIDIA Tesla T4 tested and optimized).
* **RAM**: 8GB minimum (16GB recommended for large videos).
* **CPU**: Multi-core processor (4+ cores optimal).
* **Storage**: SSD recommended for high-speed database operations.

---

## Execution Flow

To run the core pipeline via terminal or notebook:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-repo/detectper.git](https://github.com/your-repo/detectper.git)
    cd detectper
    ```
2.  **Setup Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
3.  **Process Video**:
    Update the `VIDEO_PATH` and `OUTPUT_PATH` in the main script and execute:
    ```bash
    python detectper_main.py
    ```
    The system will initialize the YOLO model, process the video, and save the results to `tracking_report.json` and the `person_tracking.db` SQLite database.

---
