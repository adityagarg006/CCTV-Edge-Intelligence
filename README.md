# Enhanced CCTV Person Detection & Tracking System

[cite_start]A high-performance surveillance analytics solution that integrates **YOLOv8** for real-time detection, **DeepSORT** for multi-object tracking, and **ResNet50** for advanced person re-identification[cite: 1, 2, 8, 9, 10]. [cite_start]This system is engineered for complexity, featuring automated entry/exit logging and comprehensive database-driven analytics[cite: 11, 86].

---

## Key Features

* [cite_start]**Intelligent Person Detection**: Utilizes **YOLOv8** for state-of-the-art, real-time person detection[cite: 8, 32].
* [cite_start]**Comprehensive Tracking**: Employs **DeepSORT** with a **MobileNet** embedder to maintain identity across frames[cite: 10, 33].
* [cite_start]**Advanced Re-identification (Re-ID)**: Uses a **ResNet50** backbone to extract 256-dimensional feature vectors, ensuring matching accuracy through L2 normalization and a weighted similarity score (70% average + 30% maximum)[cite: 34, 38, 39].
* [cite_start]**Smart Event Tracking**: Automatically detects entry/exit events with a 3-second occlusion threshold and computes visit durations[cite: 87, 90, 92].
* **Performance Engineering**: Implements batch processing (8-frame batches), multi-threading (4 workers), and dynamic memory management with CUDA optimization.
* [cite_start]**Forensic Database**: Powered by **SQLite** with indexed queries for real-time data insertion and frame-level history storage[cite: 60, 65, 67, 68].

---

## System Pipeline



1.  [cite_start]**Video Input**: The system accepts video files for processing (e.g., `.mp4`)[cite: 56].
2.  **Detection**: YOLOv8 identifies persons in extracted frames with a default confidence threshold of 0.5.
3.  [cite_start]**Tracking**: DeepSORT assigns unique IDs and maintains trajectories[cite: 130].
4.  [cite_start]**Re-ID**: Feature extraction and similarity matching (0.85 threshold) are used for robust identity recovery and registration[cite: 35, 131].
5.  [cite_start]**Analytics**: Entry/exit events are logged automatically based on appearance and disappearance logic[cite: 86, 133].
6.  [cite_start]**Storage & Reports**: Data is stored in an indexed SQLite database, and dual CSV reports (Summary + Detailed) are generated[cite: 60, 81, 134].

---

## Performance Metrics

[cite_start]Tested on a **Tesla T4 GPU** using the `palace.mp4` test video (1280x720, 30 FPS, 329 frames)[cite: 55, 56]:
* [cite_start]**Processing Speed**: 3.32 FPS achieved during technical validation[cite: 56, 112].
* [cite_start]**Accuracy**: 15 unique individuals successfully identified and registered[cite: 57, 117].
* [cite_start]**Volume**: Processed 4,459 total person detection records[cite: 57, 123].
* [cite_start]**Re-ID Consistency**: High accuracy with similarity scores ranging from 0.858 to 0.904[cite: 58, 118].

---

## Hardware & Software Requirements

### Hardware Requirements
* [cite_start]**GPU**: CUDA-compatible (NVIDIA Tesla T4 tested and optimized)[cite: 98, 116].
* [cite_start]**RAM**: 8GB minimum (16GB recommended for large videos)[cite: 100].
* [cite_start]**CPU**: Multi-core processor (4+ cores optimal)[cite: 102].
* [cite_start]**Storage**: SSD recommended for high-speed database operations[cite: 103].

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
