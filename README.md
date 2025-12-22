Enhanced CCTV Person Detection & Tracking System
A production-ready, high-performance surveillance solution that integrates YOLOv8 for real-time detection, DeepSORT for multi-object tracking, and ResNet50 for advanced person re-identification. This system is designed to handle complex environments with features like entry/exit logging and automated CSV analytics.

Key Features
Intelligent Person Detection: Utilizes YOLOv8 for state-of-the-art, real-time person detection.

Comprehensive Tracking: Employs DeepSORT with a MobileNet embedder to maintain identity across frames.

Advanced Re-identification (Re-ID): Uses a ResNet50 backbone to extract 256-dimensional feature vectors. It achieves matching accuracy through L2 normalization and a weighted similarity score (70% average + 30% maximum).

Smart Event Tracking: Automatically detects entry/exit events with a 3-second occlusion threshold and computes visit durations.

User-Friendly Interface: A Django-based web frontend featuring drag-and-drop video uploads and a real-time processing dashboard.

Production Optimized: GPU acceleration with CUDA optimization and an indexed SQLite database for efficient forensic history storage.

System Pipeline
Video Upload: User uploads video via the Django interface.

Detection: YOLOv8 identifies persons in extracted frames.

Tracking: DeepSORT assigns unique IDs and maintains trajectories.

Re-ID: Feature extraction and similarity matching (0.85 threshold) for identity recovery.

Analytics: Entry/exit events are logged with timestamps.

Storage & Reports: Data is stored in SQLite; Summary and Detailed CSV reports are generated.

Performance Metrics
Tested on a Tesla T4 GPU using palace.mp4 (1280x720, 30 FPS, 329 frames):

Processing Speed: 3.32 FPS achieved during testing.

Accuracy: 15 unique individuals successfully identified and tracked.

Volume: Processed 4,459 total person detections.

Re-ID Range: Similarity scores typically fall between 0.858 and 0.904.

Hardware & Software Requirements
Hardware
GPU: CUDA-compatible (NVIDIA Tesla T4 optimized).

RAM: 8GB minimum (16GB recommended for large videos).

CPU: Multi-core processor (4+ cores optimal).

Storage: SSD recommended for database operations.

Software Stack
Language: Python 3.8+.

Frameworks: PyTorch, Django.

Libraries: YOLOv8 (Ultralytics), DeepSORT, OpenCV 4.x.

Installation & Execution
Clone the Repository:

Bash

git clone https://github.com/your-repo/detectper.git
cd detectper
Setup Environment:

Bash

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
Run the Django Server:

Bash

python manage.py runserver
Upload & Process: Access the dashboard at http://127.0.0.1:8000, upload your video, and monitor real-time FPS and progress updates.
