# Changelog

All notable changes will be documented here following [Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Added
- Initial release of CCTV-Edge-Intelligence pipeline
- YOLOv8n + ByteTrack person detection and tracking
- OSNet x0.25 Re-ID with MobileNetV3 fallback
- FAISS IndexFlatL2 + IndexIDMap feature bank with EMA updates
- SQLite WAL persistence with batched writes
- Async frame producer with RTSP reconnect logic
- Metrics collector and analytics reporter
- pytest test suite for all modules
- GitHub Actions CI (Python 3.10–3.12)
