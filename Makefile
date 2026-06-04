SOURCE ?= 0

.PHONY: install run run-file run-rtsp lint test clean \
        bench bench-gpu bench-component bench-fast

install:
	pip install -r requirements.txt

run:
	python main.py --source $(SOURCE)

run-file:
	python main.py --source sample.mp4 --output-dir reports

run-rtsp:
	python main.py --source $${VIDEO_SOURCE} --device cuda --output-dir reports

lint:
	ruff check src/ main.py tests/

test:
	pytest tests/ --cov=src --cov-report=term-missing -v

clean:
	rm -f *.db *.db-wal *.db-shm
	rm -rf reports/ logs/ __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete

# ---------------------------------------------------------------------------
# Benchmark targets
# ---------------------------------------------------------------------------

bench:
	python benchmarks/run.py --device cpu --output-dir reports

bench-gpu:
	python benchmarks/run.py --device cuda --output-dir reports

bench-component:
	python benchmarks/run.py --device cpu --component $(COMPONENT) --output-dir reports

bench-fast:
	python benchmarks/run.py --device cpu --skip-video-gen --component feature_bank --output-dir reports
	python benchmarks/run.py --device cpu --skip-video-gen --component database --output-dir reports
