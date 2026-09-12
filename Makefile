PYTHON ?= .venv/bin/python

.PHONY: install explore train eval live yolo-train yolo-eval yolo-live clean

install:
	uv pip install torch torchvision transformers matplotlib pillow opencv-python ultralytics

explore:
	$(PYTHON) -m src.cmd.explore

train:
	$(PYTHON) -m src.cmd.train

eval:
	$(PYTHON) -m src.cmd.eval

live:
	$(PYTHON) -m src.cmd.live

yolo-train:
	$(PYTHON) -m src.cmd.train --model yolo

yolo-eval:
	$(PYTHON) -m src.cmd.eval --model yolo

yolo-live:
	$(PYTHON) -m src.cmd.live --model yolo

clean:
	rm -rf checkpoint*.pt outputs outputs_yolo __pycache__ src/**/__pycache__
