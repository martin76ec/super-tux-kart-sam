PYTHON ?= .venv/bin/python

.PHONY: install explore train eval live clean

install:
	uv pip install torch torchvision transformers matplotlib pillow opencv-python

explore:
	$(PYTHON) -m src.cmd.explore

train:
	$(PYTHON) -m src.cmd.train

eval:
	$(PYTHON) -m src.cmd.eval

live:
	$(PYTHON) -m src.cmd.live

clean:
	rm -rf checkpoint.pt outputs __pycache__ src/**/__pycache__
