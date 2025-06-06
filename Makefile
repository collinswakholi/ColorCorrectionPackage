.PHONY: install test lint build clean

install:
	pip install -e .

test:
	pytest --maxfail=1 --disable-warnings -q

lint:
	flake8 ColorCorrectionPipeline

build:
	python -m build

clean:
	rm -rf build dist *.egg-info
