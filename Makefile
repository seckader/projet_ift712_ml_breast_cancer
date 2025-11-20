.PHONY: env prepare train evaluate lint test

env:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

prepare:
	python scripts/prepare_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

lint:
	ruff check . || true

test:
	pytest -q || true
