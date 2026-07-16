.PHONY: setup test lint typecheck accept

setup:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy cadence

accept: lint typecheck test
	uv run cadence train-synthetic --config configs/test.yaml --checkpoint artifacts/checkpoints/acceptance.pt
	uv run cadence checkpoint-inspect artifacts/checkpoints/acceptance.pt
	uv run cadence remote-package --config configs/gpu-24gb.yaml --output artifacts/reports/remote-job.json
