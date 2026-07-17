.PHONY: setup setup-review test lint typecheck data-policy gpu-deps-check accept

setup:
	uv sync

setup-review:
	uv sync --group operations-ui

test:
	uv run --group operations-ui pytest

lint:
	uv run --group operations-ui ruff check .

typecheck:
	uv run --group operations-ui mypy cadence

data-policy:
	uv run cadence data-policy check

gpu-deps-check:
	uv lock --check
	python3 scripts/verify_gpu_dependencies.py

accept: data-policy lint typecheck test
	uv run cadence train-synthetic --config configs/test.yaml --checkpoint artifacts/checkpoints/acceptance.pt
	uv run cadence checkpoint-inspect artifacts/checkpoints/acceptance.pt
	uv run cadence remote-package --config configs/gpu-24gb.yaml --output artifacts/reports/remote-job.json --allow-dirty
