.PHONY: setup setup-review test lint typecheck data-policy gpu-deps-check vps-plan accept

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

vps-plan:
	./scripts/vps/prepare_private_stack.sh --expected-commit $$(git rev-parse HEAD)

accept: data-policy lint typecheck test
	uv run cadence train-synthetic --config configs/test.yaml --checkpoint artifacts/checkpoints/acceptance.pt
	uv run cadence checkpoint-inspect artifacts/checkpoints/acceptance.pt
	uv run cadence remote-package --config configs/gpu-24gb.yaml --output artifacts/reports/remote-job.json --allow-dirty
	uv run cadence first-run-freeze --config configs/first-run-v0.1.0.yaml --dataset-snapshot-handle cadence-acceptance-snapshot --output artifacts/reports/first-run-v0.1.0.json --allow-dirty
	uv run cadence first-run-validate artifacts/reports/first-run-v0.1.0.json --config configs/first-run-v0.1.0.yaml --dataset-snapshot-handle cadence-acceptance-snapshot --allow-dirty
