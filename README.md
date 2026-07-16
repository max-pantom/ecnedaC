# Cadence

Cadence is a research codebase for learning temporally aligned video and audio
representations before a later creative-reasoning model produces structured sound-design
timelines.

This repository currently implements the **local training-readiness milestone** only. It is
implemented and locally testable; it is not trained, remotely smoke-tested, evaluated on real
data, or production-ready.

## Local setup

The development machine is an Intel Mac. Install `uv`, then use the platform-pinned local stack:

```bash
uv sync
make accept
```

The default profile is CPU-only and refuses more than four samples, clips over two seconds,
more than one epoch, loader microbatches over one, more than one worker, and remote media URIs.

## Commands

```bash
uv run cadence config-check --config configs/local.yaml
uv run cadence fixture-generate --output-dir /tmp/cadence-fixtures
uv run cadence manifest-validate /path/to/manifest.jsonl
uv run cadence model-inspect --config configs/gpu-24gb.yaml
uv run cadence train-synthetic --config configs/test.yaml
uv run cadence train-contrastive --config configs/local.yaml
uv run cadence retrieval-eval --synthetic --config configs/test.yaml
uv run cadence checkpoint-inspect artifacts/checkpoints/latest.pt
uv run cadence remote-package --config configs/gpu-24gb.yaml
```

Remote scripts are dry-run-first and require both configuration/credentials and `--execute`.
No remote action is performed as part of local acceptance.

## Dataset intake pilot

Candidate URLs are persisted before any download. Unknown sources default to unverified and cannot
enter a training manifest. A typical safe flow is:

```bash
uv run cadence dataset source add https://example.com/launch-film.mp4 --submitted-by aven
uv run cadence dataset source inspect <source-id>
uv run cadence dataset source rights <source-id> --status user_owned --notes "Confirmed by user"
uv run cadence dataset source approve <source-id>
uv run cadence dataset source approve-download <source-id>
uv run cadence dataset source download <source-id>
uv run cadence dataset source eligibility <source-id> --eligible
uv run cadence dataset segments suggest <source-id>
uv run cadence dataset segment approve <segment-id>
uv run cadence dataset build launch-pilot
uv run cadence dataset report launch-pilot
uv run cadence storage report
```

See [the dataset-intake operations guide](docs/operations/dataset-intake.md) for the full safe
workflow and recovery procedures.
