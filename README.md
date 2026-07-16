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
