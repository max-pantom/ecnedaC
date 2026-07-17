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

## Launch-video research workflow

This branch also retains the earlier file-oriented launch-video pilot. Its commands live under
`cadence pilot` so they do not conflict with the audited intake workflow above. The pilot supports
batch candidate capture and direct registration of a local, lawfully obtained media file:

```bash
uv run cadence pilot source add \
  https://example.com/a https://example.com/b \
  --submitted-by max

uv run cadence pilot source add \
  --media-path /path/to/source.mp4 \
  --source-url https://example.com/launch-video \
  --creator "Example Studio" \
  --collection-method user-submitted-local-file \
  --license-status synthetic-generated
```

The remaining research flow is:

```bash
uv run cadence pilot source inspect --source all
uv run cadence pilot source approve --source <source-asset-id>
uv run cadence pilot segments suggest --source all --min-duration 4 --max-duration 10
uv run cadence pilot segments approve --clip <clip-asset-id>
uv run cadence pilot build pilot-launch-v0
uv run cadence pilot report pilot-launch-v0
```

`pilot source download` requires `yt-dlp` and only downloads approved sources. Downloading never
grants training eligibility; unverified rights remain excluded. The VPS is a lightweight dataset
coordination and preprocessing host, not a GPU training machine.
