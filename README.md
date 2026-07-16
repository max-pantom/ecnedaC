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

## Launch Video Dataset Pilot

Cadence's next milestone is the Launch Video Dataset Pilot. The initial domain is short-form
product and brand launch motion: SaaS launch films, AI product reveals, hardware reveals,
interface-driven product films, kinetic typography, feature montages, identity reveals, and
cinematic brand lockups.

URL intake records candidates first. Public visibility is not treated as training permission.
Unverified sources remain quarantined by default:

```text
rights_status = unverified
eligible_for_training = false
```

Example candidate intake:

```bash
uv run cadence dataset source add https://example.com/launch-video --submitted-by max
uv run cadence dataset source add https://example.com/a https://example.com/b --submitted-by max
```

For a local/lawful source file:

```bash
uv run cadence dataset source add \
  --media-path /path/to/source.mp4 \
  --source-url https://example.com/launch-video \
  --creator "Example Studio" \
  --collection-method user-submitted-local-file \
  --license-status synthetic-generated
```

Pilot workflow:

```bash
uv run cadence dataset source inspect --source all
uv run cadence dataset source approve --source <source-asset-id>
uv run cadence dataset source download --source <source-asset-id>
uv run cadence dataset segments suggest --source all --min-duration 4 --max-duration 10
uv run cadence dataset segments approve --clip <clip-asset-id>
uv run cadence dataset build pilot-launch-v0
uv run cadence dataset report pilot-launch-v0
```

`source download` requires `yt-dlp` and only downloads approved sources. Downloading a source does
not make it training eligible; unverified rights remain excluded.

The VPS is only a lightweight dataset coordination and preprocessing machine. Do not run GPU
training or heavy model work there.
