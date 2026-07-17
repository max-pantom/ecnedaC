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

## Canonical dataset intake

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

## Private data boundary

Git contains code, schemas, documentation, and synthetic-fixture generators only. Real source
media, extracted clips, operator registries, source queues, manifests, dataset reports, and
training artifacts stay on the VPS or in explicitly configured private object storage.

Run the repository guard before committing:

```bash
uv run cadence data-policy check
make data-policy
```

See [the private-data operations policy](docs/operations/private-data-boundary.md).

## Human review console

The private VPS review console presents the existing rights, approval, eligibility, segment, and
dataset-build operations without duplicating their service rules:

```bash
uv sync --group operations-ui
export CADENCE_REVIEW_ADMIN_SECRET="<runtime-secret-of-at-least-32-characters>"
uv run --group operations-ui cadence review-serve --config configs/vps.yaml
```

It binds to loopback by default and should be reached through an SSH tunnel. See
[the review-console operations guide](docs/operations/review-console.md).

For an explicitly authorized, short-lived Wormkey link, inspect and then execute the guarded plan:

```bash
uv run --group operations-ui cadence review-share --config configs/vps.yaml --expires 30m
uv run --group operations-ui cadence review-share --config configs/vps.yaml --expires 30m --execute
```

Executed sharing creates a separate expiring read-only login for source metadata assistance.
It never emits the administrator secret. Read-only sessions cannot mutate records or access media,
audit evidence, segments, dataset reports, storage locations, checksums, or license notes.

## Private VPS release preparation

After the human VPS-access gate is approved, a VPS operator can inspect the exact deployment plan
without changing the host:

```bash
./scripts/vps/prepare_private_stack.sh \
  --expected-commit <full-approved-40-character-sha>
```

Adding `--execute` verifies the exact clean checkout, performs a frozen dependency sync, checks the
VPS configuration and repository data policy, prepares the owner-only private runtime, and runs a
sanitized deployment doctor. It does not start the review console, open a tunnel, register sources,
download media, or train.

Metadata-only recovery controls are also dry-run or rehearsal oriented:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml backup
uv run --no-sync cadence vps --config configs/vps.yaml backup --execute
uv run --no-sync cadence vps --config configs/vps.yaml \
  restore-rehearsal <opaque-backup-id>
```

See [the private VPS deployment guide](docs/operations/vps-deployment.md).

## Retired pilot registry migration

`cadence dataset` is the only dataset workflow. The old `cadence pilot` command and its separate
file-oriented implementation have been removed. If a private VPS still has a legacy
`sources.jsonl`, preview and then execute a one-way import:

```bash
uv run cadence dataset legacy-import /private/path/to/old-pilot
uv run cadence dataset legacy-import /private/path/to/old-pilot \
  --submitted-by migration-operator --execute
```

The import preserves source identities, URLs, submitter attribution, collection method, creator,
duration, and valid checksums where available. It never copies media or trusts legacy approval and
rights fields: every imported source is unverified, pending review, and training-ineligible.
