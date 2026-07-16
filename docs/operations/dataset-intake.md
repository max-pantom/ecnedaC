# Dataset intake operations

Use these commands from the Cadence repository on the VPS. The default dataset profile is
`configs/vps.yaml`; add `dataset --config <path>` immediately after `dataset` to use another one.

## 1. Submit and inspect

```bash
uv run cadence dataset source add <url> --submitted-by aven
uv run cadence dataset source add-batch urls.txt --submitted-by aven
uv run cadence dataset source list
uv run cadence dataset source inspect <source-id>
```

`add` returns `created: true` and a source record. Repeated canonical URLs return the existing ID
with `created: false`. Batch output reports added, duplicate, and invalid counts. Inspection stores
metadata or an `unsupported` status; one unsupported URL does not stop a batch.

## 2. Record rights and approvals

```bash
uv run cadence dataset source rights <source-id> \
  --status verified_permitted --notes "Permission reference or contract location"
uv run cadence dataset source approve <source-id>
uv run cadence dataset source approve-download <source-id>
```

These are three distinct decisions. Use `source reject` or `source reject-download` when review
fails. Never select a permitted rights state without evidence.

## 3. Download and normalize

```bash
uv run cadence storage report
uv run cadence dataset source download <source-id>
```

Cadence checks both working-storage and free-disk limits before acquisition. A successful command
records the raw path, normalized path, checksum, method, duration, and status. Duplicate content is
linked to the first checksum rather than stored twice. A failure is stored on the source record and
may be retried with the same command after correcting the reported cause.

## 4. Decide training eligibility

```bash
uv run cadence dataset source eligibility <source-id> --eligible
```

This succeeds only after permitted rights, source approval, download approval, and normalization.
Use `--ineligible` to revoke inclusion immediately.

## 5. Suggest and review clips

```bash
uv run cadence dataset segments suggest <source-id>
uv run cadence dataset segments list <source-id>
uv run cadence dataset segment approve <segment-id>
uv run cadence dataset segment reject <segment-id>
```

Suggestions use scene boundaries, frame differences, RMS activity changes, onset-like changes,
silence, and configured duration limits. They remain pending until a person approves or rejects
them. Extracted candidate clips carry their own checksum.

## 6. Build and inspect a dataset

```bash
uv run cadence dataset build launch-pilot
uv run cadence dataset report launch-pilot
```

Each build creates the next `vNNNN` directory and immutable manifest/report pair. Only approved
segments from training-eligible sources are included. Splits are assigned at the source-video
level, so clips from one source cannot leak across train, validation, and test.

