# Aven dataset-intake handoff

Aven operates Cadence; she does not modify or rebuild its Python internals.

## VPS preparation

Requirements: Python 3.11.15, one CPU core, 3.8 GB RAM, Git, FFmpeg/FFprobe, and `uv`.

```bash
git pull --ff-only
uv sync --no-default-groups --group media
uv run cadence config-check --config configs/vps.yaml
uv run cadence storage report
```

Expected configuration includes `runtime.num_workers: 1`, 20 GiB maximum working storage, and
15 GiB minimum free disk.

## Safe operator sequence

```bash
uv run cadence dataset source add <url> --submitted-by aven
uv run cadence dataset source inspect <source-id>
uv run cadence dataset source rights <source-id> --status <rights-state> --notes "<evidence>"
uv run cadence dataset source approve <source-id>
uv run cadence dataset source approve-download <source-id>
uv run cadence dataset source download <source-id>
uv run cadence dataset source eligibility <source-id> --eligible
uv run cadence dataset segments suggest <source-id>
uv run cadence dataset segments list <source-id>
uv run cadence dataset segment approve <segment-id>
uv run cadence dataset build launch-pilot
uv run cadence dataset report launch-pilot
```

## Recovery

- `unsupported`: keep the record; report the URL and adapter error to the primary coding agent.
- `failed` inspection/download: inspect the stored `error_state`, correct the external condition,
  then rerun `inspect` or `download`. Commands are safe to retry.
- Storage rejection: run `cadence storage report`; do not override the guard.
- Wrong rights state: revoke eligibility first, then set the corrected rights state.
- Wrong segment decision: run the opposite approve/reject command before building again.
- Interrupted metadata write: rerun the command. Registry writes use a lock and atomic replace.
- Duplicate URL/checksum: use the returned existing/duplicate record; do not copy files manually.

Never run GPU training, scrape large URL lists, bypass source controls, edit `registry.json` by
hand, or put credentials in the repository. Escalate code defects, schema changes, and downloader
adapter changes to the primary Cadence coding agent.

