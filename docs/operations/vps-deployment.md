# Private VPS deployment and recovery

This procedure prepares the canonical intake and review stack on the existing private VPS. It does
not authorize access, enter credentials, open public exposure, register sources, download media,
provision hardware, spend money, or train.

## Required human gate

Before any command carrying `--execute`, confirm that the VPS-access approval is still valid and
identifies:

- the responsible operator and bounded operations window;
- who enters and rotates runtime credentials;
- SSH tunnelling as the review-console access mode, unless a separate short-lived public-tunnel
  approval exists;
- an opaque evidence reference stored outside Git and Linear;
- the expected console shutdown time.

Never put hostnames, credentials, private paths, source records, tunnel links, or backup contents
in Git or Linear.

## Prepare the exact release

The repository checkout and `/srv/cadence/private` must be separate directories. From the clean
repository checkout, preview the release preparation:

```bash
./scripts/vps/prepare_private_stack.sh \
  --expected-commit <full-approved-40-character-sha>
```

The preview performs no dependency sync and creates no directory. After checking the plan and
human approval, execute it:

```bash
./scripts/vps/prepare_private_stack.sh \
  --expected-commit <full-approved-40-character-sha> \
  --execute
```

The wrapper:

1. refuses a commit mismatch or dirty worktree;
2. checks the committed lock and performs a frozen sync of only `media` and `operations-ui`;
3. validates `configs/vps.yaml` and the Git data policy;
4. creates the private runtime and backup directories with mode `0700`;
5. runs the sanitized VPS doctor without requiring the console to be running.

It is idempotent. It does not fetch Git changes, start a service, expose a port, or touch dataset
records.

## Start, check, and stop the review console

Enter `CADENCE_REVIEW_ADMIN_SECRET` only in the approved runtime secret channel, then start the
console on loopback:

```bash
uv run --no-sync cadence review-serve --config configs/vps.yaml
```

From a second VPS session, require a successful loopback health check:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml doctor \
  --expected-commit <full-approved-40-character-sha> \
  --require-health
```

The doctor returns only the deployed SHA, named pass/fail checks, permission counts, and storage
capacity totals. It does not emit the private runtime path, registry contents, environment, or
credentials.

Use an SSH local port-forward for browser access. Stop the foreground console with its process
supervisor or `Ctrl-C` at the approved end time, close the SSH tunnel, and clear the administrator
secret from the operator environment. If a separately approved Wormkey session was used, stop that
foreground command too; its child processes are cleaned up together.

## Metadata backup and restore rehearsal

Preview the backup policy:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml backup
```

Create an atomic owner-only archive:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml backup --execute
```

The command returns an opaque backup ID and archive checksum. It retains the newest seven archives
by default, as configured by `vps_operations.backup_retention_count`. Archives include:

- `registry.json`, or a valid empty registry before first intake;
- JSON and JSONL dataset metadata under the managed dataset metadata roots.

They exclude source media, normalized media, candidate clips, credentials, private evidence, lock
files, and partial files.

Rehearse restoration without replacing production state:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml \
  restore-rehearsal <opaque-backup-id>
```

The rehearsal rejects links, path traversal, oversized members, duplicate members, checksum
mismatches, invalid registry schemas, and malformed JSON/JSONL. Valid content is written with
`0700`/`0600` permissions into an isolated temporary directory, validated, and removed. The
production registry is never modified.

Actual production replacement is deliberately not automated in this milestone. During a real
incident, stop the console and intake processes, preserve the damaged state, obtain explicit
recovery approval, rehearse the selected archive, and use a separately reviewed recovery change.

## Sanitized handoff

Record only:

- exact deployed Git SHA;
- names and pass/fail results of commands run;
- owner-only permission check result;
- storage limits within/outside bounds, without paths;
- opaque backup ID/evidence handle and restore-rehearsal result;
- approved access mode and confirmation that the console was stopped;
- remaining risks and whether source registration is unblocked.

Do not paste raw command output if it contains private operational information.
