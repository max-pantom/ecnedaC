# Private data boundary

Cadence uses Git to distribute implementation code, typed schemas, documentation, configuration
templates, and synthetic-fixture generators. Git must never distribute the real dataset.

## Allowed in Git

- Python and shell source code
- Schema definitions and generated type declarations
- Configuration templates without credentials or private endpoints
- Documentation containing synthetic examples
- Tests and code that generate synthetic media at test time
- Empty `.gitkeep` placeholders

## VPS or private object storage only

- Real source video or audio
- Downloaded originals and normalized media
- Extracted candidate or approved clips
- Source queues and `registry.json`
- Provenance-bearing JSONL manifests
- Dataset reports containing source or rights metadata
- Checkpoints, embeddings, evaluation outputs, and exports
- Credentials, cookies, access tokens, private contracts, or license documents

The VPS profile uses `/srv/cadence/private` as its runtime root. Local/test profiles may use
repository-relative paths only for generated synthetic fixtures. Repository ignore and index
checks also protect these conventional paths:

```text
data/intake/
data/pilots/
data/manifests/
data/cache/
artifacts/
```

The intake runtime creates private directories with mode `0700` and registry, lock, manifest,
report, and managed media files with mode `0600`.

VPS metadata recovery archives live below the private runtime root, use mode `0600`, and are also
excluded from Git. They contain the registry plus JSON/JSONL dataset metadata, but never source
media, candidate clips, credentials, or external evidence contents. The checked-in restore command
is an isolated rehearsal and never overwrites production state.

The GPU host may receive an authorized manifest and media through private storage or an explicit
VPS-to-GPU transfer. It must pull code by exact Git commit, but it must not obtain dataset contents
from Git.

The default future GPU mechanism is a temporary immutable private object-storage prefix with
separate scoped VPS-write and GPU-read credentials. Staging, checksum verification, checkpoint
durability, secret revocation, and cleanup follow
[`gpu-private-operations.md`](gpu-private-operations.md). No staging or transfer is authorized by
that documentation alone.

A temporary, explicitly executed Wormkey review tunnel does not copy runtime records into Git.
However, requested pages and previews transit a third-party TLS-terminating edge. Treat its
short-lived URL and outer credentials as secrets, and close it as soon as human review finishes.
The read-only reviewer role is authorized to receive an allowlisted source-metadata view through
that edge. It cannot receive media, storage locations, checksums, license notes, audit history, or
private evidence.

## Enforcement

`.gitignore` excludes the private roots, JSONL records, common media formats, checkpoints, and
generated artifacts. Before every commit or deployment, run:

```bash
uv run cadence data-policy check
make data-policy
```

The same check is part of `make accept`. It examines Git's tracked-file index rather than only the
working directory, so force-added private files are rejected.

If private data was already pushed, deleting it in a later commit does not remove it from Git
history. Stop using or sharing the affected identifier, assess whether credentials or private
rights information were exposed, and obtain explicit approval before rewriting published history.
