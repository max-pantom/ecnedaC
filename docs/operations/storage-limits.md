# Storage limits

The VPS profile enforces:

- Maximum Cadence intake working storage: 20 GiB.
- Minimum filesystem free space after an operation: 15 GiB.
- Unknown-size download reservation: 2 GiB.
- One processing worker.

Inspect capacity before and after work:

```bash
uv run cadence storage report
```

Downloads and segment extraction stop before starting if their reservation would cross either
limit. A `.part` download is removed after failure. Do not bypass these protections to make a job
fit; reject or remove unneeded candidate material through an approved retention procedure, or add
storage and rerun the command.

VPS pilot data lives below `/srv/cadence/private` by default. The registry, raw sources,
normalized sources, candidate segments, versioned manifests, and reports share the same 20 GiB
accounting boundary. The loader rejects a VPS intake root inside the Git worktree. R2/S3
credentials are not needed and must not be added during this milestone.

The older `cadence pilot` workflow writes below `data/pilots/`. Both `data/intake/` and
`data/pilots/` are private working directories ignored by Git. Their contents must remain on the
VPS. Only the empty `.gitkeep` placeholders are allowed in the repository.
