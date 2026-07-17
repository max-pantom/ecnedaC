# Human review console operations

The review console is a private operator interface over `DatasetIntakeService`. It does not own a
second copy of rights, approval, eligibility, or dataset-build rules.

## Install on the VPS

```bash
uv sync --no-default-groups --group media --group operations-ui
uv run cadence config-check --config configs/vps.yaml
uv run cadence data-policy check
```

The VPS configuration stores runtime data below `/srv/cadence/private`, outside the Git worktree.
Create that directory for the unprivileged Cadence operator with mode `0700`. Cadence also
enforces owner-only permissions on managed directories and files.

## Start privately

Set a random administrator secret of at least 32 characters in the process environment. Do not put
it in `.env`, shell history, system logs, or Git.

```bash
export CADENCE_REVIEW_ADMIN_SECRET="<runtime-secret>"
uv run --group operations-ui cadence review-serve --config configs/vps.yaml
```

The default listener is `127.0.0.1:8787`. From the operator laptop, create an SSH tunnel:

```bash
ssh -L 8787:127.0.0.1:8787 <vps-host>
```

Then open `http://127.0.0.1:8787`. The session cookie is signed, `HttpOnly`, and
`SameSite=Strict`. Every mutation also requires its session-bound CSRF token.

Do not expose the default console publicly. A non-loopback bind is refused unless both
`--allow-non-loopback` and `CADENCE_REVIEW_SECURE_DEPLOYMENT=true` are present. That override is
only a guardrail acknowledgement; TLS, firewalling, and upstream access control are still operator
responsibilities.

## Review sequence

1. Inspect the source metadata and preview only registered, contained media.
2. Record a rights decision with an opaque evidence reference and reason.
3. Approve or reject source relevance.
4. Approve or reject acquisition.
5. Download and normalize using the existing intake operation.
6. Explicitly enable or revoke training eligibility.
7. Generate segment suggestions and approve or reject each candidate.
8. Build a versioned private dataset and inspect its report.

The browser sends the record revision it displayed. If another operator or CLI action changed the
record first, the mutation receives a conflict response and must be reviewed again.

Rights changes to an unverified or prohibited state immediately revoke training eligibility.
Restricted, rejected, revoked, and expired rights also revoke download approval.
Evidence references identify an approved external record; they must not contain credentials,
contract text, cookies, private documents, or access tokens.

## Media safety

Preview URLs accept only registered entity IDs. The server resolves the corresponding stored path,
requires it to remain below the configured private intake root, and returns bounded byte ranges.
It never accepts an arbitrary filesystem path from the browser.

## Audit and recovery

Every review mutation appends an audit event with the actor, timestamp, action, prior state, new
state, reason, evidence reference, and resulting revision. Audit data stays in the private runtime
root and is never committed.

- Authentication failure: verify the process environment and restart; do not print the secret.
- CSRF failure: reload the page and repeat the decision.
- Revision conflict: reload the source or segment and review the intervening state change.
- Invalid state transition: resolve the unmet rights, approval, download, or normalization gate.
- Suspected path escape: stop the console and inspect the registry and intake-root ownership.
