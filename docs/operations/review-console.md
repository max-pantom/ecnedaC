# Human review console operations

The review console is a private operator interface over `DatasetIntakeService`. It does not own a
second copy of rights, approval, eligibility, or dataset-build rules.

## Install on the VPS

The preferred exact-release preparation flow is documented in
[`vps-deployment.md`](vps-deployment.md). Its wrapper is dry-run-first, requires a full approved Git
SHA, and does not start the console.

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

After starting the console, verify the loopback health endpoint and the rest of the deployment
controls together:

```bash
uv run --no-sync cadence vps --config configs/vps.yaml doctor \
  --expected-commit <full-approved-40-character-sha> \
  --require-health
```

Do not expose the default console publicly. A non-loopback bind is refused unless both
`--allow-non-loopback` and `CADENCE_REVIEW_SECURE_DEPLOYMENT=true` are present. That override is
only a guardrail acknowledgement; TLS, firewalling, and upstream access control are still operator
responsibilities.

## Temporary Wormkey link

When an SSH tunnel is inconvenient, the VPS agent can create a short-lived Wormkey link. This is
an explicit public-tunnel operation, not the default deployment mode. Node.js 18 or newer and
`npx` must already be available on the VPS.

First inspect the plan. This performs no network action:

```bash
uv run --group operations-ui cadence review-share \
  --config configs/vps.yaml \
  --expires 30m
```

Then explicitly execute it:

```bash
uv run --group operations-ui cadence review-share \
  --config configs/vps.yaml \
  --expires 30m \
  --execute
```

Cadence pins `wormkey@0.1.5`, binds the review server only to `127.0.0.1`, forces secure session
cookies, applies a login-attempt limit, and adds an independently generated outer Basic Auth
challenge before the Cadence administrator login. The expiry must be between five minutes and two
hours. Wormkey output containing owner controls is suppressed; the command emits one JSON object
containing only the share URL, expiry, temporary outer username/password, and a separate ephemeral
read-only login. A VPS agent may relay that JSON response to the operator. It never emits
`CADENCE_REVIEW_ADMIN_SECRET`.

The ephemeral login expires no later than the requested tunnel lifetime. Its signed session role
can inspect only the source queue and allowlisted source metadata: source URL, title, publisher,
platform, duration, submission metadata, inspection state, decision states, and revision. It
cannot mutate records or access media bytes, segments, dataset reports, audit events, evidence
references, license notes, storage paths/URIs, checksums, or processing errors. Server-rendered
pages hide all mutation forms for this role; API authorization independently enforces the same
boundary.

The read-only login exists for metadata assistance only. It cannot make a legal or policy
decision. The administrator remains responsible for every rights, relevance, download, and
eligibility decision. Relay the temporary credentials only through the trusted agent session and
never copy them to Git, Linear, shell history, or durable logs.

The command remains in the foreground. The link closes when it expires, the agent stops the
command, or either process fails; Cadence and Wormkey are cleaned up together. Do not add the
Wormkey overlay script to Cadence.

Wormkey is a third-party beta tunnel whose edge terminates public TLS. The provider transports the
HTTP requests and media previews that pass through the link, and the pinned npm program executes
as the VPS operator. Cadence strips its secrets from the npm subprocess environment, but this is
not equivalent to end-to-end private networking or process isolation. Use the smallest practical
expiry, relay credentials only through the trusted agent session, and close the tunnel immediately
after review.

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

Private metadata backup, isolated restore rehearsal, retention, and sanitized deployment evidence
are documented in [`vps-deployment.md`](vps-deployment.md). Those backups exclude source media,
candidate clips, credentials, and license evidence contents.
