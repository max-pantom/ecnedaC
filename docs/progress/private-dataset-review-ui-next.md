# Next milestone: private dataset operations and human review UI

Status: complete locally.

The implementation now includes versioned review/audit contracts, revision-checked service
mutations, a loopback-first FastAPI/Jinja console, signed administrator sessions, CSRF protection,
contained range-based media previews, immutable dataset build/report views, owner-only runtime
permissions, and a generated-media end-to-end acceptance test. No real media was collected, no
VPS or GPU action was executed, and no private runtime record was added to Git.

An explicitly requested follow-up adds guarded, short-lived Wormkey sharing. It is locally
validated in dry-run and simulated-process tests; no real Wormkey tunnel was opened.

## Starting point

`main` is the canonical base. It contains:

- `cadence dataset ...` as the guarded intake workflow;
- one canonical `DatasetIntakeService` registry shared by CLI and review UI;
- atomic registry persistence, storage preflight checks, download/normalization adapters, CPU segment
  suggestions, explicit approvals, versioned manifests, and synthetic acceptance tests;
- no real training, public service, or remotely executed job.

Real media and real operational metadata are private. Git contains code, schemas, documentation,
configuration templates, and synthetic generators only. Read
`docs/operations/private-data-boundary.md` before changing intake or review code.

## Goal

Build a small VPS-hosted review console for the human decisions that cannot safely be automated:

- rights classification and evidence reference;
- source relevance approval or rejection;
- download approval or rejection;
- training eligibility or revocation;
- segment preview, approval, and rejection;
- immutable dataset-build and report review.

The existing `DatasetIntakeService` remains the sole mutation authority. HTTP handlers and
templates must call it rather than reimplementing eligibility or approval rules.

## Required foundation

Before UI implementation:

1. `cadence data-policy check` must reject tracked real media, JSONL manifests, registries,
   runtime data, checkpoints, and generated artifacts.
2. The VPS intake root must live outside the Git worktree, such as `/srv/cadence/private`.
3. Real source queues, audit events, previews, manifests, and reports must remain VPS-private.
4. The existing real pilot record must be removed from the current tree. Rewriting published Git
   history requires separate explicit approval.

The later canonical-workflow cleanup removed the compatibility CLI and implementation. A guarded
one-way `cadence dataset legacy-import` command preserves private legacy source identities while
forcing rights and approval re-review.

## Proposed architecture

Use a lightweight server-rendered stack suitable for the one-core VPS:

```text
FastAPI + Jinja templates + HTMX + Uvicorn
```

Suggested modules:

```text
cadence/review/
├── app.py
├── auth.py
├── models.py
├── media.py
├── templates/
└── static/
```

Add an `operations-ui` dependency group so the existing local and training dependency groups stay
small.

The UI binds to `127.0.0.1` by default and is reached through an SSH tunnel. A non-loopback bind
must require an explicit secure-deployment configuration. Authentication uses an
environment-provided administrator secret, signed `HttpOnly` and `SameSite=Strict` cookies, CSRF
protection, and POST-only mutations.

## Review and audit models

Add structured, versioned models:

- `RightsDecision`
- `EvidenceReference`
- `ReviewDecision`
- `AuditEvent`
- `ReviewQueueItem`
- `RecordRevision`

Every mutation records actor, timestamp, previous state, new state, reason, and an opaque evidence
reference. Do not store contracts, credentials, cookies, access tokens, or private license text.
Mutation requests include `expected_revision`; stale browser submissions are rejected.

Changing rights to unverified, restricted, rejected, revoked, or expired immediately revokes
training eligibility.

## Initial routes

```text
GET  /healthz
GET  /api/v1/review/queue
GET  /api/v1/sources/{source_id}
POST /api/v1/sources/{source_id}/rights
POST /api/v1/sources/{source_id}/source-decision
POST /api/v1/sources/{source_id}/download-decision
POST /api/v1/sources/{source_id}/eligibility
GET  /api/v1/sources/{source_id}/segments
POST /api/v1/segments/{segment_id}/decision
GET  /api/v1/media/{entity_id}
POST /api/v1/datasets/build
```

Media preview handlers must resolve only registered media identifiers, enforce path containment,
and support bounded range reads without exposing arbitrary VPS paths.

## Acceptance

- CLI and UI actions produce identical service-validated state transitions.
- Every mutation creates an append-only audit event.
- Unauthenticated, CSRF-invalid, stale-revision, and path-traversal requests fail.
- Unverified or restricted media cannot become eligible or enter a manifest.
- Real runtime data remains untracked and outside the repository.
- Synthetic end-to-end tests cover submission through private dataset build.
- Ruff, strict Mypy, all existing tests, security tests, and `make accept` pass.

## Out of scope

- Public hosting
- A separate React dashboard or frontend build
- Multiple organizations or role hierarchies
- Automated legal decisions
- S3/R2 migration
- Real GPU training or real-data evaluation
- Rewriting published Git history without explicit approval
