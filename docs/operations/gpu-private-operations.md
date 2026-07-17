# Private GPU transfer and recovery runbook

This runbook is the public-safe CAD-24 procedure for a future RunPod A5000 operation. It defines
the order of work and the evidence required at each boundary. It contains no private endpoint,
credential, signed link, manifest content, filesystem location, Pod identifier, or launch
authorization.

Repository acceptance must not execute any step labelled **Human gate** or **Aven — remote
operation**. Codex owns repository code and dry-run validation. Aven owns later VPS/RunPod actions
only after the named human gate is complete.

## Roles and immutable inputs

| Role | Responsibility |
|---|---|
| Human approver | Approves provider, `$2` budget ceiling, runtime, secret channel, operator window, transfer, and termination owner |
| Codex — repository | Maintains exact config, package validation, checks, and public-safe documentation; never handles private bytes or credentials |
| Aven — remote operation | Performs separately approved VPS, private-storage, and RunPod actions; reports sanitized evidence only |
| Human launch operator | Confirms go/no-go, watches cost/runtime/health, requests clean stop, and verifies termination |

Every operation binds these immutable public-safe identities:

- exact 40-character Git commit;
- exact `uv.lock` hash;
- frozen first-run configuration and package hashes;
- opaque approved dataset snapshot handle;
- opaque human approval handle.

The real manifest, paths, checksums, source metadata, media, checkpoint contents, endpoints, and
credentials remain private. Code comes from Git at the exact commit. Dataset contents never come
from Git.

## Phase 0 — repository and approval preflight

1. **Codex — repository:** confirm the approved commit is published and its GitHub acceptance
   workflow is green.
2. **Codex — repository:** build and validate the sanitized first-run package from a clean
   worktree. Validation must report `launch_authorized: false` and `network_action: false`.
3. **Human gate:** approve one RunPod A5000, the current hourly offer at or below `$0.30`, a
   four-hour maximum, a `$2` total ceiling, zero persistent volume by default, the operator
   window, and the termination owner.
4. **Human gate:** choose a non-Linear, non-Git secret-delivery channel and approve the exact
   private-storage scopes. The synthetic smoke remains synthetic-only.
5. **Aven — remote operation:** dry-run search/create/inspect/terminate plans and compare their
   hashes and bounds with the approval. Do not create a Pod during preflight.

Stop if the commit, lock, configuration, dataset handle, package hash, provider, hardware, price,
runtime, budget, storage, operator, or approval window differs. A change requires a new package
and human decision.

## Phase 1 — private staging

The default transfer pattern is a temporary immutable private object-storage prefix. It avoids
putting dataset bytes in Git, Linear, the RunPod create request, provider environment fields, or a
public inbound service. A direct VPS-to-GPU copy is not the default and requires a separate
security review because the standard Pod plan requests no ports and no public IP.

1. **Human gate:** approve the temporary storage provider, retention deadline, encryption mode,
   and two least-privilege credentials:
   - VPS write-only access to one new immutable staging prefix;
   - GPU read-only access to that same prefix.
2. **Aven — remote operation:** on the VPS, re-run private dataset preflight against the frozen
   version. Require the approved opaque handle, 83 train rows, 22 validation rows, 0 test rows,
   approved/training-eligible records, present A/V modalities, checksum success, and no
   source-level split leak.
3. **Aven — remote operation:** upload the frozen manifest and referenced media without changing
   their bytes. Create a private inventory containing relative object key, size, and SHA-256.
   Never print or paste that inventory outside private storage.
4. **Aven — remote operation:** upload the inventory last, mark the prefix immutable for the
   operation window, and record only an opaque staging-evidence handle plus sanitized counts.
5. Remove the VPS write credential from the process environment immediately after upload.

No training eligibility, rights decision, segment decision, or manifest row may be changed during
staging. Any mismatch invalidates the staging prefix.

The default staging retention deadline is 24 hours after verified Pod termination. A longer
period requires a human decision because it expands private-data exposure and storage cost.

## Phase 2 — Runtime-only secret lifecycle

Provider API, dataset-read, and checkpoint-write credentials are separate and least privilege.
They must be short lived and limited to the approved Pod/prefix where supported.

1. **Human gate:** deliver each credential directly to Aven through the approved ephemeral
   channel. Never put it in a command argument, configuration file, `.env`, shell history, Git,
   Linear, chat, provider startup environment, logs, checkpoint, or report.
2. **Aven — remote operation:** enter credentials only into the bounded operator process
   environment with shell history disabled. Do not echo or inspect the environment.
3. Use dataset-read credentials only for download and integrity verification. Unset and revoke
   them before training begins.
4. Use checkpoint-write credentials only for the approved durable checkpoint prefix. They must
   not read the source dataset or write elsewhere.
5. Keep `RUNPOD_API_KEY` only in the control process performing an approved Pod lifecycle action;
   do not copy it into the Pod.
6. On completion or incident, unset all variables, revoke short-lived credentials, rotate any
   credential whose handling is uncertain, and verify access denial.

Sanitized logs may name the environment-variable class and success/failure, never its value,
endpoint, account, bucket, or object key.

## Phase 3 — GPU materialization and integrity verification

This phase is blocked until CAD-25 approval and an approved Pod exists. It does not authorize use
of private media for training; that later requires CAD-27.

1. **Aven — remote operation:** check out the exact approved commit in a clean worktree and
   perform the frozen `training-gpu` dependency sync.
2. Confirm CUDA/device identity, free disk, container image, Python version, dependency lock,
   checkpoint destination writeability using a synthetic sentinel, watchdog availability, and
   termination command. Do not decode private media yet.
3. Download the immutable private inventory, manifest, and media to an owner-only temporary GPU
   root. Materialize a private local manifest whose paths resolve only below that root.
4. Recompute SHA-256 for every downloaded media object and compare it privately with the frozen
   manifest/inventory. Re-run schema, modality, approval, eligibility, count, and split-isolation
   checks.
5. Confirm that the opaque dataset handle and sanitized counts match the frozen package. Delete
   the dataset-read credential from the environment and revoke it.
6. Report only pass/fail, sanitized counts, exact code SHA, package hash, and a new opaque
   integrity-evidence handle.

Checksum mismatch, missing media, extra media, path escape, row-count drift, modality failure, or
split leak is an immediate abort. Never skip, repair, or substitute a record on the GPU.

## Phase 4 — Checkpoints and clean stopping

Cadence writes an atomic local checkpoint only at a completed optimizer boundary, at most every
600 seconds, at the 210-minute soft stop, and at final completion. The durable copy uses a
run-specific private prefix and immutable step objects.

Before any later job, Aven must start a watchdog outside the training process. It uses a monotonic
240-minute deadline, requests the 210-minute clean stop, and invokes the separately approved
termination path if the process or operator misses the deadline. The human operator checks Pod
state, elapsed time, and projected compute cost at least every five minutes. Provider billing is
not an atomic kill switch, so the `$1.50` alarm preserves room to stop inside the `$2` ceiling.

After each local checkpoint:

1. **Aven — remote operation:** compute its SHA-256 and upload it under its exact global-step
   identity, never by overwriting an older step.
2. Download or remotely verify the uploaded size and checksum.
3. Update the private `latest` pointer only after verification. A partially uploaded checkpoint
   must never become latest.
4. Retain the prior verified checkpoint until the newer one has passed a fresh-process load.
5. Record only global step, success/failure, sanitized retrieval metrics, spend/runtime, and an
   opaque checkpoint-evidence handle outside private storage.

For an operator-requested stop, let the current optimizer boundary complete, write and verify a
checkpoint, stop the training process, then begin termination. If the process cannot stop cleanly,
preserve the newest previously verified durable checkpoint and terminate before exceeding the
hard limit.

## Abort and stop rules

| Trigger | Required response |
|---|---|
| OOM, decoder error, checksum mismatch, path escape, or non-finite loss/gradient | Abort; do not reduce batch size, skip data, or retry with changed inputs |
| Config, commit, lock, manifest, model, or checkpoint mismatch | Abort; require a new reviewed package |
| Checkpoint write/upload/verification failure | Stop at the next safe boundary; preserve the last verified checkpoint |
| Provider price above `$0.30/hour` | Do not create, or terminate immediately if detected after creation |
| Estimated total reaching `$1.50` | Operator alarm; stop cleanly unless the human explicitly confirms continuation within the existing `$2` cap |
| 210 minutes elapsed | Clean stop at the next optimizer boundary and durable-checkpoint verification |
| `$2` total or 240 minutes reached | Hard stop and verified termination; never extend automatically |
| Approval window expires or monitoring is lost | Clean stop if possible, otherwise terminate using the last verified checkpoint |

No error authorizes a larger GPU, longer runtime, higher spend, changed batch, skipped sample,
parameter sweep, or second real run.

## Fresh-process resume and recovery

Resume always uses a new process, the exact approved commit/config/lock/manifest, and the newest
fully verified durable checkpoint.

1. **Aven — remote operation:** download the selected immutable checkpoint and verify SHA-256.
2. Inspect checkpoint version, model metadata, configuration hash, manifest hash, lock hash,
   Git commit, epoch, next sample offset, global step, RNG state, and metrics privately.
3. Start a fresh process with the exact frozen configuration and explicit checkpoint path.
4. Require compatibility validation before any decoder or optimizer work. Cadence reconstructs
   the deterministic epoch permutation and next crop from the saved position and RNG state.
5. For the synthetic smoke, prove one bounded resumed operation and durable checkpoint before
   private-data approval. For a completed 40-step run, fresh-process load must return
   `already-complete` and perform no extra optimizer step.

If no checkpoint passes integrity and compatibility checks, do not resume. Terminate the Pod,
retain private incident evidence, and return to repository diagnosis.

## Verified termination and cleanup

1. **Human launch operator:** confirm the final or abort checkpoint is durable and identify the
   exact checkpoint that recovery would use.
2. **Aven — remote operation:** stop the process, capture only sanitized final metrics and
   runtime/spend totals, then terminate—not merely stop—the Pod.
3. Inspect the Pod ID and require a provider not-found response.
4. List persistent and network volumes separately. The default approved plan has zero persistent
   volume; any unexpected volume is an incident. Delete only resources covered by the human
   approval and verify deletion.
5. Revoke and unset provider, dataset, and checkpoint credentials. Delete GPU-local private media,
   manifests, caches, logs containing private values, and unverified checkpoints.
6. After the approved retention window, delete the temporary staging prefix and verify deletion.
7. Inspect billing. Report only sanitized total spend/runtime and an opaque billing-evidence
   handle.

The operator stays responsible until Pod absence, volume state, credential revocation, staging
retention, and durable checkpoint availability are all verified.

GPU-local data and unverified artifacts are deleted during termination. Temporary staging is
deleted within 24 hours. Raw private operational logs are retained for at most seven days unless
a human approves an incident hold. The final verified checkpoint and report remain in durable
private storage until the human records an accept/delete/extended-retention decision; they never
enter Git or Linear.

## Incident evidence boundary

Allowed in Linear: exact public Git SHA, public package/config hashes, sanitized counts, global
step, generic failure class, bounded runtime/spend, pass/fail, and opaque evidence handles.

Private only: source identifiers/URLs, manifest rows, paths, object keys, endpoints, checksums,
media, previews, rights evidence, credentials, signed links, Pod IDs, raw logs, checkpoints,
provider account/billing records, and stack traces containing any of those values.

An incident does not authorize publishing private evidence. If a secret may have entered a log or
checkpoint, revoke it first, quarantine the artifact privately, and record only the sanitized
incident class.

## Current boundary

This runbook and its tests perform no network action. No private staging prefix, credential,
RunPod Pod, transfer, checkpoint destination, spend, or training operation is created by CAD-24.
