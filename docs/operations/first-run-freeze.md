# Bounded first-run freeze

CAD-17 freezes the first real contrastive run before any private transfer, GPU provisioning,
spend, or training. The checked-in source of truth is
`configs/first-run-v0.1.0.yaml`. A generated package binds that configuration to an exact Git
commit, `uv.lock`, and an opaque approved dataset snapshot handle. The package deliberately
contains no manifest contents, private paths, storage URIs, credentials, or launch authorization.

## Frozen experiment

- Hardware: one RunPod NVIDIA RTX A5000 24 GB, CPython 3.12, `training-gpu`
- Seed: `1337`
- Dataset cardinality: 83 train rows, 22 validation rows, 0 test rows
- Video encoder: base width 64
- Audio encoder: base width 96
- Shared output: 256-dimensional temporal embeddings, 128-dimensional projections, 8 tokens
- Media: 4-second clips, 16 RGB frames at 112 square, 16 kHz mono audio, 128 mel bins
- Optimizer: AdamW, learning rate `0.0003`, weight decay `0.0001`
- Objective: symmetric InfoNCE at temperature `0.07`
- Precision: AMP FP16 on CUDA
- Batch: loader and contrastive group size 32
- Schedule: 20 deterministic epochs, two complete groups per epoch, exactly 40 optimizer steps
- Full validation retrieval: every 10 steps and at the final step
- Checkpoint: at clean optimizer boundaries, at most every 600 seconds, and at final completion

The incomplete 19-row remainder of each deterministic epoch permutation does not form a
contrastive group. It is intentionally excluded. A resumed process reconstructs the same
permutation and continues at the checkpointed next-sample offset.

## Success and abort rules

Success requires all 40 optimizer steps, finite loss and gradients, complete 22-row validation
retrieval, Recall@1 at or above the 22-row chance level in both directions, a final atomic
checkpoint, and a compatible fresh-process checkpoint load. Training-batch retrieval is
diagnostic only.

Abort immediately on an out-of-memory event, decoder error, checksum mismatch, non-finite loss or
gradient, checkpoint failure, or checkpoint/configuration/manifest/lock incompatibility. There
is no automatic batch-size reduction and no skipped media. Stop cleanly at the next optimizer
boundary after 210 minutes. The hard ceilings are 240 minutes, `$5` total, and `$0.30` per hour.
These ceilings do not authorize spend.

## Package and validate

Run these only against the approved clean commit and use the opaque private dataset handle:

```bash
uv run cadence first-run-freeze \
  --config configs/first-run-v0.1.0.yaml \
  --dataset-snapshot-handle <opaque-approved-handle> \
  --output artifacts/reports/first-run-v0.1.0.json

uv run cadence first-run-validate artifacts/reports/first-run-v0.1.0.json \
  --config configs/first-run-v0.1.0.yaml \
  --dataset-snapshot-handle <same-opaque-approved-handle>
```

Freeze rejects a dirty worktree and typed `CADENCE_*__*` configuration overrides. Validation
rejects any change to the Git commit, dependency lock, dataset handle, configuration, or package
hash. Both commands are local and make no network request. The generated document always states
`launch_authorized: false`.

`--allow-dirty` exists only for repository acceptance tests; it must never be used for the
operator handoff.
