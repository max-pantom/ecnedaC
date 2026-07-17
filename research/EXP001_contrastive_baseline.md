# EXP001 Contrastive Baseline

| Field | Value |
|---|---|
| Status | planned |
| Owner | Aven |
| Date opened | 2026-07-16 |
| Date closed | — |
| Repo commit | 36236a4 `build local contrastive training readiness` |
| Branch | main |
| Config | `configs/local.yaml`, `configs/test.yaml`, later `configs/gpu-24gb.yaml` |
| Dataset / manifest | generated fixtures for smoke tests; real manifest not yet collected |
| Hardware | current VPS for inspection only; no GPU available |
| Related code | `cadence/training/contrastive.py`, `cadence/training/runner.py`, `cadence/encoders/`, `cadence/data/contrastive.py` |

## Question

Can Cadence establish a reliable first baseline for learning temporally aligned video/audio representations before attempting a creative sound-design timeline model?

## What did we change?

Nothing has been changed yet.

This experiment records the existing baseline already present in the repository:

```text
video clip + native audio
→ video encoder global embedding
→ audio encoder global embedding
→ symmetric InfoNCE loss
→ bidirectional retrieval metrics
```

## Why?

Cadence should first learn whether motion and matching audio can be aligned in representation space.

If this fails, building a creative reasoning transformer on top would be premature. The foundation model needs grounding in timing, weight, impact, and audiovisual correspondence before it predicts structured sound-design events.

## Method

Not yet run by Aven.

The repository documents these intended local commands:

```bash
uv sync
make accept
uv run cadence train-contrastive --config configs/local.yaml
uv run cadence retrieval-eval --synthetic --config configs/test.yaml
```

For a real baseline, this must later be run against a real validated manifest, not only synthetic fixtures.

## What happened?

Aven inspected the repository on 2026-07-16 without running code.

Observed implementation state:

- compact R(2+1)D video encoder exists
- compact log-mel residual audio encoder exists
- timestamped masked temporal tokens exist
- symmetric InfoNCE exists
- retrieval metrics exist
- manifest-based contrastive training entry point exists
- checkpoint save/resume exists
- generated fixture path exists
- docs report local acceptance passed on the original Intel Mac control environment

No new training result was produced in this inspection.

| Metric | Value | Notes |
|---|---:|---|
| loss | — | not run by Aven |
| video→audio R@1 | — | not run by Aven |
| audio→video R@1 | — | not run by Aven |
| video→audio R@5 | — | not run by Aven |
| audio→video R@5 | — | not run by Aven |
| mean rank | — | not run by Aven |
| median rank | — | not run by Aven |
| latency | — | not run by Aven |
| human preference win rate | — | not applicable yet |

## Interpretation

The repo has a credible local-readiness baseline, but not a research result yet.

The current baseline proves the shape of the system more than model quality:

```text
fixtures + small encoders + one-step training + checkpointing
```

This should be treated as infrastructure readiness, not evidence that Cadence understands motion/sound relationships in real creative data.

## Should we keep it?

Decision: `keep as baseline scaffold`

Reason:

The contrastive baseline is the right first foundation because it tests audiovisual correspondence before generation. It should remain the first reproducible baseline, but it needs a real dataset manifest and real validation split before it becomes a meaningful leaderboard entry.

## Follow-ups

- [ ] Run local acceptance on the VPS without changing code.
- [ ] Create first real dataset manifest.
- [ ] Run contrastive baseline on real clips.
- [ ] Add leaderboard metrics from a real validation set.
- [ ] Compare RGB baseline against optical-flow or motion-token variants.
