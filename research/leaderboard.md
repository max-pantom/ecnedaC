# Cadence Research Leaderboard

This leaderboard tracks experiments that produce comparable evidence.

Do not add an experiment as a ranked result until it has:

- a committed repo revision
- exact config
- exact dataset/manifest
- hardware/runtime notes
- reproducible command
- linked experiment file

## Current status

No real benchmark result exists yet.

The repository currently has a local training-readiness baseline, but no real-data training or evaluation result. `EXP001` records the baseline scaffold.

## Results

| Rank | Experiment | Status | Dataset / split | Config | Commit | Primary metric | V→A R@1 | A→V R@1 | Loss | Keep? | Notes |
|---:|---|---|---|---|---|---|---:|---:|---:|---|---|
| — | [EXP001 Contrastive Baseline](EXP001_contrastive_baseline.md) | planned | generated fixtures only so far | `configs/local.yaml` / `configs/test.yaml` | `36236a4` | — | — | — | — | keep scaffold | Not a real benchmark yet |

## Metric definitions

- **V→A R@1**: percentage of videos whose correct matching audio is ranked first.
- **A→V R@1**: percentage of audio clips whose correct matching video is ranked first.
- **Loss**: symmetric InfoNCE loss unless otherwise specified.
- **Primary metric**: chosen per experiment before running. Avoid selecting metrics after seeing results.

## Open comparisons

| ID | Comparison | Why it matters | Status |
|---|---|---|---|
| EXP002 | Optical flow vs RGB | Tests whether motion-specific representation beats raw appearance | planned |
| EXP003 | Clip length | Tests timing window: short impact vs longer action context | planned |
| EXP004 | Temperature ablation | Tests sensitivity of contrastive embedding geometry | planned |
| EXP005 | Audio feature target | Tests mel baseline against richer audio embeddings | planned |
| EXP006 | Timeline event head | First bridge from representation learning to structured sound-design intent | planned |
