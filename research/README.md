# Cadence Research Log

This folder is Aven's research memory for Cadence.

Code shows what the system does now. Research logs preserve why we tried something, what happened, and whether the choice should survive.

Every experiment should be written so that a future researcher can understand the decision without reading chat history or reconstructing intent from commits.

## Experiment record format

Each experiment answers four questions:

1. **What did we change?**
2. **Why?**
3. **What happened?**
4. **Should we keep it?**

## Naming

Use stable numbered files:

```text
EXP001_contrastive_baseline.md
EXP002_optical_flow_vs_rgb.md
EXP003_clip_length.md
EXP004_temperature_ablation.md
```

Do not renumber old experiments. If an experiment is repeated, create a new file and link back to the earlier one.

## Status values

- `planned` — designed but not run
- `running` — currently executing
- `complete` — run and interpreted
- `kept` — result should become part of the default system
- `rejected` — result should not be kept
- `inconclusive` — insufficient evidence; do not treat as decision

## Rules

- Record negative results. They are valuable.
- Separate observation from interpretation.
- Include exact commit, config, dataset/manifest, hardware, and command when an experiment is actually run.
- Do not claim training results from synthetic/local smoke tests as real model quality.
- Link leaderboard rows to experiment files.
- Prefer small, controlled comparisons over vague “improvements.”
