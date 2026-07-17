# Training and readiness

The current stage permits synthetic tensors, tiny generated fixtures, one-step backward passes,
and checkpoint serialization. It prohibits dataset collection and real pretraining. A remote job
must identify its Git revision, lock hash, configuration, dataset manifest, destination, seed,
hardware, budget, and runtime.

## GPU dependency readiness

The `training-gpu` group targets CPython 3.12 on Linux x86-64 and pins matching Torch and
TorchAudio 2.11.0 CUDA 12.6 wheels by their exact official URLs. This is intentionally separate
from the mutually exclusive Intel Mac `training-local` group.

Verify the lock and official wheel listings without installing GPU packages or starting training:

```bash
make gpu-deps-check
```

The check requires network access only to the official PyTorch wheel index. It verifies both the
locked artifact sources and the presence of the CPython 3.12 Linux wheels.

## RunPod A5000 packaging

The GPU profile targets one RunPod NVIDIA RTX A5000 24 GB. Repository validation can build
redacted search/create/inspect/terminate plans without reading credentials or contacting RunPod:

```bash
uv run cadence runpod-action create --config configs/gpu-24gb.yaml
```

The synthetic smoke template is capped at 30 minutes/$1. The provisional first-run template is
capped at four hours/$2 with a `$0.30` hourly-price ceiling. These are safety ceilings, not
authorization to provision or spend. See
[RunPod GPU readiness](operations/runpod-gpu.md) for execution and termination gates.

## Frozen first real run

The exact bounded experiment is checked in as `configs/first-run-v0.1.0.yaml`. Its sanitized
package binds the experiment to an exact clean Git commit, dependency lock, and opaque approved
dataset handle without exposing private manifest details. It cannot authorize a launch.

The run is limited to 40 optimizer steps, evaluates the complete validation set every 10 steps,
checkpoints at clean boundaries, and stops at the frozen time and budget ceilings. Checksum,
decode, OOM, non-finite, and compatibility failures abort rather than silently changing the run.
See [the first-run freeze guide](operations/first-run-freeze.md).

The frozen configuration is not a transfer or launch procedure. Before any approved GPU action,
follow [the private GPU operations runbook](operations/gpu-private-operations.md) for scoped
runtime secrets, immutable staging, checksum verification, durable checkpoints, fresh-process
resume, cost alarms, verified termination, and cleanup.
