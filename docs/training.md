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
