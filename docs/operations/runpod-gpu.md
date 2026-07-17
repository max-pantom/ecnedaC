# RunPod GPU readiness

Cadence targets one RunPod Community Cloud `NVIDIA RTX A5000` with 24 GB VRAM for the first
bounded GPU checks. This document prepares validated plans only. It does not authorize API-key
entry, Pod creation, private-data transfer, spending, or training.

## Public-safe configuration

`configs/gpu-24gb.yaml` fixes these ceilings:

- one NVIDIA RTX A5000;
- an hourly-price ceiling of `$0.30`;
- a synthetic-smoke ceiling of 30 minutes and `$1`;
- a provisional first-run ceiling of four hours and `$5`;
- no exposed ports or public IP request;
- no persistent Pod volume by default;
- a PyTorch 2.11 / CUDA 12.6 image matching the locked GPU dependency group.

The budget values are maximum templates, not approvals. RunPod prices and capacity can change, so
the operator must compare the returned hourly price with both the configuration and the current
[RunPod pricing page](https://www.runpod.io/pricing) immediately before any separately approved
creation.

## Dry-run plans

These commands read no API key and perform no network request:

```bash
./scripts/remote/runpod_search.sh --config configs/gpu-24gb.yaml
./scripts/remote/runpod_create.sh --config configs/gpu-24gb.yaml
./scripts/remote/runpod_inspect.sh --config configs/gpu-24gb.yaml --pod-id placeholder-pod
./scripts/remote/runpod_terminate.sh --config configs/gpu-24gb.yaml --pod-id placeholder-pod
```

The JSON plan names `RUNPOD_API_KEY` as the future credential source but never contains its value.
It also excludes manifest/checkpoint URIs, private paths, storage credentials, environment
contents, and commands that would transfer or train on data.

## Execution gates

An executed request is a later operation and requires all of the following:

1. the matching Linear human gate is complete;
2. the exact configuration, provider, budget, runtime, and hardware have been approved;
3. `RUNPOD_API_KEY` is entered only into the bounded operator process environment;
4. `--execute` is supplied;
5. create/terminate receives a non-secret opaque `--approval-reference`;
6. terminate additionally receives `--confirm-termination`.

Never put the API key in a command argument, configuration file, `.env`, shell history, Git,
Linear, logs, or chat. Use a restricted RunPod key with only the permissions required for Pod
operations, following [RunPod API-key guidance](https://docs.runpod.io/get-started/api-keys).

Example syntax for a future approved action:

```bash
export RUNPOD_API_KEY="<entered-through-approved-secret-channel>"
./scripts/remote/runpod_create.sh \
  --config configs/gpu-24gb.yaml \
  --approval-reference <opaque-human-gate> \
  --execute
unset RUNPOD_API_KEY
```

Do not run that example during repository acceptance.

## Idempotency and cost containment

Before creating, Cadence lists existing Pods and refuses to create a second Pod with the configured
name. After creation, it checks the provider-returned hourly price. If the price exceeds the
configured ceiling, Cadence attempts immediate termination and reports failure.

The runtime ceiling is not a claim that RunPod supplies a billing kill switch. A later launch issue
must install and test an independent watchdog and operator alarm. The operator remains responsible
for verifying that the Pod is gone.

RunPod documents that stopping a Pod releases its GPU but preserves `/workspace` volume data and
continues volume-storage billing. Terminating removes non-network-volume data, while attached
network volumes persist independently. Read the current
[Pod lifecycle documentation](https://docs.runpod.io/pods/manage-pods) and
[Pod pricing documentation](https://docs.runpod.io/pods/pricing) before execution.

After every future run:

1. export only approved checkpoints/reports;
2. terminate the Pod rather than merely stopping it;
3. inspect the Pod ID and require a not-found result;
4. separately list persistent/network volumes and delete only those explicitly approved for
   deletion;
5. inspect billing and record only sanitized totals plus an opaque evidence handle.

## Current boundary

No RunPod request has been sent, no API key has been entered, no Pod has been created, no private
media has moved, and no GPU training has occurred as part of this readiness work.

The older `remote-action terminate_gpu` and `scripts/remote/terminate_gpu.sh` entry points remain
only for an explicitly configured legacy Vast.ai profile. They reject the RunPod profile so an
operator cannot accidentally send a termination command to the wrong provider.
