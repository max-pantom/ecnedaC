# Dataset specification

Cadence's current dataset milestone is the **Launch Video Dataset Pilot**.

The initial domain is not isolated UI micro-interactions. It is short-form product and brand
launch motion, including SaaS launch films, AI product reveals, hardware reveals,
interface-driven product films, kinetic typography, feature montages, identity reveals, and
cinematic brand lockups.

## Source states

Source intake has three separate states:

| State | Meaning |
|---|---|
| `candidate` | A URL/source has been discovered or submitted. It is not approved for download or training. |
| `approved_source` | The source is worth downloading/inspecting/segmenting. Rights may still be unverified. |
| `training eligible` | A reviewed clip has both modalities, acceptable sync, and verified rights for training use. |

A downloaded video is not automatically training eligible.

Publicly accessible does not imply permission for model training. Unclear sources must remain:

```text
rights_status: unverified
eligible_for_training: false
eligible_for_contrastive: false
```

Unverified sources may be used for private pipeline testing while excluded from real training
manifests by default.

## Required source and clip fields

Every source and clip must record:

- source URL
- creator or publisher where known
- collection method
- license / rights status
- source asset ID
- clip asset ID when applicable
- source timestamps when applicable
- checksum
- duration
- resolution
- fps
- audio sample rate
- split
- modality presence
- review status
- eligibility for contrastive training

The manifest schema remains JSONL and versioned as `0.1.0`. Splits occur by source asset, never by
derived clip.

## Pilot target

```text
30 to 50 source launch videos
100 to 250 extracted clips
4 to 10 seconds per clip
native aligned audio preserved
one initial domain: launch-video sound design
```

## Prioritized segment content

Prioritize clips containing:

- product reveals
- logo resolutions
- feature montages
- kinetic typography
- device rotations
- interface reveals
- camera transitions
- visual buildups
- large visual arrivals
- deliberate silence
- final brand lockups

Reject or flag:

- long talking-head sections
- static frames
- podcast footage
- dialogue-only sections
- music visualizers
- unrelated cinematic footage
- corrupt audio
- missing audio
- duplicate clips
- poor synchronization

## Cheap CPU-side segment suggestions

The VPS should only run lightweight preprocessing. Candidate suggestions use:

- scene-ish frame-difference boundaries
- frame-difference motion intensity
- audio onset/RMS activity
- RMS loudness change
- silence ratio / silence boundaries

Do not use optical flow, encoders, or neural analysis during this pilot.

## Storage limits

The VPS is only a lightweight coordination/preprocessing machine:

```text
1 CPU core
3.8 GB RAM
no GPU
Cadence storage cap: 20 GB
stop before free VPS storage drops below 15 GB
one CPU-heavy media job at a time
```

Storage records keep `path` and optional `storage_uri` fields so local filesystem pilots can later
move to S3-compatible storage such as Cloudflare R2 without changing ingestion or training records.
Do not configure R2 credentials until supplied.

## Report requirements

Dataset reports include:

- number of source videos
- number of candidate segments
- number approved
- number rejected
- total duration
- estimated disk usage
- license-status breakdown
- duplicate count
- missing-modality count
- duration distribution
- motion-intensity distribution
- audio activity distribution

## Operational intake contract

Candidate launch-video sources are tracked independently of training manifests. Rights status,
source approval, download approval, and training eligibility are separate fields. New sources are
always `unverified` and ineligible. Only an approved segment whose source has permitted rights and
explicit training eligibility can enter a versioned manifest. Launch-video category tags describe
dataset context and are not model labels.
