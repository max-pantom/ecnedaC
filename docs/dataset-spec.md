# Dataset specification

Every JSONL record uses manifest schema `0.1.0`, carries source-level provenance and split
metadata, and points to a clip with both video and native audio. The dataset trusts upstream legal
validation but requires the provenance fields and `eligible_for_contrastive=true`. Splits occur by
source asset, never by derived clip.

Candidate launch-video sources are tracked independently of training manifests. Rights status,
source approval, download approval, and training eligibility are separate fields. New sources are
always `unverified` and ineligible. Only an approved segment whose source has permitted rights and
explicit training eligibility can enter a versioned manifest. Launch-video category tags describe
dataset context and are not model labels.
