# Dataset specification

Every JSONL record uses manifest schema `0.1.0`, carries source-level provenance and split
metadata, and points to a clip with both video and native audio. The dataset trusts upstream legal
validation but requires the provenance fields and `eligible_for_contrastive=true`. Splits occur by
source asset, never by derived clip.

