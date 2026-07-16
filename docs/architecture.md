# Architecture

The current milestone is a single Python codebase containing configuration, provenance manifests,
PyAV media loading, compact video/audio encoders, temporal alignment, contrastive training, and
remote job specifications. The laptop is the development control environment. Future VPS and GPU
machines pull exact Git commits rather than receiving a separate implementation.

The dataset-intake pilot adds an atomic JSON registry under the configured intake root, a guarded
local-filesystem storage backend, replaceable download and media-processing protocols, CPU signal
analysis, explicit human approval states, and versioned JSONL manifest assembly. S3-compatible
storage remains an interface only; the pilot requires no cloud-storage credentials.
