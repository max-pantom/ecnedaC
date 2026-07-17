# Security and provenance

Credentials are never stored in the repository. Remote scripts are dry-run-first and require an
explicit execution flag. Dataset entries require source URL, license status, collection method,
and content checksum before becoming eligible for contrastive use.

Public availability never implies permission. Direct HTTP and optional `yt-dlp` adapters may only
acquire ordinary accessible sources after explicit source and download approval; Cadence contains
no authentication bypass, DRM circumvention, or restricted-media workaround.

Git is a code and schema distribution channel, not a dataset store. Real video, audio, extracted
clips, source queues, the intake registry, provenance-bearing manifests, dataset reports,
checkpoints, and evaluation outputs remain on the VPS or in private object storage. The
repository acceptance gate runs `scripts/check_repository_data_policy.py` and fails if Git tracks
one of these private artifacts.
