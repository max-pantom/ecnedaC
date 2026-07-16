# Security and provenance

Credentials are never stored in the repository. Remote scripts are dry-run-first and require an
explicit execution flag. Dataset entries require source URL, license status, collection method,
and content checksum before becoming eligible for contrastive use.

Public availability never implies permission. Direct HTTP and optional `yt-dlp` adapters may only
acquire ordinary accessible sources after explicit source and download approval; Cadence contains
no authentication bypass, DRM circumvention, or restricted-media workaround.
