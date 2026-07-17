# 0002: Atomic JSON registry and guarded local storage for the intake pilot

The one-core, 3.8 GB VPS pilot uses one atomic, file-locked JSON registry and a guarded local
filesystem tree. This avoids deploying a database before concurrent operators require one while
still making URL, rights, approval, segment, and dataset state recoverable and inspectable.

Downloader and object-storage behavior is protocol-driven. Direct HTTP and optional `yt-dlp`
adapters are replaceable. S3-compatible storage is deliberately an inactive interface until an R2
migration is authorized. All unknown rights remain unverified and training-ineligible.
