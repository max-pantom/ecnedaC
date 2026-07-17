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

Temporary Wormkey review sharing is dry-run-first and requires `--execute`. Cadence keeps the
application on loopback, adds ephemeral outer authentication and secure cookies, constrains the
tunnel lifetime, and suppresses provider owner-control URLs. It does not make Wormkey an
end-to-end private network: its edge terminates TLS and transports requested content.

RunPod plans name `RUNPOD_API_KEY` but never serialize its value. Dry runs do not read the
environment or contact RunPod. Future create/terminate execution requires an explicit execution
flag and a separate opaque human-approval reference. Plans request no public ports, public IP, or
private data. Stopped Pods can retain billable storage, so every future job requires verified
termination and a separate persistent-volume review.
