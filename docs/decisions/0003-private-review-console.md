# 0003: Service-owned private human review console

The one-core VPS review interface uses server-rendered FastAPI and Jinja pages behind a loopback
Uvicorn listener. Operators connect through an SSH tunnel. A non-loopback listener requires two
explicit safety acknowledgements and secure cookies; public hosting remains out of scope.

`DatasetIntakeService` remains the only mutation authority for CLI and HTTP callers. Review
decisions carry an expected record revision and append an immutable audit event inside the same
registry lock. The console uses a runtime-only administrator secret, signed `HttpOnly` and
`SameSite=Strict` sessions, session-bound CSRF tokens, POST-only mutations, and registered
identifier media lookup with intake-root containment.

The file-backed registry remains appropriate for one Uvicorn worker. Private directories use mode
`0700`, managed files use `0600`, and real queues, audit events, media, manifests, and reports
remain outside Git. A database, public ingress, roles, and multi-organization tenancy are deferred.

A later operator decision permits temporary Wormkey ingress as an explicit exception, not as
continuous public hosting. `cadence review-share` is dry-run-first, pins the npm package version,
keeps Uvicorn on loopback, enforces its own ephemeral outer authentication, forces secure cookies,
limits login failures and tunnel lifetime, suppresses Wormkey owner-control URLs, and tears both
processes down together. The operator accepts that Wormkey terminates TLS and transports requested
review content.
