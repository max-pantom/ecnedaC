# Source rights operations

Rights states:

- `verified_permitted`: permission has been checked and recorded.
- `user_owned`: the user owns the submitted media and confirms training use.
- `licensed`: a license explicitly covers the intended training use.
- `unverified`: default; public availability is not permission.
- `restricted`: access or terms prohibit the intended use.
- `rejected`: the source was rejected during rights review.

Every new source starts as `unverified` with `eligible_for_training: false`. Source approval means
the media is relevant. Download approval means acquisition is authorized. Training eligibility is
a later explicit decision and is permitted only for the first three rights states.

Record evidence without secrets:

```bash
uv run cadence dataset source rights <source-id> \
  --status licensed --notes "License agreement reference LIC-2026-014"
```

Do not paste credentials, private contract text, access tokens, or cookies into notes. If rights
become uncertain, revoke eligibility immediately:

```bash
uv run cadence dataset source eligibility <source-id> --ineligible
uv run cadence dataset source rights <source-id> --status unverified --notes "Review reopened"
```

Restricted and rejected states also reject download approval automatically.
