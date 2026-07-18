#!/usr/bin/env bash
set -euo pipefail

exec uv run cadence gpu-transfer checkpoint-push "$@"
