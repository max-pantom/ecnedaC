#!/usr/bin/env bash
set -euo pipefail

exec uv run cadence gpu-transfer report-pull "$@"
