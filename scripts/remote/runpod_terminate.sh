#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_runpod_action.sh" terminate "$@"
