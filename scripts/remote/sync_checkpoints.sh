#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_action.sh" sync_checkpoints "$@"

