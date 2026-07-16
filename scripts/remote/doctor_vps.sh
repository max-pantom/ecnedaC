#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/_action.sh" doctor_vps "$@"

