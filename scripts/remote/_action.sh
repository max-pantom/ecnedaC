#!/usr/bin/env bash
set -euo pipefail

action="${1:?action is required}"
shift
uv_bin="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$uv_bin" && -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
fi
if [[ -z "$uv_bin" ]]; then
  echo "uv is required; install it or set UV_BIN" >&2
  exit 1
fi
exec "$uv_bin" run cadence remote-action "$action" "$@"
