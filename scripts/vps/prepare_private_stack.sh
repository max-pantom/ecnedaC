#!/usr/bin/env bash
set -euo pipefail

expected_commit=""
config="configs/vps.yaml"
execute="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expected-commit)
      expected_commit="${2:-}"
      shift 2
      ;;
    --config)
      config="${2:-}"
      shift 2
      ;;
    --execute)
      execute="true"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "--expected-commit must be a full 40-character Git SHA" >&2
  exit 2
fi

if [[ "$execute" != "true" ]]; then
  echo "DRY RUN: verify exact commit and clean worktree"
  echo "DRY RUN: sync locked media and operations-ui dependency groups"
  echo "DRY RUN: validate VPS config and repository data policy"
  echo "DRY RUN: prepare owner-only private runtime directories"
  echo "DRY RUN: run sanitized VPS doctor without requiring a running console"
  exit 0
fi

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "current checkout does not match the approved commit" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "VPS deployment requires a clean Git worktree" >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required on the VPS" >&2
  exit 1
fi

uv lock --check
uv sync --frozen --no-default-groups --group media --group operations-ui
uv run --no-sync cadence config-check --config "$config" >/dev/null
uv run --no-sync cadence data-policy check
uv run --no-sync cadence vps --config "$config" prepare --execute
uv run --no-sync cadence vps --config "$config" doctor \
  --expected-commit "$expected_commit"
