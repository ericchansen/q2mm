#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
repo_name="$(basename "$repo_root")"
parent_dir="$(dirname "$repo_root")"
worktree_path="${1:-$parent_dir/${repo_name}-upstream-worktree}"
ref="${2:-}"

if [[ -z "$ref" ]]; then
  fixture_path="$repo_root/test/fixtures/seminario_parity/rh_enamide_reference.json"
  if [[ -f "$fixture_path" ]]; then
    ref="$(
      python - "$fixture_path" <<'PY'
import json
import sys
from pathlib import Path

fixture_path = Path(sys.argv[1])
try:
    data = json.loads(fixture_path.read_text())
    print(data.get("metadata", {}).get("upstream_commit", ""))
except Exception:
    print("")
PY
    )"
  fi
fi

if [[ -z "$ref" ]]; then
  ref="upstream/master"
fi

git -C "$repo_root" fetch upstream --quiet

if [[ -e "$worktree_path" ]]; then
  git -C "$worktree_path" rev-parse --show-toplevel >/dev/null 2>&1 || {
    echo "Path exists but is not a git worktree: $worktree_path" >&2
    exit 1
  }
  git -C "$worktree_path" fetch upstream --quiet
  git -C "$worktree_path" checkout --detach "$ref" >/dev/null
else
  git -C "$repo_root" worktree add --detach "$worktree_path" "$ref" >/dev/null
fi

commit="$(git -C "$worktree_path" rev-parse HEAD)"
echo "Ready: $worktree_path @ $commit"
