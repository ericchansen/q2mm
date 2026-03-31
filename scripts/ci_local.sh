#!/usr/bin/env bash
# ci_local.sh — Run the same checks locally that CI runs remotely.
#
# Usage:
#   scripts/ci_local.sh              # lint + core tests (fast, ~30s)
#   scripts/ci_local.sh --backend    # also run backend tests via Docker
#   scripts/ci_local.sh --all        # lint + core + all Docker backends
#
# Requires: python3, pip (for lint/core). Docker (for --backend/--all).

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "${GREEN}✓ $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; exit 1; }
header() { echo -e "\n${BOLD}── $1 ──${RESET}"; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_BACKEND=false
RUN_ALL=false

for arg in "$@"; do
    case "$arg" in
        --backend) RUN_BACKEND=true ;;
        --all)     RUN_ALL=true ;;
        -h|--help)
            echo "Usage: scripts/ci_local.sh [--backend] [--all]"
            echo "  (no args)   lint + core tests"
            echo "  --backend   also run Docker backend tests"
            echo "  --all       run everything including all backends"
            exit 0 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ── Tier 1: Lint ──
header "Lint: ruff check"
python3 -m ruff check q2mm/ test/ scripts/ && pass "ruff check" || fail "ruff check"

header "Lint: ruff format"
python3 -m ruff format --check q2mm test scripts examples && pass "ruff format" || fail "ruff format"

# ── Tier 1: Core tests ──
header "Core tests"
python3 -m pytest -m "not (openmm or tinker or jax or jax_md or psi4)" -q && pass "core tests" || fail "core tests"

if ! $RUN_BACKEND && ! $RUN_ALL; then
    echo -e "\n${GREEN}${BOLD}All fast checks passed.${RESET} Run with --backend for Docker tests."
    exit 0
fi

# ── Tier 2: Backend tests via Docker ──
if ! command -v docker &>/dev/null; then
    fail "Docker not found. Install Docker to run backend tests."
fi

REGISTRY="ghcr.io/ericchansen/q2mm"
BACKENDS=(openmm tinker jax jax-md psi4)
if $RUN_ALL; then
    BACKENDS+=(full)
fi

# Map backend name → pytest marker (jax-md → jax_md)
marker_for() {
    local b="$1"
    case "$b" in
        jax-md) echo "jax_md" ;;
        full)   echo "" ;;       # full runs all tests
        *)      echo "$b" ;;
    esac
}

for backend in "${BACKENDS[@]}"; do
    header "Docker: $backend"
    IMAGE="$REGISTRY/ci-$backend:latest"
    MARKER=$(marker_for "$backend")

    if [ -n "$MARKER" ]; then
        PYTEST_CMD="python -m pytest -m $MARKER --run-slow -q"
    else
        PYTEST_CMD="python -m pytest --run-slow -q"
    fi

    docker run --rm \
        -v "$REPO_ROOT:/work" \
        -w /work \
        --user root \
        "$IMAGE" \
        sh -c "git config --global --add safe.directory /work && pip install -e . --no-deps -q && $PYTEST_CMD" \
        && pass "$backend" || fail "$backend"
done

echo -e "\n${GREEN}${BOLD}All checks passed.${RESET}"
