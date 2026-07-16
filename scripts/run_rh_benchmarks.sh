#!/usr/bin/env bash
# Rh-enamide full benchmark re-run (post-vmap).
#
# Runs the complete Rh-enamide benchmark matrix across JAX, JAX-MD,
# and OpenMM backends to:
#   1. Verify the vmap batching change (PR #264) produces identical RMSD
#   2. Capture new timing numbers
#   3. Produce our optimized FF for comparison against Zenodo archive
#
# Also evaluates the Zenodo QFUERZA-optimized FF with our engine to
# get an apples-to-apples score on the same objective function.
#
# Usage:
#   bash scripts/run_rh_benchmarks.sh          # full run (~1-2 hours)
#   bash scripts/run_rh_benchmarks.sh --quick  # smoke test (~5 min)
#
# Results: results/rh-enamide/results/*.json

set -uo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Activate the project venv
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$(dirname "$REPO_ROOT")/q2mm/.venv/bin/activate" ]; then
    # Worktree — venv lives in the main repo
    # shellcheck disable=SC1091
    source "$(dirname "$REPO_ROOT")/q2mm/.venv/bin/activate"
fi

# Track failures but don't stop on them
FAILURES=()
run_step() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo "$name"
    echo "============================================================"
    if "$@"; then
        echo "✅ $name succeeded"
    else
        echo "❌ $name FAILED (exit $?)"
        FAILURES+=("$name")
    fi
}

QUICK=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=true
    echo "=== QUICK MODE (smoke test) ==="
fi

OUT="results/rh-enamide"
# qfuerza-zenodo data lives in the separate q2mm-data repo.
# Override with Q2MM_DATA_REPO=/path/to/q2mm-data if needed.
export Q2MM_DATA_REPO="${Q2MM_DATA_REPO:-$(dirname "$REPO_ROOT")/q2mm-data}"
QFUERZA_DIR="$Q2MM_DATA_REPO/qfuerza-zenodo"

# ── Step 0: Preflight ────────────────────────────────────────────────
run_step "STEP 0: Preflight check" \
    python3 -m q2mm.diagnostics.cli --preflight

# ── Step 1: Smoke test (JaxOpt L-BFGS, 50 iters — exercises vmap) ────
run_step "STEP 1: Smoke test — JaxOpt L-BFGS (50 iters, exercises vmap)" \
    python3 -m q2mm.diagnostics.cli --leaderboard-only \
        --system rh-enamide \
        --backend jax \
        --optimizer jaxopt-lbfgs \
        --form mm3 \
        --max-iter 50 \
        --output /tmp/rh-smoke

if $QUICK; then
    echo ""
    echo "Quick mode — skipping full matrix. Smoke test results in /tmp/rh-smoke/"
    exit 0
fi

# ── Step 2: JAX backend (primary — exercises vmap) ───────────────────
# Only the optimizers that match the archived results + jaxopt.
# Skip multi-start, basin-hopping, optax (OOM on Rh-enamide GPU).
run_step "STEP 2: JAX backend — mm3, core optimizers" \
    python3 -m q2mm.diagnostics.cli --leaderboard-only \
        --system rh-enamide \
        --backend jax \
        --form mm3 \
        --optimizer scipy-lbfgsb scipy-nm scipy-powell grad-simp jaxopt-lbfgs \
        --output "$OUT"

run_step "STEP 2b: JAX backend — harmonic, core optimizers" \
    python3 -m q2mm.diagnostics.cli --leaderboard-only \
        --system rh-enamide \
        --backend jax \
        --form harmonic \
        --optimizer scipy-lbfgsb scipy-nm scipy-powell grad-simp jaxopt-lbfgs \
        --output "$OUT"

# ── Step 3: JAX-MD backend ───────────────────────────────────────────
run_step "STEP 3: JAX-MD backend — harmonic" \
    python3 -m q2mm.diagnostics.cli --leaderboard-only \
        --system rh-enamide \
        --backend jax-md \
        --form harmonic \
        --optimizer scipy-lbfgsb scipy-nm scipy-powell grad-simp \
        --output "$OUT"

# ── Step 4: OpenMM backend ───────────────────────────────────────────
run_step "STEP 4: OpenMM backend — mm3" \
    python3 -m q2mm.diagnostics.cli --leaderboard-only \
        --system rh-enamide \
        --backend openmm \
        --form mm3 \
        --optimizer scipy-nm grad-simp \
        --output "$OUT"

# ── Step 5: Evaluate Zenodo FF with our engine ───────────────────────
run_step "STEP 5: Evaluate Zenodo QFUERZA-optimized FF on our objective" \
    python3 -c "
import json, sys, os
from pathlib import Path
sys.path.insert(0, '.')
# JAX_PLATFORMS left unset — let JAX auto-detect GPU

from q2mm.backends.mm.jax_engine import JaxBackend
from q2mm.benchmarks.systems import load_system
from q2mm.io.mm3 import load_mm3_fld
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor

backend = JaxBackend()
case = load_system('rh-enamide', backend=backend)
qfuerza_dir = Path(os.environ['Q2MM_DATA_REPO']) / 'qfuerza-zenodo'

# Load Zenodo optimized FF (final iteration)
zenodo_ff = load_mm3_fld(str(qfuerza_dir / 'rh-enamide/qfuerza/mm3_010.fld'))
print(f'Zenodo FF: {len(zenodo_ff.bonds)} bonds, {len(zenodo_ff.angles)} angles')

# Evaluate with our objective (original problem observations)
plan = ObjectivePlan.from_problem(case.problem)
evaluator = PythonObjectiveExecutor(plan, backend, zenodo_ff)
params = plan.layout.vector(zenodo_ff)
score = evaluator.value(params)
print(f'Zenodo FF score on our objective: {score:.4f}')

result = {
    'source': 'zenodo_qfuerza_mm3_010',
    'score_on_our_objective': float(score),
    'note': 'Zenodo QFUERZA-optimized FF evaluated with our JaxBackend + rh-enamide objective. Comparable to our benchmark final_score values.',
}
out = qfuerza_dir / 'rh_enamide_zenodo_ff_on_our_objective.json'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to {out}')
"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "DONE. Results in:"
echo "  $OUT/results/*.json"
echo "  $QFUERZA_DIR/rh_enamide_zenodo_ff_on_our_objective.json"
echo "============================================================"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  ${#FAILURES[@]} step(s) failed:"
    for f in "${FAILURES[@]}"; do
        echo "  ❌ $f"
    done
    echo ""
    echo "Completed steps still have valid results."
    exit 1
else
    echo ""
    echo "✅ All steps succeeded."
fi

echo ""
echo "Next: git add + commit results"
