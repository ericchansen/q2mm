#!/bin/bash
# Overnight benchmark: 5 systems × 3 optimizers (~6 hours on GPU)
set -euo pipefail
PYTHON=~/repos/q2mm/.venv/bin/python
CLI="$PYTHON -m q2mm.diagnostics.cli --backend jax"

for sys in rh-enamide heck-relay pd-allyl pd-conjugate rh-conjugate; do
  for opt in jaxopt-lbfgs optax-adam optax-adam-cosine; do
    echo "=== $(date '+%H:%M') $sys + $opt ==="
    $CLI --system "$sys" --optimizer "$opt" 2>&1 | grep -E "RMSD|Score|Wall|converged|Evals|FAILED|Skipping" || true
    echo ""
  done
done
echo "=== DONE $(date) ==="
