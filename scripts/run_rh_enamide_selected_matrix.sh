#!/usr/bin/env bash
# run_rh_enamide_selected_matrix.sh -- run rh-enamide combos 1-12 and 20.
#
# Fast-first order:
#   3, 7, 11, 2, 6, 1, 5, 4, 8, 10, 9, 12, 20
#
# Usage:
#   scripts/run_rh_enamide_selected_matrix.sh
#   scripts/run_rh_enamide_selected_matrix.sh benchmark_results/rh_enamide_2026-04-03
#
# Environment overrides:
#   PYTHON=python3.12
#   OPENMM_PLATFORM=CUDA
#   LEADERBOARD_ONLY=0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python3}"
openmm_platform="${OPENMM_PLATFORM:-CUDA}"
timestamp="$(date +%F_%H%M%S)"
output_dir="${1:-benchmark_results/rh_enamide_selected_${timestamp}}"

mkdir -p "$output_dir"

benchmark_cmd=("$python_bin" -m q2mm.diagnostics.cli)
common_args=(--system rh-enamide --output "$output_dir")

if [[ "${LEADERBOARD_ONLY:-1}" != "0" ]]; then
    common_args+=(--leaderboard-only)
fi

header() {
    printf '\n== %s ==\n' "$1"
}

run_combo() {
    local combo_id="$1"
    local label="$2"
    shift 2

    header "Combo ${combo_id} - ${label}"
    printf 'Started: %s\n' "$(date -Iseconds)"
    "$@"
    printf 'Finished: %s\n' "$(date -Iseconds)"
}

write_plan() {
    cat >"$output_dir/selected_combo_order.txt" <<EOF
rh-enamide selected matrix
output_dir=$output_dir
openmm_platform=$openmm_platform
leaderboard_only=${LEADERBOARD_ONLY:-1}

Requested combos:
  1  JAX harmonic L-BFGS-B
  2  JAX harmonic Nelder-Mead
  3  JAX harmonic Powell
  4  JAX harmonic grad-simp
  5  JAX mm3 L-BFGS-B
  6  JAX mm3 Nelder-Mead
  7  JAX mm3 Powell
  8  JAX mm3 grad-simp
  9  JAX-MD harmonic L-BFGS-B
  10 JAX-MD harmonic Nelder-Mead
  11 JAX-MD harmonic Powell
  12 JAX-MD harmonic grad-simp
  20 OpenMM mm3 grad-simp

Fast-first run order:
  3, 7, 11, 2, 6, 1, 5, 4, 8, 10, 9, 12, 20
EOF
}

write_plan

header "Preflight"
"${benchmark_cmd[@]}" --preflight

# Quick failures and lighter JAX single-shot runs first.
run_combo 3 "JAX GPU harmonic Powell" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form harmonic --optimizer Powell

run_combo 7 "JAX GPU mm3 Powell" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form mm3 --optimizer Powell

run_combo 11 "JAX-MD GPU harmonic Powell" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax-md --form harmonic --optimizer Powell

run_combo 2 "JAX GPU harmonic Nelder-Mead" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form harmonic --optimizer Nelder-Mead

run_combo 6 "JAX GPU mm3 Nelder-Mead" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form mm3 --optimizer Nelder-Mead

run_combo 1 "JAX GPU harmonic L-BFGS-B" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form harmonic --optimizer L-BFGS-B

run_combo 5 "JAX GPU mm3 L-BFGS-B" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form mm3 --optimizer L-BFGS-B

# Converged JAX cycling runs before the slower JAX-MD work.
run_combo 4 "JAX GPU harmonic grad-simp" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form harmonic --optimizer grad-simp

run_combo 8 "JAX GPU mm3 grad-simp" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax --form mm3 --optimizer grad-simp

run_combo 10 "JAX-MD GPU harmonic Nelder-Mead" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax-md --form harmonic --optimizer Nelder-Mead

run_combo 9 "JAX-MD GPU harmonic L-BFGS-B" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax-md --form harmonic --optimizer L-BFGS-B

run_combo 12 "JAX-MD GPU harmonic grad-simp" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend jax-md --form harmonic --optimizer grad-simp

# OpenMM mm3 grad-simp is last because it is expected to be the slowest.
run_combo 20 "OpenMM GPU mm3 grad-simp" \
    "${benchmark_cmd[@]}" "${common_args[@]}" \
    --backend openmm --form mm3 --optimizer grad-simp \
    --platform "$openmm_platform"

header "Done"
printf 'Saved outputs under %s\n' "$output_dir"
