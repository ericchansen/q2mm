# Rh-Enamide

This page answers one question: what does q2mm currently achieve on a realistic
large-system organometallic benchmark? The scope is intentionally explicit:
this page summarizes the Rh-enamide benchmark work that has actually been
completed and archived so far. It is not a full 24-combo Rh-enamide matrix.

## Scope

- System: 9 transition-state structures for Rh(I)-diphosphine enamide
  hydrogenation
- Size: 36-62 atoms per structure; 182 fitted parameters in the MM3 template
- QM reference: Jaguar B3LYP/LACVP**
- Current benchmark scope: selected overnight GPU matrix (13 attempted combos)
- Related pages: [GPU Acceleration](gpu.md) covers dedicated device-scaling
  analysis; [Published FF Validation](published-ff-validation.md) covers the
  literature-force-field parity check

The MM3 template is used directly on OpenMM. JAX and JAX-MD use a harmonic copy
of the same Seminario-derived parameters, so cross-form comparisons are useful
for workflow guidance but not a statement that the force fields are identical.

## Completed Rh-enamide matrix to date

The current archived run executed the selected subset defined in the runner
script (archive combos 1-12 and 20) on the RTX 5090. Default rows are grouped
by status and then by final RMSD. Use the filters to isolate backend/form/
optimizer/status slices. Successful wall-clock values come from the raw CLI
log; failed rows keep their wall-clock failure times for completeness.

<div class="benchmark-table-anchor" data-benchmark-table="rh-enamide"></div>

| Backend | Form | Optimizer | Status | Result | MAE | Wall clock |
|---------|------|-----------|--------|--------|----:|-----------:|
| OpenMM | mm3 | grad-simp | success | 173.1 -> 42.7 RMSD | 31.8 | 112,959.0 s |
| JAX | harmonic | L-BFGS-B | success | 173.5 -> 57.4 RMSD | 35.0 | 2,278.0 s |
| JAX | mm3 | L-BFGS-B | success | 173.1 -> 61.5 RMSD | 48.5 | 1,975.2 s |
| JAX | mm3 | Nelder-Mead | success | 173.1 -> 66.6 RMSD | 54.9 | 591.4 s |
| JAX | harmonic | Nelder-Mead | success | 173.5 -> 81.2 RMSD | 50.7 | 559.9 s |
| JAX-MD | harmonic | L-BFGS-B | success | 41,409.7 -> 86.7 RMSD | 48.2 | 3,261.1 s |
| JAX | harmonic | Powell | failed | Eigenvalues did not converge | - | 173.0 s |
| JAX | mm3 | Powell | failed | Eigenvalues did not converge | - | 177.0 s |
| JAX-MD | harmonic | Powell | failed | Eigenvalues did not converge | - | 211.0 s |
| JAX-MD | harmonic | Nelder-Mead | failed | Eigenvalues did not converge | - | 351.0 s |
| JAX-MD | harmonic | grad-simp | failed | Eigenvalues did not converge | - | 659.0 s |
| JAX | mm3 | grad-simp | failed | Eigenvalues did not converge | - | 836.0 s |
| JAX | harmonic | grad-simp | failed | Eigenvalues did not converge | - | 861.0 s |

## Interpretation

- OpenMM CUDA MM3 + grad-simp is the best fit in the current Rh-enamide archive,
  but it is an overnight-scale job rather than a quick screening run.
- JAX L-BFGS-B and Nelder-Mead are the most practical same-day screening runs in
  the current dataset. They cover both harmonic and MM3 workflows without the
  overnight OpenMM cost.
- JAX-MD harmonic L-BFGS-B can recover from a very poor starting point, but it
  does not yet beat the strongest JAX or OpenMM result in this selected set.
- The failure pattern is now well characterized: Powell on JAX/JAX-MD and the
  selected JAX/JAX-MD grad-simp cases all terminate with the same
  `Eigenvalues did not converge` signature.
- This page intentionally stays narrow. The older harmonic-only cycling study is
  more useful as GPU-scaling context, so it now belongs on the
  [GPU Acceleration](gpu.md) page rather than in the main Rh-enamide narrative.

## Artifacts and provenance

- Inputs:
  [Rh-enamide training set](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide)
- Archived outputs:
  [result JSONs](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results),
  [saved force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/forcefields),
  [raw timing evidence](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/logs),
  and the
  [runner script](https://github.com/ericchansen/q2mm/blob/master/scripts/run_rh_enamide_selected_matrix.sh)

Use the archived raw CLI log and timings table for end-to-end wall-clock
comparisons. The per-result `optimized.elapsed_s` values in the JSON files are
optimizer-local timings, not whole-job timings.

## Reproducing

```bash
python3 -m q2mm.diagnostics.cli --system rh-enamide --preflight
scripts/run_rh_enamide_selected_matrix.sh benchmark_results/rh_enamide_selected_<date>
```

The helper script exists because `q2mm-benchmark --system rh-enamide` defaults
to MM3-only forms; the harmonic and JAX-MD cases in this page must be requested
explicitly.
