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
- Current benchmark scope: selected overnight GPU matrix (13 original combos)
  plus 2 post-fix grad-simp re-runs
- Related pages: [GPU Acceleration](gpu.md) covers dedicated device-scaling
  analysis; [Published FF Validation](published-ff-validation.md) covers the
  literature-force-field parity check

The MM3 template is used directly on OpenMM and JAX. JAX-MD uses a harmonic
copy of the same Seminario-derived parameters, so cross-form comparisons are
useful for workflow guidance but not a statement that the force fields are
identical.

## Completed Rh-enamide matrix to date

The table below includes results from the original overnight selected-matrix run
plus the two new grad-simp runs (JAX MM3 and JAX-MD harmonic) enabled by the
robust eigendecomposition fix. Rows are grouped by status and then by final RMSD.
Use the filters to isolate backend/form/optimizer/status slices.

<div class="benchmark-table-anchor" data-benchmark-table="rh-enamide"></div>

| Backend | Form | Optimizer | Status | Result | MAE | Optimizer time |
|---------|------|-----------|--------|--------|----:|---------------:|
| OpenMM | mm3 | grad-simp | success | 173.1 → 42.7 RMSD | 31.8 | 33,223.1 s |
| JAX | mm3 | grad-simp | success | 173.1 → 42.7 RMSD | 27.4 | 1,471.3 s |
| JAX | harmonic | L-BFGS-B | success | 173.5 → 57.4 RMSD | 35.0 | 2,229.9 s |
| JAX | mm3 | L-BFGS-B | success | 173.1 → 61.5 RMSD | 48.5 | 1,921.8 s |
| JAX | mm3 | Nelder-Mead | success | 173.1 → 66.6 RMSD | 54.9 | 538.6 s |
| JAX-MD | harmonic | grad-simp | success | 41,409.7 → 80.6 RMSD | 49.2 | 1,501.7 s |
| JAX | harmonic | Nelder-Mead | success | 173.5 → 81.2 RMSD | 50.7 | 511.0 s |
| JAX-MD | harmonic | L-BFGS-B | success | 41,409.7 → 86.7 RMSD | 48.2 | 3,181.6 s |
| JAX | harmonic | Powell | failed | Eigenvalues did not converge | — | — |
| JAX | mm3 | Powell | failed | Eigenvalues did not converge | — | — |
| JAX-MD | harmonic | Powell | failed | Eigenvalues did not converge | — | — |
| JAX-MD | harmonic | Nelder-Mead | failed | Eigenvalues did not converge | — | — |

All times are `optimized.elapsed_s` from the archived JSON result files — the
time spent inside the optimizer, not total wall-clock including setup. For
grad-simp, the total wall-clock is significantly longer because each cycle
includes a full baseline evaluation; for example, OpenMM grad-simp took ~31
hours end-to-end per the [archived run log](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/run.log).

## Interpretation

- **JAX MM3 + grad-simp matches the OpenMM best-fit RMSD (42.7 cm⁻¹) in ~25
  minutes** vs OpenMM's ~9 hours of optimizer time — a **~23× speedup** for
  the same final quality. This was enabled by Hessian symmetrisation and
  bound-aware sensitivity analysis that prevent the eigenvalue-convergence
  failures that previously blocked JAX grad-simp.
- **JAX-MD harmonic + grad-simp also now succeeds**, reaching 80.6 RMSD in ~25
  minutes. This improves on JAX-MD's prior best (L-BFGS-B at 86.7 RMSD) and
  finishes in half the time. JAX-MD starts from a much worse initial point
  (41,409 vs 173 RMSD) because of force-field form differences.
- OpenMM CUDA MM3 + grad-simp remains the historical reference result at the
  same 42.7 RMSD, but it is an overnight-scale job. JAX grad-simp now delivers
  equivalent quality as a fast screening run.
- JAX L-BFGS-B and Nelder-Mead are still useful for quick single-pass screening
  when cycling is not needed.
- JAX-MD harmonic L-BFGS-B can recover from a very poor starting point, but it
  does not yet beat the strongest JAX or OpenMM result in this selected set.
- The remaining failure rows (Powell on JAX/JAX-MD) are from the pre-fix
  overnight run. Powell takes unbounded direction steps that push parameters
  well outside physical bounds, which makes it particularly hard to stabilise.
  The eigenvalue fix likely improves Powell robustness, but Powell has not been
  re-run yet.
- JAX-MD Nelder-Mead and JAX harmonic grad-simp also failed with eigenvalue
  errors in the original overnight run and have not been re-run with the fix.
  Given that JAX MM3 grad-simp and JAX-MD harmonic grad-simp both now succeed
  with zero penalty triggers, these are expected to work when re-run.

## Parameter quality

JAX MM3 grad-simp and OpenMM MM3 grad-simp reach the same final RMSD
(42.7 cm⁻¹), but the optimised force field parameters differ. A
systematic comparison of the two parameter sets against Seminario
initial estimates has not yet been committed to the repository.
[#197](https://github.com/ericchansen/q2mm/issues/197) tracked the
initial parity investigation (now closed; the gap is attributed to MM3
functional-form differences between MacroModel and OpenMM).

A secondary factor is Hessian accuracy: JAX uses analytically exact
second derivatives (`jax.hessian`), while OpenMM uses finite-difference
Hessians. The smoother landscape from exact derivatives leads to more targeted
parameter moves and less drift from the starting point.

## Artifacts and provenance

- Inputs:
  [Rh-enamide training set](https://github.com/ericchansen/q2mm/tree/master/examples/rh-enamide)
- Archived outputs:
  [result JSONs](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/results),
  [saved force fields](https://github.com/ericchansen/q2mm/tree/master/benchmarks/rh-enamide/forcefields),
  [raw run log](https://github.com/ericchansen/q2mm/blob/master/benchmarks/rh-enamide/run.log),
  and the
  [runner script](https://github.com/ericchansen/q2mm/blob/master/scripts/run_rh_enamide_selected_matrix.sh)

## Reproducing

```bash
python3 -m q2mm.diagnostics.cli --system rh-enamide --preflight
scripts/run_rh_enamide_selected_matrix.sh benchmarks/rh-enamide
```

The helper script exists because `q2mm-benchmark --system rh-enamide` defaults
to MM3-only forms; the harmonic and JAX-MD cases in this page must be requested
explicitly.
