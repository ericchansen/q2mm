# Performance Reference

Benchmarks collected on a Windows desktop (AMD Ryzen / Intel i7-class, 32 GB RAM)
with Python 3.12, OpenMM 8.x, Tinker 25.6, and Psi4 1.10. All times are
wall-clock. Your mileage will vary with hardware and molecule size.

---

## Single-Point Energy Evaluation

How long does one energy call take? This is the inner loop of optimization —
every objective function evaluation requires one energy call per molecule per
geometry.

| Backend | Molecule | Cold Start | Warm Avg | Throughput |
|---------|----------|------------|----------|------------|
| **OpenMM** | Water (3 atoms) | 248 ms | **104 ms** | ~10 eval/s |
| **Tinker** | Water (3 atoms) | 168 ms | **163 ms** | ~6 eval/s |

**Key takeaways:**

- **OpenMM is ~1.6× faster** than Tinker for single-point energy. OpenMM runs
  in-process (no subprocess overhead), while Tinker spawns a process, writes
  files, and parses output for every call.
- **Cold start** includes system/topology building (OpenMM) or first file I/O
  (Tinker). Subsequent calls are faster for OpenMM but not for Tinker (each
  call is a fresh subprocess).
- For larger molecules, OpenMM's advantage grows because its in-process
  evaluation scales better than Tinker's file-based I/O.

### Cross-Backend Parity

Both backends agree within **0.003 kcal/mol** at displaced geometries for the
same MM3 functional form. At equilibrium geometry, both give exactly
0.000 kcal/mol.

---

## Optimizer Performance

Full optimization run: perturbed water FF (4 parameters: bond k, bond eq,
angle k, angle eq) optimized to recover known true parameters.

### OpenMM Backend

| Method | Time | Evaluations | Evals/s | Initial → Final Score |
|--------|------|-------------|---------|----------------------|
| **Nelder-Mead** | 20.3 s | 189 | 9.3 | 92.9 → 0.000 |
| **L-BFGS-B** | 10.9 s | 100 | 9.1 | ~0 → 0.000 |
| **Powell** | 29.4 s | 271 | 9.2 | ~0 → 0.000 |

### Tinker Backend

| Method | Time | Evaluations | Evals/s | Initial → Final Score |
|--------|------|-------------|---------|----------------------|
| **Nelder-Mead** | 45.0 s | 275 | 6.1 | 92.9 → 0.000 |

**Key takeaways:**

- **Evaluation throughput is constant** regardless of optimizer method (~9–10
  eval/s for OpenMM, ~6 eval/s for Tinker). The bottleneck is the energy call.
- **Nelder-Mead** needs ~190–275 evaluations (derivative-free, no gradient
  needed).
- **L-BFGS-B** uses finite-difference gradients. Each "iteration" costs
  `2 × n_params + 1` evaluations. With 4 params, that's 9 evaluations per
  step, so 100 evaluations ≈ 11 gradient steps.
- **For 4 parameters**, all methods converge in under 60 seconds with either
  backend.
- **Scaling estimate**: a 20-parameter FF with 5 molecules would need
  ~5× more evaluations per objective call (5 molecules), and gradient methods
  would need ~41 evaluations per step (2×20+1). Expect ~5–15 minutes per
  optimization with OpenMM.

---

## Seminario Method

Extracting bond/angle force constants from a QM Hessian matrix.

| Molecule | Atoms | Time |
|----------|-------|------|
| Water | 3 | **0.4 ms** |
| Rh-enamide TS | ~60 | **< 50 ms** (estimated) |

The Seminario method is pure NumPy linear algebra (eigenvalue decomposition of
3×3 Hessian sub-blocks). It's effectively instant compared to everything else
in the pipeline. Even a 200-atom molecule would take < 1 second.

---

## QM Calculations (Psi4)

These are typically run once per molecule to generate reference data, not during
the optimization loop.

| Calculation | Level | Molecule | Time |
|------------|-------|----------|------|
| **Energy** | B3LYP/6-31G* | Water (3 atoms) | 1.1 s |
| **Hessian** | B3LYP/6-31G* | Water (3 atoms) | 7.8 s |

**Scaling notes:**

- QM cost scales as O(N³)–O(N⁴) with basis functions. A 30-atom organic
  molecule with 6-31G* takes ~5–30 minutes for a Hessian.
- Transition state Hessians (for TSFF work) take the same time but contain
  one negative eigenvalue along the reaction coordinate.
- Psi4 parallelizes well — set `psi4.set_num_threads(N)` for multi-core
  speedup.

---

## Test Suite Timing

| Test Category | Tests | Time | Notes |
|---------------|-------|------|-------|
| Unit tests (models, parsers) | ~45 | **< 2 s** | Pure Python, no backends |
| Integration (Seminario parity) | ~25 | **~1 s** | Fixture-based, no live QM/MM |
| Integration (scipy optimizer) | 10 | **~220 s** | Real OpenMM evaluations |
| Integration (cross-backend) | 10 | **~19 min** | OpenMM + Tinker, full pipelines |
| **Total (without cross-backend)** | **114** | **~3:47** | Default `pytest -q` |
| **Total (all)** | **124** | **~23 min** | All integration tests |

**Recommendation:** For development iteration, skip the cross-backend validation
tests:

```bash
# Fast iteration (3–4 minutes)
pytest -q --ignore=test/integration/test_optimization_validation.py

# Full validation (before PR/merge)
pytest -q
```

---

## Bottleneck Analysis

For a typical TSFF optimization workflow:

```
QM Hessian (one-time)    ████████████████  7.8 s
Seminario (one-time)     ▏                 0.0004 s
Optimization loop        ████████████████████████████████████  20–45 s
  └─ per evaluation      ██                0.1–0.16 s (energy call)
```

**The energy evaluation is the bottleneck.** Every optimization evaluation
calls the MM backend once per molecule per geometry. Strategies to speed up:

1. **Use OpenMM over Tinker** — 1.6× faster per evaluation (in-process vs
   subprocess).
2. **Reduce evaluations** — L-BFGS-B converges in fewer evaluations than
   Nelder-Mead for smooth objectives, but each "evaluation" is actually
   `2N+1` energy calls for finite-difference gradients.
3. **Fewer molecules/geometries** — each additional training point adds one
   energy call per evaluation.
4. **Analytical gradients** (future) — would eliminate the `2N+1` overhead
   for gradient methods, reducing cost from O(N) to O(1) energy calls per
   optimization step.

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `TINKER_DIR` | Tinker installation directory | `C:\tinker` |
| `TINKER_PRM` | Default Tinker parameter file | `C:\tinker\params\mm3.prm` |

If not set, Tinker paths are auto-detected from common install locations.

---

*Last updated: March 2026. Benchmarks on AMD/Intel desktop, Windows 11,
Python 3.12.*
