# Benchmarks

This section is the guide to q2mm's benchmark and validation evidence. It is
organized by reader question rather than by raw artifact location. Start here
if you want to know which pages exist, what each one covers, and how much of
the benchmark program is complete today.

## Benchmark program status

| Page | Primary question | Current scope | Status |
|------|------------------|---------------|--------|
| [Small Molecules](small-molecules.md) | How do the supported backend/form/optimizer combinations compare on a tractable system? | Full CH₃F matrix: 82 supported combos across JAX, JAX-MD, OpenMM, and Tinker (including optax, jaxopt, basin-hopping, multi-start, L2-regularized, and composed optimizers) | **Complete** |
| [Rh-Enamide](rh-enamide.md) | What does q2mm currently achieve on a realistic large-system case study? | Selected overnight GPU matrix: 13 attempted combos on the 182-parameter Rh training set | **Partial** |
| [GPU Acceleration](gpu.md) | When does GPU acceleration help, and when does CPU still win? | Dedicated CPU-vs-GPU comparisons for CH₃F and Rh-enamide on JAX/JAX-MD | **Complete for the current study set** |
| [When Analytical Wins](crossover.md) | When is the JAX analytical-gradient path worth its compile-time overhead? | Interpretive crossover analysis from existing CH₃F + Rh-enamide rows | **Complete** |
| [QFUERZA Validation](qfuerza-validation.md) | Does our Seminario/QFUERZA implementation reproduce the paper's force constants? | Cisplatin Zenodo validation: QFUERZA angles exact, bond FCs partially diverge | **Partial — Pt-Cl/N-H divergence root cause documented ([#236](https://github.com/ericchansen/q2mm/issues/236), closed)** |
| [Published FF Validation](published-ff-validation.md) | Can q2mm correctly evaluate a published force field? | Check 1 on the published Rh-enamide force field under OpenMM | **Complete; parity gap likely due to MM3 functional-form differences** |

## How to use this section

- Read [Small Molecules](small-molecules.md) if you want the only full
  backend/form/optimizer matrix currently documented in the benchmark section.
- Read [Rh-Enamide](rh-enamide.md) if you want the realistic organometallic case
  study and the large-system results that have actually been archived so far.
- Read [GPU Acceleration](gpu.md) if you want device-scaling guidance rather
  than a full benchmark matrix.
- Read [When Analytical Wins](crossover.md) if you are deciding whether to
  reach for the JAX analytical-gradient path on your problem.
- Read [Published FF Validation](published-ff-validation.md) if you want the
  correctness/parity status against a literature force field.

## What the section demonstrates today

- q2mm has one complete small-system comparison set: the CH₃F full matrix.
- q2mm has a realistic large-system Rh-enamide case study, but not yet a full
  24-combo Rh-enamide matrix.
- GPU benefit is workload-dependent: it helps on larger JAX/JAX-MD workloads,
  but small systems can still be faster on CPU.
- The published-force-field evaluation harness is in place; the Rh-enamide
  MM3 parity gap under OpenMM is attributed to functional-form differences
  between MacroModel and OpenMM.

## What is not covered yet

- A full Rh-enamide optimizer matrix across all supported combinations
  (individual combos like JaxOpt L-BFGS have been run; the full matrix
  exceeds available GPU memory for some combos).
- A broader multi-system published-force-field validation set beyond
  Rh-enamide.

## Artifacts and provenance

The docs describe one benchmark program, but the repo currently stores its
artifacts in more than one historical location:

- Current CH₃F full-matrix artifacts: `benchmarks/ch3f/`
- Archived Rh-enamide benchmark artifacts: `benchmarks/rh-enamide/`
- QFUERZA Zenodo validation data: `benchmarks/qfuerza-zenodo/`
- Dedicated GPU-study notes: `benchmarks/GPU_BENCHMARKS.md`
- Published-force-field validation artifacts: `test/fixtures/published_ff/` and
  `validation/published_ffs/README.md`

The detail pages link directly to the specific artifacts they rely on.
