"""Q2MM benchmark systems, run profiles, acceptance, and the runner.

This package composes the lower layers (models, backends, objectives,
optimizers, workflows) into the benchmark application.  Import the concrete
modules you need directly — there is no facade:

- :mod:`q2mm.benchmarks.cases` — :class:`~q2mm.benchmarks.cases.BenchmarkCase`,
  the immutable dataset/publication/reporting wrapper around an
  :class:`~q2mm.models.problem.OptimizationProblem`.
- :mod:`q2mm.benchmarks.systems` — the lazy key -> module registry
  (``load_system``, ``SYSTEM_KEYS``) with one module per scientific system.
- :mod:`q2mm.benchmarks.profiles` — the immutable
  :class:`~q2mm.benchmarks.profiles.RunProfile`, the provenance-complete
  :class:`~q2mm.benchmarks.profiles.ResolvedProfile`, and the optimizer
  catalog.
- :mod:`q2mm.benchmarks.acceptance` — the closed candidate-status vocabulary
  and the single no-progress / worsening acceptance decision.
- :mod:`q2mm.benchmarks.runner` — the one execution/result/persistence/
  promotion path (``run_profile``/``run_profiles``) shared by the CLI's
  single, batch, and matrix operations.
- :mod:`q2mm.benchmarks.cli` — the ``q2mm-benchmark`` console entry point
  (``list``/``preflight``/``single``/``batch``/``matrix``/``load``).
"""
