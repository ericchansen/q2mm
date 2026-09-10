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
  :class:`~q2mm.benchmarks.profiles.ResolvedProfile`, and runtime configuration
  adapters over the optimizer catalog and existing data-root resolver.
- :mod:`q2mm.benchmarks.acceptance` — the closed candidate-status vocabulary
  and the single no-progress / worsening and executor-ratio decisions.
- :mod:`q2mm.benchmarks.analysis` — frequency, PES-distortion, and
  optimizer-sample diagnostics, delegating objective metrics to their owner.
- :mod:`q2mm.benchmarks.records` — immutable candidate/outcome records and
  benchmark projections over canonical result and scientific-identity APIs.
- :mod:`q2mm.benchmarks.artifacts` — record storage and copy-snapshot promotion
  mechanics, reusing application serialization and file helpers.
- :mod:`q2mm.benchmarks.runner` — the one execution/promotion coordinator
  (``run_profile``/``run_profiles``) shared by single, batch, and matrix.
- :mod:`q2mm.benchmarks.cli` — the ``q2mm-benchmark`` console entry point
  (``list``/``preflight``/``single``/``batch``/``matrix``/``load``).
"""
