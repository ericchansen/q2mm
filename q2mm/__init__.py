"""Q2MM: Quantum-guided molecular mechanics force field optimization.

The package root intentionally exposes **only** version metadata.  Import
the concrete submodules you need directly — there is no top-level facade:

- :mod:`q2mm.models` — immutable domain objects (molecules, force fields,
  parameters, observations, problems, results).
- :mod:`q2mm.io` — file-format readers/writers.
- :mod:`q2mm.backends` — MM/QM engine contracts and implementations.
- :mod:`q2mm.objectives` — objective plans, executors, and metrics.
- :mod:`q2mm.optimizers` — SciPy/Optax/JaxOpt optimizers.
- :mod:`q2mm.workflows` — multi-stage parameterization workflows.
- :mod:`q2mm.benchmarks` — benchmark systems, run profiles, and the runner.

Run benchmarks from the ``q2mm-benchmark`` CLI or compose
:class:`q2mm.benchmarks.profiles.RunProfile` with
:func:`q2mm.benchmarks.runner.run_profiles`.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("q2mm")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # fallback for an uninstalled source checkout

__all__ = ["__version__"]
