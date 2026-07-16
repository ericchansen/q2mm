"""Q2MM: Quantum-guided molecular mechanics force field optimization.

Subpackages:
    models: Clean domain objects (molecules, force fields, parameters, observations, problems).
    backends: QM and MM engine integrations (OpenMM, Tinker, JAX, JAX-MD, Psi4).
    optimizers: Objective functions and optimizers (SciPy, Optax, JaxOpt).
    io: File format readers/writers (Gaussian, Jaguar, Mol2, MM3, AMBER, Tinker).
    workflows: Multi-stage parameterisation protocols (single-stage, Method E2).
    benchmarks: Scientific benchmark systems and case metadata.

Quick start
-----------

::

    from q2mm.benchmark_runner import run_benchmark

    result = run_benchmark("rh-enamide")
    print(result.summary["improvement_pct"])

See :mod:`q2mm.benchmark_runner` for the full ``run_benchmark`` API.
"""

try:
    from importlib.metadata import version

    __version__ = version("q2mm")
except Exception:
    __version__ = "0.0.0.dev0"  # fallback for editable/uninstalled

# Public API — the most commonly used classes at the top level
from q2mm.benchmark_runner import (  # noqa: E402
    BatchOutcome,
    BenchmarkRunResult,
    run_benchmark as benchmark,
    run_benchmark_batch,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField  # noqa: E402
from q2mm.models.molecule import Molecule  # noqa: E402
from q2mm.models.observations import ObservationSet  # noqa: E402
from q2mm.models.results import OptimizationResult  # noqa: E402
from q2mm.models.seminario import qfuerza_fresh, qfuerza_into  # noqa: E402
from q2mm.objectives import ObjectivePlan, PythonObjectiveExecutor  # noqa: E402

__all__ = [
    "AngleParam",
    "BatchOutcome",
    "benchmark",
    "BenchmarkRunResult",
    "BondParam",
    "ForceField",
    "ObjectivePlan",
    "ObservationSet",
    "OptimizationResult",
    "PythonObjectiveExecutor",
    "Molecule",
    "qfuerza_fresh",
    "qfuerza_into",
    "run_benchmark_batch",
    "__version__",
]
