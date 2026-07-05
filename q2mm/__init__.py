"""Q2MM: Quantum-guided molecular mechanics force field optimization.

Subpackages:
    backends: QM and MM engine integrations (OpenMM, Tinker, JAX, JAX-MD, Psi4).
    models: Clean domain objects (molecules, force fields, parameters).
    optimizers: Objective functions and optimizers (SciPy, Optax, JaxOpt).
    io: File format readers/writers (Gaussian, Jaguar, Mol2, MM3, AMBER, Tinker).
    workflows: Multi-stage parameterisation protocols (single-stage, Method E2).

Quick start
-----------

::

    from q2mm import benchmark

    result = benchmark("rh-enamide")
    print(result.summary["improvement_pct"])
    result.final_ff.to_mm3_fld("rh-enamide-optimized.fld")

See :mod:`q2mm.benchmark_runner` for the full :func:`benchmark` API.
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
from q2mm.models.molecule import Q2MMMolecule  # noqa: E402
from q2mm.models.seminario import qfuerza_fresh, qfuerza_into  # noqa: E402
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData  # noqa: E402

__all__ = [
    "AngleParam",
    "BatchOutcome",
    "benchmark",
    "BenchmarkRunResult",
    "BondParam",
    "ForceField",
    "ObjectiveFunction",
    "Q2MMMolecule",
    "qfuerza_fresh",
    "qfuerza_into",
    "ReferenceData",
    "run_benchmark_batch",
    "__version__",
]
