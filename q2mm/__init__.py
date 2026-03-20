"""Q2MM: Quantum-guided molecular mechanics force field optimization.

Subpackages
-----------
backends    — QM and MM engine integrations (OpenMM, Tinker)
models      — Clean domain objects (molecules, force fields, parameters)
optimizers  — Objective functions and scipy-based optimizers
parsers     — File format parsers (Gaussian, Jaguar, Mol2, MM3, AMBER, Tinker)
"""

try:
    from importlib.metadata import version

    __version__ = version("q2mm")
except Exception:
    __version__ = "0.0.0.dev0"  # fallback for editable/uninstalled

# Public API — the most commonly used classes at the top level
from q2mm.models.molecule import Q2MMMolecule  # noqa: E402
from q2mm.models.forcefield import ForceField, BondParam, AngleParam  # noqa: E402
from q2mm.models.seminario import estimate_force_constants  # noqa: E402
from q2mm.optimizers.objective import ReferenceData, ObjectiveFunction  # noqa: E402

__all__ = [
    "Q2MMMolecule",
    "ForceField",
    "BondParam",
    "AngleParam",
    "estimate_force_constants",
    "ReferenceData",
    "ObjectiveFunction",
    "__version__",
]
