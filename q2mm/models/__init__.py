"""Q2MM internal data models.

Provides clean, format-agnostic representations for molecules and force fields.
These decouple Q2MM's core algorithms from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod, etc.).

This package must never import :mod:`q2mm.io` (or any other outer-layer
package) — ``q2mm.io`` parsers/serializers depend on ``q2mm.models``, not
the other way around. Format-specific loaders/savers (``load_mm3_fld``,
``save_mm3_fld``, ``load_tinker_prm``, ``save_tinker_prm``, etc.) live in
``q2mm.io`` and must be imported from there directly.
"""

from q2mm.models.forcefield import ForceField, BondParam, AngleParam, CmapGrid, FunctionalForm  # noqa: F401
from q2mm.models.molecule import Molecule  # noqa: F401
