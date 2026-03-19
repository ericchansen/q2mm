"""File format parsers for Q2MM.

Provides parsers for various computational chemistry file formats and
force field parameter files. Import convenience:

    from q2mm.parsers import GaussLog, Mol2, MacroModel, Structure

Submodules:
    structures  — Atom, Bond, Angle, Torsion, Structure
    base        — File base class
    gaussian    — GaussLog parser
    mol2        — Mol2 parser
    jaguar      — JaguarIn, JaguarOut parsers
    macromodel  — MacroModel, MacroModelLog parsers
"""

from q2mm.parsers.structures import (  # noqa: F401
    Atom,
    Angle,
    Bond,
    DOF,
    Structure,
    Torsion,
    check_mm_dummy,
    get_dummy_hessian_indices,
)
from q2mm.parsers.base import File, FF  # noqa: F401
from q2mm.parsers.gaussian import GaussLog  # noqa: F401
from q2mm.parsers.mol2 import Mol2  # noqa: F401
from q2mm.parsers.jaguar import JaguarIn, JaguarOut  # noqa: F401
from q2mm.parsers.macromodel import MacroModel, MacroModelLog  # noqa: F401
from q2mm.parsers.mm3 import MM3  # noqa: F401
from q2mm.parsers.mm3 import (  # noqa: F401
    match_mm3_label,
    match_mm3_vdw,
    match_mm3_bond,
    match_mm3_angle,
    match_mm3_stretch_bend,
    match_mm3_torsion,
    match_mm3_lower_torsion,
    match_mm3_higher_torsion,
    match_mm3_improper,
)
from q2mm.parsers.amber_ff import AmberFF  # noqa: F401
from q2mm.parsers.tinker_ff import TinkerFF, TinkerMM3A  # noqa: F401
# Legacy re-exports from q2mm.models — prefer importing directly from q2mm.models
from q2mm.models.datum import Datum, remove_none, datum_sort_key  # noqa: F401
from q2mm.models.param import Param, ParamMM3, ParAMBER  # noqa: F401
from q2mm.models.hessian import (  # noqa: F401
    mass_weight_hessian,
    mass_weight_eigenvectors,
    mass_weight_force_constant,
)
