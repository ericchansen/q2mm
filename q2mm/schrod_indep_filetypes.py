"""Backward-compatibility shim.

All classes and functions formerly in this module have moved to
``q2mm.parsers``, ``q2mm.models.datum``, ``q2mm.models.param``,
and ``q2mm.models.hessian``.  This file re-exports every public name
so existing ``from q2mm.schrod_indep_filetypes import X`` continues
to work.
"""

# Structures
from q2mm.parsers.structures import (  # noqa: F401
    Atom,
    DOF,
    Bond,
    Angle,
    Torsion,
    Structure,
    check_mm_dummy,
    get_dummy_hessian_indices,
)

# Base classes
from q2mm.parsers.base import File, FF  # noqa: F401

# Parsers
from q2mm.parsers.mol2 import Mol2  # noqa: F401
from q2mm.parsers.gaussian import GaussLog  # noqa: F401
from q2mm.parsers.jaguar import JaguarIn, JaguarOut  # noqa: F401
from q2mm.parsers.macromodel import MacroModel, MacroModelLog, geo_from_points  # noqa: F401

# Force fields
from q2mm.parsers.mm3 import (  # noqa: F401
    MM3,
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

# Data / parameter classes
from q2mm.models.datum import Datum, remove_none, datum_sort_key  # noqa: F401
from q2mm.models.param import (  # noqa: F401
    ParamError,
    ParamFE,
    ParamBE,
    Param,
    ParamMM3,
    ParAMBER,
    COM_POS_START,
    P_1_START,
    P_1_END,
    P_2_START,
    P_2_END,
    P_3_START,
    P_3_END,
)

# Hessian utilities
from q2mm.models.hessian import (  # noqa: F401
    mass_weight_hessian,
    mass_weight_force_constant,
    mass_weight_eigenvectors,
)
