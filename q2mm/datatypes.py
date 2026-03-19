"""Backward-compatibility shim — import from q2mm.parsers or q2mm.models instead.

.. deprecated::
    This module re-exports symbols for backward compatibility with code that
    predates the modular package layout. New code should import directly from
    the canonical modules (q2mm.parsers.mm3, q2mm.models.datum, etc.).
"""

import warnings as _warnings

_warnings.warn(
    "q2mm.datatypes is deprecated. "
    "Import from q2mm.parsers or q2mm.models directly.",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np  # noqa: F401 — used by replace_minimum below

# Data / parameter classes
from q2mm.models.datum import Datum, remove_none, datum_sort_key  # noqa: F401
from q2mm.models.param import (  # noqa: F401
    ParamError,
    ParamFE,
    ParamBE,
    Param,
    ParamMM3,
    COM_POS_START,
    P_1_START,
    P_1_END,
    P_2_START,
    P_2_END,
    P_3_START,
    P_3_END,
)

# Force field classes
from q2mm.parsers.base import FF  # noqa: F401
from q2mm.parsers.amber_ff import AmberFF  # noqa: F401
from q2mm.parsers.tinker_ff import TinkerFF, TinkerMM3A  # noqa: F401
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

# Structures (used by old code that imports from datatypes)
from q2mm.parsers.structures import check_mm_dummy, get_dummy_hessian_indices  # noqa: F401

# Hessian utilities
from q2mm.models.hessian import mass_weight_hessian, mass_weight_eigenvectors  # noqa: F401


def replace_minimum(array, value=1):
    """Replace the minimum value in *array* with *value*.

    Kept here because it is only used by calculate.py (legacy plumbing).
    """
    min_idx = np.argmin(array)
    array[min_idx] = value
    return array
