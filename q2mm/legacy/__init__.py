"""Legacy data structures preserved for backward compatibility.

These modules contain the original Q2MM data representations (Datum, Param,
ParamMM3, etc.) used by the parser layer. New code should prefer the clean
model types in :mod:`q2mm.models`.
"""

from q2mm.legacy.datum import Datum, datum_sort_key, remove_none  # noqa: F401
from q2mm.legacy.param import Param, ParamMM3, ParAMBER  # noqa: F401
