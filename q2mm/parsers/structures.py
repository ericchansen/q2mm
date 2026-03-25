"""Deprecated compatibility shim — import from :mod:`q2mm.models.structure`.

.. deprecated::
    This module is deprecated. All classes have moved to
    :mod:`q2mm.models.structure`.  Update imports to::

        from q2mm.models.structure import Atom, Bond, Angle, Torsion, DOF, Structure

    This shim will be removed in a future release.
"""

import warnings as _warnings

_warnings.warn(
    "q2mm.parsers.structures is deprecated. Import from q2mm.models.structure instead.",
    DeprecationWarning,
    stacklevel=2,
)

from q2mm.models.structure import (  # noqa: F401, E402
    Angle,
    Atom,
    Bond,
    DOF,
    Structure,
    Torsion,
)
