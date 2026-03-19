"""Backward-compatibility shim — import from q2mm directly instead.

.. deprecated::
    This module re-exports ``linear_algebra`` as ``linalg`` for backward
    compatibility.  New code should use ``from q2mm import linear_algebra``
    or ``from q2mm.linear_algebra import <symbol>`` directly.
"""

import warnings as _warnings

_warnings.warn(
    "q2mm.core is deprecated. "
    "Import from q2mm.linear_algebra directly.",
    DeprecationWarning,
    stacklevel=2,
)

from q2mm import linear_algebra as linalg  # noqa: E402, F401

__all__ = ["linalg"]
