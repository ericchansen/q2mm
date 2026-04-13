"""Shared I/O helper utilities used by multiple format modules.

Private module — import from ``q2mm.io`` sub-modules, not directly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from q2mm import constants as co

if TYPE_CHECKING:
    from q2mm.models.forcefield import (
        AngleParam,
        BondParam,
        ForceField,
        VdwParam,
    )

from q2mm.models.identifiers import (
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)

# ---------------------------------------------------------------------------
# Format compatibility
# ---------------------------------------------------------------------------

_FORMAT_COMPATIBLE_FORMS: dict[str, set[str]] = {
    "mm3_fld": {"mm3"},
    "tinker_prm": {"mm3"},
    "openmm_xml": {"mm3"},
    "amber_frcmod": {"harmonic"},
}


def _validate_form_for_format(ff: ForceField, target_format: str) -> None:
    """Raise ``ValueError`` if the force field's functional form is incompatible with *target_format*."""
    form = getattr(ff, "functional_form", None)
    if form is None:
        return
    form_value = form.value if hasattr(form, "value") else str(form)
    allowed = _FORMAT_COMPATIBLE_FORMS.get(target_format)
    if allowed is not None and form_value not in allowed:
        raise ValueError(
            f"Cannot save a {form!r} force field to {target_format!r} format. Compatible forms: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Env-id splitting
# ---------------------------------------------------------------------------


def _split_env_id(env_id: str, expected_len: int) -> list[str]:
    parts = [part.strip() for part in env_id.split("-") if part.strip()]
    if len(parts) == expected_len:
        return parts
    return []


# ---------------------------------------------------------------------------
# Atom-type cleaning
# ---------------------------------------------------------------------------


def _clean_atom_types(atom_types: list[str] | tuple[str, ...] | None, expected_len: int) -> list[str]:
    if atom_types is None:
        return []
    if isinstance(atom_types, str):
        atom_types = [atom_types]
    cleaned = [
        str(atom_type).strip() for atom_type in atom_types if str(atom_type).strip() and str(atom_type).strip() != "-"
    ]
    return cleaned[:expected_len]


# ---------------------------------------------------------------------------
# Parameter map builders
# ---------------------------------------------------------------------------


def _build_param_maps(params: list, secondary_key: str) -> tuple[dict, dict]:
    """Build ff_row and secondary-key lookup dicts for a list of parameters."""
    by_row = {p.ff_row: p for p in params if p.ff_row is not None}
    by_key = {getattr(p, secondary_key): p for p in params if getattr(p, secondary_key, None)}
    return by_row, by_key


def _build_bond_maps(bonds: list[BondParam]) -> tuple[dict[int, BondParam], dict[str, BondParam]]:
    return _build_param_maps(bonds, "env_id")


def _build_angle_maps(angles: list[AngleParam]) -> tuple[dict[int, AngleParam], dict[str, AngleParam]]:
    return _build_param_maps(angles, "env_id")


def _build_vdw_maps(vdws: list[VdwParam]) -> tuple[dict[int, VdwParam], dict[str, VdwParam]]:
    return _build_param_maps(vdws, "atom_type")


# ---------------------------------------------------------------------------
# Export matching
# ---------------------------------------------------------------------------


def _match_for_export(param: Any, by_row: dict, by_env: dict, expected_len: int, canonicalize_fn: Callable) -> Any:
    """Match a parsed parameter to an internal param by ff_row or env_id."""
    if param.ff_row is not None and param.ff_row in by_row:
        return by_row[param.ff_row]
    atom_types = _clean_atom_types(getattr(param, "atom_types", None), expected_len)
    if len(atom_types) == expected_len:
        return by_env.get(canonicalize_fn(atom_types))
    return None


def _match_bond_for_export(
    param: Any, bond_by_row: dict[int, BondParam], bond_by_env: dict[str, BondParam]
) -> BondParam | None:
    return _match_for_export(param, bond_by_row, bond_by_env, 2, canonicalize_bond_env_id)


def _match_angle_for_export(
    param: Any,
    angle_by_row: dict[int, AngleParam],
    angle_by_env: dict[str, AngleParam],
) -> AngleParam | None:
    return _match_for_export(param, angle_by_row, angle_by_env, 3, canonicalize_angle_env_id)


# ---------------------------------------------------------------------------
# Torsion update helper (used by both MM3 and Tinker save)
# ---------------------------------------------------------------------------


def _update_torsion_param(param: Any, torsions: list) -> None:
    """Update a legacy ``df`` param from the ForceField's torsion list.

    Matches by ``ff_row`` + ``ff_col`` (periodicity).
    """
    periodicity = getattr(param, "ff_col", 1)
    ff_row = getattr(param, "ff_row", None)
    for t in torsions:
        if t.ff_row == ff_row and t.periodicity == periodicity:
            param.value = t.force_constant
            return


# ---------------------------------------------------------------------------
# Legacy parameter container
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class ParamError(Exception):
    """Raised when a parameter value is outside its allowed range."""


class Param:
    """A single force field parameter for optimization.

    Represents a parameter (bond length, angle, force constant, charge, etc.)
    from any supported force field format (MM3, AMBER, Tinker).

    Attributes:
        ptype: Parameter type ('ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2',
            'sb', 'q', 'vdwr', 'vdwfc').
        d1: First derivative w.r.t. the penalty function.
        d2: Second derivative w.r.t. the penalty function.
        ff_type: Force field unit system (constants.MM3FF, constants.AMBERFF, etc.).
        atom_labels: Atom label strings from the FF file.
        atom_types: Atom type identifiers.
        ff_col: Column index in the FF file (1, 2, or 3).
        ff_row: Row number in the FF file.
        label: FF-specific label (e.g., first 2 chars of an MM3 line).

    """

    __slots__ = [
        "_allowed_range",
        "_step",
        "_value",
        "atom_labels",
        "atom_types",
        "d1",
        "d2",
        "ff_col",
        "ff_row",
        "ff_type",
        "label",
        "ptype",
    ]

    def __init__(
        self,
        ptype: str | None = None,
        value: float | None = None,
        d1: float | None = None,
        d2: float | None = None,
        ff_type: str | None = None,
        atom_labels: list | None = None,
        atom_types: list | None = None,
        ff_col: int | None = None,
        ff_row: int | None = None,
        label: str | None = None,
    ) -> None:
        """Initialize a Param instance.

        Args:
            ptype (str | None): Parameter type identifier (e.g., ``'ae'``,
                ``'bf'``, ``'q'``).
            value (float | None): Initial parameter value.
            d1 (float | None): First derivative w.r.t. the penalty function.
            d2 (float | None): Second derivative w.r.t. the penalty function.
            ff_type (str | None): Force field unit system identifier.
            atom_labels (list | None): Atom label strings from the FF file.
            atom_types (list | None): Atom type identifiers.
            ff_col (int | None): Column index in the FF file.
            ff_row (int | None): Row number in the FF file.
            label (str | None): FF-specific label string.

        """
        self._allowed_range = None
        self._step = None
        self._value = None
        self.ptype = ptype
        self.d1 = d1
        self.d2 = d2
        self.ff_type = ff_type
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.ff_col = ff_col
        self.ff_row = ff_row
        self.label = label
        if value is not None:
            self.value = value

    def __repr__(self) -> str:
        val = self._value if self._value is not None else "?"
        if self.ff_row is not None:
            return f"Param[{self.ptype}][{self.ff_row},{self.ff_col}]({val})"
        return f"Param[{self.ptype}]({val})"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Param):
            return NotImplemented
        # Parameters with incomplete identity are not meaningfully comparable.
        if (
            self.ptype is None
            or self.ff_row is None
            or self.ff_col is None
            or other.ptype is None
            or other.ff_row is None
            or other.ff_col is None
        ):
            return NotImplemented
        return (self.ptype, self.ff_row, self.ff_col) == (other.ptype, other.ff_row, other.ff_col)

    def __hash__(self) -> int:
        """Hash based on (ptype, ff_row, ff_col) identity.

        These fields are set at construction by the FF parsers and should not
        be mutated after the Param is used in a set or as a dict key.

        Raises:
            TypeError: If any identity field is None (incomplete parameter).

        """
        if self.ptype is None or self.ff_row is None or self.ff_col is None:
            raise TypeError(
                f"Param with incomplete identity ({self.ptype}, {self.ff_row}, {self.ff_col}) is unhashable"
            )
        return hash((self.ptype, self.ff_row, self.ff_col))

    @property
    def allowed_range(self) -> tuple[float, float]:
        """Allowed value range based on parameter type.

        Force constants and charges (bf, af, df, q) allow negative values
        for transition-state force fields (TSFF). Equilibrium geometry
        parameters (be, ae) and others are non-negative.

        Returns:
            (tuple[float, float]): A ``(lower, upper)`` bound pair.

        """
        if self._allowed_range is None:
            if self.ptype in ("q", "df", "bf", "af"):
                self._allowed_range = (-float("inf"), float("inf"))
            else:
                self._allowed_range = (0.0, float("inf"))
        return self._allowed_range

    @property
    def step(self) -> float:
        """Step size for numerical differentiation.

        If the stored step is a string, it is treated as a fraction of the
        current value (e.g., ``"0.01"`` means 1% of value).

        Returns:
            (float): The absolute step size.

        """
        if self._step is None:
            try:
                self._step = co.STEPS[self.ptype]
            except KeyError:
                logger.warning(f"{self} has no default step size and none was provided!")
                raise
        if isinstance(self._step, str):
            if self.value is None:
                raise ValueError(f"{self} has a percentage-based step but value is None.")
            return float(self._step) * self.value
        return self._step

    @step.setter
    def step(self, x: float | str) -> None:
        """Set the finite-difference step (absolute float or percentage string)."""
        self._step = x

    @property
    def value(self) -> float | None:
        """Current parameter value.

        Returns:
            (float | None): The parameter value, or None if unset.

        """
        return self._value

    @value.setter
    def value(self, value: float | None) -> None:
        """Set parameter value after range validation and normalization."""
        if value is None:
            self._value = None
            return
        if self.ptype == "ae":
            value = self._normalize_angle(value)
        if self.value_in_range(value):
            self._value = value

    @staticmethod
    def _normalize_angle(value: float) -> float:
        """For equilibrium angles, fold values back into [0, 180].

        Args:
            value (float): The raw angle value in degrees.

        Returns:
            (float): The angle normalized to the range [0, 180].

        """
        if value > 180.0:
            v = value % 360.0
            return 360.0 - v if v > 180.0 else v
        return value

    def value_in_range(self, value: float) -> bool:
        """Check whether a value falls within the allowed range for this parameter type.

        Args:
            value (float): The value to validate.

        Returns:
            (bool): True if the value is within the allowed range.

        Raises:
            ParamError: If value is outside the allowed range.

        """
        lo, hi = self.allowed_range
        if lo <= value <= hi:
            return True
        raise ParamError(f"{self} value {value} outside allowed range [{lo}, {hi}]")
