"""Immutable parameter layout and active/frozen projection for Q2MM.

This module is the *only* source of truth for how a
:class:`~q2mm.models.forcefield.ForceField`'s scalar parameters are
flattened into a vector, and for which of those scalars are "active"
(optimizable) versus held fixed at a caller-chosen baseline.

- :class:`ParameterKind` / :class:`ParameterUnit` describe the physical
  role and unit of one scalar slot, plus its canonical sanity bounds and
  finite-difference step (the successor to the legacy
  ``ForceField.DEFAULT_BOUNDS`` / ``optimizers.defaults.STEPS`` tables).
- :class:`ParameterId` is a stable semantic identity for one scalar field
  of one parameter row: it never depends on Python object identity/hash
  or dict/set iteration order, so the same logical parameter gets the
  same ID across independently-constructed force fields.
- :class:`ParameterSlot` binds one :class:`ParameterId` to its full-vector
  index, kind/unit metadata, a human-readable name, and the internal
  ``(owner, owner_index, field)`` locator used to read/replace that
  scalar on a :class:`~q2mm.models.forcefield.ForceField`.
- :class:`ParameterLayout` is the ordered, immutable tuple of slots for
  one force-field *structure* (topology of bonds/angles/.../UB terms).
  It preserves the legacy full-vector order exactly: bonds (k, eq),
  angles (k, eq), torsions (k), stretch-bends (k), vdW (radius,
  epsilon), then Urey-Bradley (k, eq) for angles that carry a UB term.
- :class:`ActiveParameterSpace` is the *only* active/frozen projection.
  It stores a baseline full vector plus the active slot indices and
  provides pack/expand/bounds/steps and derived-subspace operations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.identifiers import canonicalize_torsion_env_id

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField

__all__ = [
    "ParameterKind",
    "ParameterUnit",
    "ParameterId",
    "ParameterSlot",
    "ParameterLayout",
    "ActiveParameterSpace",
    "OptSubstructureMembership",
    "opt_substructure_membership",
    "fractional_bounds",
]

# Canonical JSON/fingerprint schema version.  Bump when the slot payload
# shape changes so old and new fingerprints are never conflated.
_FINGERPRINT_SCHEMA_VERSION = 1


class ParameterUnit(str, Enum):
    """Canonical physical unit of one parameter slot."""

    KCAL_PER_MOL_PER_ANGSTROM2 = "kcal/mol/angstrom**2"
    ANGSTROM = "angstrom"
    KCAL_PER_MOL_PER_RADIAN2 = "kcal/mol/radian**2"
    DEGREE = "degree"
    KCAL_PER_MOL_PER_ANGSTROM_PER_RADIAN = "kcal/mol/angstrom/radian"
    KCAL_PER_MOL = "kcal/mol"


class ParameterKind(str, Enum):
    """Physical role of one scalar parameter slot.

    Values match the legacy per-type label strings (``ForceField.
    get_param_type_labels()`` / ``DEFAULT_BOUNDS`` keys) so downstream
    reporting code that keyed off those strings keeps working unchanged.
    """

    BOND_FORCE_CONSTANT = "bond_k"
    BOND_EQUILIBRIUM = "bond_eq"
    ANGLE_FORCE_CONSTANT = "angle_k"
    ANGLE_EQUILIBRIUM = "angle_eq"
    TORSION_FORCE_CONSTANT = "torsion_k"
    STRETCH_BEND_FORCE_CONSTANT = "sb_k"
    VDW_RADIUS = "vdw_radius"
    VDW_EPSILON = "vdw_epsilon"
    UREY_BRADLEY_FORCE_CONSTANT = "ub_k"
    UREY_BRADLEY_EQUILIBRIUM = "ub_eq"


@dataclass(frozen=True)
class _KindMeta:
    unit: ParameterUnit
    bounds: tuple[float, float]
    step: float


# Canonical sanity bounds + finite-difference steps per kind.  Bounds are
# the successor to ``ForceField.DEFAULT_BOUNDS``; steps are the successor
# to ``optimizers.defaults.STEPS`` (via ``ForceField._PARAM_TYPE_TO_STEP_KEY``).
# Bond/angle/UB force constants are non-negative because TS Hessians are
# curvature-inverted before Seminario/QFUERZA projection (see
# ``q2mm.models.hessian.invert_ts_curvature``); torsions/stretch-bends may
# legitimately be negative.
_KIND_METADATA: dict[ParameterKind, _KindMeta] = {
    ParameterKind.BOND_FORCE_CONSTANT: _KindMeta(ParameterUnit.KCAL_PER_MOL_PER_ANGSTROM2, (0.0, 3600.0), 7.2),
    ParameterKind.BOND_EQUILIBRIUM: _KindMeta(ParameterUnit.ANGSTROM, (0.5, 3.0), 0.02),
    ParameterKind.ANGLE_FORCE_CONSTANT: _KindMeta(ParameterUnit.KCAL_PER_MOL_PER_RADIAN2, (0.0, 720.0), 7.2),
    ParameterKind.ANGLE_EQUILIBRIUM: _KindMeta(ParameterUnit.DEGREE, (30.0, 180.0), 1.0),
    ParameterKind.TORSION_FORCE_CONSTANT: _KindMeta(ParameterUnit.KCAL_PER_MOL, (-20.0, 20.0), 0.1),
    ParameterKind.STRETCH_BEND_FORCE_CONSTANT: _KindMeta(
        ParameterUnit.KCAL_PER_MOL_PER_ANGSTROM_PER_RADIAN, (-50.0, 50.0), 0.2
    ),
    ParameterKind.VDW_RADIUS: _KindMeta(ParameterUnit.ANGSTROM, (0.5, 5.0), 0.1),
    ParameterKind.VDW_EPSILON: _KindMeta(ParameterUnit.KCAL_PER_MOL, (0.001, 2.0), 0.02),
    ParameterKind.UREY_BRADLEY_FORCE_CONSTANT: _KindMeta(ParameterUnit.KCAL_PER_MOL_PER_ANGSTROM2, (0.0, 500.0), 7.2),
    ParameterKind.UREY_BRADLEY_EQUILIBRIUM: _KindMeta(ParameterUnit.ANGSTROM, (1.0, 4.0), 0.02),
}

# Force-constant vs. equilibrium-type kinds, for `fractional_bounds`.
_FC_KINDS: frozenset[ParameterKind] = frozenset(
    {
        ParameterKind.BOND_FORCE_CONSTANT,
        ParameterKind.ANGLE_FORCE_CONSTANT,
        ParameterKind.TORSION_FORCE_CONSTANT,
        ParameterKind.STRETCH_BEND_FORCE_CONSTANT,
        ParameterKind.VDW_EPSILON,
        ParameterKind.UREY_BRADLEY_FORCE_CONSTANT,
    }
)
_EQ_KINDS: frozenset[ParameterKind] = frozenset(
    {
        ParameterKind.BOND_EQUILIBRIUM,
        ParameterKind.ANGLE_EQUILIBRIUM,
        ParameterKind.VDW_RADIUS,
        ParameterKind.UREY_BRADLEY_EQUILIBRIUM,
    }
)

# Urey-Bradley scalar field names — used to distinguish an angle's
# bending slots (owner="angles", field in {"force_constant",
# "equilibrium"}) from its UB slots (same owner, different fields).
_UB_FIELDS: frozenset[str] = frozenset({"ub_force_constant", "ub_equilibrium"})


@dataclass(frozen=True)
class ParameterId:
    """Stable semantic identity for one scalar field of one parameter row.

    Deterministic across independently-constructed force fields with the
    same structure: built purely from the parameter's *family*
    (``"bond"``, ``"angle"``, ``"torsion"``, ``"stretch_bend"``,
    ``"vdw"``, or ``"urey_bradley"``), its normalized row-discriminating
    identity, a 0-based *occurrence* disambiguating multiple rows that
    are still identical after all other fields are considered (genuine
    exact duplicates), and the *field* name identifying which scalar of
    the row this ID refers to (e.g. ``"force_constant"`` vs.
    ``"equilibrium"``).  Never depends on Python ``id()``/``hash()`` or
    dict/set iteration order.

    The row-discriminating ``identity`` tuple includes every field that
    can distinguish two rows sharing the same chemical element key, so a
    context-specific/bond-order/source-row variant can never be silently
    reordered or swapped with another row without changing the ID (and
    therefore the layout ``fingerprint``):

    - bond: element key, environment ID, MM3 bond order, MM3 context
      flags, source ``ff_row``.
    - angle / stretch-bend: element key, environment ID, source
      ``ff_row``.
    - torsion: canonical (direction-independent) elements, periodicity,
      improper flag, canonicalized environment ID, source ``ff_row``.
    - vdW: atom type, element, source ``ff_row``.

    ``None``-valued fields (e.g. an unset ``ff_row``) render as the
    explicit :data:`_NONE_IDENTITY_SENTINEL` token rather than Python's
    ``str(None)``.
    """

    family: str
    identity: tuple[str, ...]
    occurrence: int
    field: str


@dataclass(frozen=True)
class ParameterSlot:
    """One scalar entry in a :class:`ParameterLayout`.

    ``owner``/``owner_index``/``field`` are the internal locator used to
    read (:meth:`ParameterLayout.vector`) or replace
    (:meth:`ParameterLayout.replace`) this scalar on a
    :class:`~q2mm.models.forcefield.ForceField`: the value lives at
    ``getattr(force_field, owner)[owner_index].<field>``.
    """

    index: int
    id: ParameterId
    kind: ParameterKind
    unit: ParameterUnit
    name: str
    bounds: tuple[float, float]
    step: float
    owner: str
    owner_index: int
    field: str


def _slot_json(slot: ParameterSlot) -> dict[str, Any]:
    """Canonical (value-free) JSON payload for one slot's fingerprint contribution."""
    return {
        "index": slot.index,
        "id": {
            "family": slot.id.family,
            "identity": list(slot.id.identity),
            "occurrence": slot.id.occurrence,
            "field": slot.id.field,
        },
        "kind": slot.kind.value,
        "unit": slot.unit.value,
        "name": slot.name,
        "bounds": [float.hex(slot.bounds[0]), float.hex(slot.bounds[1])],
        "step": float.hex(slot.step),
    }


def _layout_fingerprint(slots: tuple[ParameterSlot, ...]) -> str:
    """``sha256:<hex>`` over versioned canonical JSON of *slots* metadata.

    Deterministic across processes/``PYTHONHASHSEED`` values: ASCII JSON,
    sorted object keys, fixed separators, ``allow_nan=False``, and floats
    encoded with :func:`float.hex`.  Never includes parameter *values* —
    only structural/semantic slot metadata.
    """
    payload = {"version": _FINGERPRINT_SCHEMA_VERSION, "slots": [_slot_json(slot) for slot in slots]}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    digest = hashlib.sha256(blob.encode("ascii")).hexdigest()
    return f"sha256:{digest}"


# Explicit sentinel for a ``None`` identity field (e.g. an unset
# ``ff_row``).  Chosen so it can never collide with a legitimate field
# value: bond-order symbols are one of ``"-"``/``"="``/``"*"``/``"%"``/``""``,
# MM3 context flags look like ``"O200 0000"``, and ``ff_row`` is otherwise
# always rendered as a plain non-negative integer string.
_NONE_IDENTITY_SENTINEL = "<none>"


def _identity_field(value: Any) -> str:
    """Render one semantic-identity field as a stable string.

    Uses :data:`_NONE_IDENTITY_SENTINEL` for ``None`` rather than
    relying on Python's default ``str(None) == "None"`` — an explicit,
    reserved token that cannot be produced by any real field value.
    """
    if value is None:
        return _NONE_IDENTITY_SENTINEL
    return str(value)


def _bond_identity(bond: Any) -> tuple[str, ...]:
    """Row-discriminating semantic identity for one bond.

    Includes every field that can distinguish two otherwise-same-element
    rows: the environment ID, MM3 bond order (``-``/``=``/``*``/``%``),
    MM3 context flags, and the source ``ff_row`` — so re-ordering or
    changing a context-specific/bond-order variant, or swapping two
    semantically distinct rows, is never invisible to the fingerprint.
    """
    return (
        *bond.key,
        _identity_field(bond.env_id),
        _identity_field(bond.bond_order),
        _identity_field(bond.context),
        _identity_field(bond.ff_row),
    )


def _angle_identity(angle: Any) -> tuple[str, ...]:
    """Row-discriminating semantic identity for one angle or stretch-bend.

    Includes the environment ID and source ``ff_row`` alongside the
    canonical element key.
    """
    return (*angle.key, _identity_field(angle.env_id), _identity_field(angle.ff_row))


def _torsion_identity(torsion: Any) -> tuple[str, ...]:
    """Row-discriminating semantic identity for one torsion Fourier term.

    Includes canonical (direction-independent) elements, periodicity,
    improper flag, canonicalized environment ID, and source ``ff_row``.
    """
    elements = min(torsion.elements, tuple(reversed(torsion.elements)))
    env_id = canonicalize_torsion_env_id(torsion.env_id.split("-")) if torsion.env_id else ""
    return (
        *elements,
        _identity_field(torsion.periodicity),
        _identity_field(torsion.is_improper),
        _identity_field(env_id),
        _identity_field(torsion.ff_row),
    )


def _vdw_identity(vdw: Any) -> tuple[str, ...]:
    """Row-discriminating semantic identity for one vdW atom type.

    Includes the atom type, element, and source ``ff_row``.
    """
    return (vdw.atom_type, vdw.element, _identity_field(vdw.ff_row))


def _legacy_param_identity(family: str, param: Any) -> tuple[Any, ...]:
    """Reproduce the legacy ``ForceField._param_identity`` matching key.

    Used only by :func:`opt_substructure_membership` to replicate
    ``freeze_standard_params``' multiset matching; ``ParameterId``
    identity tuples are string-only and computed independently by
    ``_build_layout``.
    """
    if family == "vdw":
        return (family, param.atom_type, param.element)
    if family == "torsion":
        elements = min(param.elements, tuple(reversed(param.elements)))
        env_id = canonicalize_torsion_env_id(param.env_id.split("-")) if param.env_id else ""
        return (family, elements, param.periodicity, param.is_improper, env_id)
    return (family, param.key, getattr(param, "env_id", ""))


def _next_occurrence(counter: dict[tuple[str, ...], int], identity: tuple[str, ...]) -> int:
    occurrence = counter.get(identity, 0)
    counter[identity] = occurrence + 1
    return occurrence


def _build_layout(force_field: ForceField) -> ParameterLayout:
    """Derive a :class:`ParameterLayout` from *force_field*'s structure.

    The **only** place the full-vector order is defined.  Preserves the
    legacy order exactly: bonds (k, eq); angles (k, eq); torsions (k);
    stretch-bends (k); vdW (radius, epsilon); then a Urey-Bradley tail
    (k, eq) for angles carrying a UB term — with no manual UB-tail logic
    anywhere outside this function.
    """
    slots: list[ParameterSlot] = []
    index = 0

    def append_slot(
        *,
        kind: ParameterKind,
        owner: str,
        owner_index: int,
        field_name: str,
        identity: tuple[str, ...],
        occurrence: int,
        name: str,
    ) -> None:
        nonlocal index
        meta = _KIND_METADATA[kind]
        pid = ParameterId(family=family, identity=identity, occurrence=occurrence, field=field_name)
        slots.append(
            ParameterSlot(
                index=index,
                id=pid,
                kind=kind,
                unit=meta.unit,
                name=name,
                bounds=meta.bounds,
                step=meta.step,
                owner=owner,
                owner_index=owner_index,
                field=field_name,
            )
        )
        index += 1

    family = "bond"
    bond_occ: dict[tuple[str, ...], int] = {}
    for i, bond in enumerate(force_field.bonds):
        identity = _bond_identity(bond)
        occurrence = _next_occurrence(bond_occ, identity)
        label = "-".join(bond.key) + (f"[{bond.env_id}]" if bond.env_id else "")
        append_slot(
            kind=ParameterKind.BOND_FORCE_CONSTANT,
            owner="bonds",
            owner_index=i,
            field_name="force_constant",
            identity=identity,
            occurrence=occurrence,
            name=f"kb_{label}",
        )
        append_slot(
            kind=ParameterKind.BOND_EQUILIBRIUM,
            owner="bonds",
            owner_index=i,
            field_name="equilibrium",
            identity=identity,
            occurrence=occurrence,
            name=f"r0_{label}",
        )

    family = "angle"
    angle_occ: dict[tuple[str, ...], int] = {}
    for i, angle in enumerate(force_field.angles):
        identity = _angle_identity(angle)
        occurrence = _next_occurrence(angle_occ, identity)
        label = "-".join(angle.key) + (f"[{angle.env_id}]" if angle.env_id else "")
        append_slot(
            kind=ParameterKind.ANGLE_FORCE_CONSTANT,
            owner="angles",
            owner_index=i,
            field_name="force_constant",
            identity=identity,
            occurrence=occurrence,
            name=f"ka_{label}",
        )
        append_slot(
            kind=ParameterKind.ANGLE_EQUILIBRIUM,
            owner="angles",
            owner_index=i,
            field_name="equilibrium",
            identity=identity,
            occurrence=occurrence,
            name=f"th0_{label}",
        )

    family = "torsion"
    torsion_occ: dict[tuple[str, ...], int] = {}
    for i, torsion in enumerate(force_field.torsions):
        identity = _torsion_identity(torsion)
        occurrence = _next_occurrence(torsion_occ, identity)
        label = "-".join(torsion.elements) + f"_n{torsion.periodicity}"
        if torsion.is_improper:
            label += "_imp"
        append_slot(
            kind=ParameterKind.TORSION_FORCE_CONSTANT,
            owner="torsions",
            owner_index=i,
            field_name="force_constant",
            identity=identity,
            occurrence=occurrence,
            name=f"kt_{label}",
        )

    family = "stretch_bend"
    sb_occ: dict[tuple[str, ...], int] = {}
    for i, sb in enumerate(force_field.stretch_bends):
        identity = _angle_identity(sb)
        occurrence = _next_occurrence(sb_occ, identity)
        label = "-".join(sb.key) + (f"[{sb.env_id}]" if sb.env_id else "")
        append_slot(
            kind=ParameterKind.STRETCH_BEND_FORCE_CONSTANT,
            owner="stretch_bends",
            owner_index=i,
            field_name="force_constant",
            identity=identity,
            occurrence=occurrence,
            name=f"ksb_{label}",
        )

    family = "vdw"
    vdw_occ: dict[tuple[str, ...], int] = {}
    for i, vdw in enumerate(force_field.vdws):
        identity = _vdw_identity(vdw)
        occurrence = _next_occurrence(vdw_occ, identity)
        label = vdw.atom_type or vdw.element
        append_slot(
            kind=ParameterKind.VDW_RADIUS,
            owner="vdws",
            owner_index=i,
            field_name="radius",
            identity=identity,
            occurrence=occurrence,
            name=f"rvdw_{label}",
        )
        append_slot(
            kind=ParameterKind.VDW_EPSILON,
            owner="vdws",
            owner_index=i,
            field_name="epsilon",
            identity=identity,
            occurrence=occurrence,
            name=f"evdw_{label}",
        )

    family = "urey_bradley"
    ub_occ: dict[tuple[str, ...], int] = {}
    for i, angle in enumerate(force_field.angles):
        if angle.ub_force_constant is None or angle.ub_equilibrium is None:
            continue
        identity = _angle_identity(angle)
        occurrence = _next_occurrence(ub_occ, identity)
        label = "-".join(angle.key) + (f"[{angle.env_id}]" if angle.env_id else "")
        append_slot(
            kind=ParameterKind.UREY_BRADLEY_FORCE_CONSTANT,
            owner="angles",
            owner_index=i,
            field_name="ub_force_constant",
            identity=identity,
            occurrence=occurrence,
            name=f"kub_{label}",
        )
        append_slot(
            kind=ParameterKind.UREY_BRADLEY_EQUILIBRIUM,
            owner="angles",
            owner_index=i,
            field_name="ub_equilibrium",
            identity=identity,
            occurrence=occurrence,
            name=f"r13_{label}",
        )

    return ParameterLayout(slots=tuple(slots))


@dataclass(frozen=True)
class ParameterLayout:
    r"""Immutable, ordered layout of one force field's scalar parameters.

    Construct via :meth:`from_force_field`. Validates unique
    :class:`ParameterId`\ s and contiguous 0-based indices at
    construction time.  Every vector/bounds/step/name/kind/unit sequence
    this class exposes has length ``len(layout)``.
    """

    slots: tuple[ParameterSlot, ...]

    def __post_init__(self) -> None:
        seen_ids: set[ParameterId] = set()
        for expected_index, slot in enumerate(self.slots):
            if slot.index != expected_index:
                raise ValueError(
                    f"ParameterLayout slots must be contiguously indexed; slot {expected_index} has index {slot.index}."
                )
            if slot.id in seen_ids:
                raise ValueError(f"Duplicate ParameterId in layout: {slot.id!r}")
            seen_ids.add(slot.id)

    def __len__(self) -> int:
        return len(self.slots)

    def __getitem__(self, index: int) -> ParameterSlot:
        return self.slots[index]

    def __iter__(self) -> Iterator[ParameterSlot]:
        return iter(self.slots)

    @classmethod
    def from_force_field(cls, force_field: ForceField) -> ParameterLayout:
        """Derive a layout from *force_field*'s current structure."""
        return _build_layout(force_field)

    @property
    def ids(self) -> tuple[ParameterId, ...]:
        """Every slot's :class:`ParameterId`, in layout order."""
        return tuple(slot.id for slot in self.slots)

    @property
    def names(self) -> tuple[str, ...]:
        """Every slot's human-readable display name, in layout order."""
        return tuple(slot.name for slot in self.slots)

    @property
    def kinds(self) -> tuple[ParameterKind, ...]:
        """Every slot's :class:`ParameterKind`, in layout order."""
        return tuple(slot.kind for slot in self.slots)

    @property
    def units(self) -> tuple[ParameterUnit, ...]:
        """Every slot's :class:`ParameterUnit`, in layout order."""
        return tuple(slot.unit for slot in self.slots)

    @property
    def bounds(self) -> np.ndarray:
        """``(len(self), 2)`` canonical sanity bounds, in layout order."""
        if not self.slots:
            return np.zeros((0, 2), dtype=float)
        return np.array([slot.bounds for slot in self.slots], dtype=float)

    @property
    def steps(self) -> np.ndarray:
        """``(len(self),)`` finite-difference steps, in layout order."""
        return np.array([slot.step for slot in self.slots], dtype=float)

    @property
    def index_by_id(self) -> Mapping[ParameterId, int]:
        """Map each slot's :class:`ParameterId` to its full-vector index."""
        return {slot.id: slot.index for slot in self.slots}

    @property
    def indices_by_kind(self) -> Mapping[ParameterKind, tuple[int, ...]]:
        """Map each :class:`ParameterKind` to the full-vector indices with that kind."""
        result: dict[ParameterKind, list[int]] = {}
        for slot in self.slots:
            result.setdefault(slot.kind, []).append(slot.index)
        return {kind: tuple(indices) for kind, indices in result.items()}

    def index_of(self, parameter_id: ParameterId) -> int:
        """Return the full-vector index of *parameter_id*.

        Raises:
            KeyError: If no slot has this ID.

        """
        return self.index_by_id[parameter_id]

    @property
    def fingerprint(self) -> str:
        """``sha256:<hex>`` over canonical, value-free slot metadata.

        See :func:`_layout_fingerprint`. Deterministic across processes
        and ``PYTHONHASHSEED`` values; unaffected by parameter values;
        changes whenever a slot's semantic identity, kind, unit, name,
        bounds, step, or ordering changes.
        """
        return _layout_fingerprint(self.slots)

    def vector(self, force_field: ForceField) -> np.ndarray:
        """Extract the current scalar values from *force_field* as a flat vector."""
        values = np.empty(len(self.slots), dtype=float)
        for slot in self.slots:
            collection = getattr(force_field, slot.owner)
            param = collection[slot.owner_index]
            values[slot.index] = getattr(param, slot.field)
        return values

    def replace(self, force_field: ForceField, vector: Sequence[float] | np.ndarray) -> ForceField:
        """Return a new :class:`ForceField` with scalars set from *vector*.

        Pure: *force_field* is not mutated.  *vector* must have length
        ``len(self)`` and follow this layout's slot order.
        """
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(self.slots),):
            raise ValueError(f"vector length {values.shape} does not match layout length {len(self.slots)}.")

        # Group per-field updates by (owner, owner_index) so a bond/angle/UB
        # row with multiple scalar slots is replaced exactly once.
        updates: dict[tuple[str, int], dict[str, float]] = {}
        touched_owners: dict[str, list[Any]] = {}
        for slot in self.slots:
            key = (slot.owner, slot.owner_index)
            updates.setdefault(key, {})[slot.field] = float(values[slot.index])
            if slot.owner not in touched_owners:
                touched_owners[slot.owner] = list(getattr(force_field, slot.owner))

        for (owner, owner_index), field_updates in updates.items():
            collection = touched_owners[owner]
            collection[owner_index] = dataclasses.replace(collection[owner_index], **field_updates)

        replacements = {owner: tuple(collection) for owner, collection in touched_owners.items()}
        return dataclasses.replace(force_field, **replacements)  # type: ignore[arg-type]


def _immutable_1d(values: Sequence[float] | np.ndarray, expected_length: int, *, name: str) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},), got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class ActiveParameterSpace:
    """The one active/frozen projection over a :class:`ParameterLayout`.

    Stores the immutable baseline full vector (values used to fill
    inactive/frozen slots on :meth:`expand`) and the sorted active
    slot indices, and provides validated full<->active conversions,
    bounds/steps for the active subset, and derived-subspace
    operations (:meth:`with_active_indices`, :meth:`with_baseline`).
    """

    layout: ParameterLayout
    baseline: np.ndarray
    active_indices: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline", _immutable_1d(self.baseline, len(self.layout), name="baseline"))
        active = np.unique(np.asarray(self.active_indices, dtype=int))
        if active.size and (int(active.min()) < 0 or int(active.max()) >= len(self.layout)):
            raise ValueError(
                f"active_indices must be in [0, {len(self.layout)}); got range [{active.min()}, {active.max()}]."
            )
        active.setflags(write=False)
        object.__setattr__(self, "active_indices", active)

    @property
    def n_full(self) -> int:
        """Length of the full parameter vector (``len(self.layout)``)."""
        return len(self.layout)

    @property
    def n_active(self) -> int:
        """Number of active (optimizable) slots."""
        return int(self.active_indices.size)

    @property
    def active_ids(self) -> tuple[ParameterId, ...]:
        """Active slots' :class:`ParameterId` values, in full-vector index order."""
        return tuple(self.layout.slots[i].id for i in self.active_indices)

    @property
    def names(self) -> tuple[str, ...]:
        """Active slots' display names, in full-vector index order."""
        return tuple(self.layout.slots[i].name for i in self.active_indices)

    @property
    def kinds(self) -> tuple[ParameterKind, ...]:
        """Active slots' :class:`ParameterKind` values, in full-vector index order."""
        return tuple(self.layout.slots[i].kind for i in self.active_indices)

    @property
    def units(self) -> tuple[ParameterUnit, ...]:
        """Active slots' :class:`ParameterUnit` values, in full-vector index order."""
        return tuple(self.layout.slots[i].unit for i in self.active_indices)

    @property
    def bounds(self) -> np.ndarray:
        """``(n_active, 2)`` canonical sanity bounds for the active subset."""
        full = self.layout.bounds
        if self.n_active == 0:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(full[self.active_indices], dtype=float)

    @property
    def steps(self) -> np.ndarray:
        """``(n_active,)`` finite-difference steps for the active subset."""
        return np.asarray(self.layout.steps[self.active_indices], dtype=float)

    def pack(self, full_vector: Sequence[float] | np.ndarray) -> np.ndarray:
        """Project a full-length vector down to the active subset."""
        full = _immutable_1d(full_vector, self.n_full, name="full_vector")
        return np.array(full[self.active_indices], dtype=float)

    def expand(self, active_vector: Sequence[float] | np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
        """Fill active slots from *active_vector*; inactive slots from *base* (default :attr:`baseline`)."""
        active = _immutable_1d(active_vector, self.n_active, name="active_vector")
        full = np.array(self.baseline if base is None else base, dtype=float, copy=True)
        if full.shape != (self.n_full,):
            raise ValueError(f"base must have shape ({self.n_full},), got {full.shape}.")
        full[self.active_indices] = active
        return full

    def with_active_indices(self, indices: Sequence[int] | np.ndarray) -> ActiveParameterSpace:
        """Return a derived space over the same layout/baseline with a different active set."""
        return dataclasses.replace(self, active_indices=np.asarray(list(indices), dtype=int))

    def with_baseline(self, vector: Sequence[float] | np.ndarray) -> ActiveParameterSpace:
        """Return a derived space over the same layout/active set with a rebased baseline."""
        return dataclasses.replace(self, baseline=np.asarray(vector, dtype=float).copy())

    def active_owner_indices(self, owner: str, *, fields: frozenset[str] | None = None) -> frozenset[int]:
        """Active :class:`ForceField` collection indices for *owner*, optionally restricted to *fields*.

        E.g. ``space.active_owner_indices("angles", fields={"force_constant", "equilibrium"})``
        returns the indices into ``force_field.angles`` whose *bending*
        slots are active — excluding indices whose only active slots are
        Urey-Bradley (``ub_force_constant``/``ub_equilibrium``).
        """
        result: set[int] = set()
        for i in self.active_indices:
            slot = self.layout.slots[int(i)]
            if slot.owner != owner:
                continue
            if fields is not None and slot.field not in fields:
                continue
            result.add(slot.owner_index)
        return frozenset(result)

    @classmethod
    def all_active(cls, layout: ParameterLayout, force_field: ForceField) -> ActiveParameterSpace:
        """Build a space with every slot active (no frozen backbone)."""
        return cls(layout=layout, baseline=layout.vector(force_field), active_indices=np.arange(len(layout)))

    @classmethod
    def from_membership(
        cls,
        layout: ParameterLayout,
        force_field: ForceField,
        membership: OptSubstructureMembership,
    ) -> ActiveParameterSpace:
        """Build a space whose active slots are those in *membership*.

        See :func:`opt_substructure_membership` for how *membership* is
        typically produced (mirroring the legacy
        ``ForceField.freeze_standard_params`` semantics).
        """
        family_active: dict[str, frozenset[int]] = {
            "bond": membership.bonds,
            "angle": membership.angles,
            "stretch_bend": membership.stretch_bends,
            "torsion": membership.torsions,
            "vdw": membership.vdws,
        }
        active: list[int] = []
        for slot in layout.slots:
            if slot.field in _UB_FIELDS:
                if slot.owner_index in membership.urey_bradley:
                    active.append(slot.index)
                continue
            active_set = family_active.get(slot.id.family)
            if active_set is not None and slot.owner_index in active_set:
                active.append(slot.index)
        return cls(
            layout=layout,
            baseline=layout.vector(force_field),
            active_indices=np.array(sorted(active), dtype=int),
        )


@dataclass(frozen=True)
class OptSubstructureMembership:
    """Per-family sets of :class:`ForceField` collection indices deemed "active".

    Produced by :func:`opt_substructure_membership`.  Each field holds
    0-based indices into the corresponding collection on the *composed*
    force field (``force_field.bonds``, ``.angles``, ...); ``urey_bradley``
    indexes into ``force_field.angles`` as well (the subset with a UB term).
    """

    bonds: frozenset[int] = field(default_factory=frozenset)
    angles: frozenset[int] = field(default_factory=frozenset)
    stretch_bends: frozenset[int] = field(default_factory=frozenset)
    torsions: frozenset[int] = field(default_factory=frozenset)
    vdws: frozenset[int] = field(default_factory=frozenset)
    urey_bradley: frozenset[int] = field(default_factory=frozenset)


def opt_substructure_membership(force_field: ForceField, opt_force_field: ForceField) -> OptSubstructureMembership:
    """Identify which of *force_field*'s parameters belong to *opt_force_field*.

    Reproduces the legacy ``ForceField.freeze_standard_params`` matching
    algorithm exactly, as a pure function: parameters are matched first
    by shared ``ff_row`` (only when both force fields share the same
    resolved ``source_path``), then by semantic chemical-identity
    multiset matching (occurrence-order, not value-based).  Used by the
    QFUERZA publication-system loaders to build an
    :class:`ActiveParameterSpace` via :meth:`ActiveParameterSpace.from_membership`
    that keeps the literature MM3 backbone frozen and only the
    OPT-substructure rows active.
    """
    same_source = (
        force_field.source_path is not None
        and opt_force_field.source_path is not None
        and force_field.source_path.resolve() == opt_force_field.source_path.resolve()
    )

    def match(attr: str, family: str) -> frozenset[int]:
        collection = getattr(force_field, attr)
        opt_collection = getattr(opt_force_field, attr)
        opt_rows = Counter(p.ff_row for p in opt_collection if p.ff_row is not None)
        opt_ids = Counter(_legacy_param_identity(family, p) for p in opt_collection)
        active: set[int] = set()
        for i, param in enumerate(collection):
            if same_source and param.ff_row is not None:
                if opt_rows[param.ff_row] > 0:
                    active.add(i)
                    opt_rows[param.ff_row] -= 1
                continue
            ident = _legacy_param_identity(family, param)
            if opt_ids[ident] > 0:
                active.add(i)
                opt_ids[ident] -= 1
        return frozenset(active)

    def match_urey_bradley() -> frozenset[int]:
        ub_indexed = [
            (i, a)
            for i, a in enumerate(force_field.angles)
            if a.ub_force_constant is not None and a.ub_equilibrium is not None
        ]
        opt_ub = [a for a in opt_force_field.angles if a.ub_force_constant is not None and a.ub_equilibrium is not None]
        opt_ub_rows = Counter(a.ff_row for a in opt_ub if a.ff_row is not None)
        opt_ub_ids = Counter(_legacy_param_identity("angle", a) for a in opt_ub)
        active: set[int] = set()
        for i, angle in ub_indexed:
            if same_source and angle.ff_row is not None:
                if opt_ub_rows[angle.ff_row] > 0:
                    active.add(i)
                    opt_ub_rows[angle.ff_row] -= 1
                continue
            ident = _legacy_param_identity("angle", angle)
            if opt_ub_ids[ident] > 0:
                active.add(i)
                opt_ub_ids[ident] -= 1
        return frozenset(active)

    return OptSubstructureMembership(
        bonds=match("bonds", "bond"),
        angles=match("angles", "angle"),
        stretch_bends=match("stretch_bends", "stretch_bend"),
        torsions=match("torsions", "torsion"),
        vdws=match("vdws", "vdw"),
        urey_bradley=match_urey_bradley(),
    )


def fractional_bounds(
    kinds: Sequence[ParameterKind],
    sanity_bounds: np.ndarray,
    vector: Sequence[float] | np.ndarray,
    *,
    fc_fraction: float | None,
    eq_fraction: float | None,
) -> np.ndarray:
    """Bounds as a fractional box around each parameter's current value.

    Successor to the legacy ``ForceField.get_fractional_bounds``.  Unlike
    *sanity_bounds* (canonical physical bounds, e.g. ``bond_k`` in
    ``(0, 3600)``), this returns a sign-aware fractional box ``(val -
    frac * abs(val), val + frac * abs(val))`` per parameter, intersected
    with *sanity_bounds*.  Falls back to *sanity_bounds* for parameters
    with ``|value| < 1e-6`` (a symmetric window would collapse to a
    point) or when the fractional window's intersection with
    *sanity_bounds* is empty.

    Args:
        kinds: Per-slot :class:`ParameterKind`, e.g. ``layout.kinds`` or
            ``space.kinds``.
        sanity_bounds: ``(n, 2)`` canonical bounds, e.g. ``layout.bounds``
            or ``space.bounds``.
        vector: Current parameter values, same length/order as *kinds*.
        fc_fraction: Fractional box width for force-constant kinds
            (``bond_k``, ``angle_k``, ``torsion_k``, ``sb_k``,
            ``vdw_epsilon``, ``ub_k``).  ``None`` uses *sanity_bounds*
            for those kinds.
        eq_fraction: Fractional box width for equilibrium kinds
            (``bond_eq``, ``angle_eq``, ``vdw_radius``, ``ub_eq``).
            ``None`` uses *sanity_bounds* for those kinds.

    Returns:
        ``(n, 2)`` bounds array, same order as *kinds*.

    """
    sanity = np.asarray(sanity_bounds, dtype=float)
    values = np.asarray(vector, dtype=float)
    if fc_fraction is None and eq_fraction is None:
        return sanity.copy()

    bounds = np.empty_like(sanity)
    for i, (kind, value, (lo, hi)) in enumerate(zip(kinds, values, sanity, strict=True)):
        if kind in _FC_KINDS:
            frac = fc_fraction
        elif kind in _EQ_KINDS:
            frac = eq_fraction
        else:  # pragma: no cover — defensive; every ParameterKind is FC or EQ
            frac = None

        if frac is None or abs(value) < 1e-6:
            bounds[i] = (lo, hi)
            continue

        window = frac * abs(value)
        new_lo = max(lo, value - window)
        new_hi = min(hi, value + window)
        bounds[i] = (lo, hi) if new_lo >= new_hi else (new_lo, new_hi)
    return bounds
