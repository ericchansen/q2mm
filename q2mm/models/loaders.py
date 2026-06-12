"""Force-field assembly strategies for the published-FF benchmark systems.

Each function in this module implements one named strategy for
turning a force-field file (or a set of training molecules) into a
ready-to-use :class:`~q2mm.models.forcefield.ForceField`.  The
strategies are deliberately *named* rather than composed at the call
site to prevent the silent-overwrite class of bugs that produced the
load_heck_relay regression (q2mm#277).

The methodology of record is **Farrugia, Helquist, Norrby & Wiest 2025**
("Rapid FF Generation via Hessian-Informed Initial Parameters and
Automated Refinement", *J. Chem. Theory Comput.* **22**, 469;
DOI 10.1021/acs.jctc.5c01751) — see AGENTS.md "Key Papers" for the
broader context.  Farrugia 2025 §"Methods" identifies four parameter
generation strategies (Approxn, FUERZA, γ-FUERZA, QFUERZA); the
loaders below correspond to the three that Q2MM ships for published
FFs:

- :func:`load_published_opt` — use literature OPT parameter values
  as-published.  Standard MM3 backbone is frozen; OPT block is active.
  Matches the "mixing parameter sources" workflow Farrugia 2025
  describes for published TSFFs.
- :func:`load_qfuerza_fresh` — no literature FF at all.  Build a brand
  new FF from one molecule's QM Hessian via QFUERZA.  For small
  single-molecule benchmarks like CH3F.
- :func:`compose_opt_with_mm3_base` — Wahlers-style composition
  primitive: append an OPT block to a standard MM3 base FF.  Returns
  the composed FF with no frozen/active partition set.
- :func:`load_published_opt_composed` — convenience wrapper that
  combines :func:`compose_opt_with_mm3_base` with the
  ``freeze_standard_params`` partition.  This is the
  :func:`load_published_opt` equivalent for Wahlers FFs that ship as
  standalone OPT-substructure-only files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from q2mm.models.forcefield import ForceField, FunctionalForm, VdwParam
from q2mm.models.seminario import qfuerza_fresh

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule


__all__ = [
    "load_published_opt",
    "load_qfuerza_fresh",
    "compose_opt_with_mm3_base",
    "load_published_opt_composed",
]


# Published metal vdW parameters for systems whose MM3 base file lacks them.
# Today only PD (sourced from Rosales 2020 Heck FF mm3.FF1.fld:1063); promote
# to its own module (e.g. ``q2mm/models/metals.py``) if this grows past a
# handful of entries.
_METAL_VDW: dict[str, VdwParam] = {
    "PD": VdwParam(atom_type="PD", radius=1.70, epsilon=0.414, element="Pd"),
}


def load_published_opt(ff_path: str | Path) -> ForceField:
    """Load a self-contained published MM3 .fld using its OPT values as-is.

    Used for FFs (e.g. Donoghue 2008 Rh-enamide, Rosales 2020 Heck relay)
    whose .fld file already contains both the standard MM3 backbone AND
    a custom OPT-substructure block with literature-fitted values.  The
    standard MM3 backbone is frozen; the OPT block is left active so an
    optimizer can refine it further.

    No QFUERZA projection is run — the published OPT values are
    preserved exactly.  This is the strategy that fixes the
    ``load_heck_relay`` regression from q2mm#277.

    Returned FF invariants:
        - ``functional_form`` is :attr:`FunctionalForm.MM3`.
        - Standard MM3 params (non-OPT) are frozen.
        - OPT-substructure params are unfrozen.
        - All param *values* come from the .fld file unchanged.

    Args:
        ff_path: Path to the published .fld file.

    Returns:
        The loaded force field with the frozen/active partition set.

    """
    ff_path = Path(ff_path)
    ff = ForceField.from_mm3_fld(str(ff_path), include_standard=True)
    opt_only = ForceField.from_mm3_fld(str(ff_path), include_standard=False)
    ff.freeze_standard_params(opt_only)
    ff.functional_form = FunctionalForm.MM3
    return ff


def load_qfuerza_fresh(
    molecule: Q2MMMolecule,
    *,
    invert_ts_curvature: bool = True,
    replace_with: float = 1.0,
) -> ForceField:
    """Build a brand-new FF from one molecule's QM Hessian via QFUERZA.

    For small single-molecule benchmarks (CH3F-style) where there is no
    published OPT block to start from.  Every parameter in the returned
    FF comes from the QFUERZA projection and is unfrozen.

    Args:
        molecule: One molecule with a QM Hessian attached.
        invert_ts_curvature: Whether to invert the TS reaction
            coordinate before projection (Limé & Norrby 2015).  Default
            ``True`` because the only system using this strategy today
            is the CH3F SN2 transition state.
        replace_with: Replacement value (Hartree/Bohr²) for the most
            negative eigenvalue when ``invert_ts_curvature=True``.
            Default ``1.0`` matches Limé & Norrby Method C.  Ignored
            when ``invert_ts_curvature=False``.

    Returns:
        Fresh force field; every parameter unfrozen.

    """
    return qfuerza_fresh(
        molecule,
        invert_ts_curvature=invert_ts_curvature,
        replace_with=replace_with,
    )


def compose_opt_with_mm3_base(
    opt_path: str | Path,
    base_path: str | Path,
    *,
    metal: str | None = None,
) -> tuple[ForceField, ForceField]:
    """Compose a Wahlers-style standalone OPT .fld with a standard MM3 base.

    Wahlers TSFFs (pd-allyl, pd 1,4-conjugate addition, rh 1,4-conjugate
    addition) ship as standalone OPT-substructure-only files (~100-500
    parameters) that depend on a separate standard MM3 base file
    (~2,500 backbone params).  This primitive concatenates the two
    into a single :class:`ForceField`, with the OPT block appearing
    first so that matching prefers OPT over the corresponding base
    entries.

    Does *not* set a frozen/active partition — use
    :func:`load_published_opt_composed` for that.

    Args:
        opt_path: Path to the standalone OPT-only .fld file.
        base_path: Path to the standard MM3 .fld file.
        metal: Optional element symbol (e.g. ``"PD"``) whose
            vdW parameters should be added to the composed FF if not
            already present.  Sources from :data:`_METAL_VDW`.

    Returns:
        ``(composed, opt_only)`` — the composed force field
        (``functional_form = MM3``, no frozen/active partition set),
        and the OPT-only FF used to build it (returned so callers
        such as :func:`load_published_opt_composed` can reuse it to
        set the frozen/active partition without re-parsing the file).

    """
    opt_ff = ForceField.from_mm3_fld(str(Path(opt_path)), include_standard=False)
    base_ff = ForceField.from_mm3_fld(str(Path(base_path)))

    composed = ForceField(
        bonds=list(opt_ff.bonds) + list(base_ff.bonds),
        angles=list(opt_ff.angles) + list(base_ff.angles),
        torsions=list(opt_ff.torsions) + list(base_ff.torsions),
        vdws=list(opt_ff.vdws) + list(base_ff.vdws),
        stretch_bends=list(opt_ff.stretch_bends) + list(base_ff.stretch_bends),
        functional_form=FunctionalForm.MM3,
    )

    if metal:
        metal_key = metal.upper()
        if metal_key in _METAL_VDW:
            has_metal_vdw = any(
                v.atom_type == metal_key or (v.element or "").capitalize() == metal.capitalize() for v in composed.vdws
            )
            if not has_metal_vdw:
                composed.vdws.append(_METAL_VDW[metal_key])

    return composed, opt_ff


def load_published_opt_composed(
    opt_path: str | Path,
    base_path: str | Path,
    *,
    metal: str | None = None,
) -> ForceField:
    """Compose Wahlers OPT + MM3 base, then freeze the standard backbone.

    The :func:`load_published_opt` equivalent for systems whose OPT
    block lives in a separate .fld file from the base MM3 parameters.
    Uses the published Wahlers OPT values as-published (no QFUERZA
    overwrite).

    Returned FF invariants:
        - ``functional_form`` is :attr:`FunctionalForm.MM3`.
        - Standard MM3 backbone params are frozen.
        - OPT-substructure params are unfrozen.
        - All param *values* come from the .fld files unchanged.

    Args:
        opt_path: Path to the standalone OPT-only .fld file (Wahlers).
        base_path: Path to the standard MM3 .fld file.
        metal: Optional element symbol for vdW injection (see
            :func:`compose_opt_with_mm3_base`).

    Returns:
        Composed force field with the frozen/active partition set.

    """
    composed, opt_only = compose_opt_with_mm3_base(opt_path, base_path, metal=metal)
    composed.freeze_standard_params(opt_only)
    return composed
