"""Force-field assembly strategies for the published-FF benchmark systems.

Each function turns a force-field file (or a set of training molecules)
into a ready-to-use immutable :class:`~q2mm.models.forcefield.ForceField`,
named rather than composed at the call site to prevent the silent-overwrite
class of bugs that produced the ``load_heck_relay`` regression (q2mm#277).
None of these apply a frozen/active partition — that is always a separate,
explicit step via ``opt_substructure_membership()``/``ActiveParameterSpace``
(see :mod:`q2mm.benchmarks.systems._assembly`).
"""

from __future__ import annotations

from pathlib import Path

from q2mm.models.forcefield import ForceField, FunctionalForm, VdwParam
from q2mm.models.molecule import Molecule
from q2mm.models.seminario import qfuerza_fresh

# Published metal vdW parameters for systems whose MM3 base file lacks them.
# Today only PD (sourced from Rosales 2020 Heck FF mm3.FF1.fld:1063); promote
# to its own module if this grows past a handful of entries.
_METAL_VDW: dict[str, VdwParam] = {
    "PD": VdwParam(atom_type="PD", radius=1.70, epsilon=0.414, element="Pd"),
}


def load_published_opt(ff_path: str | Path) -> tuple[ForceField, ForceField]:
    """Load a self-contained published MM3 .fld using its OPT values as-is.

    Used for FFs (e.g. Donoghue 2008 Rh-enamide, Rosales 2020 Heck relay)
    whose .fld file already contains both the standard MM3 backbone AND
    a custom OPT-substructure block with literature-fitted values.

    No QFUERZA projection is run — the published OPT values are
    preserved exactly.  This is the strategy that fixes the
    ``load_heck_relay`` regression from q2mm#277.

    Args:
        ff_path: Path to the published .fld file.

    Returns:
        ``(composed, opt_only)`` — the full force field (standard MM3 +
        OPT block, no frozen/active partition applied) and the
        OPT-substructure-only force field used to identify which
        parameters are OPT (via
        :func:`~q2mm.models.parameters.opt_substructure_membership`).

    """
    import dataclasses

    from q2mm.io.mm3 import load_mm3_fld

    ff_path = Path(ff_path)
    composed = load_mm3_fld(str(ff_path), include_standard=True)
    opt_only = load_mm3_fld(str(ff_path), include_standard=False)
    composed = dataclasses.replace(composed, functional_form=FunctionalForm.MM3)
    return composed, opt_only


def load_qfuerza_fresh(
    molecule: Molecule,
    *,
    functional_form: FunctionalForm,
    invert_ts_curvature: bool = False,
    replace_with: float = 1.0,
) -> ForceField:
    """Build a brand-new FF from one molecule's QM Hessian via QFUERZA.

    For small single-molecule benchmarks (CH3F-style) where there is no
    published OPT block to start from.  Every parameter in the returned
    FF comes from the QFUERZA projection.

    Args:
        molecule: One molecule with a QM Hessian attached.
        functional_form: Required — every :class:`ForceField` must carry
            an explicit form (see :func:`q2mm.models.seminario.qfuerza_fresh`).
            CH3F/CH3F-SN2 genuinely support both ``FunctionalForm.HARMONIC``
            (JAX/JAX-MD) and ``FunctionalForm.MM3`` (OpenMM/Tinker); there
            is no scientifically-correct single default across engines,
            so the caller must decide.
        invert_ts_curvature: Whether to invert the TS reaction
            coordinate before projection (Limé & Norrby 2015). Callers
            must pass ``True`` only for genuine transition states (one
            real imaginary mode to invert) and ``False`` for ground
            states. This is *not* a harmless default-True no-op for
            ground states: a real, imperfectly-converged ground-state
            Hessian routinely carries one or more tiny *spurious*
            negative eigenvalues (numerical noise on otherwise
            near-zero rigid-body modes, typically ~1e-5-1e-6
            Hartree/Bohr²) alongside its genuine positive spectrum —
            ``invert_ts_curvature=True`` would silently replace that
            noise eigenvalue with *replace_with* (default 1.0
            Hartree/Bohr², many orders of magnitude larger), corrupting
            an otherwise-real vibrational mode. See
            :class:`~q2mm.models.problem.StationaryPointKind` — the
            caller must route this from the training case's actual
            stationary-point kind, never hardcode it.
        replace_with: Replacement value (Hartree/Bohr²) for the most
            negative eigenvalue when ``invert_ts_curvature=True``.
            Default ``1.0`` matches Limé & Norrby Method C.  Ignored
            when ``invert_ts_curvature=False``.

    Returns:
        Fresh force field; every parameter is QFUERZA-derived, tagged
        with the caller-supplied *functional_form*.

    """
    return qfuerza_fresh(
        molecule,
        functional_form=functional_form,
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

    Args:
        opt_path: Path to the standalone OPT-only .fld file.
        base_path: Path to the standard MM3 .fld file.
        metal: Optional element symbol (e.g. ``"PD"``) whose
            vdW parameters should be added to the composed FF if not
            already present.  Sources from :data:`_METAL_VDW`.

    Returns:
        ``(composed, opt_only)`` — the composed force field
        (``functional_form = MM3``, no frozen/active partition applied),
        and the OPT-only FF used to build it (returned so callers
        can identify OPT-substructure membership without re-parsing
        the file).

    """
    from q2mm.io.mm3 import load_mm3_fld

    opt_ff = load_mm3_fld(str(Path(opt_path)), include_standard=False)
    base_ff = load_mm3_fld(str(Path(base_path)))

    vdws = tuple(opt_ff.vdws) + tuple(base_ff.vdws)
    if metal:
        metal_key = metal.upper()
        if metal_key in _METAL_VDW:
            has_metal_vdw = any(
                v.atom_type == metal_key or (v.element or "").capitalize() == metal.capitalize() for v in vdws
            )
            if not has_metal_vdw:
                vdws = vdws + (_METAL_VDW[metal_key],)

    composed = ForceField(
        bonds=tuple(opt_ff.bonds) + tuple(base_ff.bonds),
        angles=tuple(opt_ff.angles) + tuple(base_ff.angles),
        torsions=tuple(opt_ff.torsions) + tuple(base_ff.torsions),
        vdws=vdws,
        stretch_bends=tuple(opt_ff.stretch_bends) + tuple(base_ff.stretch_bends),
        functional_form=FunctionalForm.MM3,
    )

    return composed, opt_ff
