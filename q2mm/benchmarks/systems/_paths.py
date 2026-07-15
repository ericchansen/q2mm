"""External-data-root resolution for :mod:`q2mm.benchmarks.systems` modules.

Every published-FF/training-set system needs licensed or non-distributed
scientific data (Rh-enamide training set, Wahlers/Rosales dissertation
supporting information, the licensed MM3 base force field) located
outside the repository. This module is the *one* place that resolves
those locations — from explicit :class:`ExternalDataRoots` overrides,
then from the documented ``Q2MM_*`` environment variables — plus the
deterministic natural-sort helper used when discovering files within
those roots.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

StartingPoint = str
"""``"qfuerza"`` (canonical default) or ``"published"`` — see each system module."""


# ---------------------------------------------------------------------------
# External data roots
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExternalDataRoots:
    """Explicit locations for scientific data that Q2MM does not distribute.

    Attributes:
        rh_enamide: Directory containing ``mm3.fld`` and the
            ``rh_enamide_training_set`` directory. Environment fallback:
            ``Q2MM_RH_ENAMIDE``.
        supporting_info: Root of the extracted Wahlers/Rosales dissertation
            supporting information. Environment fallback:
            ``Q2MM_SUPPORTING_INFO``.
        mm3_base: Path to the licensed MM3 base ``.fld`` file. Environment
            fallback: ``Q2MM_MM3_BASE``.

    """

    rh_enamide: Path | None = None
    supporting_info: Path | None = None
    mm3_base: Path | None = None

    @classmethod
    def from_environment(cls) -> ExternalDataRoots:
        """Build roots from Q2MM's documented environment variables."""

        def optional_path(name: str) -> Path | None:
            value = os.environ.get(name)
            return Path(value).expanduser() if value else None

        return cls(
            rh_enamide=optional_path("Q2MM_RH_ENAMIDE"),
            supporting_info=optional_path("Q2MM_SUPPORTING_INFO"),
            mm3_base=optional_path("Q2MM_MM3_BASE"),
        )


def resolve_external_roots(roots: ExternalDataRoots | None) -> ExternalDataRoots:
    """Merge explicit roots with per-field environment fallbacks."""
    environment = ExternalDataRoots.from_environment()
    if roots is None:
        return environment
    return ExternalDataRoots(
        rh_enamide=roots.rh_enamide if roots.rh_enamide is not None else environment.rh_enamide,
        supporting_info=roots.supporting_info if roots.supporting_info is not None else environment.supporting_info,
        mm3_base=roots.mm3_base if roots.mm3_base is not None else environment.mm3_base,
    )


def resolve_rh_enamide_dir(roots: ExternalDataRoots | None = None) -> Path:
    """Resolve the externally supplied Rh-enamide dataset root."""
    path = resolve_external_roots(roots).rh_enamide
    if path is None:
        raise FileNotFoundError(
            "Rh-enamide data is not distributed with q2mm. Configure "
            "ExternalDataRoots(rh_enamide=Path(...)) via the rh_enamide loader's "
            "data_roots= argument, or set Q2MM_RH_ENAMIDE to the directory "
            "containing mm3.fld and rh_enamide_training_set/."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Configured Rh-enamide data root is not a directory: {path}")
    return path


def resolve_supporting_info_dir(roots: ExternalDataRoots | None = None) -> Path:
    """Return the explicitly configured supporting-information root."""
    path = resolve_external_roots(roots).supporting_info
    if path is None:
        raise FileNotFoundError(
            "Dissertation supporting information is not distributed with q2mm. "
            "Configure ExternalDataRoots(supporting_info=Path(...)) via the "
            "system loader's data_roots= argument, or set Q2MM_SUPPORTING_INFO."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Configured supporting-information root is not a directory: {path}")
    return path


def resolve_mm3_base_path(roots: ExternalDataRoots | None = None) -> Path:
    """Resolve the explicitly supplied licensed MM3 base force field."""
    path = resolve_external_roots(roots).mm3_base
    if path is None:
        raise FileNotFoundError(
            "The MM3 base force field is not distributed with q2mm. Configure "
            "ExternalDataRoots(mm3_base=Path(...)) via the system loader's "
            "data_roots= argument, or set Q2MM_MM3_BASE to a licensed mm3_base.fld file."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Configured MM3 base force field is not a file: {path}")
    return path


def resolve_wahlers_opt_path(
    chapter_subdir: str,
    ff_filename: str,
    roots: ExternalDataRoots | None = None,
) -> Path:
    """Resolve the path to a Wahlers chapter's standalone OPT .fld file."""
    si = resolve_supporting_info_dir(roots)
    return si / "wahlers" / "Wahlers_Jessica_Supporting_information" / chapter_subdir / ff_filename


# ---------------------------------------------------------------------------
# Deterministic sorting
# ---------------------------------------------------------------------------


def natural_sort_key(p: Path) -> list:
    """Sort key treating embedded digit runs numerically (e.g. ``ts2`` < ``ts10``)."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)]
