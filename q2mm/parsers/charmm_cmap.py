"""Parser for CHARMM CMAP correction map sections.

Reads the ``CMAP`` block from a CHARMM parameter file (``.prm``) and
returns a list of :class:`~q2mm.models.forcefield.CmapGrid` objects.

CHARMM CMAP format::

    CMAP
    CT1  N   CT1   C   N   CT1   C   CT1   24
     0.126790  0.768700  0.971260 ...
     ...  (24×24 = 576 values total)

Each CMAP entry starts with a line of 8 atom types and a grid resolution,
followed by ``resolution × resolution`` energy values (kcal/mol) in
row-major order, spanning φ from -180° to 180° and ψ from -180° to 180°.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from q2mm.models.forcefield import CmapGrid

logger = logging.getLogger(__name__)


def parse_cmap_section(text: str) -> list[CmapGrid]:
    """Parse CMAP entries from a CHARMM parameter file section.

    Args:
        text: The full text of a CHARMM ``.prm`` file, or just the CMAP
            section thereof.

    Returns:
        List of :class:`CmapGrid` objects parsed from the CMAP block.

    """
    lines = text.splitlines()
    grids: list[CmapGrid] = []

    in_cmap_section = False
    current_header: tuple[tuple[str, ...], tuple[str, ...], int] | None = None
    current_values: list[float] = []
    expected_count = 0

    for raw_line in lines:
        line = raw_line.strip()

        # Strip trailing comments
        if "!" in line:
            line = line[: line.index("!")].strip()
        if not line:
            continue

        upper = line.upper()

        # Detect CMAP section start
        if upper == "CMAP":
            in_cmap_section = True
            continue

        # Detect end of CMAP section (another section keyword)
        if in_cmap_section and re.match(
            r"^(BOND|ANGL|DIHE|IMPR|NONB|NBFI|HBON|CMAP|END|ATOM|THET|PHI|DONO|ACCE|NBON)",
            upper,
        ):
            if upper != "CMAP":
                # Finalize any in-progress grid
                if current_header is not None and len(current_values) == expected_count:
                    phi_types, psi_types, res = current_header
                    grids.append(
                        CmapGrid(
                            atom_types_phi=phi_types,
                            atom_types_psi=psi_types,
                            resolution=res,
                            energy=current_values,
                        )
                    )
                in_cmap_section = False
                current_header = None
                current_values = []
                continue

        if not in_cmap_section:
            continue

        # Try to parse as a CMAP header line: 8 atom types + resolution
        tokens = line.split()
        if len(tokens) == 9 and _is_header_line(tokens):
            # Finalize previous grid if any
            if current_header is not None:
                if len(current_values) == expected_count:
                    phi_types, psi_types, res = current_header
                    grids.append(
                        CmapGrid(
                            atom_types_phi=phi_types,
                            atom_types_psi=psi_types,
                            resolution=res,
                            energy=current_values,
                        )
                    )
                else:
                    logger.warning(
                        "Incomplete CMAP grid: expected %d values, got %d. Skipping.",
                        expected_count,
                        len(current_values),
                    )

            resolution = int(tokens[8])
            current_header = (
                (tokens[0], tokens[1], tokens[2], tokens[3]),
                (tokens[4], tokens[5], tokens[6], tokens[7]),
                resolution,
            )
            current_values = []
            expected_count = resolution * resolution
            continue

        # Otherwise, parse as grid data values
        if current_header is not None:
            for token in tokens:
                try:
                    current_values.append(float(token))
                except ValueError:
                    logger.warning("Non-numeric token in CMAP grid data: %r", token)

    # Finalize last grid
    if current_header is not None and len(current_values) == expected_count:
        phi_types, psi_types, res = current_header
        grids.append(
            CmapGrid(
                atom_types_phi=phi_types,
                atom_types_psi=psi_types,
                resolution=res,
                energy=current_values,
            )
        )
    elif current_header is not None:
        logger.warning(
            "Incomplete CMAP grid at end of file: expected %d values, got %d.",
            expected_count,
            len(current_values),
        )

    logger.info("Parsed %d CMAP grid(s).", len(grids))
    return grids


def load_cmap_from_prm(path: str | Path) -> list[CmapGrid]:
    """Load CMAP grids from a CHARMM parameter file.

    Args:
        path: Path to a CHARMM ``.prm`` parameter file.

    Returns:
        List of :class:`CmapGrid` objects.

    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_cmap_section(text)


def _is_header_line(tokens: list[str]) -> bool:
    """Check if 9 tokens look like a CMAP header (8 atom types + int)."""
    try:
        int(tokens[8])
    except ValueError:
        return False
    # Atom types are typically alphabetic with possible digits (e.g., CT1, N, C)
    return all(re.match(r"^[A-Za-z][A-Za-z0-9*]*$", t) for t in tokens[:8])
