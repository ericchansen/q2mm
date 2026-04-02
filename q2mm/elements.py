"""Periodic table data — single source of truth for element properties.

This module has **no q2mm imports**, so any module can import from it
without risking circular dependencies.

Data Sources:
    Monoisotopic masses: NUBASE2020 / AME2020 — Wang et al., Chinese
        Phys. C 45, 030003 (2021). DOI: 10.1088/1674-1137/abddaf.
    Covalent radii: Cordero et al., Dalton Trans. 2008, 2832-2838.
        DOI: 10.1039/b801115j. Values for noble gases and elements
        without covalent bonds are van der Waals or estimated values.
"""

from collections import OrderedDict
from typing import NamedTuple


class Element(NamedTuple):
    """Annotated element record from the periodic table.

    Attributes:
        z (int): Atomic number.
        symbol (str): Element symbol (e.g., ``'H'``, ``'He'``).
        mass (float): Monoisotopic mass in unified atomic mass units (u),
            from NUBASE2020.
        covalent_radius (float | None): Single-bond covalent radius in
            Angstroms from Cordero 2008, or ``None`` for elements without
            commonly observed covalent bonding.

    """

    z: int
    symbol: str
    mass: float  # monoisotopic mass (u), NUBASE2020
    covalent_radius: float | None  # single-bond covalent radius (Å), Cordero 2008


# fmt: off
# Master table — ordered by atomic number.
# Covalent radii are None for elements not commonly encountered in bonding.
_ELEMENTS: tuple[Element, ...] = (
    Element(  1, "H",    1.007825032, 0.31),
    Element(  2, "He",   4.002603250, 0.28),
    Element(  3, "Li",   7.016004049, 1.28),
    Element(  4, "Be",   9.012182135, 0.96),
    Element(  5, "B",   11.009305466, 0.84),
    Element(  6, "C",   12.000000000, 0.76),
    Element(  7, "N",   14.003074005, 0.71),
    Element(  8, "O",   15.994914622, 0.66),
    Element(  9, "F",   18.998403205, 0.57),
    Element( 10, "Ne",  19.992440176, 0.58),
    Element( 11, "Na",  22.989769675, 1.66),
    Element( 12, "Mg",  23.985041898, 1.41),
    Element( 13, "Al",  26.981538441, 1.21),
    Element( 14, "Si",  27.976926533, 1.11),
    Element( 15, "P",   30.973761512, 1.07),
    Element( 16, "S",   31.972070690, 1.05),
    Element( 17, "Cl",  34.968852707, 1.02),
    Element( 18, "Ar",  39.962383123, 1.06),
    Element( 19, "K",   38.963706861, 2.03),
    Element( 20, "Ca",  39.962591155, 1.76),
    Element( 21, "Sc",  44.955910243, 1.70),
    Element( 22, "Ti",  47.947947053, 1.60),
    Element( 23, "V",   50.943963675, 1.53),
    Element( 24, "Cr",  51.940511904, 1.39),
    Element( 25, "Mn",  54.938049636, 1.39),
    Element( 26, "Fe",  55.934942133, 1.32),
    Element( 27, "Co",  58.933200194, 1.26),
    Element( 28, "Ni",  57.935347922, 1.24),
    Element( 29, "Cu",  62.929601079, 1.32),
    Element( 30, "Zn",  63.929146578, 1.22),
    Element( 31, "Ga",  68.925580912, 1.22),
    Element( 32, "Ge",  73.921178213, 1.20),
    Element( 33, "As",  74.921596417, 1.19),
    Element( 34, "Se",  79.916521828, 1.20),
    Element( 35, "Br",  78.918337647, 1.20),
    Element( 36, "Kr",  83.911506627, 1.16),
    Element( 37, "Rb",  84.911789341, 2.20),
    Element( 38, "Sr",  87.905614339, 1.95),
    Element( 39, "Y",   88.905847902, 1.90),
    Element( 40, "Zr",  89.904703679, 1.75),
    Element( 41, "Nb",  92.906377543, 1.64),
    Element( 42, "Mo",  97.905407846, 1.54),
    Element( 43, "Tc",  97.907215692, 1.47),
    Element( 44, "Ru", 101.904349503, 1.46),
    Element( 45, "Rh", 102.905504182, 1.42),
    Element( 46, "Pd", 105.903483087, 1.39),
    Element( 47, "Ag", 106.905093020, 1.45),
    Element( 48, "Cd", 113.903358121, 1.44),
    Element( 49, "In", 114.903878328, 1.42),
    Element( 50, "Sn", 119.902196571, 1.39),
    Element( 51, "Sb", 120.903818044, 1.39),
    Element( 52, "Te", 129.906222753, 1.38),
    Element( 53, "I",  126.904468420, 1.39),
    Element( 54, "Xe", 131.904154457, 1.40),
    Element( 55, "Cs", 132.905446870, 2.44),
    Element( 56, "Ba", 137.905241273, 2.15),
    Element( 57, "La", 138.906348160, 2.07),
    Element( 58, "Ce", 139.905434035, 2.04),
    Element( 59, "Pr", 140.907647726, 2.03),
    Element( 60, "Nd", 141.907718643, 2.01),
    Element( 61, "Pm", 144.912743879, 1.99),
    Element( 62, "Sm", 151.919728244, 1.98),
    Element( 63, "Eu", 152.921226219, 1.98),
    Element( 64, "Gd", 157.924100533, 1.96),
    Element( 65, "Tb", 158.925343135, 1.94),
    Element( 66, "Dy", 163.929171165, 1.92),
    Element( 67, "Ho", 164.930319169, 1.92),
    Element( 68, "Er", 167.932367781, 1.89),
    Element( 69, "Tm", 168.934211117, 1.90),
    Element( 70, "Yb", 173.938858101, 1.87),
    Element( 71, "Lu", 174.940767904, 1.87),
    Element( 72, "Hf", 179.946548760, 1.75),
    Element( 73, "Ta", 180.947996346, 1.70),
    Element( 74, "W",  183.950932553, 1.62),
    Element( 75, "Re", 186.955750787, 1.51),
    Element( 76, "Os", 191.961479047, 1.44),
    Element( 77, "Ir", 192.962923700, 1.41),
    Element( 78, "Pt", 194.964774449, 1.36),
    Element( 79, "Au", 196.966551609, 1.36),
    Element( 80, "Hg", 201.970625604, 1.32),
    Element( 81, "Tl", 204.974412270, 1.45),
    Element( 82, "Pb", 207.976635850, 1.46),
    Element( 83, "Bi", 208.980383241, 1.48),
    Element( 84, "Po", 208.982415788, 1.40),
    Element( 85, "At", 209.987131308, 1.50),
    Element( 86, "Rn", 222.017570472, 1.50),
    Element( 87, "Fr", 223.019730712, 2.60),
    Element( 88, "Ra", 226.025402555, 2.21),
    Element( 89, "Ac", 227.027746979, 2.15),
    Element( 90, "Th", 232.038050360, 2.06),
    Element( 91, "Pa", 231.035878898, 2.00),
    Element( 92, "U",  238.050782583, 1.96),
    Element( 93, "Np", 237.048167253, 1.90),
    Element( 94, "Pu", 244.064197650, 1.87),
    Element( 95, "Am", 243.061372686, 1.80),
    Element( 96, "Cm", 247.070346811, 1.69),
    Element( 97, "Bk", 247.070298533, None),
    Element( 98, "Cf", 251.079580056, None),
    Element( 99, "Es", 252.082972247, None),
    Element(100, "Fm", 257.095098635, None),
    Element(101, "Md", 258.098425321, None),
    Element(102, "No", 259.101024000, None),
    Element(103, "Lr", 262.109692000, None),
)
# fmt: on

# ---------------------------------------------------------------------------
# Derived lookup dicts — use these instead of duplicating element data.
# ---------------------------------------------------------------------------

ATOMIC_SYMBOLS: dict[int, str] = {e.z: e.symbol for e in _ELEMENTS}
"""Atomic number → element symbol (e.g., 1 → 'H')."""

MASSES: OrderedDict[str, float] = OrderedDict((e.symbol, e.mass) for e in _ELEMENTS)
"""Symbol → monoisotopic mass (u). Ordered by Z for backward compat."""

COVALENT_RADII: dict[str, float] = {e.symbol: e.covalent_radius for e in _ELEMENTS if e.covalent_radius is not None}
"""Symbol → single-bond covalent radius (Å), Cordero et al. 2008."""

TWO_LETTER_ELEMENTS: frozenset[str] = frozenset(e.symbol for e in _ELEMENTS if len(e.symbol) == 2) | frozenset(
    {
        # Superheavy elements (Z > 103) not in MASSES but needed for
        # atom-type parsing disambiguation.
        "Bh",
        "Cn",
        "Db",
        "Ds",
        "Fl",
        "Hs",
        "Lv",
        "Mc",
        "Mt",
        "Nh",
        "Og",
        "Rf",
        "Rg",
        "Sg",
        "Ts",
    }
)
"""All two-letter element symbols (Z=1..118) for parsing disambiguation."""
