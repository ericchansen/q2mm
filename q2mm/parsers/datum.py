"""Data point container for reference and calculated values."""

from __future__ import annotations

import re


class Datum:
    """A single reference or calculated data point used in force field optimization.

    Attributes:
        val: Numerical value of the data point.
        wht: Weight for this data point in the objective function.
        typ: Data type identifier (e.g., 'b' for bond, 'a' for angle, 't' for torsion).
        com: Comment or description.
        src_1: Primary source file name.
        src_2: Secondary source file name.
        idx_1: Primary structure index.
        idx_2: Secondary structure index.
        atm_1: First atom number.
        atm_2: Second atom number.
        atm_3: Third atom number.
        atm_4: Fourth atom number.
        ff_row: Force field file row number.
    """

    __slots__ = [
        "_lbl",
        "val",
        "wht",
        "typ",
        "com",
        "src_1",
        "src_2",
        "idx_1",
        "idx_2",
        "atm_1",
        "atm_2",
        "atm_3",
        "atm_4",
        "ff_row",
    ]

    def __init__(
        self,
        lbl: str | None = None,
        val: float | None = None,
        wht: float | None = None,
        typ: str | None = None,
        com: str | None = None,
        src_1: str | None = None,
        src_2: str | None = None,
        idx_1: int | None = None,
        idx_2: int | None = None,
        atm_1: int | None = None,
        atm_2: int | None = None,
        atm_3: int | None = None,
        atm_4: int | None = None,
        ff_row: int | None = None,
    ):
        self._lbl = lbl
        self.val = val
        self.wht = wht
        self.typ = typ
        self.com = com
        self.src_1 = src_1
        self.src_2 = src_2
        self.idx_1 = idx_1
        self.idx_2 = idx_2
        self.atm_1 = atm_1
        self.atm_2 = atm_2
        self.atm_3 = atm_3
        self.atm_4 = atm_4
        self.ff_row = ff_row

    def __repr__(self) -> str:
        val_str = f"{self.val:7.4f}" if self.val is not None else "None"
        return f"{self.lbl}({val_str})"

    @property
    def lbl(self) -> str:
        """Auto-generated label from type, source, index, and atom fields."""
        if self._lbl is None:
            a = self.typ
            b = re.split(r"[.]+", self.src_1)[0] if self.src_1 else None
            c = "-".join(str(x) for x in remove_none(self.idx_1, self.idx_2))
            d = "-".join(str(x) for x in remove_none(self.atm_1, self.atm_2, self.atm_3, self.atm_4))
            self._lbl = "_".join(remove_none(a, b, c, d))
        return self._lbl


def remove_none(*args):
    """Filter out None and empty-string values."""
    return [x for x in args if x is not None and x != ""]


def datum_sort_key(datum: Datum) -> tuple:
    """Sort key ensuring calculated and reference data points align properly."""
    return (datum.typ, datum.src_1, datum.src_2, datum.idx_1, datum.idx_2)
