from __future__ import annotations
import numpy as np
import re


class Datum:
    """
    Class for a reference or calculated data point. TODO
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
        lbl=None,
        val=None,
        wht=None,
        typ=None,
        com=None,
        src_1=None,
        src_2=None,
        idx_1=None,
        idx_2=None,
        atm_1=None,
        atm_2=None,
        atm_3=None,
        atm_4=None,
        ff_row=None,
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

    def __repr__(self):
        return f"{self.lbl}({self.val:7.4f})"

    @property
    def lbl(self):
        if self._lbl is None:
            a = self.typ
            if self.src_1:
                b = re.split("[.]+", self.src_1)[0]
            # Why would it ever not have src_1?
            else:
                b = None
            c = "-".join([str(x) for x in remove_none(self.idx_1, self.idx_2)])
            d = "-".join([str(x) for x in remove_none(self.atm_1, self.atm_2, self.atm_3, self.atm_4)])
            abcd = remove_none(a, b, c, d)
            self._lbl = "_".join(abcd)
        return self._lbl


def remove_none(*args):
    return [x for x in args if (x is not None and x != "")]


def datum_sort_key(datum):
    """
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    """
    return (datum.typ, datum.src_1, datum.src_2, datum.idx_1, datum.idx_2)
