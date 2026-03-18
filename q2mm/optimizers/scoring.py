r"""Legacy scoring functions for Q2MM objective function evaluation.

Ported from q2mm.compare — the core scoring logic used by the legacy
optimizer loop. The main function is :func:`compare_data`, which computes:

.. math:: \chi^2 = w^2 (x_r - x_c)^2

where :math:`w` is a weight, :math:`x_r` is the reference data point's
value, and :math:`x_c` is the calculated (force field) value.
"""

from __future__ import annotations

from collections import defaultdict
import logging

import numpy as np

from q2mm import constants as co
from q2mm.optimizers.defaults import WEIGHTS

logger = logging.getLogger(__name__)


def data_by_type(data_iterable):
    """Group an iterable of Datum objects into a dict keyed by data type."""
    data_by_typ = {}
    for datum in data_iterable:
        if datum.typ not in data_by_typ:
            data_by_typ[datum.typ] = []
        data_by_typ[datum.typ].append(datum)
    return data_by_typ


def trim_data(dict1, dict2):
    """Remove data points not present in both datasets (within each type).

    For torsion data ('t'), matching uses filename + atom indices rather
    than exact labels, since pre/opt structures may have different indices.
    """
    for typ in dict1:
        if typ == "t":
            to_remove = []
            for d1 in dict1[typ]:
                if not any(
                    co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d1.lbl)[1]
                    and co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d1.lbl)[2]
                    for x in dict2[typ]
                ):
                    to_remove.append(d1)
            for d2 in dict2[typ]:
                if not any(
                    co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d2.lbl)[1]
                    and co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d2.lbl)[2]
                    for x in dict1[typ]
                ):
                    to_remove.append(d2)
            for datum in to_remove:
                if datum in dict1[typ] and datum in dict2[typ]:
                    raise AssertionError(
                        "The data point that is flagged to be "
                        "removed is present in both data sets."
                    )
                if datum in dict1[typ]:
                    dict1[typ].remove(datum)
                if datum in dict2[typ]:
                    dict2[typ].remove(datum)
            if to_remove:
                logger.log(20, f">>> Removed Data: {len(to_remove)}")
        dict1[typ] = np.array(dict1[typ])
        dict2[typ] = np.array(dict2[typ])
    return dict1, dict2


def compare_data(r_dict, c_dict, output=None, doprint=False):
    """Compute the legacy chi-squared objective function score.

    Scoring formula per data point:
      - Energy types (e, eo, ea, eao): score = w² × diff² / total_num_energy
      - Hessian type (h): score = w² × diff² / N_hessian
      - Other types: score = w² × diff² / N_type

    Args:
        r_dict: Reference data grouped by type (from data_by_type).
        c_dict: Calculated data grouped by type.
        output: Optional file path to write formatted output.
        doprint: If True, print formatted output to stdout.

    Returns:
        Total objective function score (float).
    """
    strings = []
    strings.append(
        "--"
        + " Label ".ljust(30, "-")
        + "--"
        + " Weight ".center(7, "-")
        + "--"
        + " R. Value ".center(11, "-")
        + "--"
        + " C. Value ".center(11, "-")
        + "--"
        + " Score ".center(11, "-")
        + "--"
        + " Row "
        + "--"
    )
    score_typ = defaultdict(float)
    num_typ = defaultdict(int)
    score_tot = 0.0
    total_num = 0
    data_types = sorted(r_dict.keys())
    total_num_energy = 0
    for typ in data_types:
        if typ in ["e", "eo", "ea", "eao"]:
            total_num_energy += len(r_dict[typ])
    for typ in data_types:
        total_num += int(len(r_dict[typ]))
        if typ in ["e", "eo", "ea", "eao"]:
            correlate_energies(r_dict[typ], c_dict[typ])
        import_weights(r_dict[typ])
        for r, c in zip(r_dict[typ], c_dict[typ]):
            if c.typ == "t":
                diff = abs(r.val - c.val)
                if diff > 180.0:
                    diff = 360.0 - diff
            else:
                diff = r.val - c.val
            if typ in ["e", "eo", "ea", "eao"]:
                score = (r.wht**2 * diff**2) / total_num_energy
            elif typ == "h":
                score = (c.wht**2 * diff**2) / len(c_dict[typ])
            else:
                score = (r.wht**2 * diff**2) / len(r_dict[typ])
            score_tot += score
            score_typ[c.typ] += score
            num_typ[c.typ] += 1
            if c.typ == "eig":
                if c.idx_1 == c.idx_2:
                    if r.val < 1100:
                        score_typ[c.typ + "-d-low"] += score
                        num_typ[c.typ + "-d-low"] += 1
                    else:
                        score_typ[c.typ + "-d-high"] += score
                        num_typ[c.typ + "-d-high"] += 1
                else:
                    score_typ[c.typ + "-o"] += score
                    num_typ[c.typ + "-o"] += 1
            if c.ff_row is None:
                strings.append(
                    f"  {c.lbl:<30}  {r.wht:>7.2f}  {r.val:>11.4f}  {c.val:>11.4f}  {score:>11.4f}  "
                )
            else:
                strings.append(
                    f"  {c.lbl:<30}  {r.wht:>7.2f}  {r.val:>11.4f}  {c.val:>11.4f}  {score:>11.4f}  {c.ff_row:>5} "
                )
    strings.append("-" * 89)
    strings.append("{:<20} {:20.4f}".format("Total score:", score_tot))
    strings.append("{:<30} {:10d}".format("Total Num. data points:", total_num))
    for k, v in num_typ.items():
        strings.append("{:<30} {:10d}".format(k + ":", v))
    strings.append("-" * 89)
    for k, v in score_typ.items():
        strings.append("{:<20} {:20.4f}".format(k + ":", v))
    if output:
        with open(output, "w") as f:
            for line in strings:
                f.write(f"{line}\n")
    if doprint:
        for line in strings:
            print(line)
    return score_tot


def correlate_energies(r_data, c_data):
    """Align calculated energies to reference by setting the minimum to zero.

    Finds the minimum in the reference dataset, then shifts all calculated
    energies so the corresponding calculated value is zero.

    Both datasets must be aligned (same ordering).
    """
    for indices in select_group_of_energies(c_data):
        zero, zero_ind = min((x.val, i) for i, x in enumerate(r_data[indices]))
        zero_ind = indices[zero_ind]
        zero = c_data[zero_ind].val
        for ind in indices:
            c_data[ind].val -= zero


def select_group_of_energies(data):
    """Yield index arrays for each group of energies in the dataset."""
    for energy_type in ["e", "eo"]:
        indices = np.where([x.typ == energy_type for x in data])[0]
        unique_group_nums = set([x.idx_1 for x in data[indices]])
        for unique_group_num in unique_group_nums:
            more_indices = np.where(
                [x.typ == energy_type and x.idx_1 == unique_group_num for x in data]
            )[0]
            yield more_indices


def import_weights(data):
    """Set weights on data points from the default WEIGHTS table.

    Only sets weights on data points where ``datum.wht is None``.
    Eigenvalue data gets special handling based on diagonal/off-diagonal
    and frequency value.
    """
    for datum in data:
        if datum.wht is None:
            if datum.typ == "eig":
                if datum.idx_1 == datum.idx_2 == 1:
                    datum.wht = WEIGHTS["eig_i"]
                elif datum.idx_1 == datum.idx_2:
                    if datum.val < 1100:
                        datum.wht = WEIGHTS["eig_d_low"]
                    else:
                        datum.wht = WEIGHTS["eig_d_high"]
                elif datum.idx_1 != datum.idx_2:
                    datum.wht = WEIGHTS["eig_o"]
            else:
                datum.wht = WEIGHTS[datum.typ]
