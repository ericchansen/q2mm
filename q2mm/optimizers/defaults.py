"""Default optimization parameters: step sizes and data weights.

These were originally in q2mm.constants and are used by the legacy
optimizer loop for numerical differentiation and scoring.
"""

from __future__ import annotations

# Step sizes for numerical differentiation by parameter type.
# Float values: x_new = x +/- step
# String values: x_new = x +/- (x * step)  (percentage)
STEPS = {
    "ae": 1.0,
    "af": 0.1,
    "be": 0.02,
    "bf": 0.1,
    "df": 0.1,
    "imp1": 0.2,
    "imp2": 0.2,
    "op_b": 0.2,
    "sb": 0.2,
    "q": 0.1,
    "q_p": 0.05,
    "vdwr": 0.1,
    "vdwfc": 0.02,
}

# Default weights for different data types in the objective function.
WEIGHTS = {
    "a": 2.00,
    "b": 100.00,
    "t": 1.00,
    "h": 0.031,
    "h12": 0.031,
    "h13": 0.031,
    "h14": 0.31,
    "eig_i": 0.00,
    "eig_d_low": 0.10,
    "eig_d_high": 0.10,
    "eig_o": 0.05,
    "e": 20.00,
    "e1": 20.00,
    "eo": 100.00,
    "e1o": 100.00,
    "ea": 20.00,
    "eao": 100.00,
    "q": 10.00,
    "qh": 10.00,
    "qa": 10.00,
    "esp": 10.00,
    "p": 10.00,
}
