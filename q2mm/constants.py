"""
Constants and variables used throughout Q2MM.

This module contains unit conversions, logging configuration, and other
physical constants.  Element data (masses, radii, symbols) is centralized
in :mod:`q2mm.elements`; ``MASSES`` is re-exported here for backward
compatibility.  Optimization defaults (WEIGHTS, STEPS) live in
:mod:`q2mm.optimizers.defaults`; regex patterns used by the parsers are
defined inline below, and this module is now their canonical location
following the removal of :mod:`q2mm.parsers._patterns`.
"""

import logging
import math
from collections import OrderedDict

# GAUSSIAN ENERGIES
GAUSSIAN_ENERGIES = ["HF"]

# LOGGING SETTINGS
# Kept for backwards compatibility. Do not call logging.config.dictConfig()
# with this dict at import time — configure logging once in the application
# entry point instead.
LOG_SETTINGS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "bare": {"format": "%(message)s"},
        "basic": {"format": "%(name)s %(message)s"},
        "simple": {"format": "%(asctime)s:%(name)s:%(levelname)s %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "bare", "level": logging.INFO},
        "root_file_handler": {
            "class": "logging.FileHandler",
            "filename": "root.log",
            "formatter": "basic",
            "level": 20,
        },
    },
    "loggers": {
        "__main__": {"level": 5, "propagate": True},
        "calculate": {"level": 20, "propagate": True},
        "compare": {"level": 10, "propagate": True},
        "constants": {"level": 20, "propagate": True},
        "datatypes": {"level": 20, "propagate": True},
        "filetypes": {"level": 20, "propagate": True},
        "gradient": {"level": 20, "propagate": True},
        "loop": {"level": 5, "propagate": True},
        "opt": {"level": 5, "propagate": True},
        "parameters": {"level": 20, "propagate": True},
        "simplex": {"level": 5, "propagate": True},
        "seminario": {"level": 20, "propagate": True},
    },
    "root": {"level": "NOTSET", "propagate": True, "handlers": ["console", "root_file_handler"]},
}

# --- Energy conversions ---
HARTREE_TO_KJMOL = 2625.5
HARTREE_TO_J = 4.359744650e-18
HARTREE_TO_KCALMOL = 627.51
KCAL_TO_KJ = 4.184

# --- Length conversions ---
# CODATA 2018: https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
BOHR_TO_ANG = 0.529177210903
CM_TO_ANG = 10**8

# --- Force constant conversions ---
FORCE_CONVERSION = 15.569141
EIGENVALUE_CONVERSION = 53.0883777868
AU_TO_MDYNA = 15.569141
AU_TO_MDYN_ANGLE = 4.3598
KJ_TO_DYNCM = 10**10
KJMOLA2_TO_MDYNA = 1.0 / (6.022140857e3)
MDYNA_TO_KJMOLA2 = 6.022140857e2
KJMOLA_TO_MDYN = 1.0 / (6.022140857e2)
MM3_STR = 601.99392

# --- Derived Hessian unit conversions ---
# All derived from base CODATA constants above to avoid inconsistency.
# Canonical internal Hessian unit: Hartree/Bohr² (atomic units).
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# Hartree/Bohr² ↔ kJ/(mol·Å²)
HESSIAN_AU_TO_KJMOLA2 = HARTREE_TO_KJMOL / (BOHR_TO_ANG**2)
KJMOLA2_TO_HESSIAN_AU = 1.0 / HESSIAN_AU_TO_KJMOLA2

# Hartree/Bohr² ↔ kJ/mol/nm² (OpenMM native; 1 nm = 10 Å → 1 nm² = 100 Å²)
HESSIAN_AU_TO_KJMOLNM2 = HESSIAN_AU_TO_KJMOLA2 * 100.0
KJMOLNM2_TO_HESSIAN_AU = 1.0 / HESSIAN_AU_TO_KJMOLNM2

# Legacy alias (deprecated — prefer HESSIAN_AU_TO_KJMOLA2)
HESSIAN_CONVERSION = HESSIAN_AU_TO_KJMOLA2

# --- Physical constants ---
AVO = 6.022140857e23
AMU_TO_KG = 1.66053906660e-27
SPEED_OF_LIGHT_MS = 299792458.0
RAD_TO_DEG = 180.0 / math.pi

# --- MM3 functional form coefficients ---
MM3_BOND_C3 = 2.55
MM3_BOND_C4 = (7.0 / 12.0) * 2.55**2
MM3_ANGLE_C3 = -0.014
MM3_ANGLE_C4 = 5.6e-5
MM3_ANGLE_C5 = -7.0e-7
MM3_ANGLE_C6 = 9.0e-10

# UNIT SYSTEM LABELS
AMBERFF = "KCALMOLA"
MM3FF = "MDYNA"
TINKERFF = "NOT IMPLEMENTED"
GAUSSIAN = "AU"
KJMOLA = "KJMOLA"

# MASSES — re-exported from q2mm.elements (single source of truth).
from q2mm.elements import MASSES  # noqa: E402

# ELECTRONIC STRUCTURE METHODS
gaussian_methods = ["b3lyp", "m06", "m062x", "m06L"]

# CHELPG RADII
CHELPG_RADII = OrderedDict(
    [
        ("H", 1.45),
        ("C", 1.50),
        ("N", 1.70),
        ("O", 1.70),
        ("F", 1.70),
        ("Pd", 2.40),
        ("Ir", 2.40),
        ("Ru", 2.40),
        ("Rh", 2.40),
        ("S", 2.00),
    ]
)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# ---------------------------------------------------------------------------
# STEPS and WEIGHTS are now in q2mm.optimizers.defaults.
# Regex patterns are now in q2mm.parsers._patterns.
# MacroModel constants are now in q2mm.parsers.macromodel.
#
# These cannot be re-exported here because the parsers package imports
# constants, creating circular dependencies.  Import from the canonical
# locations instead:
#   from q2mm.optimizers.defaults import STEPS, WEIGHTS
#   from q2mm.parsers.macromodel import COM_FORM, ...
#
# For the most common case (co.STEPS / co.WEIGHTS), we re-export only
# optimizers.defaults since it has no dependency on parsers.
from q2mm.optimizers.defaults import STEPS, WEIGHTS  # noqa: E402, F401

# ---------------------------------------------------------------------------
# Regex patterns — canonical location is q2mm.parsers._patterns,
# but defined here too for backward compatibility with old code using co.RE_*
# ---------------------------------------------------------------------------
import re  # noqa: E402

RE_FLOAT = r"[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_SMILES = r"[\w\-\=\(\)\.\+\[\]\*]+"
RE_SPLIT_ATOMS = r"[\s\-\(\)\=\.\[\]\*]+"
RE_SUB = r"[\w\s\-\.\*\(\)\%\=\,]+"
RE_BOND = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+({RE_FLOAT})\s+{RE_FLOAT}\s+\w+"
    rf"\s+\d+\s+({RE_SUB})\s+(\d+)"
)
RE_ANGLE = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+{RE_FLOAT}\s+"
    rf"({RE_FLOAT})\s+{RE_FLOAT}\s+{RE_FLOAT}\s+\w+\s+\d+\s+({RE_SUB})\s+(\d+)"
)
RE_TORSION = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+{RE_FLOAT}\s+"
    rf"({RE_FLOAT})\s+{RE_FLOAT}\s+\w+\s+\d+({RE_SUB})\s+(\d+)"
)
RE_T_LBL = re.compile(r"\At_(\S+)_\d+_(\d+-\d+-\d+-\d+)")
