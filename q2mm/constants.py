"""
Constants and variables used throughout Q2MM.

This module contains unit conversions, atomic masses, logging configuration,
and other physical constants. Optimization defaults (WEIGHTS, STEPS) have been
moved to q2mm.optimizers.defaults; regex patterns to q2mm.parsers._patterns;
MacroModel constants to q2mm.parsers.macromodel. Backward-compatible re-exports
are provided at the bottom of this module.
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

# MASSES
MASSES = OrderedDict(
    [
        ("H", 1.007825032),
        ("He", 4.002603250),
        ("Li", 7.016004049),
        ("Be", 9.012182135),
        ("B", 11.009305466),
        ("C", 12.000000000),
        ("N", 14.003074005),
        ("O", 15.994914622),
        ("F", 18.998403205),
        ("Ne", 19.992440176),
        ("Na", 22.989769675),
        ("Mg", 23.985041898),
        ("Al", 26.981538441),
        ("Si", 27.976926533),
        ("P", 30.973761512),
        ("S", 31.972070690),
        ("Cl", 34.968852707),
        ("Ar", 39.962383123),
        ("K", 38.963706861),
        ("Ca", 39.962591155),
        ("Sc", 44.955910243),
        ("Ti", 47.947947053),
        ("V", 50.943963675),
        ("Cr", 51.940511904),
        ("Mn", 54.938049636),
        ("Fe", 55.934942133),
        ("Co", 58.933200194),
        ("Ni", 57.935347922),
        ("Cu", 62.929601079),
        ("Zn", 63.929146578),
        ("Ga", 68.925580912),
        ("Ge", 73.921178213),
        ("As", 74.921596417),
        ("Se", 79.916521828),
        ("Br", 78.918337647),
        ("Kr", 83.911506627),
        ("Rb", 84.911789341),
        ("Sr", 87.905614339),
        ("Y", 88.905847902),
        ("Zr", 89.904703679),
        ("Nb", 92.906377543),
        ("Mo", 97.905407846),
        ("Tc", 97.907215692),
        ("Ru", 101.904349503),
        ("RH", 102.905504182),
        ("Rh", 102.905504182),
        ("Pd", 105.903483087),
        ("Ag", 106.905093020),
        ("Cd", 113.903358121),
        ("In", 114.903878328),
        ("Sn", 119.902196571),
        ("Sb", 120.903818044),
        ("Te", 129.906222753),
        ("I", 126.904468420),
        ("Xe", 131.904154457),
        ("Cs", 132.905446870),
        ("Ba", 137.905241273),
        ("La", 138.906348160),
        ("Ce", 139.905434035),
        ("Pr", 140.907647726),
        ("Nd", 141.907718643),
        ("Pm", 144.912743879),
        ("Sm", 151.919728244),
        ("Eu", 152.921226219),
        ("Gd", 157.924100533),
        ("Tb", 158.925343135),
        ("Dy", 163.929171165),
        ("Ho", 164.930319169),
        ("Er", 167.932367781),
        ("Tm", 168.934211117),
        ("Yb", 173.938858101),
        ("Lu", 174.940767904),
        ("Hf", 179.946548760),
        ("Ta", 180.947996346),
        ("W", 183.950932553),
        ("Re", 186.955750787),
        ("Os", 191.961479047),
        ("Ir", 192.962923700),
        ("Pt", 194.964774449),
        ("Au", 196.966551609),
        ("Hg", 201.970625604),
        ("Tl", 204.974412270),
        ("Pb", 207.976635850),
        ("Bi", 208.980383241),
        ("Po", 208.982415788),
        ("At", 209.987131308),
        ("Rn", 222.017570472),
        ("Fr", 223.019730712),
        ("Ra", 226.025402555),
        ("Ac", 227.027746979),
        ("Th", 232.038050360),
        ("Pa", 231.035878898),
        ("U", 238.050782583),
        ("Np", 237.048167253),
        ("Pu", 244.064197650),
        ("Am", 243.061372686),
        ("Cm", 247.070346811),
        ("Bk", 247.070298533),
        ("Cf", 251.079580056),
        ("Es", 252.082972247),
        ("Fm", 257.095098635),
        ("Md", 258.098425321),
        ("No", 259.101024000),
        ("Lr", 262.109692000),
    ]
)

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
#   from q2mm.parsers._patterns import RE_FLOAT, RE_BOND, ...
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

# MacroModel constants — canonical location is q2mm.parsers.macromodel
COM_FORM = " {0:4}{1:>8}{2:>7}{3:>7}{4:>7}{5:>11.4f}{6:>11.4f}{7:>11.4f}{8:>11.4f}\n"
LABEL_SUITE = r"SUITE_\w+"
LABEL_MACRO = "MMOD_MACROMODEL"
LIC_SUITE = re.compile(rf"(?<!_){LABEL_SUITE}\s+(\d+)\sof\s\d+\s" r"tokens\savailable")
LIC_MACRO = re.compile(rf"{LABEL_MACRO}\s+(\d+)\sof\s\d+\stokens\s" "available")
MIN_SUITE_TOKENS = 2
MIN_MACRO_TOKENS = 2
