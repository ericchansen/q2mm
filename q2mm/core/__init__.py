"""Core optimization engine for Q2MM.

Re-exports the main optimization modules for convenient access:

    from q2mm.core import optimizer, gradient, simplex, objective, linalg, parameters
"""

from q2mm import gradient
from q2mm import simplex
from q2mm import compare as objective
from q2mm import loop as optimizer
from q2mm import opt
from q2mm import parameters
from q2mm import linear_algebra as linalg
