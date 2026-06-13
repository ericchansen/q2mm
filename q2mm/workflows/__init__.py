"""Multi-stage force field parameterization workflows.

A *workflow* composes preliminary parameter estimation, one or more
optimization stages, and post-processing into a single ``run()``
entry point.  Distinct from the search algorithms in
:mod:`q2mm.optimizers` — those define *how* to search the parameter
space; workflows define *what sequence of search problems to solve*.

Implementations:

- :class:`q2mm.workflows.SingleStageWorkflow` — the standard Q2MM
  workflow: one optimization pass against the
  :class:`~q2mm.optimizers.objective.ObjectiveFunction` built from
  ``SystemData.reference``.  Equivalent to the current default
  behavior used throughout the codebase; this class makes the pattern
  composable and substitutable.
- (Planned) ``MethodE2Workflow`` — Limé & Norrby 2015 two-stage TSFF
  protocol: Method D Round 1, lock force-constants that decay to
  zero/negative, Method C Round 2 on the remainder.  Forthcoming
  per Phase 9.D of the QFUERZA-recovery work.

References
----------
- Limé, E.; Norrby, P.-O. *J. Comput. Chem.* **2015**, *36*, 244–250.
  DOI 10.1002/jcc.23797.
- Farrugia, M. et al. *J. Chem. Theory Comput.* **2025**, *22*, 469.
  DOI 10.1021/acs.jctc.5c01751.

"""

from __future__ import annotations

from q2mm.workflows.base import StageResult, Workflow, WorkflowResult
from q2mm.workflows.single_stage import SingleStageWorkflow

__all__ = [
    "SingleStageWorkflow",
    "StageResult",
    "Workflow",
    "WorkflowResult",
]
