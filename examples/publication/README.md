# Publication case studies

These six scripts exercise real multi-molecule repository benchmark problems
through the package-root `q2mm.evaluate`, `q2mm.optimize`, and `q2mm.save`
functions. Repository loaders perform parsing and delegate problem construction
to `q2mm.prepare`; they are teaching helpers, not a second public SDK.

Every invocation requires an explicit output root and the external scientific
roots needed by that system. `--bounded-ci` preserves the complete case
membership, force-field composition, observations, and active/frozen partition,
then enters the optimizer once. It is not a convergence run.

The structured JSON output records citation URLs, source and reproduction
status, case order, stationary point, force-field blocks, parameter counts,
QFUERZA audit, initial/final objective categories, resolved execution choices,
saved paths, and blocked exact rows.

No recovered dissertation archive or standalone MM3 base is copied into this
tree. The Rh-enamide inputs are tracked in the source repository but excluded
from wheel and sdist artifacts; redistribution/licensing is not established.
