# Publication case studies

These six scripts exercise real multi-molecule repository benchmark problems
through the canonical settings/build/capture/evaluate/optimize/run/save lifecycle.
The package-root exports are the same objects as their owning modules.
Repository loaders perform parsing and delegate problem construction
to `q2mm.prepare`; they are teaching helpers, not a second public SDK.

Every invocation requires an explicit output root and the external scientific
roots needed by that system. `--bounded-ci` preserves the complete case
membership, force-field composition, observations, and active/frozen partition,
then enters the optimizer once using a deterministic echo backend. It is a
software smoke, not native-backend validation or a convergence run.

Native execution defaults to JAX with analytical gradients. A different MM
backend requires an explicit executor choice, for example
`--backend openmm --executor python`; incompatible JAX-objective/backend
combinations fail typed validation. The explicit Python choice retains each
system's GS/TS tolerance and scientific bounds, including Heck's tighter
force-constant fraction. It is not a backend-name fallback.

The structured JSON output records citation URLs, source and reproduction
status, case order, stationary point, force-field blocks, parameter counts,
QFUERZA audit, initial/final objective categories, resolved execution choices,
saved paths, and blocked exact rows.

No recovered dissertation archive or standalone MM3 base is copied into this
tree. The Rh-enamide inputs are tracked in the source repository but excluded
from wheel and sdist artifacts; redistribution/licensing is not established.
