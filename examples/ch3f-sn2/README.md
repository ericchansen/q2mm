# CH3F identity-SN2 transition state

This canonical small transition-state example uses the installed q2mm SN2
resource and makes transition-state semantics explicit. `q2mm.prepare` derives
curvature inversion from `stationary_point="transition_state"`; the script does
not expose a contradictory switch.

```bash
python run.py --bounded-ci --output-root ./output
```

Omit `--bounded-ci` for the JAX scientific workflow. The bounded mode constructs
the real SN2 problem and enters its optimizer once, but does not claim
convergence. The source-only regeneration utility is
`scripts/generate_sn2_reference.py`; generated reference payloads have one
canonical location, `q2mm/data/sn2/`.
