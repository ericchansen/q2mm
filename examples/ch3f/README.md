# CH3F and bring-your-own FCHK

This is the canonical fresh ground-state smoke example. It uses the installed
CH3F reference resource by default, or a caller-owned Gaussian formatted
checkpoint with `--fchk`. Both paths call the package-root
canonical preparation, immutable settings, component construction, pre-run
capture, full-vector evaluation, numerical optimization, run assembly and
persistence functions.

```bash
python run.py --bounded-ci --output-root ./output
python run.py --fchk /path/to/molecule.fchk --stationary-point ground_state \
  --bounded-ci --output-root ./output
```

`--bounded-ci` enters the real objective once with a deterministic echo backend;
it is a software-path check, not native-backend validation or a convergence
claim. Omit it to use the
documented JAX scientific workflow. SciPy is an optional optimizer dependency,
not part of q2mm core.

Backend and executor are independent explicit choices. With a non-JAX backend,
pass both, for example `--backend openmm --executor python` (install the
backend and SciPy extras first). This uses the Python objective and SciPy's
finite differences without changing the ground-state optimizer defaults.
`--backend openmm` alone is rejected rather than silently changing the executor.
