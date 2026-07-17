# CH3F and bring-your-own FCHK

This is the canonical fresh ground-state smoke example. It uses the installed
CH3F reference resource by default, or a caller-owned Gaussian formatted
checkpoint with `--fchk`. Both paths call the package-root
`q2mm.prepare`, `q2mm.evaluate`, `q2mm.optimize`, and `q2mm.save` functions.

```bash
python run.py --bounded-ci --output-root ./output
python run.py --fchk /path/to/molecule.fchk --stationary-point ground_state \
  --bounded-ci --output-root ./output
```

`--bounded-ci` enters the real objective once with a deterministic echo backend;
it is a software-path check, not a convergence claim. Omit it to use the
documented JAX scientific workflow. SciPy is an optional optimizer dependency,
not part of q2mm core.
