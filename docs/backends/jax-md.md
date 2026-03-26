# JaxMDEngine

Built on the [JAX-MD](https://github.com/jax-md/jax-md) library, this
engine adds **periodic boundary conditions**, **neighbor lists**, and
**configurable 1-4 scaling** on top of JAX's differentiable energy
functions.

---

## Installation

```bash
pip install jax jaxlib jax-md
```

For GPU support:

```bash
pip install jax[cuda12] jax-md
```

---

## Configuration

```python
from q2mm.backends.mm import JaxMDEngine

engine = JaxMDEngine(
    box=(100.0, 100.0, 100.0),   # simulation box dimensions (Å)
    coulomb=None,                 # CoulombHandler; default: CutoffCoulomb(r_cut=12.0)
    nb_options=None,              # NonbondedOptions; default: r_cut=12.0
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box` | `tuple[float, float, float]` | `(100.0, 100.0, 100.0)` | Periodic box dimensions in Å |
| `coulomb` | `CoulombHandler \| None` | `CutoffCoulomb(r_cut=12.0)` | Electrostatic handler |
| `nb_options` | `NonbondedOptions \| None` | `NonbondedOptions(r_cut=12.0)` | Non-bonded cutoff and options |

---

## Supported Energy Terms

| Term | Supported |
|------|:---------:|
| Bonds (harmonic) | ✅ |
| Angles (harmonic) | ✅ |
| Torsions (cosine) | ✅ |
| Improper torsions | ❌ |
| vdW (LJ 12-6) | ✅ |
| Electrostatics | Infrastructure only (charges zeroed) |
| 1-4 scaling | ✅ Configurable (default: AMBER 0.5) |
| Periodic boundaries | ✅ |
| Neighbor lists | ✅ (jax-md native) |

**Functional forms:** Harmonic only.

---

## Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | — |
| `energy_breakdown()` | ✅ | Per-term decomposition |
| `minimize()` | ✅ | JAX gradients + SciPy L-BFGS-B |
| `hessian()` | ✅ | **Analytical** via `jax.hessian` |
| `frequencies()` | ✅ | From analytical Hessian |
| `energy_and_param_grad()` | ✅ | **Analytical** via `jax.grad` |
| `batched_energy()` | ✅ | **Vectorized** via `jax.vmap` |
| `supports_runtime_params()` | ✅ | — |
| `supports_analytical_gradients()` | ✅ | — |

---

## GPU Support

JaxMDEngine runs on whichever device JAX selects.  To use a GPU:

1. Install the CUDA-enabled JAX: `pip install jax[cuda12]`
2. Verify: `python -c "import jax; print(jax.default_backend())"`

The engine name includes the backend string (e.g., `JAX-MD (OPLSAA, gpu)`).

---

## Limitations

- **Harmonic only** — MM3 support is tracked in
  [#91](https://github.com/ericchansen/q2mm/issues/91).
- **Electrostatics zeroed** — Coulomb energy is computed with zero charges;
  charge optimization is not yet supported.
- **No improper torsions** — topology arrays are empty.
- **64-bit mode forced** — importing this module enables `jax_enable_x64`
  globally, which affects all JAX code in the process.

---

## Example

```python
from q2mm.backends.mm import JaxMDEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

mol = Q2MMMolecule.from_xyz("molecule.xyz")
ff = ForceField.create_for_molecule(mol)

engine = JaxMDEngine(box=(50.0, 50.0, 50.0))
e = engine.energy(mol, ff)
print(f"JAX-MD energy: {e:.4f} kcal/mol")

# Analytical parameter gradients
e, grad = engine.energy_and_param_grad(mol, ff)
print(f"Energy: {e:.4f}, grad shape: {grad.shape}")
```

---

## See Also

- [JaxEngine](jax-engine.md) — simpler, no periodic boundaries
- [Engine comparison table](index.md#engine-overview)
- [GPU benchmarks](../benchmarks/gpu.md)
- [API Reference: JaxMDEngine](../reference/q2mm/backends/mm/jax_md_engine.md)
