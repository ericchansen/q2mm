# JAX Backends

Q2MM provides two JAX-based backends — `JaxEngine` and `JaxMDEngine` — both
offering **analytical gradients** via `jax.grad` and **JIT compilation** for
fast iterative optimization.

---

## JaxEngine

A lightweight, pure-JAX implementation of harmonic bond, angle, torsion,
and Lennard-Jones energy terms. Best for small-to-medium molecules where
periodic boundaries and neighbor lists are not needed.

### Installation

```bash
pip install jax jaxlib
```

For GPU support, install the CUDA-enabled `jaxlib`:

```bash
pip install jax[cuda12]
```

!!! tip "Verify installation"
    ```python
    import jax
    print(jax.__version__)
    print(jax.default_backend())  # "cpu" or "gpu"
    ```

### Configuration

```python
from q2mm.backends.mm import JaxEngine

engine = JaxEngine()
```

JaxEngine has no constructor parameters. It runs on whichever JAX backend
is active (`cpu` or `gpu`), detected via `jax.default_backend()`.

### Supported Energy Terms

| Term | Supported |
|------|:---------:|
| Bonds (harmonic) | ✅ |
| Angles (harmonic) | ✅ |
| Torsions (cosine) | ✅ |
| Improper torsions | ❌ |
| vdW (LJ 12-6) | ✅ |
| Electrostatics | ❌ |
| 1-4 scaling | ❌ Not implemented |

**Functional forms:** Harmonic only. MM3 support is tracked in
[#91](https://github.com/ericchansen/q2mm/issues/91).

### Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | Pure JAX |
| `minimize()` | ✅ | JAX gradients + SciPy L-BFGS-B |
| `hessian()` | ✅ | **Analytical** via `jax.hessian` |
| `frequencies()` | ✅ | From analytical Hessian |
| `energy_and_param_grad()` | ✅ | **Analytical** via `jax.grad` |
| `supports_runtime_params()` | ✅ | — |
| `supports_analytical_gradients()` | ✅ | — |

### Limitations

- **No 1-4 pair scaling** — non-bonded energies differ from OpenMM/JAX-MD
  for molecules with 1-4 interactions. See the
  [compatibility notes](index.md#when-becomes).
- **No periodic boundaries** — gas-phase only.
- **Harmonic only** — cannot evaluate MM3 force fields.

---

## JaxMDEngine

Built on the [JAX-MD](https://github.com/jax-md/jax-md) library, this
engine adds **periodic boundary conditions**, **neighbor lists**, and
**configurable 1-4 scaling** on top of JAX's differentiable energy
functions.

### Installation

```bash
pip install jax jaxlib jax-md
```

For GPU support:

```bash
pip install jax[cuda12] jax-md
```

### Configuration

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

### Supported Energy Terms

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

### Capabilities

| Method | Supported | Notes |
|--------|:---------:|-------|
| `energy()` | ✅ | — |
| `energy_breakdown()` | ✅ | Per-term decomposition |
| `minimize()` | ✅ | JAX gradients + SciPy L-BFGS-B |
| `hessian()` | ✅ | **Analytical** via `jax.hessian` |
| `frequencies()` | ✅ | From analytical Hessian |
| `energy_and_param_grad()` | ✅ | **Analytical** via `jax.grad` |
| `supports_runtime_params()` | ✅ | — |
| `supports_analytical_gradients()` | ✅ | — |

### Limitations

- **Harmonic only** — MM3 support is tracked in
  [#91](https://github.com/ericchansen/q2mm/issues/91).
- **Electrostatics zeroed** — Coulomb energy is computed with zero charges;
  charge optimization is not yet supported.
- **No improper torsions** — topology arrays are empty.
- **64-bit mode forced** — importing this module enables `jax_enable_x64`
  globally, which affects all JAX code in the process.

---

## GPU Support

Both JAX backends run on whichever device JAX selects. To use a GPU:

1. Install the CUDA-enabled JAX: `pip install jax[cuda12]`
2. Verify: `python -c "import jax; print(jax.default_backend())"`

The engine name includes the backend string (e.g., `JAX (harmonic, gpu)`
or `JAX-MD (OPLSAA, gpu)`).

!!! info "Performance"
    JAX engines are 5–10× faster than OpenMM and ~1000× faster than Tinker
    per energy evaluation. Combined with analytical gradients, this makes
    them the preferred choice for large-scale optimization. See the
    [benchmarks](../benchmarks/index.md) for detailed comparisons.

---

## Example

```python
from q2mm.backends.mm import JaxEngine, JaxMDEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

mol = Q2MMMolecule.from_xyz("molecule.xyz")
ff = ForceField.from_amber_frcmod("params.frcmod", mol)

# JaxEngine — simple, no periodic boundaries
engine = JaxEngine()
e = engine.energy(mol, ff)
print(f"JAX energy: {e:.4f} kcal/mol")

# JaxMDEngine — periodic boundaries, neighbor lists
engine_md = JaxMDEngine(box=(50.0, 50.0, 50.0))
e_md = engine_md.energy(mol, ff)
print(f"JAX-MD energy: {e_md:.4f} kcal/mol")

# Analytical parameter gradients (both engines)
e, grad = engine.energy_and_param_grad(mol, ff)
print(f"Energy: {e:.4f}, grad shape: {grad.shape}")
```

---

## See Also

- [Engine comparison table](index.md#engine-overview)
- [Parameter transferability](index.md#parameter-transferability)
- [Benchmarks](../benchmarks/index.md)
- [API Reference: JaxEngine](../reference/q2mm/backends/mm/jax_engine.md)
- [API Reference: JaxMDEngine](../reference/q2mm/backends/mm/jax_md_engine.md)
