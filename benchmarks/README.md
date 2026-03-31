# Q2MM Benchmark Data

Archived results from `q2mm-benchmark` runs, committed for scientific
reproducibility.

## Directory Layout

```
benchmarks/
├── ch3f/                        # CH₃F (fluoromethane) — 1 molecule, 8 params
│   ├── results/                 # JSON result files
│   │   ├── ch3f_jax_harmonic_cpu_lbfgsb.json
│   │   ├── ch3f_openmm_mm3_gpu_powell.json
│   │   └── ...
│   └── forcefields/             # Optimized force fields in native formats
│       ├── ch3f_jax_harmonic_cpu_lbfgsb.fld
│       ├── ch3f_jax_harmonic_cpu_lbfgsb.prm
│       └── ...
├── rh-enamide/                  # Rh-enamide — 9 molecules, 182 params
│   ├── results/
│   │   ├── rh-enamide_jax-md_oplsaa_cpu_lbfgsb.json
│   │   └── ...
│   └── forcefields/
│       └── ...
└── README.md
```

## File Naming Convention

Pattern: `{system}_{engine}_{ff}_{device}_{optimizer}.{ext}`

All segments are **lowercase**, separated by underscores.  Hyphens only
appear within naturally hyphenated names (e.g. `jax-md`, `rh-enamide`,
`nelder-mead`).

| Segment | Values | Description |
|---------|--------|-------------|
| `system` | `ch3f`, `rh-enamide` | Molecular system |
| `engine` | `jax`, `jax-md`, `openmm`, `tinker` | Compute engine |
| `ff` | `harmonic`, `oplsaa`, `mm3` | Force field type |
| `device` | `cpu`, `gpu` | Execution device |
| `optimizer` | `lbfgsb`, `nelder-mead`, `powell`, `cycling` | Optimization strategy |

Glob examples:

```bash
ls *_gpu_*.json          # all GPU results
ls *_lbfgsb.*            # all L-BFGS-B results + FFs
ls rh-enamide_*.json     # all Rh-enamide results
ls *_jax-md_*.json       # all JAX-MD results
```

## Systems

| System | Molecules | Atoms | Parameters | QM Level |
|--------|----------:|------:|-----------:|----------|
| **CH₃F** | 1 | 5 | 8 | B3LYP/6-31+G(d) (Psi4) |
| **Rh-enamide** | 9 | 36–62 | 182 | B3LYP/LACVP** (Jaguar) |

## Force Field Formats

Each optimized force field is saved in all formats compatible with its
functional form:

| Functional Form | Formats |
|-----------------|---------|
| MM3             | `.fld` (Schrödinger MM3), `.prm` (Tinker), `.xml` (OpenMM) |
| Harmonic        | `.frcmod` (AMBER) |

In this archive, CH₃F force fields (including harmonic-engine variants
like JAX and JAX-MD) are stored in `.fld`, `.prm`, and `.xml` formats
because the underlying force field template uses MM3 functional form.
The engine's functional form determines the energy expression, but the
serialized parameters are the same.

Rh-enamide force fields from JAX/JAX-MD use `.frcmod` (AMBER harmonic
format) because those engines operate on a harmonic copy of the force
field.  OpenMM uses the native MM3 formats (`.fld`, `.prm`, `.xml`).

## GPU Acceleration

See [GPU_BENCHMARKS.md](GPU_BENCHMARKS.md) for detailed GPU vs CPU
benchmarks on NVIDIA RTX 5090.  Summary: JAX-MD OPLSAA achieves
**5.61× per-evaluation speedup** on the Rh-enamide system; JAX harmonic
achieves **2.08×**.  Small molecules (CH₃F) are faster on CPU.

## Reproducing

```bash
# Re-run CH₃F benchmarks (default system)
q2mm-benchmark --output benchmarks/ch3f

# Run Rh-enamide benchmarks (slow — ~3 min per optimizer on OpenMM)
q2mm-benchmark --system rh-enamide --output benchmarks/rh-enamide

# Quick run with limited iterations
q2mm-benchmark --system rh-enamide --max-iter 2 --output benchmarks/rh-enamide

# Run only specific backends/optimizers
q2mm-benchmark --system rh-enamide --backend jax --optimizer L-BFGS-B

# Load and display saved results without re-running
q2mm-benchmark --load benchmarks/rh-enamide/results
```

## QM Reference

**CH₃F:**

- **Level of theory**: B3LYP/6-31+G(d)
- **Software**: Psi4
- **Reference data**: `examples/sn2-test/qm-reference/`

**Rh-enamide:**

- **Level of theory**: B3LYP/LACVP** (Hay-Wadt ECP for Rh)
- **Software**: Jaguar (Schrödinger)
- **Reference data**: `examples/rh-enamide/`
