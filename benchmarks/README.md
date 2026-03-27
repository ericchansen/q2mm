# Q2MM Benchmark Data

Archived results from `q2mm-benchmark` runs, committed for scientific
reproducibility.

## Directory Layout

```
benchmarks/
├── ch3f/                        # CH₃F (fluoromethane) — 1 molecule, 8 params
│   ├── results/                 # JSON result files (BenchmarkResult format)
│   │   ├── OpenMM_L-BFGS-B.json
│   │   └── ...
│   └── forcefields/             # Optimized force fields in native formats
│       ├── OpenMM_L-BFGS-B.fld  # MM3 .fld format
│       ├── OpenMM_L-BFGS-B.prm  # Tinker .prm format
│       ├── OpenMM_L-BFGS-B.xml  # OpenMM XML format
│       └── ...
├── rh-enamide/                  # Rh-enamide — 9 molecules, 182 params
│   ├── results/                 # JSON result files (BenchmarkResult format)
│   │   ├── JAX_(harmonic,_cpu)_L-BFGS-B.json
│   │   ├── JAX-MD_(OPLSAA,_cpu)_L-BFGS-B.json
│   │   ├── OpenMM_(CUDA)_Nelder-Mead.json
│   │   └── ...
│   └── forcefields/             # Optimized force fields
│       ├── JAX_(harmonic,_cpu)_L-BFGS-B.frcmod
│       ├── OpenMM_(CUDA)_Nelder-Mead.fld
│       └── ...
└── README.md
```

## Systems

| System | Molecules | Atoms | Parameters | QM Level |
|--------|----------:|------:|-----------:|----------|
| **CH₃F** | 1 | 5 | 8 | B3LYP/6-31+G(d) (Psi4) |
| **Rh-enamide** | 9 | 36–62 | 182 | B3LYP/LACVP** (Jaguar) |

## File Naming

Files follow the pattern `{Backend}_{Optimizer}.{ext}`:

- **Backend**: `JAX_(harmonic,_cpu)`, `JAX-MD_(OPLSAA,_cpu)`,
  `OpenMM_(CUDA)`, `OpenMM`, `Tinker`
- **Optimizer**: `L-BFGS-B`, `Nelder-Mead`, `Powell`

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

## Reproducing

```bash
# Re-run CH₃F benchmarks (default system)
q2mm-benchmark --output benchmarks/ch3f

# Run Rh-enamide benchmarks (slow — ~3 min per optimizer on OpenMM)
q2mm-benchmark --system rh-enamide --output benchmarks/rh-enamide/

# Quick run with limited iterations
q2mm-benchmark --system rh-enamide --max-iter 2 --output benchmarks/rh-enamide/

# Run only specific backends/optimizers
q2mm-benchmark --system rh-enamide --backend jax --optimizer Nelder-Mead

# Load and display saved results without re-running
q2mm-benchmark --load benchmarks/rh-enamide/
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
