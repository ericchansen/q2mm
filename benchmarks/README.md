# Q2MM Benchmark Data

Archived results from `q2mm-benchmark` runs, committed for scientific
reproducibility.

## Directory Layout

```
benchmarks/
├── ch3f/                        # CH₃F (fluoromethane) system
│   ├── results/                 # JSON result files from q2mm-benchmark
│   │   ├── OpenMM_L-BFGS-B.json
│   │   └── ...
│   └── forcefields/             # Optimized force fields in native formats
│       ├── OpenMM_L-BFGS-B.fld  # MM3 .fld format
│       ├── OpenMM_L-BFGS-B.prm  # Tinker .prm format
│       ├── OpenMM_L-BFGS-B.xml  # OpenMM XML format
│       └── ...
└── README.md
```

## File Naming

Files follow the pattern `{Backend}_{Optimizer}.{ext}`:

- **Backend**: `JAX_(harmonic)`, `JAX-MD_(OPLSAA)`, `OpenMM`, `Tinker`
- **Optimizer**: `L-BFGS-B`, `Nelder-Mead`, `Powell`

## Force Field Formats

Each optimized force field is saved in all formats compatible with its
functional form:

| Functional Form | Formats |
|-----------------|---------|
| MM3             | `.fld` (Schrödinger MM3), `.prm` (Tinker), `.xml` (OpenMM) |
| Harmonic        | `.frcmod` (AMBER) |

CH₃F uses the MM3 functional form, so all FF files are in `.fld`,
`.prm`, and `.xml` formats.

## Reproducing

```bash
# Re-run the full benchmark matrix (saves to ./benchmark_results/ by default)
q2mm-benchmark

# Re-run and save to a specific directory
q2mm-benchmark --output benchmarks/ch3f

# Load and display saved results without re-running
q2mm-benchmark --load benchmarks/ch3f
```

## QM Reference

- **Molecule**: CH₃F (fluoromethane)
- **Level of theory**: B3LYP/6-31+G(d)
- **Software**: Psi4
- **Reference data**: `examples/sn2-test/qm-reference/`
