# Agent Instructions

## Before Every Commit

Run the **exact same** lint and format checks that CI runs. If either
fails, fix the issues before committing.

```bash
python -m ruff check q2mm/ test/ scripts/
python -m ruff format --check q2mm test scripts examples
```

## Testing

Run core tests (no backend-specific markers):

```bash
python -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"
```

Backend tests require Docker containers. Use `scripts/ci_local.sh --all`
to run the full CI matrix locally.

## Git

- GPG signing is broken (expired key). Use `git -c commit.gpgsign=false commit`.
- Never push directly to `main` — always use a feature branch + PR.
- Use conventional commit prefixes: `feat`, `fix`, `docs`, `refactor`,
  `chore`, `test`, `ci`, `perf`.
