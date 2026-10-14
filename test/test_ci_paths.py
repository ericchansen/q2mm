"""Representative native CI path decisions and fail-closed Git handling."""

import subprocess

import pytest

from scripts import ci_paths


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("q2mm/settings.py", set(ci_paths.JOB_COMPONENTS)),
        ("q2mm/backends/mm/openmm.py", {"openmm-cpu", "parity-openmm-jax", "parity-openmm-tinker"}),
        ("q2mm/backends/mm/tinker.py", {"tinker", "parity-openmm-tinker"}),
        ("q2mm/backends/mm/jax_engine.py", {"parity-openmm-jax", "parity-jax-jaxmd"}),
        ("q2mm/backends/mm/jax_md_engine.py", {"jax-md-cpu", "parity-jax-jaxmd"}),
        ("q2mm/backends/qm/psi4.py", {"psi4-reference", "parity-psi4-qcengine"}),
        ("q2mm/backends/reference/qcengine.py", {"parity-psi4-qcengine"}),
        (".github/envs/openmm-jax.yml", {"parity-openmm-jax"}),
        (".github/envs/openmm.yml", {"openmm-cpu"}),
        ("docs/backends/openmm.md", set()),
        ("q2mm/optimizers/scipy_opt.py", set()),
        ("test/test_tinker_writers.py", {"tinker"}),
        ("test/integration/test_qcengine_psi4_parity.py", {"parity-psi4-qcengine"}),
        ("docs/platform-support.md", set()),
        ("q2mm/backends/reference/ase.py", set()),
        ("q2mm/io/qcelemental.py", set()),
        ("test/test_jax_executor.py", set()),
    ],
)
def test_representative_path_decisions(path: str, expected: set[str]) -> None:
    assert {job for job, affected in ci_paths.classify([path]).items() if affected} == expected


def test_windows_separators_match_repository_paths() -> None:
    assert ci_paths.classify([r".\q2mm\backends\mm\tinker.py"]) == ci_paths.classify(["q2mm/backends/mm/tinker.py"])


def test_invalid_git_range_fails_instead_of_claiming_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(command: list[str], **kwargs: object) -> str:
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(subprocess, "check_output", fail)
    with pytest.raises(subprocess.CalledProcessError):
        ci_paths.main(["--base", "missing", "--head", "HEAD"])
