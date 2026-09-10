"""Tinker request controls must reach the native minimizer explicitly."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from q2mm.backends.contracts import EvaluationError, MinimizationRequest, PreparationRequest
from q2mm.backends.mm.tinker import PreparedTinker, TinkerBackend
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm, VdwParam
from test._shared import make_diatomic


def _case(backend: TinkerBackend) -> tuple[PreparedTinker, np.ndarray]:
    ff = ForceField(
        bonds=(BondParam(("H", "H"), 0.74, 100.0),),
        vdws=(VdwParam("H", 1.2, 0.02),),
        functional_form=FunctionalForm.MM3,
    )
    molecule = make_diatomic(distance=1.2, bond_tolerance=2.5)
    prepared = backend.prepare(PreparationRequest(case_id="bounded-h2", molecule=molecule, force_field=ff))
    return prepared, prepared.layout.vector(ff)


@pytest.mark.parametrize("limit", [None, 1, np.int64(7), 2**31 - 1])
@pytest.mark.parametrize("tolerance", [None, 0.005])
def test_iteration_limit_reaches_only_its_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int | None, tolerance: float | None
) -> None:
    backend = TinkerBackend(tinker_dir=str(tmp_path), params_file=str(tmp_path / "unused.prm"))
    prepared, parameters = _case(backend)
    requests: list[tuple[str, list[str] | None]] = []

    def run(
        executable: str, xyz: str, args: list[str] | None = None, stdin: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert executable == "minimize"
        assert stdin is None
        path = Path(xyz)
        requests.append((path.with_suffix(".key").read_text(), args))
        Path(xyz + "_2").write_text(path.read_text(), encoding="utf-8")
        return subprocess.CompletedProcess([executable], 0, stdout="Final Function Value : 1.25\n", stderr="")

    monkeypatch.setattr(backend, "_run_tinker", run)
    result = prepared.minimize(MinimizationRequest(parameters, max_iterations=limit, tolerance=tolerance))
    assert result.energy == 1.25
    key, args = requests[0]
    controls = [line for line in key.splitlines() if line.upper().startswith("MAXITER ")]
    assert controls == ([] if limit is None else [f"MAXITER {limit}"])
    assert args == [str(0.01 if tolerance is None else tolerance)]
    assert key.splitlines()[0].startswith("parameters ")

    prepared.minimize(MinimizationRequest(parameters))
    assert "MAXITER" not in requests[1][0].upper()
    assert requests[1][1] == ["0.01"]


@pytest.mark.parametrize("limit", [True, 1.5, np.float64(2), float("nan"), float("inf"), 2**31])
def test_unrepresentable_limit_fails_before_input_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: Any
) -> None:
    backend = TinkerBackend(tinker_dir=str(tmp_path), params_file=str(tmp_path / "unused.prm"))
    prepared, parameters = _case(backend)

    def reject_io(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid iteration limit reached input creation")

    monkeypatch.setattr(backend, "_write_tinker_xyz", reject_io)
    with pytest.raises(EvaluationError, match="max_iterations"):
        prepared.minimize(MinimizationRequest(parameters, max_iterations=limit))


@pytest.mark.tinker
@pytest.mark.integration
def test_native_minimize_stops_at_requested_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    from test.backend_fixtures import optional_test_backend

    backend = optional_test_backend("tinker")
    if backend is None:
        pytest.skip("Tinker is not available")
    assert isinstance(backend, TinkerBackend)
    prepared, parameters = _case(backend)
    output: list[str] = []
    run = backend._run_tinker

    def capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = run(*args, **kwargs)
        output.append(result.stdout)
        return result

    monkeypatch.setattr(backend, "_run_tinker", capture)
    result = prepared.minimize(MinimizationRequest(parameters, max_iterations=1, tolerance=1e-12))
    assert np.isfinite(result.energy)
    assert "IterLimit" in output[0]
