"""JAX-MD request plumbing; only the marked test executes the native minimizer."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from q2mm.backends.contracts import (
    EnergyRequest,
    EvaluationError,
    MinimizationRequest,
    PreparationRequest,
    PreparedBackend,
)
from q2mm.backends.mm import jax_md_engine
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Bond, Molecule
from q2mm.models.parameters import ParameterLayout
from test.backend_fixtures import load_test_backend
from test.test_backend_term_coverage import _unit_backend


def _system() -> tuple[Molecule, ForceField]:
    mol = Molecule(
        symbols=("H", "H"),
        geometry=((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
        bonds=(Bond(0, 1, ("H", "H"), 0.8),),
    )
    ff = ForceField(
        functional_form=FunctionalForm.HARMONIC,
        bonds=(BondParam(("H", "H"), equilibrium=0.74, force_constant=10.0),),
    )
    return mol, ff


def _session(monkeypatch: pytest.MonkeyPatch) -> tuple[PreparedBackend, np.ndarray, Mock]:
    backend, _ = _unit_backend("jax-md", monkeypatch)
    mol, ff = _system()
    native = Mock(return_value=(0.25, list(mol.symbols), mol.geometry.copy()))
    monkeypatch.setattr(backend, "_evaluate_minimize", native)
    session = backend.prepare(PreparationRequest(case_id="controls", molecule=mol, force_field=ff))
    return session, ParameterLayout.from_force_field(ff).vector(ff), native


@pytest.mark.parametrize("tolerance", [1e-12, 0.25, 0.0, -1.0, float("inf"), float("nan")])
def test_explicit_tolerance_is_rejected_before_native_minimization(
    monkeypatch: pytest.MonkeyPatch, tolerance: float
) -> None:
    session, params, native = _session(monkeypatch)
    with pytest.raises(EvaluationError, match="JAX-MD.*tolerance"):
        session.minimize(MinimizationRequest(parameters=params, max_iterations=1, tolerance=tolerance))
    native.assert_not_called()


@pytest.mark.parametrize("limit,expected", [(None, 200), (1, 1), (7, 7)])
def test_omitted_tolerance_preserves_existing_request_controls(
    monkeypatch: pytest.MonkeyPatch, limit: int | None, expected: int
) -> None:
    session, params, native = _session(monkeypatch)
    result = session.minimize(MinimizationRequest(parameters=params, max_iterations=limit))
    native.assert_called_once()
    assert native.call_args.kwargs == {"max_iterations": expected}
    assert result.energy == 0.25
    assert result.symbols == ("H", "H")
    np.testing.assert_array_equal(result.coordinates, session.molecule.geometry)


def test_explicit_none_keeps_native_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    session, params, native = _session(monkeypatch)
    session.minimize(MinimizationRequest(parameters=params, tolerance=None))
    assert native.call_args.kwargs == {"max_iterations": 200}


@pytest.mark.parametrize("limit", [None, 2])
def test_native_helper_does_not_override_scipy_stopping_defaults(
    monkeypatch: pytest.MonkeyPatch, limit: int | None
) -> None:
    mol, ff = _system()
    backend = object.__new__(jax_md_engine.JaxMdBackend)
    params = ParameterLayout.from_force_field(ff).vector(ff)
    monkeypatch.setattr(backend, "_params_and_coords", Mock(return_value=(params, mol.geometry)))
    monkeypatch.setattr(jax_md_engine, "jax", Mock())
    state = Mock(molecule=mol)
    terminal = SimpleNamespace(x=mol.geometry.flatten(), fun=0.25)
    with patch("scipy.optimize.minimize", return_value=terminal) as solver:
        if limit is None:
            backend._evaluate_minimize(state, ff)
        else:
            backend._evaluate_minimize(state, ff, max_iterations=limit)
    solver.assert_called_once()
    assert solver.call_args.kwargs["method"] == "L-BFGS-B"
    assert solver.call_args.kwargs["options"] == {"maxiter": 200 if limit is None else limit}
    assert "tol" not in solver.call_args.kwargs
    assert callable(solver.call_args.kwargs["jac"])


@pytest.mark.jax_md
def test_native_tolerance_rejection_and_one_iteration_default_path() -> None:
    """Exercise the real backend without asserting convergence or a force threshold."""
    mol, ff = _system()
    backend = load_test_backend("jax-md", box=(50.0, 50.0, 50.0))
    session = backend.prepare(PreparationRequest(case_id="native-controls", molecule=mol, force_field=ff))
    params = ParameterLayout.from_force_field(ff).vector(ff)
    baseline = session.energy(EnergyRequest(parameters=params)).energy
    with patch.object(backend, "_evaluate_minimize", wraps=backend._evaluate_minimize) as native:
        with pytest.raises(EvaluationError, match="JAX-MD.*tolerance"):
            session.minimize(MinimizationRequest(parameters=params, max_iterations=1, tolerance=1e-12))
        native.assert_not_called()
        terminal = session.minimize(MinimizationRequest(parameters=params, max_iterations=1))
        native.assert_called_once()
        assert native.call_args.kwargs == {"max_iterations": 1}
    assert np.isfinite(terminal.energy)
    assert terminal.coordinates.shape == (2, 3)
    assert np.all(np.isfinite(terminal.coordinates))
    assert session.energy(EnergyRequest(parameters=params)).energy == pytest.approx(baseline)
