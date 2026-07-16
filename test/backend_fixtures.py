"""Test-only fixtures for the prepared-session backend API.

These helpers keep test configuration separate from the production backend
catalog.  In particular, Tinker needs both executables and an MM3 parameter
file: the production catalog intentionally probes only import/executable
dependencies, while tests must not treat that cheap probe as proof that the
default backend can be constructed.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from q2mm.backends.contracts import (
    Backend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendUnavailableError,
    Capability,
    PreparationRequest,
)
from q2mm.backends.registry import available_backends, load_backend
from q2mm.models.parameters import ParameterLayout


def load_test_backend(name: str, **kwargs: object) -> Backend:
    """Load *name* using backend configuration supplied to the test process."""
    configured = dict(kwargs)
    if name == "tinker":
        tinker_dir = os.environ.get("TINKER_DIR")
        params_file = os.environ.get("TINKER_PRM")
        if tinker_dir is not None:
            configured.setdefault("tinker_dir", tinker_dir)
        if params_file is not None:
            configured.setdefault("params_file", params_file)
    return load_backend(name, **configured)


def optional_test_backend(name: str) -> Backend | None:
    """Load a test backend, returning ``None`` only for typed unavailability."""
    # Explicit TINKER_DIR/TINKER_PRM configuration must bypass the catalog's
    # PATH-only dependency probe, just like production ``load_backend`` does.
    if name != "tinker" and name not in available_backends():
        return None
    try:
        return load_test_backend(name)
    except BackendUnavailableError:
        return None


def backend_is_usable(name: str) -> bool:
    """Return whether *name* has enough configuration for its test suite.

    Most backends need only the production catalog's cheap dependency probe.
    Tinker additionally requires a successful configured load because an
    ``analyze`` executable can be present without the required ``mm3.prm``.
    """
    if name == "tinker":
        return optional_test_backend(name) is not None
    return name in available_backends()


def param_vector(force_field: Any) -> np.ndarray:
    """Return the full parameter vector for *force_field* (length == len(layout))."""
    return ParameterLayout.from_force_field(force_field).vector(force_field)


def prepare_case(backend: Any, molecule: Any, force_field: Any, case_id: str = "0") -> Any:
    """Return an MM :class:`PreparedBackend` session for *molecule* under *force_field*."""
    return backend.prepare(PreparationRequest(case_id=case_id, molecule=molecule, force_field=force_field))


def qm_prepare_case(backend: Any, molecule: Any, case_id: str = "0") -> Any:
    """Return a QM :class:`PreparedBackend` session for *molecule* (no force field)."""
    return backend.prepare(PreparationRequest(case_id=case_id, molecule=molecule))


class MockLayout:
    """Minimal parameter layout exposing only ``__len__`` for test doubles.

    The :class:`~q2mm.backends.contracts.AbstractPreparedBackend` central result
    validation needs ``len(layout)`` to check gradient/Jacobian dimensions; test
    doubles that override the ``_evaluate`` hooks directly only need this.
    """

    def __init__(self, n_params: int) -> None:
        self._n = int(n_params)

    def __len__(self) -> int:
        return self._n


def mock_molecule(symbols: Any) -> Any:
    """Return a real :class:`Molecule` exposing ``.symbols`` and ``.geometry``.

    ``ObjectivePlan`` validates that every case molecule is a real
    ``Molecule`` instance, so the test double is a genuine (immutable)
    ``Molecule``.  Atoms are spread 10 A apart along x so no spurious bonds
    or overlap-angle warnings are perceived; the concrete element/geometry
    values are irrelevant to the tests, which only read ``.symbols`` /
    ``.geometry`` shapes.
    """
    from q2mm.models.molecule import Molecule

    syms = tuple(symbols)
    geometry = np.zeros((len(syms), 3), dtype=float)
    geometry[:, 0] = np.arange(len(syms), dtype=float) * 10.0
    return Molecule(symbols=syms, geometry=geometry, name="mock_mol")


def mock_backend_info(
    *,
    param_grad: bool = False,
    hess_jac: bool = False,
    batched: bool = False,
    forms: tuple[str, ...] = ("harmonic",),
) -> BackendInfo:
    """Build a :class:`BackendInfo` for MagicMock backends in unit tests."""
    caps = {Capability.ENERGY, Capability.HESSIAN, Capability.FREQUENCIES, Capability.MINIMIZE}
    if param_grad:
        caps.add(Capability.PARAMETER_GRADIENT)
    if hess_jac:
        caps.add(Capability.HESSIAN_PARAMETER_JACOBIAN)
    if batched:
        caps.add(Capability.BATCHED_ENERGY)
        caps.add(Capability.BATCHED_HESSIAN)
    return BackendInfo(
        name="mock",
        role=BackendRole.MM,
        capabilities=frozenset(caps),
        functional_forms=frozenset(forms),
        provenance=BackendProvenance(backend="mock", role=BackendRole.MM),
    )
