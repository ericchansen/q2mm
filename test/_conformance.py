"""Reusable capability-conformance drivers for MM and reference backends.

The single entry point, :func:`assert_capability_conformance`, executes **only**
the capabilities a backend declares in its
:class:`~q2mm.backends.contracts.BackendInfo` and proves that every *drivable*
capability it does not declare raises a typed
:class:`~q2mm.backends.contracts.UnsupportedCapabilityError` — the base class
guards each operation before dispatch, so the backend's implementation hook is
never invoked for an undeclared capability.

Scope (intentionally precise):

* :func:`assert_capability_conformance` covers **MM** backends.
  :func:`assert_reference_capability_conformance` covers reference prepared
  sessions without force fields.
* The *drivable prepared-session* capabilities are ``ENERGY``, ``MINIMIZE``,
  ``HESSIAN``, ``FREQUENCIES``, ``PARAMETER_GRADIENT``,
  ``COORDINATE_GRADIENT``,
  ``HESSIAN_PARAMETER_JACOBIAN``, and ``BATCHED_ENERGY``.  ``BATCHED_HESSIAN`` is
  driven through the backend-level
  :func:`~q2mm.backends.contracts.prepare_hessian_batches` surface (it is not a
  prepared-session method).
* ``REUSABLE_STATE`` is a *non-method* capability: when declared and selected it
  is exercised by invoking a declared and already-executed prepared-session
  capability (preferring ``ENERGY``) a second time on the SAME prepared session.
  It is **not** asserted-unsupported when undeclared (there is no wrapper to
  invoke), and if it is selected but no drivable capability was executed to
  demonstrate reuse, that is a conformance failure rather than a silent
  omission.

It always executes ``ENERGY`` (the universal cheap capability) and lets callers
restrict which other declared capabilities to actually run via ``execute``, so
applying it to a built-in backend does not duplicate the heavier integration
engine suite.  Every failure message names the offending backend and capability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from q2mm.backends.contracts import (
    BackendRole,
    BatchedEnergyRequest,
    BatchedEnergyResult,
    BatchedHessianRequest,
    BatchedHessianResult,
    Capability,
    CoordinateGradientResult,
    EnergyRequest,
    EnergyResult,
    FrequencyRequest,
    FrequencyResult,
    GeometryResult,
    HessianJacobianRequest,
    HessianJacobianResult,
    HessianRequest,
    HessianResult,
    MinimizationRequest,
    ParameterGradientRequest,
    ParameterGradientResult,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
    UnsupportedCapabilityError,
    prepare_hessian_batches,
)
from q2mm.models.parameters import ParameterLayout


class ConformanceError(AssertionError):
    """Raised when a backend violates its declared-capability contract."""


@dataclass(frozen=True)
class ConformanceOutcome:
    """Summary of a conformance run.

    Args:
        backend: Backend name.
        executed: Capabilities actually executed and validated.
        unsupported_verified: Undeclared drivable capabilities proven to raise
            :class:`~q2mm.backends.contracts.UnsupportedCapabilityError`.

    """

    backend: str
    executed: tuple[Capability, ...]
    unsupported_verified: tuple[Capability, ...]


#: Drivable MM prepared-session capabilities: capability -> (method name,
#: request builder from a full parameter vector, expected result type).
_MM_DRIVERS: dict[Capability, tuple[str, Callable[[np.ndarray], object], type]] = {
    Capability.ENERGY: ("energy", lambda vec: EnergyRequest(parameters=vec), EnergyResult),
    Capability.MINIMIZE: ("minimize", lambda vec: MinimizationRequest(parameters=vec), GeometryResult),
    Capability.HESSIAN: ("hessian", lambda vec: HessianRequest(parameters=vec), HessianResult),
    Capability.FREQUENCIES: ("frequencies", lambda vec: FrequencyRequest(parameters=vec), FrequencyResult),
    Capability.PARAMETER_GRADIENT: (
        "parameter_gradient",
        lambda vec: ParameterGradientRequest(parameters=vec),
        ParameterGradientResult,
    ),
    Capability.COORDINATE_GRADIENT: (
        "coordinate_gradient",
        lambda _vec: ReferenceCoordinateGradientRequest(),
        CoordinateGradientResult,
    ),
    Capability.HESSIAN_PARAMETER_JACOBIAN: (
        "hessian_parameter_jacobian",
        lambda vec: HessianJacobianRequest(parameters=vec),
        HessianJacobianResult,
    ),
    Capability.BATCHED_ENERGY: (
        "batched_energy",
        lambda vec: BatchedEnergyRequest(parameter_matrix=np.asarray(vec, dtype=float).reshape(1, -1)),
        BatchedEnergyResult,
    ),
}

_REFERENCE_DRIVERS: dict[Capability, tuple[str, Callable[[], object], type]] = {
    Capability.ENERGY: ("energy", ReferenceEnergyRequest, EnergyResult),
    Capability.COORDINATE_GRADIENT: (
        "coordinate_gradient",
        ReferenceCoordinateGradientRequest,
        CoordinateGradientResult,
    ),
    Capability.HESSIAN: ("hessian", ReferenceHessianRequest, HessianResult),
    Capability.FREQUENCIES: ("frequencies", ReferenceFrequencyRequest, FrequencyResult),
    Capability.GEOMETRY_OPTIMIZATION: (
        "optimize_geometry",
        ReferenceGeometryOptimizationRequest,
        GeometryResult,
    ),
}


def _should_run(capability: Capability, execute: frozenset[Capability] | None) -> bool:
    """Return whether a declared *capability* should actually be executed."""
    return execute is None or capability in execute or capability is Capability.ENERGY


def assert_capability_conformance(
    backend: object,
    *,
    molecule: object,
    force_field: object,
    execute: frozenset[Capability] | None = None,
) -> ConformanceOutcome:
    """Execute declared MM capabilities and prove undeclared drivable ones raise.

    Args:
        backend: An MM backend exposing ``info`` and ``prepare``.
        molecule: A molecule compatible with *force_field*.
        force_field: The base force field to prepare with.
        execute: Optional subset of declared capabilities to actually run.
            When ``None``, every declared drivable capability is executed.
            ``ENERGY`` is always executed when declared, regardless of this
            argument.

    Returns:
        A :class:`ConformanceOutcome` summarizing what ran and what was proven
        unsupported.

    Raises:
        ConformanceError: If a declared capability fails to execute or returns
            the wrong result type, or if an undeclared drivable capability does
            not raise :class:`~q2mm.backends.contracts.UnsupportedCapabilityError`.

    """
    info = backend.info  # type: ignore[attr-defined]
    if info.role is not BackendRole.MM:
        raise ConformanceError(
            f"{info.name}: capability conformance helper supports MM backends only (role={info.role.value})."
        )

    prepared = backend.prepare(  # type: ignore[attr-defined]
        PreparationRequest(case_id="conformance", molecule=molecule, force_field=force_field)
    )
    vector = ParameterLayout.from_force_field(force_field).vector(force_field)

    executed: list[Capability] = []
    unsupported: list[Capability] = []
    #: Prepared-session capabilities that actually executed, with the method and
    #: request builder needed to demonstrate state reuse (see REUSABLE_STATE).
    executed_drivers: dict[Capability, tuple[str, Callable[[np.ndarray], object], type]] = {}

    # --- Prepared-session drivable capabilities -----------------------------
    for capability, (method_name, build_request, result_type) in _MM_DRIVERS.items():
        method = getattr(prepared, method_name)
        if info.supports(capability):
            if not _should_run(capability, execute):
                continue
            try:
                result = method(build_request(vector))
            except Exception as exc:  # noqa: BLE001 - surface as a named conformance failure
                raise ConformanceError(
                    f"{info.name}: declared capability {capability.value} failed to execute: {exc!r}"
                ) from exc
            if not isinstance(result, result_type):
                raise ConformanceError(
                    f"{info.name}: capability {capability.value} returned "
                    f"{type(result).__name__}, expected {result_type.__name__}."
                )
            executed.append(capability)
            executed_drivers[capability] = (method_name, build_request, result_type)
        else:
            try:
                method(build_request(vector))
            except UnsupportedCapabilityError:
                unsupported.append(capability)
            except Exception as exc:  # noqa: BLE001 - wrong error type is a conformance failure
                raise ConformanceError(
                    f"{info.name}: undeclared capability {capability.value} raised "
                    f"{type(exc).__name__}, expected UnsupportedCapabilityError."
                ) from exc
            else:
                raise ConformanceError(
                    f"{info.name}: undeclared capability {capability.value} did not raise UnsupportedCapabilityError."
                )

    # --- BATCHED_HESSIAN (backend-level surface, not a prepared method) ------
    if info.supports(Capability.BATCHED_HESSIAN):
        if _should_run(Capability.BATCHED_HESSIAN, execute):
            try:
                batches = prepare_hessian_batches(backend, [prepared])  # type: ignore[arg-type]
                for batch in batches:
                    batch_result = batch.hessians(BatchedHessianRequest(parameters=vector))
                    if not isinstance(batch_result, BatchedHessianResult):
                        raise ConformanceError(
                            f"{info.name}: BATCHED_HESSIAN returned {type(batch_result).__name__}, "
                            "expected BatchedHessianResult."
                        )
            except ConformanceError:
                raise
            except Exception as exc:  # noqa: BLE001 - surface as a named conformance failure
                raise ConformanceError(
                    f"{info.name}: declared capability batched_hessian failed to execute: {exc!r}"
                ) from exc
            executed.append(Capability.BATCHED_HESSIAN)
    else:
        try:
            prepare_hessian_batches(backend, [prepared])  # type: ignore[arg-type]
        except UnsupportedCapabilityError:
            unsupported.append(Capability.BATCHED_HESSIAN)
        except Exception as exc:  # noqa: BLE001 - wrong error type is a conformance failure
            raise ConformanceError(
                f"{info.name}: undeclared capability batched_hessian raised "
                f"{type(exc).__name__}, expected UnsupportedCapabilityError."
            ) from exc
        else:
            raise ConformanceError(
                f"{info.name}: undeclared capability batched_hessian did not raise UnsupportedCapabilityError."
            )

    # --- REUSABLE_STATE (non-method: reuse the SAME session twice) -----------
    # When declared and selected, demonstrate reuse by invoking a declared and
    # *already executed* prepared-session capability a second time on the SAME
    # prepared session (preferring ENERGY when available).  It is not
    # asserted-unsupported when undeclared (there is no wrapper to invoke).  If
    # it is selected but nothing was executed to demonstrate reuse, that is a
    # conformance failure rather than a silent omission.
    if info.supports(Capability.REUSABLE_STATE) and (execute is None or Capability.REUSABLE_STATE in execute):
        if not executed_drivers:
            raise ConformanceError(
                f"{info.name}: reusable_state was selected but no drivable prepared-session capability "
                "was executed to demonstrate session reuse."
            )
        if Capability.ENERGY in executed_drivers:
            drive_capability = Capability.ENERGY
        else:
            drive_capability = sorted(executed_drivers, key=lambda cap: cap.value)[0]
        drive_method_name, drive_build_request, drive_result_type = executed_drivers[drive_capability]
        drive_method = getattr(prepared, drive_method_name)
        try:
            first = drive_method(drive_build_request(vector))
            second = drive_method(drive_build_request(vector))
        except Exception as exc:  # noqa: BLE001 - surface as a named conformance failure
            raise ConformanceError(
                f"{info.name}: reusable_state failed reusing {drive_capability.value} on the same session: {exc!r}"
            ) from exc
        if not (isinstance(first, drive_result_type) and isinstance(second, drive_result_type)):
            raise ConformanceError(
                f"{info.name}: reusable_state reuse of {drive_capability.value} did not return two "
                f"{drive_result_type.__name__} results."
            )
        executed.append(Capability.REUSABLE_STATE)

    if info.supports(Capability.ENERGY) and Capability.ENERGY not in executed:
        raise ConformanceError(f"{info.name}: declared ENERGY was not executed.")

    return ConformanceOutcome(
        backend=info.name,
        executed=tuple(executed),
        unsupported_verified=tuple(unsupported),
    )


def assert_reference_capability_conformance(
    backend: object,
    *,
    molecule: object,
    execute: frozenset[Capability] | None = None,
) -> ConformanceOutcome:
    """Drive declared reference capabilities and verify undeclared operations."""
    info = backend.info  # type: ignore[attr-defined]
    if info.role is not BackendRole.REFERENCE:
        raise ConformanceError(
            f"{info.name}: reference conformance helper requires reference role (role={info.role.value})."
        )
    prepared = backend.prepare(  # type: ignore[attr-defined]
        PreparationRequest(case_id="reference-conformance", molecule=molecule)
    )
    executed: list[Capability] = []
    unsupported: list[Capability] = []
    for capability, (method_name, build_request, result_type) in _REFERENCE_DRIVERS.items():
        method = getattr(prepared, method_name)
        if info.supports(capability):
            if execute is not None and capability not in execute:
                continue
            try:
                result = method(build_request())
            except Exception as exc:  # noqa: BLE001
                raise ConformanceError(
                    f"{info.name}: declared reference capability {capability.value} failed: {exc!r}"
                ) from exc
            if not isinstance(result, result_type):
                raise ConformanceError(
                    f"{info.name}: reference capability {capability.value} returned "
                    f"{type(result).__name__}, expected {result_type.__name__}."
                )
            executed.append(capability)
        else:
            try:
                method(build_request())
            except UnsupportedCapabilityError:
                unsupported.append(capability)
            except Exception as exc:  # noqa: BLE001
                raise ConformanceError(
                    f"{info.name}: undeclared reference capability {capability.value} raised "
                    f"{type(exc).__name__}, expected UnsupportedCapabilityError."
                ) from exc
            else:
                raise ConformanceError(
                    f"{info.name}: undeclared reference capability {capability.value} "
                    "did not raise UnsupportedCapabilityError."
                )
    return ConformanceOutcome(
        backend=info.name,
        executed=tuple(executed),
        unsupported_verified=tuple(unsupported),
    )
