"""Psi4 reference backend.

Wraps the Psi4 Python API for reference calculations: energy, Hessian, geometry
optimization, and vibrational frequencies.

Requires: ``conda install psi4 -c conda-forge``

Psi4 is a reference backend: it consumes no force field, so its
:class:`~q2mm.backends.contracts.BackendInfo` declares **no** functional
forms.  Method and basis are fixed when the backend is constructed.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendUnavailableError,
    Capability,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
    FrequencyResult,
    FrequencyUnit,
    GeometryResult,
    HessianResult,
    HessianUnit,
    LengthUnit,
    PreparationRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
    readonly_array,
)
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.molecule import Molecule

try:
    import psi4 as _psi4

    _HAS_PSI4 = True
except ImportError:
    _psi4 = None
    _HAS_PSI4 = False


def _make_psi4_geometry(atoms: list[str], coords: np.ndarray, charge: int = 0, multiplicity: int = 1) -> object:
    """Create a Psi4 molecule object from atoms and coordinates.

    Args:
        atoms: Element symbols.
        coords: Cartesian coordinates, shape ``(N, 3)``, in Å.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.

    Returns:
        A Psi4 ``Molecule`` object.

    """
    geom_str = f"    {charge} {multiplicity}\n"
    for atom, (x, y, z) in zip(atoms, coords, strict=False):
        geom_str += f"    {atom} {x:.10f} {y:.10f} {z:.10f}\n"
    return _psi4.geometry(geom_str)


class Psi4Backend:
    """Reference backend using Psi4.

    Args:
        method: DFT functional or method (default: "b3lyp")
        basis: Basis set (default: "6-31+G(d)")
        memory: Memory allocation (default: "2 GB")
        n_threads: Number of threads (default: 4)
        charge: Molecular charge (default: 0)
        multiplicity: Spin multiplicity (default: 1)

    """

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "6-31+G(d)",
        memory: str = "2 GB",
        n_threads: int = 4,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        """Initialize the Psi4 backend.

        Raises:
            BackendUnavailableError: If Psi4 is not installed.

        """
        if not _HAS_PSI4:
            raise BackendUnavailableError("Psi4 is not installed. Install via: conda install psi4 -c conda-forge")
        self._method = method
        self._basis = basis
        self._charge = charge
        self._multiplicity = multiplicity
        _psi4.set_memory(memory)
        _psi4.set_num_threads(n_threads)
        self._tmpdir = tempfile.mkdtemp(prefix="q2mm_psi4_")
        _psi4.core.set_output_file(os.path.join(self._tmpdir, "psi4_output.dat"), False)
        version = getattr(_psi4, "__version__", "")
        self._provenance = BackendProvenance(
            backend="psi4",
            role=BackendRole.REFERENCE,
            version=str(version),
            details={
                "implementation": {"name": "Psi4", "version": str(version)},
                "model": {"method": method, "basis": basis},
                "calculator": {"charge": charge, "multiplicity": multiplicity},
                "config": {"memory": memory, "n_threads": n_threads},
            },
        )
        self._info = BackendInfo(
            name=f"Psi4 ({method}/{basis})",
            role=BackendRole.REFERENCE,
            capabilities=frozenset(
                {
                    Capability.ENERGY,
                    Capability.HESSIAN,
                    Capability.FREQUENCIES,
                    Capability.GEOMETRY_OPTIMIZATION,
                }
            ),
            functional_forms=frozenset(),
            provenance=self._provenance,
        )

    @property
    def info(self) -> BackendInfo:
        """Immutable capability declaration for this backend."""
        return self._info

    def prepare(self, request: PreparationRequest) -> PreparedPsi4:
        """Build a prepared reference session for one training case.

        Args:
            request: Preparation request carrying the molecule.  ``force_field``
                is ignored — Psi4 is a reference backend.

        Returns:
            PreparedPsi4: A per-case QM session.

        """
        return PreparedPsi4(backend=self, case_id=request.case_id, molecule=request.molecule)

    # -- internal Psi4 plumbing --------------------------------------------

    def _load_molecule(self, molecule: Molecule) -> object:
        atoms = list(molecule.symbols)
        coords = np.asarray(molecule.geometry, dtype=float)
        mol = _make_psi4_geometry(atoms, coords, self._charge, self._multiplicity)
        ref = "rhf" if self._multiplicity == 1 else "uhf"
        _psi4.set_options({"basis": self._basis, "reference": ref})
        return mol

    def _evaluate_energy(self, molecule: Molecule) -> float:
        mol = self._load_molecule(molecule)
        return float(_psi4.energy(self._method, molecule=mol))

    def _evaluate_hessian(self, molecule: Molecule) -> np.ndarray:
        mol = self._load_molecule(molecule)
        _, wfn = _psi4.frequency(self._method, molecule=mol, return_wfn=True)
        return np.array(wfn.hessian())

    def _evaluate_frequencies(self, molecule: Molecule) -> list[float]:
        mol = self._load_molecule(molecule)
        _, wfn = _psi4.frequency(self._method, molecule=mol, return_wfn=True)
        return list(np.array(wfn.frequencies()))

    def _evaluate_optimize(self, molecule: Molecule, opt_type: str) -> tuple[float, list[str], np.ndarray]:
        mol = self._load_molecule(molecule)
        _psi4.set_options({"opt_type": opt_type, "geom_maxiter": 100})
        energy = float(_psi4.optimize(self._method, molecule=mol))
        coords_bohr = mol.geometry().np
        coords_ang = coords_bohr * BOHR_TO_ANG
        atoms = [mol.symbol(i) for i in range(mol.natom())]
        return energy, atoms, coords_ang

    def close(self) -> None:
        """Clean up temporary files created by Psi4."""
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> Psi4Backend:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and clean up temporary files."""
        self.close()

    def __del__(self) -> None:
        """Destructor — clean up temporary files."""
        self.close()


class PreparedPsi4(AbstractPreparedBackend):
    """Prepared Psi4 reference session for a single molecule."""

    def __init__(self, *, backend: Psi4Backend, case_id: str, molecule: Molecule) -> None:
        super().__init__(
            info=backend.info,
            case_id=case_id,
            molecule=molecule,
            force_field=None,
            layout=None,
        )
        self._backend = backend

    def _energy(self, request: ReferenceEnergyRequest) -> EnergyResult:
        try:
            value = self._backend._evaluate_energy(self.molecule)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Psi4 energy evaluation failed: {exc}") from exc
        return EnergyResult(energy=value, unit=EnergyUnit.HARTREE, provenance=self._backend.info.provenance)

    def _hessian(self, request: ReferenceHessianRequest) -> HessianResult:
        try:
            hess = self._backend._evaluate_hessian(self.molecule)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Psi4 Hessian evaluation failed: {exc}") from exc
        return HessianResult(
            hessian=readonly_array(hess), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=self._backend.info.provenance
        )

    def _frequencies(self, request: ReferenceFrequencyRequest) -> FrequencyResult:
        try:
            freqs = self._backend._evaluate_frequencies(self.molecule)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Psi4 frequency evaluation failed: {exc}") from exc
        return FrequencyResult(
            frequencies=readonly_array(freqs), unit=FrequencyUnit.INVERSE_CM, provenance=self._backend.info.provenance
        )

    def _optimize_geometry(self, request: ReferenceGeometryOptimizationRequest) -> GeometryResult:
        try:
            energy, atoms, coords = self._backend._evaluate_optimize(self.molecule, request.opt_type)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Psi4 geometry optimization failed: {exc}") from exc
        return GeometryResult(
            energy=energy,
            energy_unit=EnergyUnit.HARTREE,
            symbols=tuple(atoms),
            coordinates=readonly_array(coords),
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=self._backend.info.provenance,
        )
