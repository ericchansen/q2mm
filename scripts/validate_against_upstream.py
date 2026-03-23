#!/usr/bin/env python3
"""Validate current Q2MM behavior against pinned fixtures or live upstream code."""

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import numbers
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from collections.abc import Callable
from typing import Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants, seminario_bond_fc
from q2mm.parsers.jaguar import JaguarIn
from q2mm.parsers.macromodel import MacroModel
from q2mm.parsers.mol2 import Mol2

DEFAULT_WORKTREE = REPO_ROOT.parent / f"{REPO_ROOT.name}-upstream-worktree"
FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "seminario_parity"

RH_FIXTURE_PATH = FIXTURE_DIR / "rh_enamide_reference.json"
SN2_FIXTURE_PATH = FIXTURE_DIR / "sn2_reference.json"
OPTIMIZATION_FIXTURE_DIR = REPO_ROOT / "test" / "fixtures"
OPTIMIZATION_GOLDEN_PATH = OPTIMIZATION_FIXTURE_DIR / "optimization_golden.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MM3_PATH = RH_DIR / "mm3.fld"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"
MOL2_PATH = TRAINING_SET_DIR / "mol2" / "1_zdmp.mol2"
RH_DIRECT_HESSIAN_PATH = JAG_DIR / "1ZDMPfromJCTCSI_loner1.01.in"

SN2_QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"
SN2_XYZ_PATH = SN2_QM_REF / "sn2-ts-optimized.xyz"
SN2_HESSIAN_PATH = SN2_QM_REF / "sn2-ts-hessian.npy"


Mode = Literal["fixture", "live"]
Status = Literal["passed", "failed", "blocked"]


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    description: str
    category: str
    supported_modes: tuple[Mode, ...]
    fixture_systems: tuple[str, ...] = ()
    blocked_reason: str | None = None
    requires_worktree: bool = True


@dataclass
class CaseResult:
    case_id: str
    description: str
    category: str
    mode: str
    status: Status
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)
    blocked_reason: str | None = None


CASE_MATRIX: dict[str, CaseDefinition] = {
    "seminario-sn2-bond": CaseDefinition(
        case_id="seminario-sn2-bond",
        description="SN2 direct bond projections match legacy references.",
        category="seminario",
        supported_modes=("fixture", "live"),
        fixture_systems=("sn2",),
    ),
    "seminario-rh-direct-bond": CaseDefinition(
        case_id="seminario-rh-direct-bond",
        description="Rh-enamide direct bond projections match legacy references.",
        category="seminario",
        supported_modes=("fixture", "live"),
        fixture_systems=("rh-enamide",),
    ),
    "seminario-rh-pipeline": CaseDefinition(
        case_id="seminario-rh-pipeline",
        description="Rh-enamide full parameter pipeline matches legacy references.",
        category="seminario",
        supported_modes=("fixture", "live"),
        fixture_systems=("rh-enamide",),
    ),
    "calculate-compare-d_rhod": CaseDefinition(
        case_id="calculate-compare-d_rhod",
        description="Legacy calculate.py / compare.py workflow on d_rhod fixtures.",
        category="objective",
        supported_modes=(),
        blocked_reason=(
            "Blocked in this environment: requires Schrödinger MacroModel runtime and the d_rhod fixture directory."
        ),
    ),
    "optimization-endpoint": CaseDefinition(
        case_id="optimization-endpoint",
        description="Optimization endpoint comparison (objective, energy, geometry).",
        category="optimization",
        supported_modes=("fixture",),
        blocked_reason=None,
        requires_worktree=False,
    ),
    "openmm-tinker-mm3-shared": CaseDefinition(
        case_id="openmm-tinker-mm3-shared",
        description="OpenMM vs Tinker MM3 parity on shared bond-only and vdW-only cases.",
        category="backend",
        supported_modes=("fixture", "live"),
        requires_worktree=False,
    ),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _int_keyed_map(values: dict[str, float | None]) -> dict[int, float | None]:
    return {int(key): value for key, value in values.items()}


class _DisableLogging:
    def __enter__(self):
        self._previous = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        return self

    def __exit__(self, exc_type, exc, tb):
        logging.disable(self._previous)
        return False


def _blocked_result(case: CaseDefinition, mode: str, reason: str | None = None) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="blocked",
        blocked_reason=reason or case.blocked_reason,
    )


def _run_sn2_bond_case(fixture_dir: Path, mode: Mode) -> CaseResult:
    case = CASE_MATRIX["seminario-sn2-bond"]
    fixture = _load_json(fixture_dir / "sn2_reference.json")
    molecule = Q2MMMolecule.from_xyz(SN2_XYZ_PATH, name="sn2_ts", bond_tolerance=1.5)
    hessian = np.load(str(SN2_HESSIAN_PATH))
    scaling = float(fixture["metadata"]["dft_scaling"])

    max_abs_diff = 0.0
    details: list[str] = []
    for bond in fixture["bonds"]:
        actual = seminario_bond_fc(
            bond["atom_i"],
            bond["atom_j"],
            molecule.geometry,
            hessian,
            au_units=True,
            dft_scaling=scaling,
        )
        expected = float(bond["legacy_force_constant_mdyn_a"])
        diff = abs(actual - expected)
        max_abs_diff = max(max_abs_diff, diff)
        if diff >= 1e-8:
            details.append(f"{bond['label']}: expected {expected:.12f}, got {actual:.12f}, diff {diff:.3e}")

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="passed" if not details else "failed",
        metrics={
            "compared_bonds": len(fixture["bonds"]),
            "max_abs_diff_mdyn_a": max_abs_diff,
            "upstream_commit": fixture["metadata"]["upstream_commit"],
        },
        details=details,
    )


def _run_rh_direct_case(fixture_dir: Path, mode: Mode) -> CaseResult:
    case = CASE_MATRIX["seminario-rh-direct-bond"]
    fixture = _load_json(fixture_dir / "rh_enamide_reference.json")
    structure = Mol2(str(MOL2_PATH)).structures[0]
    hessian = JaguarIn(str(RH_DIRECT_HESSIAN_PATH)).get_hessian(len(structure.atoms))
    coordinates = np.array([[atom.x, atom.y, atom.z] for atom in structure.atoms], dtype=float)
    scaling = float(fixture["metadata"]["dft_scaling"])

    max_abs_diff = 0.0
    details: list[str] = []
    for bond in fixture["direct_bonds"]:
        actual = seminario_bond_fc(
            bond["atom_i"],
            bond["atom_j"],
            coordinates,
            hessian,
            au_units=True,
            dft_scaling=scaling,
        )
        expected = float(bond["legacy_force_constant_mdyn_a"])
        diff = abs(actual - expected)
        max_abs_diff = max(max_abs_diff, diff)
        if diff >= 1e-8:
            details.append(f"{bond['label']}: expected {expected:.12f}, got {actual:.12f}, diff {diff:.3e}")

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="passed" if not details else "failed",
        metrics={
            "compared_bonds": len(fixture["direct_bonds"]),
            "max_abs_diff_mdyn_a": max_abs_diff,
            "upstream_commit": fixture["metadata"]["upstream_commit"],
        },
        details=details,
    )


def _run_rh_pipeline_case(fixture_dir: Path, mode: Mode) -> CaseResult:
    case = CASE_MATRIX["seminario-rh-pipeline"]
    fixture = _load_json(fixture_dir / "rh_enamide_reference.json")
    structures = MacroModel(str(MMO_PATH)).structures
    hessian_files = sorted(JAG_DIR.glob("*.in"))
    hessians = [
        JaguarIn(str(path)).get_hessian(len(structure.atoms)) for structure, path in zip(structures, hessian_files)
    ]
    molecules = [
        Q2MMMolecule.from_structure(
            structure,
            hessian=hessian,
            name=f"rh_enamide_{index + 1}",
        )
        for index, (structure, hessian) in enumerate(zip(structures, hessians))
    ]
    clean_start = ForceField.from_mm3_fld(MM3_PATH)
    with _DisableLogging():
        clean_estimated = estimate_force_constants(
            molecules,
            forcefield=clean_start,
            zero_torsions=True,
            au_hessian=True,
            invalid_policy="skip",
        )

    fixture_bf = _int_keyed_map(fixture["parameters"]["bond_force_constants_mdyn_a"])
    fixture_be = _int_keyed_map(fixture["parameters"]["bond_equilibria_angstrom"])
    fixture_af = _int_keyed_map(fixture["parameters"]["angle_force_constants_mdyn_a_rad2"])
    fixture_ae = _int_keyed_map(fixture["parameters"]["angle_equilibria_degrees"])
    starting_bonds = {param.ff_row: param for param in clean_start.bonds}
    starting_angles = {param.ff_row: param for param in clean_start.angles}

    max_bond_fc_diff = 0.0
    max_bond_eq_diff = 0.0
    max_angle_fc_diff = 0.0
    max_angle_eq_diff = 0.0
    details: list[str] = []

    for bond_param in clean_estimated.bonds:
        assert bond_param.ff_row is not None
        fixture_force_constant = fixture_bf[bond_param.ff_row]
        expected_fc = (
            starting_bonds[bond_param.ff_row].force_constant
            if fixture_force_constant is None
            else fixture_force_constant
        )
        fc_diff = abs(bond_param.force_constant - expected_fc)
        eq_diff = abs(bond_param.equilibrium - fixture_be[bond_param.ff_row])
        max_bond_fc_diff = max(max_bond_fc_diff, fc_diff)
        max_bond_eq_diff = max(max_bond_eq_diff, eq_diff)
        if fc_diff >= 1e-8 or eq_diff >= 1e-8:
            details.append(f"bond row {bond_param.ff_row}: fc diff {fc_diff:.3e}, eq diff {eq_diff:.3e}")

    for angle_param in clean_estimated.angles:
        assert angle_param.ff_row is not None
        fixture_force_constant = fixture_af[angle_param.ff_row]
        expected_fc = (
            starting_angles[angle_param.ff_row].force_constant
            if fixture_force_constant is None
            else fixture_force_constant
        )
        fc_diff = abs(angle_param.force_constant - expected_fc)
        eq_diff = abs(angle_param.equilibrium - fixture_ae[angle_param.ff_row])
        max_angle_fc_diff = max(max_angle_fc_diff, fc_diff)
        max_angle_eq_diff = max(max_angle_eq_diff, eq_diff)
        if fc_diff >= 1e-8 or eq_diff >= 1e-8:
            details.append(f"angle row {angle_param.ff_row}: fc diff {fc_diff:.3e}, eq diff {eq_diff:.3e}")

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="passed" if not details else "failed",
        metrics={
            "bond_param_count": len(clean_estimated.bonds),
            "angle_param_count": len(clean_estimated.angles),
            "max_bond_force_constant_diff": max_bond_fc_diff,
            "max_bond_equilibrium_diff": max_bond_eq_diff,
            "max_angle_force_constant_diff": max_angle_fc_diff,
            "max_angle_equilibrium_diff": max_angle_eq_diff,
            "upstream_commit": fixture["metadata"]["upstream_commit"],
        },
        details=details,
    )


def _run_openmm_tinker_case(fixture_dir: Path, mode: Mode) -> CaseResult:
    case = CASE_MATRIX["openmm-tinker-mm3-shared"]
    try:
        from q2mm.backends.mm.openmm import OpenMMEngine
        from q2mm.backends.mm.tinker import TinkerEngine
    except (ImportError, FileNotFoundError) as exc:
        return _blocked_result(case, mode, str(exc))

    try:
        openmm = OpenMMEngine()
        tinker = TinkerEngine()
    except (ImportError, FileNotFoundError) as exc:
        return _blocked_result(case, mode, str(exc))

    forcefield = ForceField.from_tinker_prm(Path(tinker._params_file))
    bond_molecule = Q2MMMolecule(
        symbols=["C", "H"],
        atom_types=["1", "5"],
        geometry=np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]], dtype=float),
        name="CH-bond",
        bond_tolerance=1.5,
    )
    vdw_molecule = Q2MMMolecule(
        symbols=["F", "F"],
        atom_types=["11", "11"],
        geometry=np.array([[0.0, 0.0, 0.0], [3.50, 0.0, 0.0]], dtype=float),
        name="F2-nonbonded",
        bond_tolerance=0.5,
    )

    comparisons = {
        "bond_energy_kcal_mol": (bond_molecule, 1.0e-3),
        "vdw_energy_kcal_mol": (vdw_molecule, 1.0e-3),
    }
    details: list[str] = []
    metrics: dict[str, float | int | str | None] = {
        "tinker_params_file": str(tinker._params_file),
    }
    max_abs_diff = 0.0
    for label, (molecule, tolerance) in comparisons.items():
        openmm_energy = openmm.energy(molecule, forcefield)
        tinker_energy = tinker.energy(molecule)
        diff = abs(openmm_energy - tinker_energy)
        max_abs_diff = max(max_abs_diff, diff)
        metrics[f"{label}_openmm"] = openmm_energy
        metrics[f"{label}_tinker"] = tinker_energy
        metrics[f"{label}_abs_diff"] = diff
        if diff > tolerance:
            details.append(f"{label}: OpenMM {openmm_energy:.6f}, Tinker {tinker_energy:.6f}, diff {diff:.3e}")

    metrics["max_abs_diff"] = max_abs_diff
    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="passed" if not details else "failed",
        metrics=metrics,
        details=details,
    )


def _run_optimization_endpoint_case(fixture_dir: Path, mode: Mode) -> CaseResult:
    """Validate optimization endpoint against golden fixture."""
    case = CASE_MATRIX["optimization-endpoint"]
    if not OPTIMIZATION_GOLDEN_PATH.exists():
        return _blocked_result(
            case,
            mode,
            f"Golden fixture not found at {OPTIMIZATION_GOLDEN_PATH}. "
            "Run: python scripts/generate_optimization_fixtures.py",
        )

    try:
        from q2mm.backends.mm.openmm import OpenMMEngine
        from q2mm.models.forcefield import AngleParam, BondParam
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer
    except ImportError as exc:
        return _blocked_result(case, mode, f"Import error: {exc}")

    golden = _load_json(OPTIMIZATION_GOLDEN_PATH)

    # Reproduce the same water problem used to generate the fixture
    engine = OpenMMEngine()
    true_ff = ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=503.6, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=57.6, equilibrium=104.5)],
    )

    def _water(angle_deg=104.5, bond_length=0.96):
        theta = np.deg2rad(angle_deg)
        return Q2MMMolecule(
            symbols=["O", "H", "H"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [bond_length, 0.0, 0.0],
                    [bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0],
                ]
            ),
            name="water",
            bond_tolerance=1.5,
        )

    mols = [_water(104.5, 0.96), _water(115.0, 0.96), _water(104.5, 1.05)]
    ref = ReferenceData()
    for i, mol in enumerate(mols):
        ref.add_energy(engine.energy(mol, true_ff), weight=1.0, molecule_idx=i)
    freqs = engine.frequencies(mols[0], true_ff)
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:
            ref.add_frequency(f, data_idx=j, weight=0.001, molecule_idx=0)

    guess_ff = ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=611.5, equilibrium=1.01)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=79.1, equilibrium=109.5)],
    )

    obj = ObjectiveFunction(guess_ff, engine, mols, ref)
    opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
    result = opt.optimize(obj)

    details: list[str] = []
    score_tol = 1e-6
    param_tol = 1e-6

    initial_diff = abs(result.initial_score - golden["initial_score"])
    if initial_diff > score_tol * max(abs(golden["initial_score"]), 1.0):
        details.append(
            f"initial_score: expected {golden['initial_score']:.8f}, "
            f"got {result.initial_score:.8f}, diff {initial_diff:.3e}"
        )

    final_diff = abs(result.final_score - golden["final_score"])
    if final_diff > score_tol * max(abs(golden["final_score"]), 1.0):
        details.append(
            f"final_score: expected {golden['final_score']:.8f}, got {result.final_score:.8f}, diff {final_diff:.3e}"
        )

    max_param_diff = 0.0
    for i, (actual, expected) in enumerate(zip(result.final_params, golden["final_params"])):
        diff = abs(actual - expected)
        max_param_diff = max(max_param_diff, diff)
        denom = max(abs(expected), 1e-8)
        if diff / denom > param_tol:
            details.append(f"param[{i}]: expected {expected:.8f}, got {actual:.8f}, diff {diff:.3e}")

    return CaseResult(
        case_id=case.case_id,
        description=case.description,
        category=case.category,
        mode=mode,
        status="passed" if not details else "failed",
        metrics={
            "initial_score_diff": initial_diff,
            "final_score_diff": final_diff,
            "max_param_diff": max_param_diff,
            "improvement": result.improvement,
            "n_evaluations": result.n_evaluations,
        },
        details=details,
    )


CASE_RUNNERS: dict[str, Callable[[Path, Mode], CaseResult]] = {
    "seminario-sn2-bond": _run_sn2_bond_case,
    "seminario-rh-direct-bond": _run_rh_direct_case,
    "seminario-rh-pipeline": _run_rh_pipeline_case,
    "openmm-tinker-mm3-shared": _run_openmm_tinker_case,
    "optimization-endpoint": _run_optimization_endpoint_case,
}


def _fixture_systems_for_cases(case_ids: list[str]) -> list[str]:
    systems: list[str] = []
    for case_id in case_ids:
        for system in CASE_MATRIX[case_id].fixture_systems:
            if system not in systems:
                systems.append(system)
    return systems


def _ensure_live_fixture_dir(case_ids: list[str], worktree: Path, output_dir: Path) -> Path:
    systems = _fixture_systems_for_cases(case_ids)
    args = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "regenerate_parity_fixtures.py"),
        "--worktree",
        str(worktree),
        "--output-dir",
        str(output_dir),
    ]
    if systems:
        args.extend(["--systems", *systems])
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, args)
    return output_dir


def _selected_case_ids(requested: list[str] | None) -> list[str]:
    if requested:
        unknown = [case_id for case_id in requested if case_id not in CASE_MATRIX]
        if unknown:
            raise ValueError(f"Unknown case ids: {', '.join(unknown)}")
        return requested
    return list(CASE_MATRIX)


def _run_mode(case_ids: list[str], mode: Mode, fixture_dir: Path) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case_id in case_ids:
        case = CASE_MATRIX[case_id]
        if mode not in case.supported_modes:
            results.append(_blocked_result(case, mode))
            continue
        results.append(CASE_RUNNERS[case_id](fixture_dir, mode))
    return results


def _summarize(results: list[CaseResult]) -> dict[str, int]:
    return {
        "passed": sum(result.status == "passed" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "blocked": sum(result.status == "blocked" for result in results),
    }


def _print_results(results: list[CaseResult]) -> None:
    print("=" * 78)
    print("Old-vs-New Validation Summary")
    print("=" * 78)
    print(f"{'Case':<28} {'Mode':<8} {'Status':<8} Details")
    print("-" * 78)
    for result in results:
        numeric_diffs = [
            float(value) for key, value in result.metrics.items() if "diff" in key and isinstance(value, numbers.Real)
        ]
        detail = result.blocked_reason or (f"max diff {max(numeric_diffs):.3e}" if numeric_diffs else "")
        print(f"{result.case_id:<28} {result.mode:<8} {result.status:<8} {detail}")
        for line in result.details[:5]:
            print(f"{'':<48} {line}")
        if len(result.details) > 5:
            print(f"{'':<48} ... {len(result.details) - 5} more")
    print("-" * 78)
    summary = _summarize(results)
    print(f"Passed: {summary['passed']}  Failed: {summary['failed']}  Blocked: {summary['blocked']}")


def _write_report(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _summarize(results),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fixture", "live", "both"),
        default="fixture",
        help="Validation mode: pinned fixtures, live upstream, or both.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Run only a specific case id (repeatable).",
    )
    parser.add_argument(
        "--worktree",
        type=Path,
        default=DEFAULT_WORKTREE,
        help="Path to upstream worktree for live mode.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path for a structured JSON report.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List validation cases and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_cases:
        for case in CASE_MATRIX.values():
            modes = ",".join(case.supported_modes) if case.supported_modes else "blocked"
            print(f"{case.case_id}: [{modes}] {case.description}")
        return 0

    case_ids = _selected_case_ids(args.cases)
    results: list[CaseResult] = []

    if args.mode in ("fixture", "both"):
        results.extend(_run_mode(case_ids, "fixture", FIXTURE_DIR))

    if args.mode in ("live", "both"):
        if not args.worktree.exists():
            runnable_without_worktree = [
                case_id
                for case_id in case_ids
                if "live" in CASE_MATRIX[case_id].supported_modes and not CASE_MATRIX[case_id].requires_worktree
            ]
            if runnable_without_worktree:
                results.extend(_run_mode(runnable_without_worktree, "live", FIXTURE_DIR))
            for case_id in case_ids:
                case = CASE_MATRIX[case_id]
                if case_id in runnable_without_worktree:
                    continue
                reason = f"Live mode requested but worktree does not exist: {args.worktree}"
                results.append(_blocked_result(case, "live", reason))
        else:
            with TemporaryDirectory() as tempdir:
                worktree_cases = [
                    case_id
                    for case_id in case_ids
                    if "live" in CASE_MATRIX[case_id].supported_modes and CASE_MATRIX[case_id].requires_worktree
                ]
                non_worktree_cases = [
                    case_id
                    for case_id in case_ids
                    if "live" in CASE_MATRIX[case_id].supported_modes and not CASE_MATRIX[case_id].requires_worktree
                ]
                if worktree_cases:
                    fixture_dir = _ensure_live_fixture_dir(worktree_cases, args.worktree, Path(tempdir))
                    results.extend(_run_mode(worktree_cases, "live", fixture_dir))
                if non_worktree_cases:
                    results.extend(_run_mode(non_worktree_cases, "live", FIXTURE_DIR))

    _print_results(results)
    if args.report_json:
        _write_report(args.report_json, results)

    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
