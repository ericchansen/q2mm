"""Guard focused benchmark ownership and diagnostic behavior without native runtimes."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from q2mm.backends.contracts import PreparationRequest
from q2mm.benchmarks import acceptance, analysis, artifacts, profiles, records, runner
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from test._shared import REPO_ROOT
from test.test_architecture_doc import _imported_dotted_modules

_OWNED_DEFINITIONS = {
    profiles: (
        "ConfigurationError",
        "resolve_optimizer",
        "_resolve_workflow",
        "_classify_backend",
        "_norm_path",
        "_load_kwargs",
        "_default_form",
    ),
    acceptance: ("classify_ratio",),
    analysis: (
        "real_frequencies",
        "frequency_rmsd",
        "frequency_mae",
        "_mm_real_frequencies",
        "_frequency_analysis",
        "compute_distortions",
        "_pes_distortion_summary",
        "_mean_ci95",
        "_score_interval_summary",
    ),
    records: (
        "sanitize_for_json",
        "_git_info",
        "build_run_provenance",
        "_data_provenance",
        "result_to_dict",
        "_resolved_summary",
        "_initial_summary",
        "_optimization_summary",
        "CandidateResult",
        "LoadedCandidate",
        "RunOutcome",
    ),
    artifacts: (
        "write_json",
        "read_json",
        "_candidate_path",
        "persist_candidate",
        "_ff_extension",
        "_serialize_ff",
        "_opposite_ext",
        "promote_candidate",
        "load_candidates",
    ),
}


def test_moved_definitions_have_one_owner_and_public_exports_are_direct() -> None:
    root = REPO_ROOT / "q2mm" / "benchmarks"
    definitions = {}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions[path.stem] = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for owner, names in _OWNED_DEFINITIONS.items():
        short_name = owner.__name__.rsplit(".", 1)[-1]
        for name in names:
            assert {module for module, found in definitions.items() if name in found} == {short_name}
            value = getattr(owner, name)
            assert value.__module__ == owner.__name__
            if not name.startswith("_"):
                assert getattr(runner, name) is value
    assert runner.RunProfile is profiles.RunProfile
    assert runner.DATA_DIR_FOR_SYSTEM is profiles.DATA_DIR_FOR_SYSTEM
    assert runner.REPO_ROOT == records.REPO_ROOT == REPO_ROOT
    assert definitions["runner"] == {
        "ExecutionError",
        "_is_jax_backend",
        "_terminal",
        "run_profile",
        "_run_profile_inner",
        "_execute",
        "run_profiles",
    }
    assert not hasattr(runner, "_build_evaluator_factory")
    assert not hasattr(runner, "_write_json_atomic")


def test_existing_canonical_projection_and_gradient_resolution_are_reused() -> None:
    from q2mm._result_serialization import result_payload
    from q2mm.optimizers.catalog import expected_result_gradient

    assert records.result_payload is result_payload
    assert runner._expected_result_gradient is expected_result_gradient
    tree = ast.parse((REPO_ROOT / "q2mm" / "benchmarks" / "artifacts.py").read_text(encoding="utf-8"))
    sdk_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "q2mm.application.persistence"
        for alias in node.names
    }
    assert {"save", "_cleanup_files", "_temp_sibling"} <= sdk_imports


@pytest.mark.parametrize(
    "module,forbidden",
    [
        ("analysis", {"profiles", "records", "artifacts", "runner"}),
        ("profiles", {"analysis", "records", "artifacts", "runner"}),
        ("records", {"analysis", "artifacts", "runner"}),
        ("artifacts", {"analysis", "profiles", "runner"}),
    ],
)
def test_owner_dependencies_do_not_point_back_to_coordinator(module: str, forbidden: set[str]) -> None:
    imported = _imported_dotted_modules(REPO_ROOT / "q2mm" / "benchmarks" / f"{module}.py")
    for name in forbidden:
        prefix = f"q2mm.benchmarks.{name}"
        assert not any(target == prefix or target.startswith(prefix + ".") for target in imported)


def test_owner_imports_do_not_load_optional_runtimes() -> None:
    code = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        forbidden = {"jax", "jaxlib", "scipy", "openmm", "psi4", "optax", "jaxopt"}

        class RejectRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("eager optional runtime: " + fullname)
                return None

        sys.meta_path.insert(0, RejectRuntime())
        for name in ("profiles", "analysis", "records", "artifacts", "runner", "cli"):
            importlib.import_module("q2mm.benchmarks." + name)
        assert not forbidden.intersection(sys.modules)
        print("lazy benchmark owners")
        """
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "lazy benchmark owners"


def test_frequency_diagnostics_keep_filtering_truncation_and_core_formulas() -> None:
    threshold = analysis.REAL_FREQUENCY_THRESHOLD
    np.testing.assert_array_equal(
        analysis.real_frequencies([threshold + 2, threshold, threshold - 1, threshold + 1, float("nan")]),
        [threshold + 1, threshold + 2],
    )
    assert analysis.frequency_rmsd([1, 2], [4, 6, 999]) == pytest.approx(np.sqrt(12.5))
    assert analysis.frequency_mae([1, 2], [4, 6, 999]) == 3.5
    assert np.isnan(analysis.frequency_rmsd([], [1]))
    assert np.isnan(analysis.frequency_mae([1], []))


def test_pes_diagnostics_keep_mode_cutoff_displacements_and_topology() -> None:
    from q2mm.constants import BOHR_TO_ANG

    molecule = Molecule(
        symbols=("H",),
        geometry=((0.0, 0.0, 0.0),),
        atom_types=("H1",),
        bonds=(),
        angles=(),
        torsions=(),
        charge=0,
        multiplicity=2,
        name="analysis-probe",
    )
    ff = ForceField(functional_form=FunctionalForm.HARMONIC)
    seen = []
    backend = Mock()

    def prepare(request: PreparationRequest) -> SimpleNamespace:
        structure = request.molecule
        seen.append(structure)
        energy = 3.0 + float(np.sum(structure.geometry**2))
        return SimpleNamespace(energy=lambda _request: SimpleNamespace(energy=energy))

    backend.prepare.side_effect = prepare
    modes = {"eigenvalues": np.array([0.0, 1e-3, 1.0]), "eigenvectors": np.eye(3), "masses_amu": np.ones(1)}
    results, equilibrium_energy, elapsed = analysis.compute_distortions(molecule, ff, backend, modes)
    assert equilibrium_energy == 3.0
    assert elapsed >= 0.0
    assert [mode["mode_idx"] for mode in results] == [2]
    assert [point["d_ang"] for point in results[0]["displacements"]] == [0.05, 0.10, 0.15]
    assert len(seen) == 4
    for structure in seen:
        assert structure.bonds == structure.angles == structure.torsions == ()
        assert structure.atom_types == molecule.atom_types
        assert (structure.name, structure.charge, structure.multiplicity) == ("analysis-probe", 0, 2)
    for point, structure in zip(results[0]["displacements"], seen[1:], strict=True):
        displacement = point["d_ang"]
        expected_qm = 0.5 * (displacement / BOHR_TO_ANG) ** 2 * 627.5094740631
        assert point["e_qm"] == pytest.approx(expected_qm)
        assert point["e_mm"] == pytest.approx(displacement**2)
        assert point["pct_err"] == pytest.approx((displacement**2 - expected_qm) / expected_qm * 100.0)
        np.testing.assert_allclose(structure.geometry, [[0.0, 0.0, displacement]])
