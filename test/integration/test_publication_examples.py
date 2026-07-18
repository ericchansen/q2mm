from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

from q2mm.benchmarks.systems._paths import ExternalDataRoots
from test._shared import REPO_ROOT

pytestmark = [pytest.mark.integration, pytest.mark.external_data]

_EXAMPLES = REPO_ROOT / "examples" / "publication"
_ROWS = (
    ("rh-enamide", 9, "partial_repository_reproduction"),
    ("heck-relay", 23, "executable_archive_reproduction"),
    ("pd-allyl", 21, "partial_repository_reproduction"),
    ("pd-conjugate", 10, "partial_repository_reproduction"),
    ("rh-conjugate", 10, "sdk_software_path_demonstration"),
    ("ferrocene", 7, "partial_repository_reproduction"),
)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import publication example {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _roots_or_skip() -> ExternalDataRoots:
    roots = ExternalDataRoots.from_environment()
    missing = []
    if roots.supporting_info is None or not roots.supporting_info.is_dir():
        missing.append("Q2MM_SUPPORTING_INFO")
    if roots.mm3_base is None or not roots.mm3_base.is_file():
        missing.append("Q2MM_MM3_BASE")
    if roots.rh_enamide is None or not roots.rh_enamide.is_dir():
        missing.append("Q2MM_RH_ENAMIDE")
    if missing:
        pytest.skip(f"publication example matrix unavailable; configure {', '.join(missing)}")
    return roots


@pytest.mark.parametrize(("key", "case_count", "status"), _ROWS)
def test_every_publication_example_runs_real_bounded_problem(
    key: str,
    case_count: int,
    status: str,
    tmp_path: Path,
) -> None:
    roots = _roots_or_skip()
    module = _load(_EXAMPLES / key / "run.py", f"bounded_publication_{key.replace('-', '_')}")
    output = tmp_path / key
    result = module.run(
        output_root=output,
        supporting_info=roots.supporting_info,
        mm3_base=roots.mm3_base,
        rh_enamide=roots.rh_enamide,
        bounded_ci=True,
    )

    assert result["case_count"] == case_count
    assert len(result["case_order"]) == case_count
    assert result["source_status"]["status"] == status
    assert result["optimization"]["iterations"] == 1
    assert result["optimization"]["convergence_claim"] is False
    assert result["execution"]["resolved_bounds"] == {"mode": "bounded_ci_no_parameter_update"}
    assert result["optimization"]["proof"]["proof_status"] in {
        "blocked_methodology",
        "bounded_software_path",
    }
    assert result["parameter_counts"]
    assert result["initial"]["categories"] == result["final"]["categories"]
    assert Path(result["saved"]["force_field"]).parent == output.resolve()
    assert Path(result["saved"]["manifest"]).parent == output.resolve()
    if key == "heck-relay":
        assert result["execution"]["scientific_default_bounds"]["fc_fraction"] == 0.05
        assert any(row["case_order"].count("prrts1") == 1 for row in result["blocked_rows"])
    if key == "rh-conjugate":
        assert "developmental" in result["note"].lower()
    if key == "ferrocene":
        assert result["stationary_point"] == "ground_state"
        assert result["force_field_composition"]["starting_point"] == "published"
        assert any(row["starting_point"] == "qfuerza" for row in result["blocked_rows"])
