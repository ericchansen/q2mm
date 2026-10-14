"""Image inventory and exact build-input behavior, not copied image policies."""

from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import ci_image


def test_canonical_environment_discovery() -> None:
    images = ci_image.image_inputs()
    assert set(images) == {path.stem for path in ci_image.ENVS.glob("*.yml")}
    assert all(entry["python"].count(".") == 1 for entry in images.values())


def test_only_actual_build_inputs_change_references_and_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envs = tmp_path / "envs"
    envs.mkdir()
    for name, python in (("first", "3.14"), ("second", "3.15")):
        (envs / f"{name}.yml").write_text(f"dependencies:\n  - python={python}\n")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM example\n")
    monkeypatch.setattr(ci_image, "ENVS", envs)
    monkeypatch.setattr(ci_image, "DOCKERFILE", dockerfile)
    before = ci_image.image_inputs()
    assert before == ci_image.image_inputs()
    assert (before["first"]["python"], before["second"]["python"]) == ("3.14", "3.15")
    assert ci_image.selected_images(before, ["scripts/ci_image.py", "docs/index.md"]) == []
    assert ci_image.selected_images(before, [".github\\envs\\first.yml"]) == ["first"]
    (envs / "first.yml").write_text("dependencies:\n  - python=3.15\n")
    changed = ci_image.image_inputs()
    assert changed["first"]["reference"] != before["first"]["reference"]
    assert changed["second"] == before["second"]
    dockerfile.write_text("FROM another\n")
    assert all(ci_image.image_inputs()[key]["reference"] != changed[key]["reference"] for key in changed)
    assert ci_image.selected_images(changed, [".github/docker/Dockerfile"]) == ["first", "second"]
    (envs / "first.yml").write_text("dependencies:\n  - python\n")
    with pytest.raises(ValueError, match="pin a Python"):
        ci_image.image_inputs()


def test_local_helper_syntax_and_unknown_job() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    root = Path(__file__).parents[1]
    subprocess.run([bash, "-n", "scripts/ci_local.sh"], cwd=root, check=True)
    result = subprocess.run(
        [bash, "scripts/ci_local.sh", "--job", "unknown"],
        cwd=root,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2 and "Unknown responsibility" in result.stderr
