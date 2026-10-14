from __future__ import annotations

import dataclasses
import html
import re
import subprocess
import sys
from pathlib import Path

import pytest

from q2mm.benchmarks.publications import publication_records
from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata
from scripts.gen_publication_pages import projected_pages, render_publication_sections, replace_generated_section
from test._shared import REPO_ROOT


def test_projection_preserves_every_record_and_target_without_loading_cases() -> None:
    records = publication_records()
    systems = tuple(map(system_metadata, SYSTEM_KEYS))
    rendered = render_publication_sections(records, systems)
    assert rendered == render_publication_sections(records, systems)
    inventory = rendered["publication-inventory"]
    profiles = rendered["publication-profiles"]
    assert sum(line.startswith("#### ") for line in profiles.splitlines()) == len(records)
    assert sum(line.startswith("| [") for line in inventory.splitlines()) == len(SYSTEM_KEYS)
    for system in systems:
        assert f"(`{system.key}`)" in inventory
        assert html.escape(system.name, quote=False) in inventory
        for form in system.default_forms:
            assert f"`{form}`" in inventory
    for record in records:
        system_section = profiles.split(f'<a id="{record.system}"></a>', 1)[1].split("<a id=", 1)[0]
        start = record.starting_point or "not specified"
        section = system_section.split(f"#### `{record.objective_profile.identifier}` / `{start}`", 1)[1]
        section = section.split("\n#### ", 1)[0]
        assert f"| `{start}` | {len(record.authoritative_case_ids)} | `{record.stationary_point}` " in section
        assert f"| `{record.status.value}` | {'Yes' if record.provisionable else 'No'} |" in section
        membership = ", ".join(f"`{case}`" for case in record.authoritative_case_ids) or "None recorded."
        assert f"**Authoritative case IDs (record order):** {membership}" in section
        for target in record.targets:
            assert f"[`{target.category.value}`]" in section
            assert f"| `{target.disposition.value}` | {html.escape(target.details, quote=False)} |" in section
        for blocker in record.blockers:
            assert f"- **Blocker:** {html.escape(blocker, quote=False)}" in section
        for note in record.notes:
            assert f"- **Note:** {html.escape(note, quote=False)}" in section


def test_rendering_projects_changed_metadata_instead_of_handwritten_statuses() -> None:
    record = dataclasses.replace(
        publication_records()[0],
        authoritative_case_ids=("case-B", "case-A"),
        blockers=("New | blocker\nwith detail",),
        notes=(),
        provisionable=False,
    )
    system = dataclasses.replace(system_metadata(record.system), name="Updated name", default_forms=("new-form",))
    rendered = render_publication_sections((record,), (system,))
    assert "| 2 |" in rendered["publication-profiles"]
    assert "`case-B`, `case-A`" in rendered["publication-profiles"]
    assert "| No |" in rendered["publication-profiles"]
    assert "New \\| blocker<br>with detail" in rendered["publication-profiles"]
    assert "Updated name" in rendered["publication-inventory"]
    assert "`new-form`" in rendered["publication-inventory"]


def test_projection_links_resolve_in_source_docs_and_every_status_row_has_evidence() -> None:
    sections = render_publication_sections(publication_records(), tuple(map(system_metadata, SYSTEM_KEYS)))
    evidence_root = "https://github.com/ericchansen/q2mm/blob/master/"
    for text in sections.values():
        for link in re.findall(r"\]\(([^)]+)\)", text):
            path_text, _, anchor = link.partition("#")
            path = (
                REPO_ROOT / path_text.removeprefix(evidence_root)
                if path_text.startswith(evidence_root)
                else REPO_ROOT / "docs" / "benchmarks" / path_text
            )
            assert path.is_file(), link
            if anchor:
                source = path.read_text(encoding="utf-8")
                headings = re.findall(r"^#+ (.+)$", source, flags=re.MULTILINE)
                slugs = {re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-") for heading in headings}
                assert anchor in slugs or f'<a id="{anchor}"></a>' in source, link
        for line in text.splitlines():
            if re.search(r"\| `(implemented|available|omitted|blocked)` \|", line):
                assert evidence_root in line
    profiles = sections["publication-profiles"]
    for line in profiles.splitlines():
        if re.search(r"\| `(published|qfuerza|not specified)` \|", line):
            assert "[Source/status details](https://" in line


def test_committed_snapshots_are_current() -> None:
    for page, projected in projected_pages(REPO_ROOT).items():
        assert projected == (REPO_ROOT / "docs" / page).read_text(encoding="utf-8")


def test_stale_snapshot_is_rejected_and_refresh_preserves_surrounding_text(tmp_path: Path) -> None:
    for page, projected in projected_pages(REPO_ROOT).items():
        destination = tmp_path / "docs" / page
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(projected, encoding="utf-8")
    index = tmp_path / "docs" / "benchmarks" / "index.md"
    index.write_text("Handwritten introduction\n" + index.read_text(encoding="utf-8").replace("`ch3f`", "`stale`"))
    with pytest.raises(ValueError, match="Stale publication snapshot.*--write"):
        projected_pages(tmp_path)
    refreshed = projected_pages(tmp_path, check=False)
    assert refreshed[Path("benchmarks", "index.md")].startswith("Handwritten introduction\n")
    assert "`stale`" not in refreshed[Path("benchmarks", "index.md")]
    for page, text in refreshed.items():
        (tmp_path / "docs" / page).write_text(text, encoding="utf-8")
    assert projected_pages(tmp_path) == refreshed


@pytest.mark.parametrize(
    "text",
    [
        "no markers",
        "<!-- BEGIN GENERATED test -->\n<!-- BEGIN GENERATED test -->\n<!-- END GENERATED test -->",
        "<!-- END GENERATED test -->\n<!-- BEGIN GENERATED test -->",
    ],
)
def test_invalid_snapshot_markers_fail_explicitly(text: str) -> None:
    with pytest.raises(ValueError, match="marker pair"):
        replace_generated_section(text, "test", "table")


def test_pure_import_and_full_projection_do_not_import_docs_engines_or_load_data() -> None:
    code = """
import importlib.abc
import pathlib
import sys

class DenyEngines(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = (
            "jax", "jaxlib", "openmm", "psi4", "scipy", "mkdocs",
            "properdocs", "mkdocs_gen_files", "q2mm.backends.mm",
            "q2mm.backends.qm", "q2mm.io",
        )
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Unexpected engine/docs/data-reader import: " + fullname)

sys.meta_path.insert(0, DenyEngines())
from scripts.gen_publication_pages import projected_pages, render_publication_sections
assert not any(name == "q2mm" or name.startswith("q2mm.") for name in sys.modules)
assert render_publication_sections((), ())["publication-profiles"] == "\\n"

import numpy
import q2mm.benchmarks.systems as systems

def reject_load(*args, **kwargs):
    raise AssertionError("Metadata rendering must not load scientific data")

numpy.load = reject_load
systems.load_system = reject_load
original_open = pathlib.Path.open
def guarded_open(path, *args, **kwargs):
    if path.suffix in {".fld", ".log", ".xyz", ".npy", ".npz", ".fchk"}:
        reject_load()
    return original_open(path, *args, **kwargs)
pathlib.Path.open = guarded_open
assert len(projected_pages(pathlib.Path.cwd())) == 2
print("pure import and full projection are data/engine/docs-runtime free")
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "data/engine/docs-runtime free" in result.stdout
