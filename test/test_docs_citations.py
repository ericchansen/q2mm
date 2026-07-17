"""Citation audit: methodology pages must cite the primary QFUERZA/TS DOIs.

Every rewritten QFUERZA / transition-state methodology page must carry a
clickable link to the Zotero-verified primary sources, and the superseded
JCTC DOI must never stand in for the transition-state reference:

- QFUERZA: Farrugia et al. 2025 — https://doi.org/10.1021/acs.jctc.5c01751
- TS Hessian inversion / Method E2: Limé & Norrby (J. Comput. Chem. 36, 244;
  Zotero item UATW7GLJ) — https://doi.org/10.1002/jcc.23797
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test._shared import REPO_ROOT

_FARRUGIA = "https://doi.org/10.1021/acs.jctc.5c01751"
_LIME_NORRBY = "https://doi.org/10.1002/jcc.23797"
_WRONG_TS_DOI = "10.1021/acs.jctc.5b00461"
_PUBLICATION_SOURCES = {
    "XDS9K3C4": "https://doi.org/10.1021/acs.jctc.5c01751",
    "JXH5HHS6": "https://doi.org/10.1021/ct800132a",
    "2NHVUNW5": "https://doi.org/10.1021/jacs.0c01979",
    "QVKE99W3": "https://doi.org/10.1038/s41467-021-27065-2",
    "R62E6EGV": "https://doi.org/10.1021/acs.joc.1c00136",
    "SXWNJTQ2": "https://doi.org/10.1021/acs.joc.2c01553",
    "AAZ6I5V3": "https://doi.org/10.7274/k930bv76q4n",
    "QCQ6Z5MR": "https://doi.org/10.7274/rj430290902",
}

# Methodology pages whose scientific claims rest on QFUERZA + TS inversion.
_QFUERZA_PAGES = (
    REPO_ROOT / "docs" / "how-it-works" / "theory.md",
    REPO_ROOT / "docs" / "benchmarks" / "qfuerza-recovery.md",
)

# Current docs + agent instructions that must not cite the wrong TS DOI.
_CURRENT_TEXT_FILES = (
    *(REPO_ROOT / "docs").rglob("*.md"),
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "README.md",
)


@pytest.mark.parametrize("page", _QFUERZA_PAGES, ids=lambda p: p.name)
def test_methodology_page_cites_both_primary_dois(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    assert _FARRUGIA in text, f"{page.name} is missing the QFUERZA (Farrugia 2025) DOI link {_FARRUGIA}"
    assert _LIME_NORRBY in text, f"{page.name} is missing the TS-inversion (Limé & Norrby) DOI link {_LIME_NORRBY}"


def test_agents_cites_correct_ts_doi() -> None:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert _LIME_NORRBY in text, f"AGENTS.md must cite the TS reference DOI {_LIME_NORRBY}"


def test_wrong_ts_doi_absent_from_current_docs_and_instructions() -> None:
    offenders = [
        f.relative_to(REPO_ROOT).as_posix()
        for f in _CURRENT_TEXT_FILES
        if f.is_file() and _WRONG_TS_DOI in f.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"the superseded TS DOI {_WRONG_TS_DOI} must not appear in current docs/instructions: {offenders}"
    )


def test_references_use_zotero_validated_publication_metadata() -> None:
    text = (REPO_ROOT / "docs" / "references.md").read_text(encoding="utf-8")
    for key, url in _PUBLICATION_SOURCES.items():
        assert f"`{key}`" in text
        assert url in text
    assert "University of Notre Dame, **2021**" in text
    assert "University of Notre Dame, **2019**" in text


def test_publication_system_pages_and_nav_cover_every_executable_row() -> None:
    keys = ("rh-enamide", "heck-relay", "pd-allyl", "pd-conjugate", "rh-conjugate", "ferrocene")
    nav = (REPO_ROOT / "properdocs.yml").read_text(encoding="utf-8")
    coverage = (REPO_ROOT / "docs" / "benchmarks" / "published-ff-validation.md").read_text(encoding="utf-8")
    for key in keys:
        page = REPO_ROOT / "docs" / "systems" / f"{key}.md"
        assert page.is_file()
        assert f"systems/{key}.md" in nav
        assert f"(../systems/{key}.md)" in coverage


def test_byo_docs_use_root_facade_and_current_example_paths() -> None:
    files = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "getting-started.md",
        REPO_ROOT / "docs" / "tutorial.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for symbol in ("q2mm.prepare", "q2mm.evaluate", "q2mm.optimize", "q2mm.save", "load_fchk_molecule"):
        assert symbol in combined
    assert "examples/publication/rh-enamide" in combined
    assert "examples/" + "rh-enamide" not in combined
    assert "examples/" + "sn2-test" not in combined
    assert "examples/" + "ethane" not in combined
    assert "SciPy is optional" in combined


def test_rh_source_policy_is_narrow_and_factual() -> None:
    files = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "systems" / "rh-enamide.md",
        REPO_ROOT / "examples" / "publication" / "rh-enamide" / "README.md",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "excluded from wheel and sdist" in combined
    assert "redistribution/licensing is not established" in combined.lower()
    assert "https://doi.org/10.1021/ct800132a" in combined
