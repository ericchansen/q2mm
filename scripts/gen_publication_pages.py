"""Project canonical publication metadata into checked, GitHub-readable docs.

Run ``python scripts/gen_publication_pages.py --write`` after metadata changes.
The docs build and core tests reject stale snapshots; rendering needs neither
the docs runtime nor scientific data.
"""

from __future__ import annotations

import argparse
import html
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from q2mm.benchmarks.systems import SystemMetadata
    from q2mm.models.publication import PublicationMetadata

_EVIDENCE = "https://github.com/ericchansen/q2mm/blob/master/validation/published_ffs/README.md"
_EVIDENCE_ANCHORS = {
    "pd-conjugate": "pd-14-conjugate-addition",
    "rh-conjugate": "rh-14-conjugate-addition",
    "osmium-dihydroxylation": "blocked-historical-records",
    "ru-ketone-hydrogenation": "blocked-historical-records",
    "sulfone": "blocked-historical-records",
}
_PAGES = {
    "publication-inventory": Path("benchmarks", "index.md"),
    "publication-profiles": Path("benchmarks", "published-ff-validation.md"),
}


def _cell(value: str) -> str:
    return html.escape(value, quote=False).replace("|", r"\|").replace("\n", "<br>")


def render_publication_sections(
    records: Sequence[PublicationMetadata], systems: Sequence[SystemMetadata]
) -> dict[str, str]:
    """Render pure Markdown projections in registry and canonical-record order.

    Only documentation routes live here; membership, claims, target dispositions,
    blockers, notes, names, and functional forms come from the supplied records.
    """
    names = {system.key: system.name for system in systems}
    publication_keys = {record.system for record in records}
    inventory = [
        "| Registered system | Default functional forms | Publication coverage |",
        "|---|---|---|",
    ]
    for system in systems:
        page = "small-molecules" if system.key in {"ch3f", "ch3f-sn2"} else system.key
        coverage = (
            f"[Profile/start records](published-ff-validation.md#{system.key})"
            if system.key in publication_keys
            else "No canonical publication record"
        )
        inventory.append(
            f"| [{_cell(system.name)}](../systems/{page}.md) (`{system.key}`) "
            f"| {', '.join(f'`{form}`' for form in system.default_forms)} | {coverage} |"
        )

    profiles: list[str] = []
    for key in dict.fromkeys(record.system for record in records):
        evidence = f"{_EVIDENCE}#{_EVIDENCE_ANCHORS.get(key, key)}"
        profiles.extend(
            [
                f'<a id="{key}"></a>',
                "",
                f"### {_cell(names.get(key, key))} profiles",
                "",
                f"[Source evidence and limitations]({evidence})"
                + (f" · [System guide](../systems/{key}.md)" if key in names else ""),
                "",
            ]
        )
        for record in records:
            if record.system != key:
                continue
            start = record.starting_point or "not specified"
            profiles.extend(
                [
                    f"#### `{record.objective_profile.identifier}` / `{start}`",
                    "",
                    "| Starting point | Authoritative cases | Stationary point | Claim | Provisionable | Evidence |",
                    "|---|---:|---|---|---|---|",
                    f"| `{start}` | {len(record.authoritative_case_ids)} | `{record.stationary_point}` "
                    f"| `{record.status.value}` | {'Yes' if record.provisionable else 'No'} "
                    f"| [Source/status details]({evidence}) |",
                    "",
                    "**Authoritative case IDs (record order):** "
                    + (", ".join(f"`{case}`" for case in record.authoritative_case_ids) or "None recorded."),
                    "",
                    "| Target category | Disposition | Canonical detail |",
                    "|---|---|---|",
                ]
            )
            profiles.extend(
                f"| [`{target.category.value}`]({evidence}) | `{target.disposition.value}` | {_cell(target.details)} |"
                for target in record.targets
            )
            profiles.append("")
            profiles.extend(f"- **Blocker:** {_cell(blocker)}" for blocker in record.blockers)
            profiles.extend(f"- **Note:** {_cell(note)}" for note in record.notes)
            profiles.append("")
    return {
        "publication-inventory": "\n".join(inventory) + "\n",
        "publication-profiles": "\n".join(profiles).rstrip() + "\n",
    }


def replace_generated_section(text: str, section: str, rendered: str) -> str:
    """Replace exactly one marked section, rejecting missing or ambiguous markers."""
    begin = f"<!-- BEGIN GENERATED {section} -->"
    end = f"<!-- END GENERATED {section} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair for {section!r}.")
    before, _, remainder = text.partition(begin)
    _, found_end, after = remainder.partition(end)
    if not found_end:
        raise ValueError(f"Reversed marker pair for {section!r}.")
    return f"{before}{begin}\n\n{rendered.rstrip()}\n\n{end}{after}"


def projected_pages(root: Path, *, check: bool = True) -> dict[Path, str]:
    """Read canonical metadata and reject stale source snapshots by default."""
    from q2mm.benchmarks.publications import publication_records
    from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata

    sections = render_publication_sections(publication_records(), tuple(map(system_metadata, SYSTEM_KEYS)))
    pages: dict[Path, str] = {}
    for section, page in _PAGES.items():
        text = (root / "docs" / page).read_text(encoding="utf-8")
        projected = replace_generated_section(text, section, sections[section])
        if check and projected != text:
            raise ValueError(
                f"Stale publication snapshot: docs/{page.as_posix()}. "
                "Run python scripts/gen_publication_pages.py --write and commit the updated docs."
            )
        pages[page] = projected
    return pages


def build_publication_pages(root: Path) -> None:
    """Check source freshness and publish the same projection with gen-files."""
    import mkdocs_gen_files

    for page, text in projected_pages(root).items():
        with mkdocs_gen_files.open(page, "w") as stream:
            stream.write(text)
        mkdocs_gen_files.set_edit_path(page, Path("docs") / page)


def main() -> None:
    """Refresh committed snapshots or check them without importing docs tooling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh the marked source-doc sections.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    for page, text in projected_pages(root, check=not args.write).items():
        if args.write:
            (root / "docs" / page).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
elif __name__ == "<run_path>":
    build_publication_pages(Path(__file__).resolve().parent.parent)
