"""Generate API reference pages automatically from the q2mm package.

Called by mkdocs-gen-files at build time.  Walks every ``.py`` file under
``q2mm/``, creates a corresponding Markdown page with a mkdocstrings
``::: module.path`` directive, and writes a literate-nav SUMMARY.md so
the sidebar stays in sync with the source tree.

New modules are picked up automatically — no manual ``.md`` files or
``mkdocs.yml`` edits required.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "q2mm"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __main__, private helpers, and __pycache__
    if parts[-1] == "__main__":
        continue
    if parts[-1].startswith("_") and parts[-1] != "__init__":
        continue
    if "__pycache__" in parts:
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
