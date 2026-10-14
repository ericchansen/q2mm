"""Generate API reference pages automatically from the q2mm package.

Called by mkdocs-gen-files during ``properdocs build``.  Walks every ``.py``
file under ``q2mm/``, creates a corresponding Markdown page with a
mkdocstrings ``::: module.path`` directive, and writes a literate-nav
SUMMARY.md so the sidebar stays in sync with the source tree.

New modules are picked up automatically — no manual ``.md`` files or
``properdocs.yml`` edits required.
"""

import ast
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
        if ident in {"q2mm", "q2mm.application"}:
            fd.write(
                "# Canonical public exports\n\n"
                "These lazy exports are the same objects as their owning modules, "
                "not a second execution API. See the "
                "[application lifecycle](../../how-it-works/optimization-guide.md"
                "#canonical-application-lifecycle).\n\n"
                if ident == "q2mm"
                else "# Application services\n\nLazy exports retain their canonical owners.\n\n"
            )
            tree = ast.parse(path.read_text(encoding="utf-8"))
            declaration = next(
                node
                for node in tree.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in {"_LAZY", "_LAZY_EXPORTS"} for target in node.targets
                )
            )
            exports = ast.literal_eval(declaration.value)
            for name, (owner, attribute) in exports.items():
                if owner.rsplit(".", 1)[-1].startswith("_"):
                    fd.write(f"- [`{name}`](#{owner}.{attribute})\n")
                    continue
                owner_path = root.joinpath(*owner.split("."))
                relative = Path(*owner.split("."))
                target = relative / "index.md" if owner_path.is_dir() else relative.with_suffix(".md")
                # Root and application pages are one and two levels below reference/.
                prefix = "../" if ident == "q2mm" else "../../"
                fd.write(f"- [`{name}`]({prefix}{target.as_posix()}#{owner}.{attribute})\n")
            fd.write("\n")
            for owner, attribute in exports.values():
                if owner.rsplit(".", 1)[-1].startswith("_"):
                    fd.write(f"::: {owner}.{attribute}\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
