"""Developmental Rh 1,4-conjugate-addition case study."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_NAME = "_q2mm_publication_example_runner"
_PATH = Path(__file__).parents[1] / "_runner.py"
if _NAME in sys.modules:
    _RUNNER = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(_NAME, _PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"Cannot load shared publication runner: {_PATH}")
    _RUNNER = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = _RUNNER
    _SPEC.loader.exec_module(_RUNNER)

CONFIG = _RUNNER.PublicationExample(
    key="rh-conjugate",
    expected_status="sdk_software_path_demonstration",
    required_roots=("supporting_info", "mm3_base"),
    note="Wahlers 2021 Chapter 6 labels this force field developmental; this is an SDK path demonstration.",
)


def run(**kwargs: Any) -> dict[str, Any]:
    """Run the developmental ten-case Rh-conjugate profile."""
    return _RUNNER.run_publication(CONFIG, **kwargs)


def main() -> int:
    """Run the command-line case study."""
    return _RUNNER.main_for(CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
