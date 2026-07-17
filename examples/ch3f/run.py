"""Fresh ground-state force-field example using installed q2mm."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_NAME = "_q2mm_small_example_runner"
_PATH = Path(__file__).parents[1] / "_small_runner.py"
if _NAME in sys.modules:
    _RUNNER = sys.modules[_NAME]
else:
    _SPEC = importlib.util.spec_from_file_location(_NAME, _PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise RuntimeError(f"Cannot load shared example runner: {_PATH}")
    _RUNNER = importlib.util.module_from_spec(_SPEC)
    sys.modules[_NAME] = _RUNNER
    _SPEC.loader.exec_module(_RUNNER)

CONFIG = _RUNNER.SmallExample(
    key="ch3f",
    stationary_point="ground_state",
    geometry_name="ch3f-optimized.xyz",
    hessian_name="ch3f-hessian.npy",
    charge=0,
    bond_tolerance=1.5,
)


def run(**kwargs: Any) -> dict[str, Any]:
    """Run the CH3F example; pass ``fchk=`` to use a caller-owned molecule."""
    return _RUNNER.run_small(CONFIG, **kwargs)


def main() -> int:
    """Run the command-line example."""
    return _RUNNER.main_for(CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
