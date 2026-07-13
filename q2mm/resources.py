"""Access to scientific resources distributed with Q2MM."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def sn2_reference_dir() -> Path:
    """Return the installed CH3F/SN2 reference-data directory.

    Q2MM wheels are installed as ordinary filesystem packages.  Requiring a
    filesystem-backed resource keeps the existing path-oriented scientific I/O
    APIs usable while avoiding any lookup relative to a repository checkout.

    Returns:
        Path: Directory containing the packaged CH3F/SN2 reference files.

    Raises:
        RuntimeError: If the package is loaded by a non-filesystem importer.

    """
    resource = files("q2mm").joinpath("data", "sn2")
    if not isinstance(resource, Path):
        raise RuntimeError(
            "Q2MM's scientific resources require a filesystem-backed installation; "
            "install the wheel with pip instead of importing it from a zip archive."
        )
    return resource


def validate_sn2_resources() -> None:
    """Verify packaged CH3F/SN2 files against their SHA-256 manifest.

    Raises:
        RuntimeError: If metadata is invalid, a file is missing, or a checksum
            or size does not match.

    """
    resource_dir = sn2_reference_dir()
    metadata_path = resource_dir / "manifest.json"
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        entries = metadata["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid packaged SN2 resource manifest: {metadata_path}") from exc

    if not isinstance(entries, list):
        raise RuntimeError(f"Invalid packaged SN2 resource manifest: {metadata_path}")

    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid packaged SN2 resource manifest entry: {entry!r}")
        try:
            name = entry["name"]
            expected_size = entry["size"]
            expected_sha256 = entry["sha256"]
        except KeyError as exc:
            raise RuntimeError(f"Incomplete packaged SN2 resource manifest entry: {entry!r}") from exc
        if not isinstance(name, str) or not isinstance(expected_size, int) or not isinstance(expected_sha256, str):
            raise RuntimeError(f"Invalid packaged SN2 resource manifest entry: {entry!r}")
        names.append(name)

    if len(names) != len(set(names)):
        raise RuntimeError(f"Packaged SN2 resource manifest contains duplicate file names: {metadata_path}")

    declared = set(names)
    actual = {path.name for path in resource_dir.iterdir() if path.is_file() and path.name != metadata_path.name}
    if declared != actual:
        raise RuntimeError(
            "Packaged SN2 resource manifest coverage differs: "
            f"missing={sorted(actual - declared)}, extra={sorted(declared - actual)}"
        )

    for entry in entries:
        name = entry["name"]
        expected_size = entry["size"]
        expected_sha256 = entry["sha256"]
        path = resource_dir / name
        if not path.is_file():
            raise RuntimeError(f"Packaged SN2 resource is missing: {path}")
        content = path.read_bytes()
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if len(content) != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Packaged SN2 resource failed integrity validation: {name} "
                f"(expected size={expected_size}, sha256={expected_sha256}; "
                f"got size={len(content)}, sha256={actual_sha256})"
            )
