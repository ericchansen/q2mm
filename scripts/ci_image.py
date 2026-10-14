#!/usr/bin/env python3
"""Resolve native CI image inputs, not registry availability.

``python scripts/ci_image.py openmm --field reference`` supplies the same
input-addressed tag to image builds and local/CI test jobs. The environment
files are the sole inventory. ``--list`` prints their keys; ``--selected``
accepts changed repository paths and emits affected image keys as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / ".github" / "docker" / "Dockerfile"
ENVS = ROOT / ".github" / "envs"


def image_reference(key: str, environment: str) -> str:
    """Hash only the exact Dockerfile and selected environment bytes."""
    digest = hashlib.sha256(DOCKERFILE.read_bytes() + b"\0" + (ENVS / environment).read_bytes())
    return f"ghcr.io/ericchansen/q2mm/ci-{key}:inputs-{digest.hexdigest()}"


def image_inputs() -> dict[str, dict[str, str]]:
    """Discover environments and derive interpreter and image reference."""
    images = {}
    for environment in sorted(ENVS.glob("*.yml")):
        key = environment.stem
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key):
            raise ValueError(f"invalid image key: {key!r}")
        text = environment.read_text(encoding="utf-8")
        python = re.search(r"^\s*-\s*python=(\d+\.\d+)\s*$", text, re.MULTILINE)
        if python is None:
            raise ValueError(f"{key}: environment must pin a Python major.minor version")
        images[key] = {
            "environment": environment.name,
            "python": python[1],
            "reference": image_reference(key, environment.name),
        }
    if not images:
        raise ValueError(f"no environments found in {ENVS}")
    return images


def selected_images(images: dict[str, dict[str, str]], paths: list[str]) -> list[str]:
    """Rebuild only for Dockerfile changes or an image's own environment."""
    changed = {path.replace("\\", "/").removeprefix("./") for path in paths}
    if ".github/docker/Dockerfile" in changed:
        return sorted(images)
    return sorted(key for key, entry in images.items() if f".github/envs/{entry['environment']}" in changed)


def main() -> int:
    """Print an inventory field, affected keys, or input check status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--field", choices=("reference", "environment", "python"), default="reference")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--check", action="store_true")
    operation.add_argument("--list", action="store_true")
    operation.add_argument("--selected", nargs="*", metavar="CHANGED_PATH")
    args = parser.parse_args()
    if args.image is not None and (args.check or args.list or args.selected is not None):
        parser.error("choose an image or --check/--list/--selected")
    try:
        images = image_inputs()
        if args.check:
            print("CI image inputs OK (registry publication not checked)")
        elif args.list:
            print(json.dumps(sorted(images)))
        elif args.selected is not None:
            print(json.dumps(selected_images(images, args.selected)))
        elif args.image in images:
            print(images[args.image][args.field])
        else:
            parser.error(f"choose an image: {', '.join(images)}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
