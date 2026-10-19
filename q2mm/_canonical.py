"""Dependency-neutral strict JSON serialization and fingerprinting.

This private module is the single implementation used by application and
benchmark persistence.  Callers choose whether unknown objects are rejected
(``strict=True``) or rendered as strings for the benchmark's legacy identity
contract.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from q2mm._provenance import _credential_like_key


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented as safe strict JSON."""


def path_identity(value: str | Path | None) -> str | None:
    """Return the filename portion of a path or None."""
    if value is None:
        return None
    return Path(value).name


def scientific_value(value: Any) -> Any:
    """Recursively normalize scientific model structures for fingerprinting."""
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        payload: dict[str, Any] = {}
        for item in dataclasses.fields(value):
            if not item.init and item.name in {
                "bonds_explicit",
                "angles_explicit",
                "torsions_explicit",
                "improper_torsions",
            }:
                continue
            field_value = getattr(value, item.name)
            payload[item.name] = (
                path_identity(field_value) if item.name in {"source_path", "path"} else scientific_value(field_value)
            )
        return payload
    if isinstance(value, Mapping):
        return {str(key): scientific_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [scientific_value(item) for item in value]
    return value


def json_value(
    value: Any,
    *,
    strict: bool = True,
    stringify_unknown: bool = False,
    coerce_keys: bool = False,
    screen_secrets: bool = False,
    _path: str = "$",
) -> Any:
    """Return a recursively normalized strict-JSON value.

    Non-finite floating-point values use stable string sentinels so serialized
    output always complies with RFC 8259.  In strict mode mapping keys must be
    strings and unknown objects are rejected.
    """

    def normalize(current: Any, path: str) -> Any:
        if isinstance(current, Mapping):
            result: dict[str, Any] = {}
            for raw_key, item in current.items():
                if not isinstance(raw_key, str):
                    if not coerce_keys:
                        raise CanonicalizationError(f"JSON object key at {path} must be a string, got {raw_key!r}.")
                    key = str(raw_key)
                else:
                    key = raw_key
                if screen_secrets and _credential_like_key(key):
                    raise CanonicalizationError(f"Secret-like field is not permitted at {path}.{key}.")
                result[key] = normalize(item, f"{path}.{key}")
            return result
        if isinstance(current, (list, tuple)):
            return [normalize(item, f"{path}[{index}]") for index, item in enumerate(current)]
        if isinstance(current, (set, frozenset)):
            if strict:
                raise CanonicalizationError(f"Unordered collection is not permitted at {path}.")
            return sorted(
                (normalize(item, f"{path}[]") for item in current),
                key=lambda item: canonical_json(item, strict=False, stringify_unknown=True, coerce_keys=True),
            )
        if isinstance(current, Enum):
            return normalize(current.value, path)
        if isinstance(current, bool) or current is None or isinstance(current, (str, int)):
            return current
        if isinstance(current, (float, np.floating)):
            scalar = float(current)
            if math.isnan(scalar):
                return "NaN"
            if math.isinf(scalar):
                return "Infinity" if scalar > 0 else "-Infinity"
            return scalar
        if isinstance(current, np.integer):
            return int(current)
        if isinstance(current, np.ndarray):
            return normalize(current.tolist(), path)
        if isinstance(current, Path):
            if strict:
                raise CanonicalizationError(f"Filesystem path is not permitted in canonical JSON at {path}.")
            return str(current)
        if stringify_unknown:
            return str(current)
        if not strict:
            return current
        raise CanonicalizationError(f"Unsupported JSON value at {path}: {type(current).__name__}.")

    return normalize(value, _path)


def canonical_json(
    payload: Any,
    *,
    strict: bool = True,
    stringify_unknown: bool = False,
    coerce_keys: bool = False,
    screen_secrets: bool = False,
) -> str:
    """Serialize *payload* to deterministic ASCII strict JSON."""
    normalized = json_value(
        payload,
        strict=strict,
        stringify_unknown=stringify_unknown,
        coerce_keys=coerce_keys,
        screen_secrets=screen_secrets,
    )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_fingerprint(
    payload: Any,
    *,
    strict: bool = True,
    stringify_unknown: bool = False,
    coerce_keys: bool = False,
    screen_secrets: bool = False,
) -> str:
    """Return ``sha256:<hex>`` over :func:`canonical_json`."""
    blob = canonical_json(
        payload,
        strict=strict,
        stringify_unknown=stringify_unknown,
        coerce_keys=coerce_keys,
        screen_secrets=screen_secrets,
    )
    return f"sha256:{hashlib.sha256(blob.encode('ascii')).hexdigest()}"
