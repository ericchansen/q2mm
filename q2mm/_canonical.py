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


def _screen_key(key: str, path: str) -> None:
    if _credential_like_key(key):
        raise CanonicalizationError(f"Secret-like field is not permitted at {path}.{key}.")


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
            if item.name in {"source_path", "path"}:
                payload[item.name] = path_identity(field_value)
            else:
                payload[item.name] = scientific_value(field_value)
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
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                if not coerce_keys:
                    raise CanonicalizationError(f"JSON object key at {_path} must be a string, got {raw_key!r}.")
                key = str(raw_key)
            else:
                key = raw_key
            if screen_secrets:
                _screen_key(key, _path)
            result[key] = json_value(
                item,
                strict=strict,
                stringify_unknown=stringify_unknown,
                coerce_keys=coerce_keys,
                screen_secrets=screen_secrets,
                _path=f"{_path}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            json_value(
                item,
                strict=strict,
                stringify_unknown=stringify_unknown,
                coerce_keys=coerce_keys,
                screen_secrets=screen_secrets,
                _path=f"{_path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        if strict:
            raise CanonicalizationError(f"Unordered collection is not permitted at {_path}.")
        return sorted(
            (
                json_value(
                    item,
                    strict=False,
                    stringify_unknown=stringify_unknown,
                    coerce_keys=coerce_keys,
                    screen_secrets=screen_secrets,
                    _path=f"{_path}[]",
                )
                for item in value
            ),
            key=lambda item: canonical_json(item, strict=False, stringify_unknown=True, coerce_keys=True),
        )
    if isinstance(value, Enum):
        return json_value(
            value.value,
            strict=strict,
            stringify_unknown=stringify_unknown,
            coerce_keys=coerce_keys,
            screen_secrets=screen_secrets,
            _path=_path,
        )
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, np.floating):
        return json_value(float(value), _path=_path)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return json_value(
            value.tolist(),
            strict=strict,
            stringify_unknown=stringify_unknown,
            coerce_keys=coerce_keys,
            screen_secrets=screen_secrets,
            _path=_path,
        )
    if isinstance(value, Path):
        if strict:
            raise CanonicalizationError(f"Filesystem path is not permitted in canonical JSON at {_path}.")
        return str(value)
    if stringify_unknown:
        return str(value)
    if not strict:
        return value
    raise CanonicalizationError(f"Unsupported JSON value at {_path}: {type(value).__name__}.")


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
