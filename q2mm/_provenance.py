"""Strict immutable JSON data used by provenance records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import NoReturn

_CREDENTIAL_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|auth(?:orization)?|credential|"
    r"password|passwd|private[_-]?key|secret|session[_-]?key|token)(?:$|[_-])",
    re.IGNORECASE,
)
_CREDENTIAL_DATA = (
    re.compile(r"^\s*(?:basic|bearer)\s+\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"(?:^|[\s;])(?:api[_-]?key|access[_-]?key|authorization|credential|"
        r"password|passwd|private[_-]?key|secret|token)\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
_CREDENTIAL_KEY_TERMS = (
    "accesskey",
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "privatekey",
    "secret",
    "sessionkey",
    "token",
)


class FrozenJSONMapping(dict[str, object]):
    """A recursively immutable mapping that remains JSON serializable."""

    @staticmethod
    def _immutable() -> NoReturn:
        raise TypeError("provenance mappings are immutable")

    def __delitem__(self, key: str) -> None:
        self._immutable()

    def __ior__(self, value: object) -> FrozenJSONMapping:  # type: ignore[override,misc]
        self._immutable()

    def __setitem__(self, key: str, value: object) -> None:
        self._immutable()

    def clear(self) -> None:
        self._immutable()

    def pop(self, key: str, default: object = None) -> object:
        self._immutable()

    def popitem(self) -> tuple[str, object]:
        self._immutable()

    def setdefault(self, key: str, default: object = None) -> object:
        self._immutable()

    def update(self, *args: object, **kwargs: object) -> None:
        self._immutable()


def _freeze_json_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _CREDENTIAL_DATA):
            raise ValueError(f"{path} contains credential-like data.")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        return freeze_json_mapping(value, path=path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{path} contains non-JSON-safe value of type {type(value).__name__}.")


def _credential_like_key(key: str) -> bool:
    """Return whether a key can identify secret material in common styles."""
    if _CREDENTIAL_KEY.search(key):
        return True
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return any(normalized.startswith(term) or normalized.endswith(term) for term in _CREDENTIAL_KEY_TERMS)


def freeze_json_mapping(value: Mapping[str, object], *, path: str) -> FrozenJSONMapping:
    """Validate and recursively freeze a JSON-safe string-keyed mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    frozen: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path} keys must be strings; got {key!r}.")
        if _credential_like_key(key):
            raise ValueError(f"{path} contains credential-like key {key!r}.")
        frozen[key] = _freeze_json_value(item, path=f"{path}.{key}")
    return FrozenJSONMapping(frozen)
