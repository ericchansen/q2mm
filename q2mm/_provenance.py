"""Strict immutable JSON data used by provenance records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, NoReturn

_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
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


def _contains_absolute_path(value: object) -> bool:
    """Check nested values for platform-independent absolute-path spellings."""
    if isinstance(value, str):
        return value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


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
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _CREDENTIAL_DATA):
            raise ValueError(f"{path} contains credential-like data.")
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


def thaw_json_value(value: Any) -> Any:
    """Recursively convert frozen JSON mappings and tuples into mutable dicts and lists."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def _credential_like_key(key: str) -> bool:
    """Return whether a key can identify secret material in common styles."""
    lowered = key.casefold()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    return "auth" in re.split(r"[^a-z0-9]+", lowered) or any(term in normalized for term in _CREDENTIAL_KEY_TERMS)


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
