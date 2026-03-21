"""Helpers for extracting and matching clean-model identifiers."""

from __future__ import annotations

from collections.abc import Sequence

from q2mm.elements import TWO_LETTER_ELEMENTS


def _extract_element(atom_type: str) -> str:
    """Extract an element symbol from an atom label or atom type."""
    s = atom_type.strip()
    if not s:
        return s

    letters = "".join(ch for ch in s if ch.isalpha())
    if not letters:
        return s

    if len(letters) >= 2:
        candidate = letters[:2].title()
        if candidate in TWO_LETTER_ELEMENTS:
            return candidate

    return letters[0].upper()


def canonicalize_bond_env_id(atom_types: Sequence[str]) -> str:
    """Canonicalize bond atom-type labels so bond direction does not matter."""
    cleaned = [item.strip() for item in atom_types if item and item.strip() and item.strip() != "-"]
    if len(cleaned) < 2:
        return "-".join(cleaned)
    return "-".join(sorted(cleaned[:2]))


def canonicalize_angle_env_id(atom_types: Sequence[str]) -> str:
    """Canonicalize angle atom-type labels with fixed center and sorted outers."""
    cleaned = [item.strip() for item in atom_types if item and item.strip() and item.strip() != "-"]
    if len(cleaned) < 3:
        return "-".join(cleaned)
    outer = sorted([cleaned[0], cleaned[2]])
    return "-".join([outer[0], cleaned[1], outer[1]])
