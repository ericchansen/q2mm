#!/usr/bin/env python3
"""Basic utility methods for use in Q2MM."""


# region Atom Type Handling


def convert_atom_type(atom_type: str) -> str:
    """Convert a raw atom type string to a normalized Q2MM atom type.

    Strips non-alphanumeric characters and converts to uppercase.

    Args:
        atom_type (str): Raw atom type string from a force field file.

    Returns:
        (str): Normalized uppercase alphanumeric atom type.

    """
    q2mm_atom_type = "".join(filter(str.isalnum, atom_type))
    q2mm_atom_type = q2mm_atom_type.upper()
    # TODO: MF Add a check to verify it is included in atom.typ here,
    # exception should be caught, propagated, and handled here to avoid
    # silent failure within MacroModel upon FF export (or other silent or loud failures).
    return q2mm_atom_type


# endregion Atom Type Handling
