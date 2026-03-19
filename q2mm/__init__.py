"""Q2MM: Quantum-guided molecular mechanics force field optimization.

Subpackages:
    core        — Backend-agnostic optimization engine
    backends    — QM and MM engine integrations
    io          — File format parsers
    forcefields — Force field representations
    cli         — Command-line interface
"""

try:
    from importlib.metadata import version

    __version__ = version("q2mm")
except Exception:
    __version__ = "0.0.0.dev0"  # fallback for editable/uninstalled
