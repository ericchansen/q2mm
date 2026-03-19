"""Q2MM internal data models.

Provides clean, format-agnostic representations for molecules and force fields.
These decouple Q2MM's core algorithms from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod, etc.).
"""

from q2mm.models.ff_io import load_mm3_fld, load_tinker_prm, save_mm3_fld, save_tinker_prm
