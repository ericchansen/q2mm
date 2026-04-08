"""File format I/O for Q2MM.

Consolidates all file format readers and writers into one package.
Force field formats (MM3, Tinker, AMBER, OpenMM), quantum chemistry
output formats (Gaussian, Jaguar, FCHK), molecular mechanics output
(MacroModel), structure files (Mol2), and reference data (YAML).
"""

from q2mm.io._helpers import (  # noqa: F401  — private re-exports for sibling sub-modules
    Param,
    ParamError,
    _FORMAT_COMPATIBLE_FORMS,
    _build_angle_maps,
    _build_bond_maps,
    _build_param_maps,
    _build_vdw_maps,
    _clean_atom_types,
    _match_angle_for_export,
    _match_bond_for_export,
    _match_for_export,
    _split_env_id,
    _update_torsion_param,
    _validate_form_for_format,
)

from q2mm.io.mm3 import load_mm3_fld, save_mm3_fld  # noqa: F401, E402
from q2mm.io.tinker import load_tinker_prm, save_tinker_prm  # noqa: F401, E402
from q2mm.io.amber import load_amber_frcmod, save_amber_frcmod  # noqa: F401, E402
from q2mm.io.openmm import save_openmm_xml  # noqa: F401, E402
from q2mm.io.gaussian import GaussLog  # noqa: F401, E402
from q2mm.io.fchk import parse_fchk  # noqa: F401, E402
from q2mm.io.jaguar import JaguarIn, JaguarOut  # noqa: F401, E402
from q2mm.io.macromodel import MacroModel, MacroModelLog  # noqa: F401, E402
from q2mm.io.mol2 import Mol2  # noqa: F401, E402
from q2mm.io.reference import load_reference_yaml, save_reference_yaml  # noqa: F401, E402
from q2mm.io.cmap import parse_cmap_section, load_cmap_from_prm  # noqa: F401, E402
