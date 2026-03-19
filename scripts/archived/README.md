# Archived Scripts

These scripts are preserved for reference but are **not actively maintained**.
They were part of the legacy Q2MM workflow and depend on external tools
(Schrodinger Suite, MacroModel, OpenBabel, PubChem API) rather than the
core Q2MM package.

## tools/

Standalone utility scripts for force field formatting, MAE file cleanup,
energy analysis, and cluster job submission. Most do not import from q2mm.

## smiles_to_catvs/

Workflow for converting SMILES strings to conformer databases via PubChem
and MacroModel. Requires `pubchempy`, `requests`, `openbabel`, and
Schrodinger MacroModel.

## smiles_to_catvs_example/

Working example demonstrating the smiles_to_catvs pipeline with sample
ligand structures and templates.
