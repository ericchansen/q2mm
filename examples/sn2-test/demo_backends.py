"""Quick demonstration of the prepared-session backend API for MM and QM."""

if __name__ == "__main__":
    from q2mm.backends.contracts import EnergyRequest, PreparationRequest, QMEnergyRequest
    from q2mm.backends.registry import available_backends, load_backend
    from q2mm.io.xyz import load_xyz
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.models.parameters import ParameterLayout
    from q2mm.models.seminario import qfuerza_fresh
    from q2mm.resources import sn2_reference_dir

    ch3f_xyz = sn2_reference_dir() / "ch3f-optimized.xyz"
    molecule = load_xyz(ch3f_xyz, bond_tolerance=1.5)

    # MM: Tinker (MM3). Requires the Tinker executables on PATH.
    if "tinker" in available_backends():
        backend = load_backend("tinker")
        ff = qfuerza_fresh(molecule, functional_form=FunctionalForm.MM3)
        prepared = backend.prepare(PreparationRequest(case_id="ch3f", molecule=molecule, force_field=ff))
        params = ParameterLayout.from_force_field(ff).vector(ff)
        result = prepared.energy(EnergyRequest(parameters=params))
        print(f"Tinker: {backend.info.name}")
        print(f"  CH3F energy: {result.energy:.4f} {result.unit.value}")
    else:
        print("Tinker not available on PATH - skipping MM demo.")

    # QM: Psi4.
    if "psi4" in available_backends():
        qm = load_backend("psi4", charge=0)  # CH3F is neutral
        prepared_qm = qm.prepare(PreparationRequest(case_id="ch3f", molecule=molecule))
        e = prepared_qm.energy(QMEnergyRequest())
        print(f"Psi4: {qm.info.name}")
        print(f"  CH3F energy: {e.energy:.10f} {e.unit.value}")
    else:
        print("Psi4 not available in this Python - use conda run -n q2mm")

    print("\nBackend tests complete!")
