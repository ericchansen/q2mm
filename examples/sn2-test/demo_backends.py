"""Quick test of both backends."""

if __name__ == "__main__":
    # Test TinkerEngine (doesn't need conda)
    from q2mm.backends.mm.tinker import TinkerEngine
    from q2mm.resources import sn2_reference_dir

    ch3f_xyz = sn2_reference_dir() / "ch3f-optimized.xyz"
    engine = TinkerEngine()
    print(f"TinkerEngine: {engine.name}, available={engine.is_available()}")
    energy = engine.energy(ch3f_xyz)
    print(f"  CH3F energy: {energy:.4f} kcal/mol")

    # Test Psi4Engine
    try:
        from q2mm.backends.qm.psi4 import Psi4Engine

        psi4_engine = Psi4Engine(charge=0)  # CH3F is neutral
        print(f"Psi4Engine: {psi4_engine.name}, available={psi4_engine.is_available()}")
        e = psi4_engine.energy(ch3f_xyz)
        print(f"  CH3F energy: {e:.10f} Ha")
    except ImportError:
        print("Psi4 not available in this Python — use conda run -n q2mm")

    print("\nBackend tests complete!")
