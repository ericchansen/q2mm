#!/usr/bin/env bash
# Local fast checks, or the exact bounded native/pair responsibility used by CI.
# --job skips fast checks and uses one pinned prebuilt image (pulling if absent).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
JOBS=(openmm-cpu tinker jax-md-cpu psi4-reference parity-openmm-jax parity-openmm-tinker parity-jax-jaxmd parity-psi4-qcengine)
job=""
run_backends=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --job)
            [ "$#" -ge 2 ] || { echo "--job requires a responsibility name" >&2; exit 2; }
            job="$2"; shift 2 ;;
        --all) run_backends=true; shift ;;
        -h|--help)
            echo "Usage: bash scripts/ci_local.sh [--all|--job NAME]"
            echo "No args: lint, format, mypy, environment/image parity, core tests."
            echo "--all: fast checks plus all eight bounded image jobs."
            echo "--job NAME: only that responsibility; use the input-pinned image."
            printf '  %s\n' "${JOBS[@]}"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

run_job() {
    local name="$1" image_key contract="" preflight
    local exclude="not (nightly or validation or external_data)"
    local markers="not (cross_backend or nightly or validation or external_data)"
    local -a selectors
    local openmm_check="import openmm; assert openmm.__version__ == '8.6.1'; s=openmm.System(); s.addParticle(1); i=openmm.VerletIntegrator(0.001); c=openmm.Context(s,i,openmm.Platform.getPlatformByName('CPU')); assert c.getPlatform().getName() == 'CPU'"
    local jax_check="import jax; assert jax.devices() and all(d.platform == 'cpu' for d in jax.devices())"
    local tinker_check="from test.backend_fixtures import load_test_backend; load_test_backend('tinker')"
    case "$name" in
        openmm-cpu)
            image_key=openmm; contract=openmm; preflight="$openmm_check"
            selectors=(test/test_openmm_platform.py test/test_openmm_xml.py
                test/integration/test_openmm_backend.py test/integration/test_openmm_export.py) ;;
        tinker)
            image_key=tinker; contract=tinker; preflight="$tinker_check"
            selectors=(test/test_tinker_writers.py test/test_tinker_minimize_controls.py
                test/test_tinker_hessian_parser.py test/test_tinker_boundary_contracts.py
                test/integration/test_tinker_backend.py) ;;
        jax-md-cpu)
            image_key=jax-md; contract=jax_md
            preflight="$jax_check; import jax_md; from test.backend_fixtures import load_test_backend; load_test_backend('jax-md')"
            selectors=(test/test_jaxmd_minimize_controls.py
                test/integration/test_jax_md_backend.py::TestJaxMdBackendLJNumeric
                test/integration/test_jax_md_backend.py::TestJaxMdBackendBreakdown) ;;
        psi4-reference)
            image_key=psi4
            preflight="import psi4; from test.backend_fixtures import load_test_backend; load_test_backend('psi4')"
            selectors=(test/integration/test_psi4_backend.py::TestPsi4BackendAvailability
                test/integration/test_psi4_backend.py::TestPsi4TinyReference) ;;
        parity-openmm-jax)
            image_key=openmm-jax; markers="cross_backend and $exclude"
            preflight="$openmm_check; $jax_check"
            selectors=(test/test_backend_contracts.py::test_openmm_jax_energy_hessian_parity
                test/test_mm3_jax.py::TestMM3ParityJaxVsOpenMM
                test/test_urey_bradley.py::TestUreyBradleyParity
                test/integration/test_jax_openmm_improper_parity.py
                test/benchmarks/test_cross_engine.py::TestEnergyParity::test_openmm_jax_mm3_energy_parity
                test/benchmarks/test_cross_engine.py::TestFrequencyParity::test_openmm_jax_mm3_frequency_parity) ;;
        parity-openmm-tinker)
            image_key=openmm-tinker; markers="cross_backend and $exclude"
            preflight="$openmm_check; $tinker_check"
            selectors=(test/benchmarks/test_cross_engine.py::TestEnergyParity::test_mm3_energy_parity
                test/benchmarks/test_cross_engine.py::TestFrequencyParity::test_mm3_frequency_parity
                'test/integration/test_engine_contract.py::TestTorsionEnergy::test_torsion_energy_matches_openmm[tinker]') ;;
        parity-jax-jaxmd)
            image_key=jax-md; markers="cross_backend and $exclude"
            preflight="$jax_check; import jax_md; from test.backend_fixtures import load_test_backend; load_test_backend('jax-md')"
            selectors=(test/integration/test_jax_md_backend.py::TestJaxMdBackendParity) ;;
        parity-psi4-qcengine)
            image_key=psi4; markers="cross_backend and $exclude"
            preflight="import psi4, qcengine; from test.backend_fixtures import load_test_backend; load_test_backend('psi4')"
            selectors=(test/integration/test_qcengine_psi4_parity.py) ;;
        *) echo "Unknown responsibility: $name" >&2; return 2 ;;
    esac

    command -v docker >/dev/null || { echo "Docker is required for $name" >&2; return 1; }
    local image expected_python
    image="$("$PYTHON" scripts/ci_image.py "$image_key" --field reference)"
    expected_python="$("$PYTHON" scripts/ci_image.py "$image_key" --field python)"
    echo "── $name: $image"
    docker run --pull=missing --rm --init --user "$(id -u):$(id -g)" \
        -v "$REPO_ROOT:/work" -w /work \
        -e JAX_PLATFORMS=cpu -e OPENMM_DEFAULT_PLATFORM=CPU -e Q2MM_USE_GPU=0 \
        -e HOME=/tmp/q2mm-ci-home -e SETUPTOOLS_SCM_PRETEND_VERSION_FOR_Q2MM=0.0.0+ci \
        -e CI_JOB="$name" -e CI_PYTHON="$expected_python" \
        -e CI_PREFLIGHT="$preflight" -e CI_MARKERS="$markers" -e CI_CONTRACT="$contract" \
        "$image" bash -ec '
            mkdir -p "$HOME"
            python -m venv --system-site-packages /tmp/q2mm-ci-venv
            export PATH="/tmp/q2mm-ci-venv/bin:$PATH"
            python -m pip install -e . --no-deps --no-build-isolation
            python -m pip check
            python -c "import sys; assert sys.version_info[:2] == tuple(map(int, sys.argv[1].split(\".\"))), sys.version" "$CI_PYTHON"
            python -c "$CI_PREFLIGHT"
            mkdir -p build/ci
            report="build/ci/$CI_JOB.xml"
            python -m pytest "$@" --run-integration -m "$CI_MARKERS" --junitxml="$report" -q
            python scripts/check_test_execution.py "$report"
            if [ -n "$CI_CONTRACT" ]; then
                report="build/ci/$CI_JOB-contract.xml"
                contracts=(test/integration/test_engine_contract.py)
                if [ "$CI_CONTRACT" = tinker ]; then
                    # The shared He2 vdW fixture is not a supported Tinker term.
                    # Preserve it locally; this job exercises supported terms.
                    contracts=()
                    for group in TestEngineMetadata TestBondEnergy TestAngleEnergy TestHessian \
                        TestFrequencies TestMinimize TestRealMolecule TestTorsionEnergy; do
                        contracts+=("test/integration/test_engine_contract.py::$group")
                    done
                fi
                python -m pytest "${contracts[@]}" --run-integration \
                    -m "$CI_CONTRACT and not (cross_backend or nightly or validation or external_data)" \
                    --junitxml="$report" -q
                python scripts/check_test_execution.py "$report"
            fi
        ' -- "${selectors[@]}"
}

if [ -n "$job" ]; then
    $run_backends && { echo "Use --job or --all, not both" >&2; exit 2; }
    run_job "$job"
    exit 0
fi

"$PYTHON" -m ruff check q2mm/ test/ scripts/
"$PYTHON" -m ruff format --check q2mm test scripts examples
"$PYTHON" -m mypy q2mm/
"$PYTHON" scripts/check_env_dep_parity.py
"$PYTHON" scripts/ci_image.py --check
"$PYTHON" -m pytest test/ -x -q -m "not (openmm or tinker or jax or jax_md or psi4)"
if $run_backends; then
    for name in "${JOBS[@]}"; do
        run_job "$name"
    done
fi
