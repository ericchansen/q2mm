# Base Dockerfile for Q2MM CI images.
#
# Build backend-specific images:
#   docker build --build-arg ENV_FILE=openmm.yml -t ghcr.io/ericchansen/q2mm/ci-openmm .
#   docker build --build-arg ENV_FILE=tinker.yml -t ghcr.io/ericchansen/q2mm/ci-tinker .
#   docker build --build-arg ENV_FILE=jax.yml    -t ghcr.io/ericchansen/q2mm/ci-jax    .
#   docker build --build-arg ENV_FILE=psi4.yml   -t ghcr.io/ericchansen/q2mm/ci-psi4   .
#   docker build --build-arg ENV_FILE=full.yml   -t ghcr.io/ericchansen/q2mm/ci-full   .
#
# Run locally:
#   docker run --rm -v $PWD:/workspace -w /workspace ghcr.io/ericchansen/q2mm/ci-openmm \
#       pytest -m openmm --run-slow

FROM mambaorg/micromamba:1.5-jammy

ARG ENV_FILE=openmm.yml

COPY .github/envs/${ENV_FILE} /tmp/env.yml

RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

# Ensure conda env is activated
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /workspace
