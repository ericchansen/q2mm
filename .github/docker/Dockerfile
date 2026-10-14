# The environment inventory supplies ENV_FILE and the input-addressed tag:
# python scripts/ci_image.py openmm --field reference
# docker build -f .github/docker/Dockerfile --build-arg ENV_FILE=openmm.yml -t <reference> .github
FROM mambaorg/micromamba:1.5-jammy

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /opt/q2mm-image && chown "$MAMBA_USER" /opt/q2mm-image
USER $MAMBA_USER

ARG ENV_FILE=openmm.yml
COPY --chown=$MAMBA_USER envs/${ENV_FILE} /opt/q2mm-image/environment.yml
RUN micromamba install -y -n base -f /opt/q2mm-image/environment.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}" \
    JAX_PLATFORMS=cpu \
    OPENMM_DEFAULT_PLATFORM=CPU \
    Q2MM_USE_GPU=0

# The conda Tinker package omits parameter data. Only its matching official
# archive supplies mm3.prm; neither the archive nor that data belongs in git.
RUN if [ "$ENV_FILE" = "tinker.yml" ] || [ "$ENV_FILE" = "openmm-tinker.yml" ]; then \
      curl --fail --location --retry 3 \
        https://dasher.wustl.edu/tinker/downloads/tinker-8.11.3.tar.gz \
        --output /opt/q2mm-image/tinker-8.11.3.tar.gz && \
      echo "8dfbc9fb8f26742d91139187657e2c905744b0243538f81b75bc04cdc2606ff7  /opt/q2mm-image/tinker-8.11.3.tar.gz" | sha256sum --check - && \
      mkdir -p "$MAMBA_ROOT_PREFIX/params" && \
      tar -xOzf /opt/q2mm-image/tinker-8.11.3.tar.gz tinker/params/mm3.prm > "$MAMBA_ROOT_PREFIX/params/mm3.prm" && \
      test -s "$MAMBA_ROOT_PREFIX/params/mm3.prm" && \
      rm /opt/q2mm-image/tinker-8.11.3.tar.gz; \
    fi && \
    python -m pip check

ENV TINKER_DIR="${MAMBA_ROOT_PREFIX}/bin" \
    TINKER_PRM="${MAMBA_ROOT_PREFIX}/params/mm3.prm"
WORKDIR /work
