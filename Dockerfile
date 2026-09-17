# syntax=docker/dockerfile:1.7
#
# Reproducible Blue Team RL + project tooling image.
#
# WHAT THIS IMAGE CONTAINS
#   - The triad_rl trainer/planner package and all 64 test modules
#   - The full published evidence tree (RL/BlueTeam/{Results,Checkpoints,Examples})
#     so `pytest` verifies archived artifacts offline, with no network
#   - The stdlib-only scripts in Tools/ (geometry generation, release verification,
#     benchmark summarisation, red-team wire client)
#
# WHAT IT DOES NOT CONTAIN
#   Unreal Engine, Content/ or SourceAssets/. Unreal cannot be redistributed and
#   its Linux container requires an Epic entitlement; see docker/Dockerfile.unreal.
#   Excluding those trees is deliberate: this image builds from a plain `git clone`
#   with no `git lfs pull` and no Epic account.
#
# The repository layout is preserved at /workspace because the test suite resolves
# its repo root as Path(__file__).resolve().parents[4] and loads Tools/*.py from it.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Determinism knobs. The single-thread BLAS/OMP settings are not arbitrary: the
# published runs were executed with "OpenBLAS/OMP threads set to one"
# (RL/BlueTeam/TEMPORAL_RESULTS.md). Leaving these unset lets the BLAS thread
# count vary with host core count, which perturbs float reduction order.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /workspace

# Created here so the runtime stage can COPY --chown without an extra 184 MB
# layer just to fix ownership.
RUN useradd --create-home --uid 1000 istana


# ---------------------------------------------------------------------------
# Dependency layer. Cached independently of project source so edits to the
# codebase do not trigger a dependency reinstall.
# ---------------------------------------------------------------------------
FROM base AS deps

ARG CONSTRAINTS=docker/constraints.txt
COPY docker/constraints.txt docker/constraints-current.txt docker/

# All four dependencies ship manylinux wheels, so no compiler is required.
RUN python -m pip install --upgrade pip==24.3.1 && \
    python -m pip install -c "${CONSTRAINTS}" numpy gymnasium pettingzoo pytest


# ---------------------------------------------------------------------------
# Runtime image.
# ---------------------------------------------------------------------------
FROM deps AS runtime

ARG CONSTRAINTS=docker/constraints.txt

# Project source. .dockerignore keeps Content/ and SourceAssets/ out.
# --chown so the unprivileged user can write pytest caches and tool output in
# place, without a second full-size layer.
COPY --chown=istana:istana . /workspace

# Install triad_rl itself without re-resolving the pinned dependency set.
RUN python -m pip install --no-deps -c "${CONSTRAINTS}" ./RL/BlueTeam/Python && \
    python -c "import triad_rl, numpy; print('triad_rl ok, numpy', numpy.__version__)"

# The exec bit does not survive a checkout on Windows hosts, so set it here.
RUN mkdir -p /workspace/runs && \
    chown istana:istana /workspace/runs && \
    chmod +x /workspace/docker/entrypoint.sh /workspace/docker/unreal-build.sh

USER istana

LABEL org.opencontainers.image.title="Istana Open — Blue Team RL and tooling" \
      org.opencontainers.image.description="Reproducible CPU-only sensor-placement trainer, planner, evidence verifier and project tools." \
      org.opencontainers.image.source="https://github.com/Jenjenson/IstanaOpen" \
      org.opencontainers.image.licenses="MIT"

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["help"]
