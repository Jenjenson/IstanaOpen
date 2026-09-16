#!/usr/bin/env bash
# Build and test the Istana Open native modules inside Epic's Unreal container.
#
# Mirrors what Tools/build.ps1 does on Windows, using the Linux BatchFiles.
# Unverified: see the header of docker/Dockerfile.unreal.
set -euo pipefail

UE_ROOT="${UE_ROOT:-/home/ue4/UnrealEngine}"
PROJECT="${PROJECT:-/workspace/IstanaOpen.uproject}"
BUILD="${UE_ROOT}/Engine/Build/BatchFiles/Linux/Build.sh"
EDITOR_CMD="${UE_ROOT}/Engine/Binaries/Linux/UnrealEditor-Cmd"
REPORT="/workspace/Saved/Automation/Docker"

if [ ! -f "${PROJECT}" ]; then
  echo "error: ${PROJECT} not found. Bind-mount the repository at /workspace." >&2
  exit 1
fi

if [ ! -x "${BUILD}" ]; then
  echo "error: Unreal build script not found at ${BUILD}." >&2
  echo "       Set UE_ROOT to your engine root for this image tag." >&2
  exit 1
fi

# The .uasset/.umap files are Git LFS objects. A pointer file starts with this
# marker; building against pointers fails in confusing ways, so check early.
map="/workspace/Content/Maps/Istana.umap"
if [ -f "${map}" ] && head -c 43 "${map}" | grep -q "^version https://git-lfs.github.com/spec/v1"; then
  echo "error: Content/ holds unresolved Git LFS pointers." >&2
  echo "       Run 'git lfs install && git lfs pull' on the host, then retry." >&2
  exit 1
fi

build_target() {
  echo "==> Building $1 (Linux Development)"
  "${BUILD}" "$1" Linux Development -project="${PROJECT}" -waitmutex
}

run_tests() {
  echo "==> Automation: Istana.Simulation (-nullrhi, headless)"
  mkdir -p "${REPORT}"
  # -nullrhi keeps this CPU-only: these are simulation/contract tests, never a
  # rendering or FPS benchmark.
  "${EDITOR_CMD}" "${PROJECT}" \
    -ExecCmds="Automation RunTests Istana.Simulation;quit" \
    -unattended -nopause -nosplash -nullrhi -nop4 \
    -ReportExportPath="${REPORT}" \
    -testexit="Automation Test Queue Empty"
  echo "Automation report: ${REPORT#/workspace/}"
}

case "${1:-all}" in
  editor) build_target IstanaOpenEditor ;;
  game)   build_target IstanaOpen ;;
  test)   run_tests ;;
  all)
    build_target IstanaOpenEditor
    build_target IstanaOpen
    run_tests
    ;;
  shell)  exec bash ;;
  *)
    echo "usage: unreal-build {all|editor|game|test|shell}" >&2
    exit 2
    ;;
esac
