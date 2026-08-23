#!/usr/bin/env bash
set -euo pipefail

# Build the official FasterCap console solver without installing anything into
# the system.  Prerequisites on Debian/Ubuntu:
#   cmake g++ git libwxgtk3.2-dev libeigen3-dev

destination=${1:-build/fastercap}
source_root=${destination}/src
build_root=${destination}/build

mkdir -p "${source_root}" "${build_root}"

clone_at_commit() {
    repo_url=$1
    checkout=$2
    commit=$3
    if [[ ! -d "${checkout}/.git" ]]; then
        git clone "${repo_url}" "${checkout}"
    fi
    git -C "${checkout}" fetch --depth 1 origin "${commit}"
    git -C "${checkout}" checkout --detach "${commit}"
}

clone_at_commit https://github.com/ediloren/FasterCap.git \
    "${source_root}/FasterCap" b42179a8fdd25ab42fe45527282b4a738d7e7f87
clone_at_commit https://github.com/ediloren/Geometry.git \
    "${source_root}/Geometry" de03ffebfd5013b96102bd60f71c8fe8b73870e2
clone_at_commit https://github.com/ediloren/LinAlgebra.git \
    "${source_root}/LinAlgebra" 627132d70bfd7eadd727f930286938a5a01d9914

if ! command -v wx-config >/dev/null 2>&1; then
    echo "wx-config is missing; install the wxWidgets development package" >&2
    exit 1
fi

# Upstream requests wx 3.0 explicitly.  Current distributions ship 3.2; use
# the installed ABI while retaining upstream's static/debug choices.
wx_version=$(wx-config --version)
sed -i "s/--version=3\.0/--version=${wx_version}/" \
    "${source_root}/FasterCap/CMakeLists.txt"

cmake -S "${source_root}/FasterCap" -B "${build_root}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFASTFIELDSOLVERS_HEADLESS=ON
cmake --build "${build_root}" --parallel

solver=$(realpath "${build_root}/FasterCap")
"${solver}" -bv
echo "FasterCap executable: ${solver}"
