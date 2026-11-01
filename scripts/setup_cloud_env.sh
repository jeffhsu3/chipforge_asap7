#!/usr/bin/env bash
# Install the external tools chipforge_asap7's DRC, LVS, extraction and SPICE
# flows use, for a fresh Ubuntu 24.04 cloud session (paste into the cloud
# environment's setup script, or run as root from the repository root).
#
# Everything is pinned and idempotent: rerunning skips finished steps.
#
#   ngspice 42      apt; loads OSDI 0.3 libraries
#   OpenVAF 23.5.0  built from source against LLVM 15; emits OSDI 0.3, which
#                   ngspice 42 needs (OpenVAF-Reloaded emits 0.4, which only
#                   ngspice >= 44 reads)
#   BSIM-CMG 107    Verilog-A source, sparse checkout of Xyce
#   ASAP7 models    sparse checkout of the PDK at the commit asap7 pins
#   ASAP7 release   standard cells and SRAM add-on, for LVS and references
#   FasterCap       scripts/build_fastercap.sh into build/fastercap
#   gdscheck        jeffhsu3/gdscheck asap7 branch, the DRC engine
#   Icarus Verilog  apt; runs the BROM Verilog-model tests
#
# The model card and BSIM-CMG source sit beside the repository, where
# chipforge_asap7.devices.spice finds them; gdscheck sits there too, where
# the minimum-area proofs read its pinned rules.
set -euo pipefail

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
home=$(dirname "${repo}")
src=${home}/src

apt-get update -qq
apt-get install -y --no-install-recommends \
    ngspice iverilog p7zip-full \
    libwxgtk3.2-dev libeigen3-dev cmake g++ \
    llvm-15 llvm-15-dev libclang-15-dev clang-15 clang-tools-15 lld-15

# Clone `url` at `commit` into `dir`, checking out only `paths` (all if none).
sparse_clone() {
    local url=$1 dir=$2 commit=$3
    shift 3
    if [[ ! -d ${dir}/.git ]]; then
        git clone -q --filter=blob:none --no-checkout "${url}" "${dir}"
    fi
    if (($#)); then
        git -C "${dir}" sparse-checkout set "$@"
    fi
    git -C "${dir}" checkout -q "${commit}"
}

# ASAP7 BSIM-CMG model cards (asap7 pins the PDK submodule at this commit).
sparse_clone https://github.com/The-OpenROAD-Project/asap7_pdk_r1p7.git \
    "${home}/asap7/asap7_pdk_r1p7" 58d72c9d291e186a77468586ab0c43d8a21eda6a models

# The released standard cells and SRAM add-on the LVS and reference tests
# compare against, at the commits asap7 pins. The standard cells skip their
# multi-gigabyte CCS libraries; nothing here reads them.
sparse_clone https://github.com/The-OpenROAD-Project/asap7sc7p5t_28.git \
    "${home}/asap7/asap7sc7p5t_28" f970bd3c3292b79ae4d022a3ec80533534614066 \
    CDL GDS LIB/NLDM
sparse_clone https://github.com/The-OpenROAD-Project/asap7_sram_0p0.git \
    "${home}/asap7/asap7_sram_0p0" 522eeccbccefcd66e61893fa1059df24d95e9f86

# BSIM-CMG 107 Verilog-A.
sparse_clone https://github.com/Xyce/Xyce.git "${home}/Xyce" \
    e5b7b9bb5b221630c66654c059fc2d33524f5b21 utils/ADMS/examples/bsimcmg_107.0.0

# OpenVAF 23.5.0. Its rust-toolchain file pins Rust 1.64, which rustup fetches.
openvaf=${src}/OpenVAF-23.5.0
sparse_clone https://github.com/pascalkuthe/OpenVAF.git "${openvaf}" \
    d4079e776f4b54b23e158b7857c4e238e5cacd05
if [[ ! -x ${openvaf}/target/release/openvaf ]]; then
    (cd "${openvaf}" && PATH=/usr/lib/llvm-15/bin:${PATH} \
        LLVM_LINK_SHARED=1 LLVM_CONFIG=llvm-config-15 \
        cargo build --release --bin openvaf)
fi
ln -sf "${openvaf}/target/release/openvaf" /usr/local/bin/openvaf

# gdscheck, the DRC engine.
sparse_clone https://github.com/jeffhsu3/gdscheck.git "${home}/gdscheck" \
    0b383c3b07edd6247586aca812e392928dc3b872
if [[ ! -x ${home}/gdscheck/target/release/gdscheck ]]; then
    (cd "${home}/gdscheck" && cargo build --release)
fi
ln -sf "${home}/gdscheck/target/release/gdscheck" /usr/local/bin/gdscheck

# FasterCap, where verification.pex finds it.
if [[ ! -x ${repo}/build/fastercap/build/FasterCap ]]; then
    (cd "${repo}" && scripts/build_fastercap.sh build/fastercap)
fi

# Python environment, including the pex (KPEX) and gds extras.
(cd "${repo}" && uv sync --all-extras --frozen)

echo "simulation tools ready:"
ngspice -v 2>&1 | grep -m1 ngspice
openvaf --version
command -v gdscheck iverilog
echo "FasterCap: ${repo}/build/fastercap/build/FasterCap"
