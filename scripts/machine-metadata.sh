#!/usr/bin/env bash
# Machine and toolchain metadata to keep with benchmark results: the review
# handoff's Section 6.2 command set plus compiler, platform, memory, and
# power-mode details. A selector this OS or CPU doesn't have prints
# "unavailable" instead of failing, so the script works on any host.
set -uo pipefail

# kv <label> <command...>: first line of the command's output, or unavailable.
kv() {
  local label=$1
  shift
  local out
  out=$("$@" 2>/dev/null | head -n 1)
  if [ -n "$out" ]; then
    printf '%-36s %s\n' "$label" "$out"
  else
    printf '%-36s %s\n' "$label" "unavailable"
  fi
}

echo "## Machine metadata"
echo
echo "Captured: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo

echo "### Operating system"
kv "sw_vers ProductName" sw_vers -productName
kv "sw_vers ProductVersion" sw_vers -productVersion
kv "sw_vers BuildVersion" sw_vers -buildVersion
kv "uname -srm" uname -srm
kv "uname -m" uname -m
echo

echo "### CPU"
kv "machdep.cpu.brand_string" sysctl -n machdep.cpu.brand_string
kv "hw.ncpu" sysctl -n hw.ncpu
kv "hw.memsize" sysctl -n hw.memsize
kv "hw.optional.arm.FEAT_DotProd" sysctl -n hw.optional.arm.FEAT_DotProd
kv "hw.optional.arm.FEAT_CSSC" sysctl -n hw.optional.arm.FEAT_CSSC
kv "hw.optional.arm.FEAT_SME" sysctl -n hw.optional.arm.FEAT_SME
kv "hw.optional.arm.FEAT_SME2" sysctl -n hw.optional.arm.FEAT_SME2
kv "hw.optional.arm.FEAT_SME2p1" sysctl -n hw.optional.arm.FEAT_SME2p1
# 64 here is 64 bytes: a 512-bit maximum streaming vector length.
kv "hw.optional.arm.sme_max_svl_b" sysctl -n hw.optional.arm.sme_max_svl_b
echo

echo "### Toolchain"
kv "rustc" rustc -Vv
rustc -Vv 2>/dev/null | sed -n '2,$s/^/  /p'
kv "cargo" cargo -V
kv "clang" clang --version
echo

echo "### Power"
# pmset has no stable machine-readable power-mode query; read the line out
# of `pmset -g` (0 = automatic, 1 = low power, 2 = high power on Apple
# silicon) and tolerate its absence.
powermode=$(pmset -g 2>/dev/null | awk '$1 == "powermode" {print $2}')
case "${powermode:-}" in
  0) kv "pmset powermode" echo "0 (automatic)" ;;
  1) kv "pmset powermode" echo "1 (low power)" ;;
  2) kv "pmset powermode" echo "2 (high power)" ;;
  *) kv "pmset powermode" echo "${powermode:-unavailable}" ;;
esac
kv "AC/battery" pmset -g batt
