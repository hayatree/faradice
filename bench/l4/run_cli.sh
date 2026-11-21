#!/usr/bin/env bash
# L4 efficiency runs on real stacks (L5, L6): fixed samples per net, one run
# per variant; prints `CLI|variant|case|net|total|sigma|walks|steps|cpu_s`.
# Usage: bench/l4/run_cli.sh OUTDIR [WALKS]
set -euo pipefail
out=${1:?outdir}; walks=${2:-4000000}
bin=${FARADICE_BIN:-target/release/faradice}
root=$(cd "$(dirname "$0")/../.." && pwd)
declare -A V=(
  [default]=""
  [K8]="interface_offsets = 8"
  [K32_M64]=$'interface_offsets = 32\nlayered_resolution = 64'
  [M64]="layered_resolution = 64"
  [eta_quarter]="band_fraction = 0.25"
  [alpha_0.5]="landing_alpha = 0.5"
  [alpha_0.85]="landing_alpha = 0.85"
)
cases=(
  "L5|$root/bench/compare/cases/l5_wires_in_stack.faradice.toml|\"a\", \"cross\""
  "L6|$root/bench/compare/cases/l6_sky130_stack_bus.faradice.toml|\"m1_1\", \"m3_0\""
)
mkdir -p "$out"
for v in default K8 K32_M64 M64 eta_quarter alpha_0.5 alpha_0.85; do
  for c in "${cases[@]}"; do
    IFS='|' read -r name file nets <<<"$c"
    cfg="$out/$v.$name.toml"
    cat >"$cfg" <<CFG
schema_version = 1
[domain]
kind = "box"
margin = 2.0
[ground]
substrate = "plane"
substrate_z = 0.0
[extract]
nets = [$nets]
[accuracy]
preset = "reference"
min_walks = $walks
max_walks = $walks
[coupling]
symmetrize = "none"
[walk]
${V[$v]}
[run]
seed = 1
CFG
    res="$out/$v.$name"
    /usr/bin/time -o "$res.time" -f "%U" "$bin" solve "$file" --config "$cfg" --out "$res" >/dev/null 2>&1
    cpu=$(tail -1 "$res.time")
    python3 - "$res" "$v" "$name" "$cpu" <<'PY'
import csv, json, sys
res, v, name, cpu = sys.argv[1:]
d = json.load(open(f"{res}/result.json"))
rows = {(r["net"], r["kind"]): r for r in csv.DictReader(open(f"{res}/report.csv")) if r["kind"] == "total"}
for t in d["targets"]:
    r = rows[(t["net"], "total")]
    print(f"CLI|{v}|{name}|{t['net']}|{r['value_ff']}|{r['sigma_ff']}|{t['walks']}|{t['total_steps']/t['walks']:.2f}|{cpu}")
PY
  done
done
