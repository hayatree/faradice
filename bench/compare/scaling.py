#!/usr/bin/env python3
"""Scaling comparison: a SKY130-like met2 bus of N bits between met1 and met3
cross lines (as in cases/v8_sky130_bus), for growing N.

For every N it times
  * faradice, all nets (full matrix) and one target net (the middle bit),
  * FasterCap on the same problem (open space + substrate plane),
  * optionally Palace on the grounded-box problem (``--palace-max-bits``),
and reports the accuracy against the reference next to the time.

Usage: scaling.py [--bits 4,8,16,32,64] [--out DIR] [--palace-max-bits 16]
"""

import argparse
import json
import os
import time
from pathlib import Path

import compare as C

M1 = (1.3761, 1.3761 + 0.36)
M2 = (2.0061, 2.0061 + 0.36)
M3 = (2.7861, 2.7861 + 0.845)


def bus(n, path):
    """Writes an n-bit bus structure; returns the name of the middle bit."""
    width = 0.28 * n
    lines = [
        f"# {n}-bit met2 bus (0.14 µm width/space, 8 µm long) between three met1",
        "# and two met3 cross lines; SKY130 heights, uniform eps_r = 4.0.",
        "schema_version = 1",
        'units = "um"',
        "",
        "[dielectric]",
        "eps_r = 4.0",
        "",
    ]

    def box(net, b):
        lines.extend(["[[conductor]]", f'net = "{net}"', "boxes = [[" + ", ".join(f"{v:.4f}" for v in b) + "]]", ""])

    for i in range(n):
        y = 0.28 * i
        box(f"b{i:03}", (0.0, y, M2[0], 8.0, y + 0.14, M2[1]))
    for i, x in enumerate((1.5, 3.93, 6.36)):
        box(f"m1_{i}", (x, -1.0, M1[0], x + 0.14, width + 1.0, M1[1]))
    for i, x in enumerate((2.5, 5.2)):
        box(f"m3_{i}", (x, -1.0, M3[0], x + 0.3, width + 1.0, M3[1]))
    path.write_text("\n".join(lines))
    return f"b{n // 2:03}"


def errors(fr, ref, ext=None):
    rows = C.compare(fr, ref, ext)
    tot = max(abs(r[5]) for r in rows if r[1] == "total")
    cpl = max((abs(r[5]) for r in rows if r[1] != "total" and r[8]), default=0.0)
    return tot, cpl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bits", default="4,8,16,32,64")
    ap.add_argument("--out", type=Path, default=Path("scaling_out"))
    ap.add_argument("--preset", default="reference")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--fastercap-error", type=float, default=1e-3)
    ap.add_argument("--fastercap-timeout", type=float, default=1800.0, help="seconds")
    ap.add_argument("--palace-max-bits", type=int, default=16)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--palace-max-dofs", type=int, default=1_500_000)
    ap.add_argument("--palace-ranks", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    out = a.out.resolve()
    (out / "cases").mkdir(parents=True, exist_ok=True)

    results = []
    for n in map(int, a.bits.split(",")):
        case = out / "cases" / f"bus{n:03}.faradice.toml"
        mid = bus(n, case)
        eps_r, nets = C.load_structure(case)
        w = out / f"bus{n:03}"
        r = {"bits": n, "nets": len(nets)}
        fr_all = C.run_faradice(case, "plane", w / "plane" / "faradice_all", a.preset, a.margin, a.threads)
        fr_one = C.run_faradice(case, "plane", w / "plane" / "faradice_one", a.preset, a.margin, a.threads, [mid])
        r["faradice_all_s"] = fr_all["wall_s"]
        r["faradice_one_s"] = fr_one["wall_s"]
        t = time.perf_counter()
        try:
            fc = C.run_fastercap(eps_r, nets, w / "plane" / "fastercap", a.fastercap_error, a.fastercap_timeout)
            r["fastercap_s"] = time.perf_counter() - t
            r["fastercap_err_total"], r["fastercap_err_coupling"] = errors(fr_all, fc["matrix"])
        except C.SolverFailed as e:
            r["fastercap_note"] = f"FasterCap: {e}"
        if n <= a.palace_max_bits:
            fb = C.run_faradice(case, "box", w / "box" / "faradice_all", a.preset, a.margin, a.threads)
            r["faradice_box_s"] = fb["wall_s"]
            try:
                pa = C.run_palace(eps_r, nets, w / "box" / "palace", a.margin, a.palace_ranks, 10, a.palace_max_dofs, 2)
                r["palace_s"] = pa["wall_s"] + pa["mesh_wall_s"]
                r["palace_note"] = pa["note"]
                r["palace_err_total"], r["palace_err_coupling"] = errors(fb, pa["matrix"], pa["extrapolated"])
            except C.SolverFailed as e:
                r["palace_note"] = f"Palace: {e}"
        results.append(r)
        print(json.dumps(r), flush=True)
        (out / "scaling.json").write_text(json.dumps(results, indent=1))

    def f(v, unit="s"):
        return "—" if v is None else (f"{v:.1f} {unit}" if unit == "s" else f"{v * 100:.2f} %")

    md = [
        "| bits | nets | faradice all nets | faradice 1 net | FasterCap | FasterCap / faradice | max error vs FasterCap (total / coupling) "
        "| faradice box | Palace | Palace / faradice | max error vs Palace (total / coupling) |",
        "|---" * 11 + "|",
    ]
    for r in results:
        g = r.get
        fr = "—" if g("fastercap_s") is None else f"{g('fastercap_s') / r['faradice_all_s']:.1f}×"
        pr = "—" if g("palace_s") is None else f"{g('palace_s') / g('faradice_box_s'):.1f}×"
        md.append(
            f"| {r['bits']} | {r['nets']} | {f(r['faradice_all_s'])} | {f(r['faradice_one_s'])} | {f(g('fastercap_s'))} "
            f"| {fr} "
            f"| {f(g('fastercap_err_total'), '%')} / {f(g('fastercap_err_coupling'), '%')} "
            f"| {f(g('faradice_box_s'))} | {f(g('palace_s'))} | {pr} "
            f"| {f(g('palace_err_total'), '%')} / {f(g('palace_err_coupling'), '%')} |"
        )
    notes = [f"- {r['bits']} bits: {r[k]}" for r in results for k in ("fastercap_note", "palace_note") if r.get(k)]
    if notes:
        md += ["", *notes]
    (out / "scaling.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
