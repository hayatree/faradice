#!/usr/bin/env python3
"""Compares faradice with external reference solvers run in Docker.

Two problem settings, each solved identically by faradice and one reference:

* ``plane``: open space above an infinite grounded substrate plane at z = 0.
  Reference: FasterCap (BEM, free space). The plane is modelled exactly by
  odd images: every conductor is mirrored below z = 0 and, with images held at
  the opposite potential, C_plane = C[real, real] - C[real, image].
* ``box``: a grounded box (conductor bounding box plus a margin, bottom face
  on the substrate plane), faradice's default domain. Reference: Palace (FEM)
  with adaptive mesh refinement on the same box.

The reference tools are only run as separate processes inside the
``faradice-bench/*`` images (see ``docker/``); nothing is linked into faradice.
Only the Python standard library is needed on the host.

Usage:
  compare.py CASE.faradice.toml [--solvers fastercap,palace] [--out DIR]
             [--preset reference] [--margin 2.0] [--threads N]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EPS0 = 8.8541878128e-12  # F/m


# ---------------------------------------------------------------- structures


def load_structure(path):
    """Returns (eps_r, [(net, [boxes])]) in µm, preserving file order."""
    s = tomllib.loads(Path(path).read_text())
    if s.get("units", "um") != "um":
        sys.exit("only units = \"um\" is supported")
    nets = {}
    for c in s.get("conductor", []):
        nets.setdefault(c["net"], []).extend([list(map(float, b)) for b in c["boxes"]])
    return float(s["dielectric"]["eps_r"]), list(nets.items())


def load_layers(path):
    """Stratified dielectric of a structure: ([z_top, ...], [eps per layer, top last])."""
    d = tomllib.loads(Path(path).read_text())["dielectric"]
    layers = d.get("layer", [])
    return [float(l["z_top"]) for l in layers], [float(l["eps_r"]) for l in layers] + [float(d["eps_r"])]


def load_regions(path):
    """Dielectric boxes and conformal shells of a structure as disjoint labelled
    boxes [(box, eps_r)], in µm.

    Follows faradice's rules (design M2c §6.2): a conformal rule grows each box
    of its tagged conductors laterally by `side` and upwards by `top`; explicit
    boxes win over shells, later rules over earlier ones, and every conductor
    masks the medium. The result is a coordinate-compressed partition merged
    greedily into boxes."""
    s = tomllib.loads(Path(path).read_text())
    d = s["dielectric"]
    conductors = [list(map(float, b)) for c in s.get("conductor", []) for b in c["boxes"]]
    sources = []  # (boxes, eps), increasing precedence
    for r in d.get("conformal", []):
        side, top = float(r["side"]), float(r.get("top", 0.0))
        # Grown faces are rounded to 1e-9 µm against float noise; faces that
        # do not grow stay exactly on the conductor.
        def grow(v, d):
            return round(v + d, 9) if d else v

        grown = [
            [grow(b[0], -side), grow(b[1], -side), b[2], grow(b[3], side), grow(b[4], side), grow(b[5], top)]
            for c in s.get("conductor", [])
            if c.get("layer") == r["layer"]
            for b in (list(map(float, v)) for v in c["boxes"])
        ]
        sources.append((grown, float(r["eps_r"])))
    for b in d.get("box", []):
        sources.append(([list(map(float, b["min"])) + list(map(float, b["max"]))], float(b["eps_r"])))
    if not sources:
        return []
    every = [b for bs, _ in sources for b in bs]
    if not every:
        return []  # rules without tagged conductors (faradice warns)
    lo = [min(b[a] for b in every) for a in range(3)]
    hi = [max(b[a + 3] for b in every) for a in range(3)]
    cs = []
    for a in range(3):
        vals = {lo[a], hi[a]}
        for b in every + conductors:
            vals |= {v for v in (b[a], b[a + 3]) if lo[a] < v < hi[a]}
        cs.append(sorted(vals))

    def contains(b, p):
        return all(b[a] < p[a] < b[a + 3] for a in range(3))

    label = {}
    for i in range(len(cs[0]) - 1):
        for j in range(len(cs[1]) - 1):
            for k in range(len(cs[2]) - 1):
                p = [0.5 * (cs[a][m] + cs[a][m + 1]) for a, m in enumerate((i, j, k))]
                if any(contains(b, p) for b in conductors):
                    continue
                for q in range(len(sources) - 1, -1, -1):
                    if any(contains(b, p) for b in sources[q][0]):
                        label[(i, j, k)] = q
                        break
    out = []
    while label:
        i, j, k = min(label)
        q = label[(i, j, k)]
        i1 = i + 1
        while label.get((i1, j, k)) == q:
            i1 += 1
        j1 = j + 1
        while all(label.get((x, j1, k)) == q for x in range(i, i1)):
            j1 += 1
        k1 = k + 1
        while all(label.get((x, y, k1)) == q for x in range(i, i1) for y in range(j, j1)):
            k1 += 1
        for x in range(i, i1):
            for y in range(j, j1):
                for z in range(k, k1):
                    del label[(x, y, z)]
        out.append(([cs[0][i], cs[1][j], cs[2][k], cs[0][i1], cs[1][j1], cs[2][k1]], sources[q][1]))
    return out


def bbox(nets):
    lo = [min(b[a] for _, bs in nets for b in bs) for a in range(3)]
    hi = [max(b[a + 3] for _, bs in nets for b in bs) for a in range(3)]
    return lo, hi


def union_faces(boxes):
    """Exterior quads of the union of axis-aligned boxes (shared faces removed)."""
    cs = [sorted({b[a] for b in boxes} | {b[a + 3] for b in boxes}) for a in range(3)]
    idx = [{v: i for i, v in enumerate(c)} for c in cs]
    occ = set()
    for b in boxes:
        r = [range(idx[a][b[a]], idx[a][b[a + 3]]) for a in range(3)]
        occ.update((i, j, k) for i in r[0] for j in r[1] for k in r[2])
    quads = []
    for a in range(3):
        u, v = [x for x in range(3) if x != a]
        for cell in occ:
            for d in (-1, 1):
                nb = list(cell)
                nb[a] += d
                if tuple(nb) in occ:
                    continue
                w = cs[a][cell[a] + (1 if d > 0 else 0)]
                u0, u1 = cs[u][cell[u]], cs[u][cell[u] + 1]
                v0, v1 = cs[v][cell[v]], cs[v][cell[v] + 1]
                quad = []
                for pu, pv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                    p = [0.0] * 3
                    p[a], p[u], p[v] = w, pu, pv
                    quad.append(p)
                quads.append(quad)
    return quads


# ------------------------------------------------------------------ faradice


def faradice_bin():
    exe = ROOT / "target" / "release" / "faradice"
    if not exe.exists():
        subprocess.run(["cargo", "build", "--release", "-q", "-p", "faradice"], cwd=ROOT, check=True)
    return exe


def run_faradice(case, setting, work, preset, margin, threads, targets=("*",), reuse=False, confidence=None, total_only=False):
    work.mkdir(parents=True, exist_ok=True)
    domain = '[domain]\nkind = "open"\n' if setting == "plane" else f'[domain]\nkind = "box"\nmargin = {margin}\n'
    cfg = work / "faradice.toml"
    cfg.write_text(
        "schema_version = 1\n"
        + domain
        + '[ground]\nsubstrate = "plane"\nsubstrate_z = 0.0\nnets = []\n'
        + '[power]\nnets = []\n'
        + f"[extract]\nnets = {json.dumps(list(targets))}\n"
        + f'[accuracy]\npreset = "{preset}"\n'
        + (f"confidence_sigma = {confidence}\n" if confidence else "")
        + ("coupling_track = { rel = 0.0 }\n" if total_only else "")
        + '[coupling]\nsymmetrize = "none"\n'
        + f"[run]\nseed = 1\nthreads = {threads}\n"
    )
    out = work / "out"
    wall_file = work / "wall_s"
    # A reused run must have the same configuration and structure.
    stamp = cfg.read_text() + Path(case).read_text()
    stamp_file = work / "inputs.stamp"
    if reuse and (out / "result.json").exists() and wall_file.exists():
        if not stamp_file.exists() or stamp_file.read_text() != stamp:
            sys.exit(f"--reuse: {work} was run with other inputs; rerun without --reuse")
        wall = float(wall_file.read_text())
    else:
        t = time.perf_counter()
        subprocess.run(
            [faradice_bin(), "solve", case, "-c", cfg, "-o", out], check=True, stdout=subprocess.DEVNULL
        )
        wall = time.perf_counter() - t
        wall_file.write_text(f"{wall}\n")
        stamp_file.write_text(stamp)
    rep = json.loads((out / "report.json").read_text())
    res = {t["net"]: t for t in json.loads((out / "result.json").read_text())["targets"]}
    rows = {}
    for n in rep["nets"]:
        t = res[n["net"]]
        rows[n["net"]] = {
            "total": (n["total"]["value"], n["total"]["sigma"]),
            "couplings": {c["net"]: (c["c"]["value"], c["c"]["sigma"]) for c in n["couplings"]},
            "walks": n["walks"],
            "steps_per_walk": t["total_steps"] / t["walks"] if t["walks"] else 0.0,
            "aborted_walks": n["aborted_walks"],
            "converged": n["converged"],
        }
    return {"rows": rows, "wall_s": wall}


# ----------------------------------------------------------------- FasterCap


# Memory cap of every solver container (no swap). A reference solver that
# outgrows it is OOM-killed inside its container instead of taking the host
# down; --oom-score-adj makes the kernel pick it first under host pressure.
MEMORY = os.environ.get("FARADICE_BENCH_MEMORY", "24g")


class SolverFailed(Exception):
    pass


def docker(image, work, args, extra=(), log=None, check=True, timeout=None):
    """Runs `image` on `work`; returns (stdout, wall seconds, exit code).
    Raises SolverFailed if the run exceeds `timeout` seconds."""
    limits = ["--memory", MEMORY, "--memory-swap", MEMORY, "--oom-score-adj", "1000"]
    user = ["-u", f"{os.getuid()}:{os.getgid()}"]
    name = f"faradice-bench-{os.getpid()}-{time.monotonic_ns()}"
    cmd = ["docker", "run", "--rm", "--name", name, *limits, *extra, "-v", f"{work}:/work", *user, image, *args]
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        raise SolverFailed(f"{image} did not finish within {timeout:.0f} s") from None
    wall = time.perf_counter() - t
    (work / (log or f"{image.split('/')[-1]}.log")).write_text(p.stdout + p.stderr)
    if check and p.returncode != 0:
        oom = " (killed: out of memory?)" if p.returncode == 137 else ""
        raise SolverFailed(f"{image} failed with exit {p.returncode}{oom}; see {work}")
    return p.stdout, wall, p.returncode


def run_fastercap(eps_r, nets, work, rel_err, timeout=None):
    """Open space + substrate plane via odd images; returns Maxwell C (fF) by net."""
    work.mkdir(parents=True, exist_ok=True)
    lines = ["0 faradice structure with odd images below z = 0 (coordinates in m)"]
    for i, (_, boxes) in enumerate(nets):
        for tag, sign in (("n", 1.0), ("m", -1.0)):
            for q in union_faces(boxes):
                pts = [(x, y, sign * z) for x, y, z in q]
                lines.append(f"Q {tag}{i} " + " ".join(f"{v * 1e-6:.12g}" for p in pts for v in p))
    (work / "geom.qui").write_text("\n".join(lines) + "\n")
    (work / "case.lst").write_text(f"C geom.qui {eps_r} 0 0 0\n")
    out, wall, _ = docker("faradice-bench/fastercap", work, ["case.lst", "-b", f"-a{rel_err}", "-ap", "-g"], timeout=timeout)
    block = out.rsplit("Capacitance matrix is:", 1)[1].splitlines()
    dim = int(re.search(r"Dimension (\d+) x", block[1]).group(1))
    names, mat = [], []
    for line in block[2 : 2 + dim]:
        parts = line.split()
        names.append(parts[0].split("_", 1)[1])
        mat.append([float(v) for v in parts[1:]])
    pos = {n: k for k, n in enumerate(names)}
    c = {}
    for i, (ni, _) in enumerate(nets):
        c[ni] = {
            nj: (mat[pos[f"n{i}"]][pos[f"n{j}"]] - mat[pos[f"n{i}"]][pos[f"m{j}"]]) * 1e15
            for j, (nj, _) in enumerate(nets)
        }
    return {"matrix": c, "wall_s": wall, "settings": f"-a{rel_err} -ap -g"}


# -------------------------------------------------------------------- Palace


def run_palace(eps_r, nets, work, margin, ranks, amr_its, max_dofs, order, layers=None, regions=()):
    """Grounded box (same as faradice's box domain); returns Maxwell C (fF) by net.

    `layers` = (interface heights, permittivities bottom-up) for a stratified
    dielectric; `eps_r` is used otherwise. `regions` = disjoint dielectric boxes
    [(box, eps_r)] overriding the stack (`load_regions`)."""
    work.mkdir(parents=True, exist_ok=True)
    lo, hi = bbox(nets)
    dom = [lo[0] - margin, lo[1] - margin, 0.0, hi[0] + margin, hi[1] + margin, hi[2] + margin]
    feat = min(min(b[a + 3] - b[a] for a in range(3)) for _, bs in nets for b in bs)
    zs, eps = layers if layers and layers[0] else ([], [eps_r])
    # Slabs inside the domain, bottom-up, with their permittivities.
    inside = [z for z in zs if dom[2] < z < dom[5]]
    slab_eps = []
    for k in range(len(inside) + 1):
        zc = 0.5 * (([dom[2]] + inside)[k] + (inside + [dom[5]])[k])
        slab_eps.append(next((e for z, e in zip(zs, eps) if zc < z), eps[-1]))
    if inside:
        feat = min(feat, min(b - a for a, b in zip([dom[2]] + inside, inside + [dom[5]])))
    # Each distinct box permittivity gets its own volume attribute after the slabs.
    # Clip the boxes to the domain (faradice ignores the medium outside it).
    clipped = []
    for b, e in regions:
        c = [max(b[a], dom[a]) for a in range(3)] + [min(b[a + 3], dom[a + 3]) for a in range(3)]
        if all(c[a] < c[a + 3] for a in range(3)):
            clipped.append((c, e))
    regions = clipped
    box_eps = sorted({e for _, e in regions})
    boxes = [b + [len(slab_eps) + 1 + box_eps.index(e)] for b, e in regions]
    thin = min((min(b[a + 3] - b[a] for a in range(3)) for b, _ in regions), default=None)
    spec = {
        "domain": dom,
        "nets": [bs for _, bs in nets],
        "layers": inside,
        "boxes": boxes,
        "lc_near": feat / 2,
        "lc_box": thin,
        "lc_far": margin,
    }
    (work / "spec.json").write_text(json.dumps(spec))
    shutil.copy(HERE / "palace_mesh.py", work / "palace_mesh.py")
    _, mesh_wall, _ = docker(
        "faradice-bench/palace", work, ["python", "palace_mesh.py", "spec.json", "mesh.msh"], log="gmsh.log"
    )
    config = {
        "Problem": {"Type": "Electrostatic", "Verbose": 2, "Output": "postpro"},
        "Model": {
            "Mesh": "mesh.msh",
            "L0": 1.0e-6,
            "Refinement": {
                "Tol": 1e-6,
                "MaxIts": amr_its,
                "MaxSize": max_dofs,
                "Nonconformal": False,
                "UpdateFraction": 0.7,
                "SaveAdaptIterations": True,
            },
        },
        "Domains": {
            "Materials": [{"Attributes": [1 + k], "Permittivity": e} for k, e in enumerate(slab_eps)]
            + [{"Attributes": [len(slab_eps) + 1 + k], "Permittivity": e} for k, e in enumerate(box_eps)]
        },
        "Boundaries": {
            "Ground": {"Attributes": [2]},
            "Terminal": [{"Index": i + 1, "Attributes": [3 + i]} for i in range(len(nets))],
        },
        "Solver": {
            "Order": order,
            "Device": "CPU",
            "Electrostatic": {"Save": 0},
            "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1e-10, "MaxIts": 1000},
        },
    }
    (work / "palace.json").write_text(json.dumps(config, indent=1))
    shutil.rmtree(work / "postpro", ignore_errors=True)
    _, wall, code = docker(
        "faradice-bench/palace",
        work,
        ["palace", "-np", str(ranks), "palace.json"],
        ["--shm-size=8g", "-e", "OMPI_MCA_rmaps_base_oversubscribe=1"],
        check=False,
    )

    def read_c(path):
        rows = [r.split(",") for r in path.read_text().splitlines()[1:] if r.strip()]
        return [[float(v) * 1e15 for v in r[1:]] for r in rows]

    # Completed AMR meshes are saved in iterationNN/; the final mesh's results
    # are in postpro/ itself, written only when Palace finishes.
    iters = sorted((work / "postpro").glob("iteration*/terminal-C.csv"), key=lambda p: int(p.parent.name[9:]))
    history = [read_c(p) for p in iters]
    note = None
    if code == 0:
        history.append(read_c(work / "postpro" / "terminal-C.csv"))
    elif history:
        note = f"Palace stopped (exit {code}) on AMR mesh {len(history) + 1}; using mesh {len(history)}"
    else:
        raise SolverFailed(f"Palace failed with exit {code} before finishing a mesh; see {work}")
    mat = history[-1]
    names = [n for n, _ in nets]
    c = {ni: {nj: mat[i][j] for j, nj in enumerate(names)} for i, ni in enumerate(names)}
    # Convergence of the diagonal over the AMR iterations (last two meshes).
    drift = None
    if len(history) >= 2:
        drift = max(abs(history[-1][i][i] / history[-2][i][i] - 1) for i in range(len(names)))
    # FEM capacitance converges geometrically under AMR: extrapolate each entry
    # from the last three meshes when the differences shrink consistently.
    ext = None
    if len(history) >= 3:
        ext = {}
        for i, ni in enumerate(names):
            ext[ni] = {}
            for j, nj in enumerate(names):
                c1, c2, c3 = (h[i][j] for h in history[-3:])
                r = (c3 - c2) / (c2 - c1) if c2 != c1 else 0.0
                ext[ni][nj] = c3 + (c3 - c2) * r / (1 - r) if 0.0 < r < 0.9 else c3
    return {
        "matrix": c,
        "extrapolated": ext,
        "wall_s": wall,
        "mesh_wall_s": mesh_wall,
        "amr_meshes": len(history),
        "last_amr_change": drift,
        "settings": f"order {order}, AMR {amr_its} its, max {max_dofs} dofs, {ranks} MPI ranks, margin {margin} um"
        + (f", {len(regions)} dielectric boxes" if regions else ""),
        "note": note,
    }


# --------------------------------------------------------------------- RWCap


def media_partition(dom, eps_r, layers, regions):
    """Disjoint dielectric blocks [(box, eps_r)] filling `dom`: the slabs of the
    stratified stack with the boxes of `regions` (disjoint, from `load_regions`)
    cut out of them and added as blocks of their own."""
    zs, eps = layers if layers and layers[0] else ([], [eps_r])
    inside = [z for z in zs if dom[2] < z < dom[5]]
    cuts = [dom[2]] + inside + [dom[5]]
    out = []
    for z0, z1 in zip(cuts, cuts[1:]):
        zc = 0.5 * (z0 + z1)
        e_slab = next((e for z, e in zip(zs, eps) if zc < z), eps[-1])
        slab = [dom[0], dom[1], z0, dom[3], dom[4], z1]
        here = []
        for b, e in regions:
            c = [max(b[a], slab[a]) for a in range(3)] + [min(b[a + 3], slab[a + 3]) for a in range(3)]
            if all(c[a] < c[a + 3] for a in range(3)):
                here.append((c, e))
        if not here:
            out.append((slab, e_slab))
            continue
        # Coordinate-compressed cells of the slab, labelled by region or slab,
        # merged greedily into boxes (as in `load_regions`).
        cs = [sorted({slab[a], slab[a + 3]} | {v for b, _ in here for v in (b[a], b[a + 3])}) for a in range(3)]
        label = {}
        for i in range(len(cs[0]) - 1):
            for j in range(len(cs[1]) - 1):
                for k in range(len(cs[2]) - 1):
                    p = [0.5 * (cs[a][m] + cs[a][m + 1]) for a, m in enumerate((i, j, k))]
                    label[(i, j, k)] = next(
                        (e for b, e in here if all(b[a] < p[a] < b[a + 3] for a in range(3))), e_slab
                    )
        while label:
            i, j, k = min(label)
            q = label[(i, j, k)]
            i1 = i + 1
            while label.get((i1, j, k)) == q:
                i1 += 1
            j1 = j + 1
            while all(label.get((x, j1, k)) == q for x in range(i, i1)):
                j1 += 1
            k1 = k + 1
            while all(label.get((x, y, k1)) == q for x in range(i, i1) for y in range(j, j1)):
                k1 += 1
            for x in range(i, i1):
                for y in range(j, j1):
                    for z in range(k, k1):
                        del label[(x, y, z)]
            out.append(([cs[0][i], cs[1][j], cs[2][k], cs[0][i1], cs[1][j1], cs[2][k1]], q))
    return out


def cap3d(window, media, nets, master):
    """A .cap3d case (RWCap's input format): the medium blocks, the conductors
    (the ground among them) in `window`, extracting the row of `master`."""

    def block(b, pad):
        return [
            f"{pad}<block>",
            f"{pad}    basepoint({b[0]:.12g}, {b[1]:.12g}, {b[2]:.12g})",
            f"{pad}    v1({b[3] - b[0]:.12g}, 0, 0)",
            f"{pad}    v2(0, {b[4] - b[1]:.12g}, 0)",
            f"{pad}    hvector(0, 0, {b[5] - b[2]:.12g})",
            f"{pad}</block>",
        ]

    lines = ["<cap3d>"]
    for k, (b, e) in enumerate(media):
        lines += ["<medium>", f"    name medium{k}", *block(b, "    "), f"    diel {e:.12g}", "</medium>"]
    for i, (_, boxes) in enumerate(nets):
        lines += ["<conductor>", f"    name n{i}"]
        for b in boxes:
            lines += block(b, "    ")
        lines.append("</conductor>")
    lines += ["<task>", "    <capacitance>", f"        n{master}", "    </capacitance>", "</task>"]
    lines += [
        "<window>",
        f"    v1({window[0]:.12g}, {window[1]:.12g}, {window[2]:.12g})",
        f"    v2({window[3]:.12g}, {window[4]:.12g}, {window[5]:.12g})",
        "</window>",
        "</cap3d>",
    ]
    return "\n".join(lines) + "\n"


def run_rwcap(eps_r, nets, work, margin, threads, self_err, coup_err, layers=None, regions=()):
    """Grounded box (same as faradice's box domain), one RWCap run per net;
    returns Maxwell C (fF) by net.

    The box is an explicit grounded conductor: six plates of thickness `t`
    just outside the domain, inside a slightly larger window. (A window
    marked `dirichlet` absorbed no charge in RWCap v4: every row summed to
    zero.)"""
    work.mkdir(parents=True, exist_ok=True)
    lo, hi = bbox(nets)
    dom = [lo[0] - margin, lo[1] - margin, 0.0, hi[0] + margin, hi[1] + margin, hi[2] + margin]
    t = 0.05 * margin
    window = [v - t for v in dom[:3]] + [v + t for v in dom[3:]]
    x0, y0, z0, x1, y1, z1 = window
    ground = [
        [x0, y0, z0, x1, y1, dom[2]],
        [x0, y0, dom[5], x1, y1, z1],
        [x0, y0, dom[2], dom[0], y1, dom[5]],
        [dom[3], y0, dom[2], x1, y1, dom[5]],
        [dom[0], y0, dom[2], dom[3], dom[1], dom[5]],
        [dom[0], dom[4], dom[2], dom[3], y1, dom[5]],
    ]
    media = media_partition(window, eps_r, layers, regions)
    names = [n for n, _ in nets]
    with_ground = nets + [("ground", ground)]
    c, wall, elapsed, walks = {}, 0.0, 0.0, {}
    for i, ni in enumerate(names):
        (work / f"n{i}.cap3d").write_text(cap3d(window, media, with_ground, i))
        args = ["-f", f"/work/n{i}.cap3d", "-p", f"{self_err}", "-n", str(threads)]
        if coup_err:
            args += ["-c", f"{coup_err}"]
        args += ["--out-file", f"/work/n{i}.out", "--log-file", f"/work/n{i}.log"]
        _, w, _ = docker("faradice-bench/rwcap", work, args, log=f"n{i}.stdout")
        wall += w
        text = (work / f"n{i}.out").read_text()
        row = {m.group(1): float(m.group(2)) * 1e15 for m in re.finditer(r"Capacitance on (\S+) = (\S+) F", text)}
        if f"n{i}" not in row:
            raise SolverFailed(f"RWCap wrote no capacitance for {ni}; see {work}")
        c[ni] = {nj: row.get(f"n{j}", 0.0) for j, nj in enumerate(names)}
        m = re.search(r"RWCap has run (\d+) walks \(([\d.]+) hops/walk\)", text)
        if m:
            walks[ni] = (int(m.group(1)), float(m.group(2)))
        m = re.search(r"Elapsed time: ([\d.]+)sec", text)
        elapsed += float(m.group(1)) if m else w
    return {
        "matrix": c,
        "wall_s": wall,
        "solver_s": elapsed,
        "walks": walks,
        "settings": f"-p {self_err}" + (f" -c {coup_err}" if coup_err else "") + f", {threads} threads, margin {margin} um, {len(media)} medium blocks"
        + f"; {elapsed:.2f} s reported by RWCap, {sum(n for n, _ in walks.values())} walks",
    }


# ------------------------------------------------------------------- compare


def compare(fr, ref, ext=None, major=0.05):
    """Rows: (net, quantity, faradice, sigma, reference, rel error, error/sigma,
    rel error vs the extrapolated reference or None, is_major)."""
    rows = []
    for net, r in fr["rows"].items():
        items = [("total", r["total"], ref[net][net], ext and ext[net][net])]
        items += [(f"C({o})", cs, -ref[net][o], ext and -ext[net][o]) for o, cs in r["couplings"].items()]
        cref = ref[net][net]
        for q, (v, s), cr, ce in items:
            is_major = q == "total" or cr >= major * cref
            e = v / cr - 1 if cr else 0.0
            rows.append((net, q, v, s, cr, e, (v - cr) / s if s else 0.0, ce and v / ce - 1, is_major))
    return rows


def table(title, rows, fr, ref):
    out = [f"### {title}", ""]
    out.append(f"faradice {fr['wall_s']:.2f} s wall; reference {ref['wall_s']:.2f} s wall ({ref['settings']})")
    if ref.get("note"):
        out.append(f"; **{ref['note']}**")
    if ref.get("last_amr_change") is not None:
        out.append(f"; Palace AMR meshes {ref['amr_meshes']}, last change of the diagonal {ref['last_amr_change'] * 100:.3f} %")
    has_ext = any(r[7] is not None for r in rows)
    head = "| net | quantity | faradice (fF) | σ (fF) | reference (fF) | error | error/σ |"
    out += ["", head + (" error vs extrapolated |" if has_ext else ""), "|---" * (8 if has_ext else 7) + "|"]
    for net, q, v, s, r, e, z, ee, maj in rows:
        if not maj:
            continue
        line = f"| {net} | {q} | {v:.5f} | {s:.5f} | {r:.5f} | {e * 100:+.3f} % | {z:+.1f} |"
        out.append(line + (f" {ee * 100:+.3f} % |" if has_ext else ""))
    tot = [abs(r[5]) for r in rows if r[1] == "total"]
    cpl = [abs(r[5]) for r in rows if r[1] != "total" and r[8]]
    out += ["", f"max |error|: total {max(tot) * 100:.3f} %, major couplings (≥ 5 % of total) {max(cpl, default=0) * 100:.3f} %", ""]
    stats = [
        f"{net} {r['steps_per_walk']:.1f}" + ("" if r["converged"] else " (not converged)")
        for net, r in fr["rows"].items()
        if "steps_per_walk" in r
    ]
    if stats:
        out += ["faradice steps per walk: " + ", ".join(stats), ""]
    aborted = {net: r["aborted_walks"] for net, r in fr["rows"].items() if r.get("aborted_walks")}
    if aborted:
        out += [f"**faradice aborted walks: {aborted}**", ""]
    return "\n".join(out), max(tot), max(cpl, default=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path)
    ap.add_argument("--solvers", default="fastercap,palace")
    ap.add_argument("--out", type=Path, default=Path("compare_out"))
    ap.add_argument("--preset", default="reference")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument(
        "--confidence-sigma", type=float, help="faradice accuracy.confidence_sigma (1 matches RWCap's -p and -c)"
    )
    ap.add_argument(
        "--total-only",
        action="store_true",
        help="stop on the total only: faradice tracks no coupling, RWCap runs without -c",
    )
    ap.add_argument("--reuse", action="store_true", help="reuse a previous faradice run in --out (same settings)")
    ap.add_argument("--fastercap-error", type=float, default=1e-3)
    ap.add_argument("--fastercap-timeout", type=float, default=1800.0, help="seconds")
    ap.add_argument("--palace-amr", type=int, default=10)
    ap.add_argument("--palace-max-dofs", type=int, default=1_500_000)
    ap.add_argument("--palace-ranks", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--palace-order", type=int, default=2)
    ap.add_argument("--rwcap-self", type=float, default=1e-3, help="RWCap -p (relative error of the total)")
    ap.add_argument("--rwcap-coupling", type=float, default=1e-2, help="RWCap -c (relative error of major couplings)")
    a = ap.parse_args()

    case = a.case.resolve()
    eps_r, nets = load_structure(case)
    layers = load_layers(case)
    regions = load_regions(case)
    name = case.name.removesuffix(".faradice.toml")
    out = (a.out / name).resolve()
    solvers = a.solvers.split(",")
    dielectric = f"eps_r = {eps_r}" if not layers[0] else f"{len(layers[0])} dielectric layers"
    md = [f"## {name}", "", f"{len(nets)} nets, {dielectric}, faradice preset `{a.preset}`, {a.threads} threads", ""]
    if regions:
        md.append(f"{len(regions)} disjoint dielectric boxes (explicit boxes and conformal shells)\n")
    if (layers[0] or regions) and "fastercap" in solvers:
        solvers.remove("fastercap")
        md.append("FasterCap skipped: inhomogeneous dielectrics are compared with Palace only.\n")
    summary = {"case": name, "results": {}}
    def attempt(key, title, setting, solve, use_ext):
        fr = run_faradice(
            case, setting, out / setting / "faradice", a.preset, a.margin, a.threads, reuse=a.reuse,
            confidence=a.confidence_sigma,
            total_only=a.total_only,
        )
        try:
            ref = solve()
        except SolverFailed as e:
            md.append(f"### {title}\n\n**{e}**\n")
            summary["results"][key] = {"faradice": fr, "error": str(e)}
            return
        text, et, ec = table(title, compare(fr, ref["matrix"], ref.get("extrapolated") if use_ext else None), fr, ref)
        md.append(text)
        summary["results"][key] = {"faradice": fr, "reference": ref, "max_total_err": et, "max_major_coupling_err": ec}

    if "fastercap" in solvers:
        attempt(
            "fastercap",
            "Open space + substrate plane: faradice vs FasterCap",
            "plane",
            lambda: run_fastercap(eps_r, nets, out / "plane" / "fastercap", a.fastercap_error, a.fastercap_timeout),
            False,
        )
    if "palace" in solvers:
        attempt(
            "palace",
            "Grounded box: faradice vs Palace",
            "box",
            lambda: run_palace(
                eps_r,
                nets,
                out / "box" / "palace",
                a.margin,
                a.palace_ranks,
                a.palace_amr,
                a.palace_max_dofs,
                a.palace_order,
                layers,
                regions,
            ),
            True,
        )
    if "rwcap" in solvers:
        attempt(
            "rwcap",
            "Grounded box: faradice vs RWCap v4",
            "box",
            lambda: run_rwcap(
                eps_r,
                nets,
                out / "box" / "rwcap",
                a.margin,
                a.threads,
                a.rwcap_self,
                0.0 if a.total_only else a.rwcap_coupling,
                layers,
                regions,
            ),
            False,
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
