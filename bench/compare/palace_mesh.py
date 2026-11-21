#!/usr/bin/env python3
"""Meshes a grounded box with box-shaped conductor holes for Palace (runs inside
the faradice-bench/palace container, which provides the Gmsh Python module).

Usage: palace_mesh.py SPEC.json MESH.msh

SPEC.json: {"domain": [x0, y0, z0, x1, y1, z1],
            "nets": [[[x0, y0, z0, x1, y1, z1], ...], ...],
            "layers": [z_1, z_2, ...],      # optional interface heights
            "boxes": [[x0, y0, z0, x1, y1, z1, attr], ...],  # optional
            "lc_near": float, "lc_box": float or null, "lc_far": float}

Physical groups: volume 1 + k = dielectric layer k (bottom-up; a single
volume 1 without layers), volume `attr` = the dielectric boxes with that
attribute, surface 2 = grounded domain faces, surface 3 + i = conductor
surface of net i. The boxes must be disjoint and must not overlap conductors;
they override the layers they lie in.
"""

import json
import sys

import gmsh


def inside(bb, box, eps):
    """`True` if bounding box `bb` (xmin, ymin, zmin, xmax, ymax, zmax) lies in `box`."""
    return all(box[a] - eps <= bb[a] and bb[a + 3] <= box[a + 3] + eps for a in range(3))


def main():
    spec = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    dom = spec["domain"]
    nets = spec["nets"]
    size = max(dom[a + 3] - dom[a] for a in range(3))
    eps = 1e-5 * size  # OCC pads bounding boxes by ~1e-7

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("faradice")
    occ = gmsh.model.occ
    # The domain as a stack of slabs split at the interface heights.
    zs = [dom[2]] + sorted(z for z in spec.get("layers", []) if dom[2] < z < dom[5]) + [dom[5]]
    slabs = [
        (3, occ.addBox(dom[0], dom[1], zs[k], dom[3] - dom[0], dom[4] - dom[1], zs[k + 1] - zs[k]))
        for k in range(len(zs) - 1)
    ]
    tools = []
    for boxes in nets:
        for b in boxes:
            tools.append((3, occ.addBox(b[0], b[1], b[2], b[3] - b[0], b[4] - b[1], b[5] - b[2])))
    vols, _ = occ.cut(slabs, tools)
    boxes = spec.get("boxes", [])
    pieces = [(3, occ.addBox(b[0], b[1], b[2], b[3] - b[0], b[4] - b[1], b[5] - b[2])) for b in boxes]
    box_of = {}
    if len(vols) > 1 or pieces:
        # Conformal interfaces between the slabs and the boxes. The pieces of
        # each input box keep that box's attribute.
        n = len(vols)
        vols, parts = occ.fragment(vols, pieces)
        for b, part in zip(boxes, parts[n:]):
            for _, v in part:
                box_of[v] = b[6]
    occ.synchronize()
    layer_of = {}
    for _, v in vols:
        if v in box_of:
            continue
        bb = gmsh.model.getBoundingBox(3, v)
        zc = 0.5 * (bb[2] + bb[5])
        layer_of[v] = max(k for k in range(len(zs) - 1) if zs[k] <= zc)

    ground, per_net = [], [[] for _ in nets]
    for _, s in gmsh.model.getBoundary(vols, combined=True, oriented=False):
        bb = gmsh.model.getBoundingBox(2, s)
        on_domain = any(
            abs(bb[a] - dom[a]) < eps and abs(bb[a + 3] - dom[a]) < eps
            or abs(bb[a] - dom[a + 3]) < eps and abs(bb[a + 3] - dom[a + 3]) < eps
            for a in range(3)
        )
        if on_domain:
            ground.append(s)
            continue
        owner = [i for i, boxes in enumerate(nets) if any(inside(bb, b, eps) for b in boxes)]
        if len(owner) != 1:
            sys.exit(f"surface {s} {bb}: owned by nets {owner}")
        per_net[owner[0]].append(s)

    for k in range(len(zs) - 1):
        members = [v for v, layer in layer_of.items() if layer == k]
        if members:
            gmsh.model.addPhysicalGroup(3, members, 1 + k)
    for attr in sorted(set(box_of.values())):
        gmsh.model.addPhysicalGroup(3, [v for v, a in box_of.items() if a == attr], attr)
    gmsh.model.addPhysicalGroup(2, ground, 2)
    for i, surfs in enumerate(per_net):
        if not surfs:
            sys.exit(f"net {i} has no surface")
        gmsh.model.addPhysicalGroup(2, surfs, 3 + i)

    # Fine near conductors, coarse far away; Palace AMR refines from there.
    conductor_surfs = [s for surfs in per_net for s in surfs]
    f = gmsh.model.mesh.field
    dist = f.add("Distance")
    f.setNumbers(dist, "SurfacesList", conductor_surfs)
    thr = f.add("Threshold")
    f.setNumber(thr, "InField", dist)
    f.setNumber(thr, "SizeMin", spec["lc_near"])
    f.setNumber(thr, "SizeMax", spec["lc_far"])
    f.setNumber(thr, "DistMin", 0.0)
    f.setNumber(thr, "DistMax", 0.25 * size)
    if spec.get("lc_box"):
        # Thin dielectric boxes (conformal shells) get cells no larger than
        # their thickness near their faces.
        box_surfs = [s for _, s in gmsh.model.getBoundary([(3, v) for v in box_of], combined=False, oriented=False)]
        dbox = f.add("Distance")
        f.setNumbers(dbox, "SurfacesList", box_surfs)
        tbox = f.add("Threshold")
        f.setNumber(tbox, "InField", dbox)
        f.setNumber(tbox, "SizeMin", spec["lc_box"])
        f.setNumber(tbox, "SizeMax", spec["lc_far"])
        f.setNumber(tbox, "DistMin", 0.0)
        f.setNumber(tbox, "DistMax", 0.25 * size)
        thr_min = f.add("Min")
        f.setNumbers(thr_min, "FieldsList", [thr, tbox])
        thr = thr_min
    f.setAsBackgroundMesh(thr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write(out)
    gmsh.finalize()


if __name__ == "__main__":
    main()
