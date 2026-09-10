#!/usr/bin/env python3
"""Generate original, redistributable Istana-inspired exterior geometry.

No downloaded meshes, imagery, textures, crests or flags are used. Architectural
proportions are an artistic reconstruction from public exterior references, not
survey measurements. All generated geometry and this script are dedicated CC0.

Run: python Tools/generate_istana.py
Output: SourceAssets/Architecture/*.obj (centimetres, Z up, front is -Y).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

TAU = math.tau
MATERIALS = {
    "plaster": {"color": (0.87, 0.845, 0.77), "roughness": 0.76, "metallic": 0},
    "stone": {"color": (0.64, 0.60, 0.50), "roughness": 0.9, "metallic": 0},
    "slate": {"color": (0.085, 0.12, 0.145), "roughness": 0.68, "metallic": 0},
    "glass": {"color": (0.055, 0.105, 0.115), "roughness": 0.12, "metallic": 0.25},
    "dark_wood": {"color": (0.065, 0.079, 0.063), "roughness": 0.65, "metallic": 0},
    "metal": {"color": (0.105, 0.115, 0.10), "roughness": 0.38, "metallic": 0.8},
}


def sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normal(a, b, c):
    n = cross(sub(b, a), sub(c, a))
    length = math.sqrt(sum(v * v for v in n))
    if length < 1e-9:
        return None
    return tuple(v / length for v in n)


class Mesh:
    def __init__(self, name):
        self.name = name
        self.faces = []
        self.bounds_min = [float("inf")] * 3
        self.bounds_max = [-float("inf")] * 3
        self.labels = Counter()

    def face(self, points, material="plaster", label="detail"):
        n = normal(*points[:3])
        if n is None:
            return
        assert material in MATERIALS
        assert all(math.isfinite(v) for p in points for v in p)
        for p in points:
            for i in range(3):
                self.bounds_min[i] = min(self.bounds_min[i], p[i])
                self.bounds_max[i] = max(self.bounds_max[i], p[i])
        self.faces.append((tuple(points), material, label, n))
        self.labels[label] += len(points) - 2

    def box(self, x, y, z, sx, sy, sz, material="plaster", label="detail", angle=0):
        assert sx > 0 and sy > 0 and sz > 0
        c, s = math.cos(angle), math.sin(angle)
        p = []
        for dz in (-sz / 2, sz / 2):
            for dx, dy in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)):
                p.append((x + dx * c - dy * s, y + dx * s + dy * c, z + dz))
        for ids in ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.face([p[i] for i in ids], material, label)

    def prism_xz(self, polygon, y, depth, material="plaster", label="detail"):
        """Extrude a counterclockwise x,z polygon along Y; convex caps only."""
        front = [(x, y - depth / 2, z) for x, z in polygon]
        back = [(x, y + depth / 2, z) for x, z in polygon]
        # CCW in X/Z yields outward negative-Y front normals.
        self.face(front, material, label)
        self.face(list(reversed(back)), material, label)
        for i in range(len(polygon)):
            j = (i + 1) % len(polygon)
            self.face((front[j], front[i], back[i], back[j]), material, label)

    def lathe(self, x, y, z, profile, material="plaster", label="detail", sides=20):
        for i in range(sides):
            a, b = TAU * i / sides, TAU * (i + 1) / sides
            for (za, ra), (zb, rb) in zip(profile, profile[1:]):
                pts = [(x + ra * math.cos(a), y + ra * math.sin(a), z + za),
                       (x + ra * math.cos(b), y + ra * math.sin(b), z + za),
                       (x + rb * math.cos(b), y + rb * math.sin(b), z + zb),
                       (x + rb * math.cos(a), y + rb * math.sin(a), z + zb)]
                if ra == 0:
                    pts = [pts[0], pts[2], pts[3]]
                if rb == 0:
                    pts = [pts[0], pts[1], pts[2]]
                self.face(pts, material, label)
        for zi, ri, reverse in ((profile[0][0], profile[0][1], True), (profile[-1][0], profile[-1][1], False)):
            if ri:
                p = [(x + ri * math.cos(TAU * i / sides), y + ri * math.sin(TAU * i / sides), z + zi) for i in range(sides)]
                self.face(list(reversed(p)) if reverse else p, material, label)

    def beam(self, a, b, thickness, material="plaster", label="detail"):
        axis = sub(b, a)
        length = math.sqrt(sum(v * v for v in axis))
        u = tuple(v / length for v in axis)
        seed = (0, 0, 1) if abs(u[2]) < .95 else (0, 1, 0)
        v = cross(u, seed)
        mag = math.sqrt(sum(k * k for k in v))
        v = tuple(k / mag * thickness / 2 for k in v)
        w = cross(u, v)
        pts = []
        for p in (a, b):
            for sv, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                pts.append(tuple(p[i] + sv * v[i] + sw * w[i] for i in range(3)))
        for ids in ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.face([pts[i] for i in ids], material, label)

    def write(self, directory):
        path = directory / (self.name + ".obj")
        triangles = 0
        index = 1
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# CC0 original procedural architecture; centimetres; Z up; front -Y\n")
            f.write("mtllib Istana.mtl\no " + self.name + "\ns off\n")
            last_material = None
            last_label = None
            for points, material, label, n in self.faces:
                if label != last_label:
                    f.write("g " + label + "\n")
                    last_label = label
                if material != last_material:
                    f.write("usemtl " + material + "\n")
                    last_material = material
                dominant = max(range(3), key=lambda axis: abs(n[axis]))
                uv_axes = ((1, 2), (0, 2), (0, 1))[dominant]
                for p in points:
                    f.write("v %.4f %.4f %.4f\n" % tuple(v * 100 for v in p))
                for p in points:
                    f.write("vt %.5f %.5f\n" % (p[uv_axes[0]], p[uv_axes[1]]))
                for _ in points:
                    f.write("vn %.6f %.6f %.6f\n" % n)
                for i in range(1, len(points) - 1):
                    ids = (index, index + i, index + i + 1)
                    f.write("f " + " ".join(f"{j}/{j}/{j}" for j in ids) + "\n")
                    triangles += 1
                index += len(points)
        assert path.stat().st_size < 50_000_000, f"Split oversized part {path}"
        return {"file": path.name, "vertices": index - 1, "triangles": triangles,
                "size_bytes": path.stat().st_size,
                "bounds_cm": [[round(v * 100, 3) for v in self.bounds_min], [round(v * 100, 3) for v in self.bounds_max]],
                "materials": sorted(set(f[1] for f in self.faces)), "components_triangles": dict(self.labels)}


def ring_band(mesh, x, y, width, depth, z, height, projection=.15, material="plaster", label="cornices"):
    mesh.box(x, y - depth / 2, z, width + 2 * projection, 2 * projection, height, material, label)
    mesh.box(x, y + depth / 2, z, width + 2 * projection, 2 * projection, height, material, label)
    mesh.box(x - width / 2, y, z, 2 * projection, depth, height, material, label)
    mesh.box(x + width / 2, y, z, 2 * projection, depth, height, material, label)


def cornice(mesh, x, y, width, depth, z, scale=1):
    for dz, h, projection in ((0, .16, .10), (.15, .14, .20), (.32, .18, .38), (.49, .13, .27), (.63, .14, .45)):
        ring_band(mesh, x, y, width, depth, z + dz * scale, h * scale, projection * scale)
    for side in (-1, 1):
        for i in range(int(width / .52)):
            px = x - width / 2 + .26 + i * .52
            mesh.box(px, y + side * (depth / 2 + .17), z + .2 * scale, .16, .28, .18 * scale, label="dentil_mouldings")


def column(mesh, x, y, base, height, order="doric", radius=.3):
    cap_h = .52 if order == "corinthian" else .32
    mesh.box(x, y, base + .10, radius * 2.85, radius * 2.85, .20, label=order + "_plinth")
    profile = [(.20, radius * 1.24), (.28, radius * 1.24), (.34, radius * 1.08),
               (.41, radius * 1.1), (.48, radius), (height * .35, radius * 1.015),
               (height - cap_h - .13, radius * .82), (height - cap_h, radius * .98),
               (height - cap_h + .08, radius * 1.1), (height - .13, radius * 1.3)]
    mesh.lathe(x, y, base, profile, label=order + "_columns", sides=24)
    mesh.box(x, y, base + height - .065, radius * 2.95, radius * 2.95, .13, label=order + "_abacus")
    if order in ("ionic", "corinthian"):
        # Paired volutes are modelled as small lobed scroll solids.
        for dx in (-radius * .9, radius * .9):
            for dy in (-radius * .84, radius * .84):
                mesh.lathe(x + dx, y + dy, base + height - .3, [(0, .12), (.06, .16), (.15, .13), (.22, .06)], label="capital_scrolls", sides=10)
    if order == "corinthian":
        for i in range(8):
            a = TAU * i / 8
            dx, dy = radius * math.cos(a), radius * math.sin(a)
            mesh.lathe(x + dx, y + dy, base + height - .49, [(0, .045), (.15, .10), (.26, .12), (.36, .025)], label="capital_acanthus", sides=8)


def arch(mesh, trim, x, y, base, width, spring, top, depth=.55):
    r = width / 2
    count = 20
    # Fill above the open arch, preserving a real hole through the arcade.
    for i in range(count):
        a, b = math.pi * i / count, math.pi * (i + 1) / count
        xa, za = x + r * math.cos(a), spring + r * math.sin(a)
        xb, zb = x + r * math.cos(b), spring + r * math.sin(b)
        mesh.prism_xz([(xb, zb), (xa, za), (xa, top), (xb, top)], y, depth, label="arcade_spandrels")
        outer = r + .18
        trim.prism_xz([(xa, za), (x + outer * math.cos(a), spring + outer * math.sin(a)),
                       (x + outer * math.cos(b), spring + outer * math.sin(b)), (xb, zb)], y - depth / 2 - .025, .16, label="archivolts")
    for side in (-1, 1):
        trim.box(x + side * (r + .09), y - depth / 2 - .025, (base + spring) / 2, .18, .16, spring - base, label="arch_jambs")
        trim.box(x + side * r, y - depth / 2 - .05, spring - .025, .4, .29, .15, label="arch_imposts")
    trim.prism_xz([(x - .22, spring + r - .17), (x + .22, spring + r - .17), (x + .3, spring + r + .34), (x - .3, spring + r + .34)], y - depth / 2 - .12, .25, label="keystones")


def window(mesh, x, y, base, width=2.2, height=3.5, angle=0, arched=False):
    """A tall timber window with recessed glass, real slats and masonry reveals."""
    c, s = math.cos(angle), math.sin(angle)

    def box(dx, dy, z, sx, sy, sz, mat="dark_wood", label="window_joinery"):
        mesh.box(x + dx * c - dy * s, y + dx * s + dy * c, z, sx, sy, sz, mat, label, angle)

    # Renderable window fills stay in front of the wall, surrounded by deep trim.
    frame_h = height - width * .24 if arched else height
    box(0, -.035, base + frame_h / 2, width, .1, frame_h, "glass", "recessed_glazing")
    for side in (-1, 1):
        box(side * (width / 2 + .09), -.09, base + frame_h / 2, .19, .27, frame_h + .2, "plaster", "window_reveals")
        box(side * width * .20, -.115, base + frame_h / 2, .075, .11, frame_h)
        shutter_w = width * .265
        shutter_x = side * width * .36
        box(shutter_x, -.115, base + frame_h / 2, shutter_w, .12, frame_h)
        for edge in (-1, 1):
            box(shutter_x + edge * (shutter_w / 2 - .035), -.2, base + frame_h / 2, .07, .11, frame_h)
        for j in range(max(1, int((frame_h - .22) / .14))):
            z = base + .15 + j * .14
            # Bevelled wedge slats catch light without using bitmap textures.
            p = [(shutter_x - shutter_w / 2 + .06, -.25, z),
                 (shutter_x + shutter_w / 2 - .06, -.25, z),
                 (shutter_x + shutter_w / 2 - .06, -.15, z + .07),
                 (shutter_x - shutter_w / 2 + .06, -.15, z + .07)]
            pts = [(x + px * c - py * s, y + px * s + py * c, pz) for px, py, pz in p]
            mesh.face(pts, "dark_wood", "individual_shutter_louvres")
            mesh.face(list(reversed(pts)), "dark_wood", "individual_shutter_louvres")
        box(shutter_x, -.245, base + frame_h * .48, shutter_w, .07, .12)
        box(side * width * .17, -.22, base + frame_h * .45, .035, .05, .22, "metal", "window_handles")
    for fraction in (.28, .58, .86):
        box(0, -.14, base + frame_h * fraction, width * .41, .08, .065)
    box(0, -.105, base + frame_h / 2, .075, .09, frame_h)
    box(0, -.13, base + frame_h, width + .45, .36, .15, "plaster", "window_hoods")
    box(0, -.15, base - .05, width + .5, .48, .16, "plaster", "window_sills")
    if arched:
        r = width / 2
        spring = base + frame_h
        for j in range(16):
            a, b = math.pi * j / 16, math.pi * (j + 1) / 16
            for outer, inner, mat in ((r, 0, "glass"), (r + .14, r, "plaster")):
                p = [(inner * math.cos(a), spring + inner * .48 * math.sin(a)),
                     (outer * math.cos(a), spring + outer * .48 * math.sin(a)),
                     (outer * math.cos(b), spring + outer * .48 * math.sin(b)),
                     (inner * math.cos(b), spring + inner * .48 * math.sin(b))]
                if inner == 0:
                    p = p[:3]
                mesh.face([(x + px * c + .1 * s, y + px * s - .1 * c, pz) for px, pz in p], mat, "arched_fanlights")


def balustrade(mesh, a, b, z, height=.88, spacing=.43):
    distance = math.dist(a, b)
    count = max(1, int(distance / spacing))
    profile = [(0, .10), (.09, .10), (.14, .065), (.24, .105), (.37, .125), (.49, .055), (.64, .05), (.70, .1)]
    for i in range(count + 1):
        t = i / count
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        mesh.lathe(x, y, z + .10, [(h * height / .88, r) for h, r in profile], label="turned_balusters", sides=8)
    mesh.beam((*a, z + .07), (*b, z + .07), .20, label="balustrade_lower_rail")
    mesh.beam((*a, z + height), (*b, z + height), .23, label="balustrade_coping")


def frustum(mesh, x, y, z, width, depth, top_width, top_depth, height, material="slate", label="mansard_roof"):
    bottom = [(x - width / 2, y - depth / 2, z), (x + width / 2, y - depth / 2, z), (x + width / 2, y + depth / 2, z), (x - width / 2, y + depth / 2, z)]
    top = [(x - top_width / 2, y - top_depth / 2, z + height), (x + top_width / 2, y - top_depth / 2, z + height), (x + top_width / 2, y + top_depth / 2, z + height), (x - top_width / 2, y + top_depth / 2, z + height)]
    for i in range(4):
        j = (i + 1) % 4
        mesh.face((bottom[i], bottom[j], top[j], top[i]), material, label)
    mesh.face(top, material, label)
    mesh.face(list(reversed(bottom)), material, label)


def pediment(mesh, trim, x, y, width, z, rise=2.1, depth=.9):
    mesh.prism_xz([(x - width / 2, z), (x + width / 2, z), (x, z + rise)], y, depth, label="pediment_tympanums")
    for side in (-1, 1):
        trim.beam((x + side * (width / 2 + .25), y - depth / 2 - .08, z - .05), (x, y - depth / 2 - .08, z + rise + .1), .25, label="pediment_raking_cornice")
    trim.box(x, y - depth / 2, z, width + .7, .45, .22, label="pediment_base_moulding")


def build():
    names = ("Istana_Structure", "Istana_Colonnades", "Istana_Cornices", "Istana_Windows_West", "Istana_Windows_Centre", "Istana_Windows_East", "Istana_Roof", "Istana_Balustrades", "Istana_Terraces")
    parts = {name: Mesh(name) for name in names}
    structure, columns, trim, west, centre, east, roof, rails, ground = parts.values()
    # Foundation and principal floor levels. Internal dimensions are metres.
    ground.box(0, 1, .31, 103, 34, .62, "stone", "foundation_plinth")
    ground.box(0, -1, .86, 102, 34, .5, "plaster", "plinth_upper_course")
    ground.box(0, -15.2, 1.04, 102, 4.5, .28, "stone", "front_terrace")
    for i in range(7):
        # Continuous shallow steps in front of the central approach.
        ground.box(0, -22.6 + i * .43, .08 + i * .15, 23.5, .52, .16 + i * .30, "stone", "central_broad_stair")
    for x in (-12.1, 12.1):
        for i in range(7):
            ground.box(x, -22.6 + i * .43, .33 + i * .15, .55, .52, .65 + i * .30, "plaster", "stair_cheekwalls")
        ground.lathe(x, -19.8, 1.5, [(0, .32), (.12, .4), (.3, .29), (.8, .25), (.95, .34), (1.1, .25), (1.25, 0)], label="stair_newel_finials", sides=16)

    # Two side wings, each with seven deeply recessed arcade bays.
    for side, windows in ((-1, west), (1, east)):
        cx = side * 25.8
        wing_width, core_depth = 28.4, 21.0
        structure.box(cx, 3.5, 6.52, wing_width, core_depth, 10.8, label="wing_wall_mass")
        for z, thickness in ((1.2, .25), (6.7, .35), (12.25, .34)):
            structure.box(cx, -9.55, z, wing_width, 5.4, thickness, label="verandah_floors_and_soffits")
        for z in (1.3, 6.65, 12.3):
            ring_band(trim, cx, 1.0, wing_width, 26, z, .25, .21)
        cornice(trim, cx, 1.0, wing_width, 26, 12.25)
        x0 = cx - wing_width / 2
        bay = wing_width / 7
        for j in range(8):
            x = x0 + j * bay
            # Ground order: substantial Doric piers and moulded pilasters.
            columns.box(x, -12.0, 2.87, .70, .72, 3.3, label="doric_arcade_piers")
            columns.box(x, -12.03, 1.33, 1.0, 1.0, .30, label="doric_pier_bases")
            columns.box(x, -12.03, 4.48, 1.0, .97, .24, label="doric_pier_capitals")
            column(columns, x, -12.0, 6.9, 4.95, "ionic", .245)
            # Pilasters frame the upper back wall behind the open gallery.
            columns.box(x, -7.11, 9.2, .36, .20, 4.5, label="upper_verandah_pilasters")
        for j in range(7):
            x = x0 + (j + .5) * bay
            arch(structure, trim, x, -12.0, 1.25, bay - .72, 4.38, 6.55, .66)
            window(windows, x, -7.08, 1.56, 2.25, 4.0, arched=True)
            window(windows, x, -7.08, 7.1, 2.32, 4.55)
            # Low, delicate upper gallery balustrades leave the columns exposed.
            balustrade(rails, (x - bay / 2 + .47, -12.02), (x + bay / 2 - .47, -12.02), 6.94, .82)
            for z in (1.8, 7.05):
                window(windows, x, 14.03, z, 2.3, 3.7, math.pi)
        frustum(roof, cx, 1.0, 12.94, wing_width + .25, 26.2, wing_width - 7.0, 17.0, 2.1, label="wing_hipped_roofs")
        # Front parapet screens the low hip roof, like the actual silhouette.
        balustrade(rails, (x0, -11.85), (x0 + wing_width, -11.85), 13.0, .82)

        # Twin pedimented terminal pavilions.
        px, py, pw, pd = side * 45.55, 1.0, 11.1, 30.0
        structure.box(px, 2, 6.7, pw, 25, 11.2, label="end_pavilion_walls")
        structure.box(px, -13.15, 6.7, pw, 2.4, .38, label="end_pavilion_upper_floor")
        structure.box(px, -13.15, 12.25, pw, 2.4, .38, label="end_pavilion_ceiling")
        for dx in (-4.6, -1.55, 1.55, 4.6):
            column(columns, px + dx, -14.0, 1.2, 4.8, "doric", .28)
            column(columns, px + dx, -14.0, 6.9, 4.95, "ionic", .26)
        for dx in (-3.05, 0, 3.05):
            window(windows, px + dx, -10.55, 1.65, 1.95, 3.75, arched=True)
            window(windows, px + dx, -10.55, 7.08, 1.96, 4.48)
        cornice(trim, px, py, pw, pd, 12.25)
        pediment(structure, trim, px, -14.0, pw - .35, 13.0, 2.2)
        frustum(roof, px, py + 1, 13.0, pw, pd - 1, 5.0, pd - 8, 2.0, label="terminal_pavilion_roofs")
        # East/west elevations, visible when orbiting around the site.
        side_angle = side * math.pi / 2
        for yy in (-7.5, -2.5, 2.5, 7.5, 12.0):
            for zz in (1.6, 7.05):
                window(windows, px + side * (pw / 2 + .025), yy, zz, 2.25, 3.9, side_angle)
        for xx in (px - 3.5, px, px + 3.5):
            for zz in (1.7, 7.05):
                window(windows, xx, 14.53, zz, 2.1, 3.8, math.pi)

    # Projecting central double-height portico with three broad arched openings.
    structure.box(0, 1.8, 6.7, 22.5, 29.5, 11.2, label="central_main_block")
    structure.box(0, -16.2, 1.22, 23.5, 6.5, .32, "stone", "portico_floor")
    structure.box(0, -16.1, 6.7, 23, 6.5, .40, label="portico_gallery_floor")
    structure.box(0, -16.1, 12.35, 23, 6.5, .40, label="portico_entablature")
    for x in (-10.8, -7.2, -3.6, 0, 3.6, 7.2, 10.8):
        if x in (-10.8, -3.6, 3.6, 10.8):
            columns.box(x, -19.1, 3.06, 1.1, 1.3, 3.7, label="central_portico_piers")
            columns.box(x, -19.1, 1.37, 1.45, 1.6, .32, label="portico_pier_plinths")
        column(columns, x, -19.05, 6.9, 5.0, "corinthian", .28)
    for x in (-7.2, 0, 7.2):
        arch(structure, trim, x, -19.08, 1.25, 6.1, 3.34, 6.5, .85)
        window(centre, x, -13.02, 1.4, 3.25, 4.35, arched=True)
    for x in (-9, -5.4, -1.8, 1.8, 5.4, 9):
        window(centre, x, -13.02, 7.05, 2.65, 4.65)
        balustrade(rails, (x - 1.48, -19.10), (x + 1.48, -19.10), 6.97, .72)
    for x in (-11.0, 11.0):
        column(columns, x, -15.25, 6.9, 5.0, "corinthian", .28)
        # The first-floor corner pilasters reinforce the projecting block.
        columns.box(x, -19.0, 9.5, .9, .8, 5.1, label="portico_corner_pilasters")
    cornice(trim, 0, -2.0, 22.6, 34.1, 12.30, 1.16)
    for x in (-10.9, -7.2, -3.6, 0, 3.6, 7.2, 10.9):
        columns.box(x, -19.08, 13.56, .8, .8, .83, label="central_parapet_piers")
        columns.lathe(x, -19.08, 13.98, [(0, .27), (.15, .29), (.3, .2), (.47, .12), (.69, 0)], label="roof_urn_finials", sides=12)
    for x in (-9.05, -5.4, -1.8, 1.8, 5.4, 9.05):
        rails.box(x, -19.06, 13.48, 2.9, .35, .5, label="pierced_appearance_parapet_panels")
        rails.box(x, -19.27, 13.48, 2.45, .05, .22, "stone", "parapet_recessed_panels")
    ring_band(trim, 0, -2.0, 22.6, 34.1, 14.0, .19, .24)

    # The recessed third stage and its distinct steep slate mansard roof.
    structure.box(0, 1.0, 16.0, 18.8, 20.0, 5.5, label="third_stage_tower")
    for front in (True, False):
        y = -9.025 if front else 11.025
        angle = 0 if front else math.pi
        for x in (-6.7, -3.35, 0, 3.35, 6.7):
            window(centre, x, y, 14.15, 2.1, 3.75, angle)
        for x in (-8.9, -5.05, -1.7, 1.7, 5.05, 8.9):
            columns.box(x, y - (.14 if front else -.14), 16.12, .39, .27, 4.65, label="tower_corinthian_pilasters")
            columns.box(x, y - (.15 if front else -.15), 18.25, .66, .42, .35, label="tower_pilaster_capitals")
    for side in (-1, 1):
        for y in (-5.7, -1.7, 2.3, 6.3):
            window(centre, side * 9.43, y, 14.15, 2.1, 3.8, side * math.pi / 2)
    cornice(trim, 0, 1, 18.8, 20, 18.70, 1.15)
    frustum(roof, 0, 1, 19.60, 19.8, 21.0, 15.0, 16.2, 3.65)
    frustum(roof, 0, 1, 23.25, 15.0, 16.2, 11.8, 13.0, .85, label="mansard_upper_slope")
    # Raised standing seams, spaced like slate courses, create readable roof detail.
    for i in range(1, 12):
        t = i / 12
        z = 19.60 + 3.65 * t
        w = 19.8 + (15.0 - 19.8) * t
        d = 21 + (16.2 - 21) * t
        ring_band(roof, 0, 1, w, d, z, .027, .025, "slate", "mansard_slate_courses")
    for x in (-6.3, 0, 6.3):
        y = -8.50
        structure.box(x, y, 20.75, 1.85, 1.6, 2.18, label="front_dormer_cheeks")
        window(centre, x, y - .82, 20.0, 1.2, 1.55)
        pediment(structure, trim, x, y - .89, 2.12, 21.88, .7, .3)
        frustum(roof, x, y + .1, 22.0, 2.2, 2.1, .7, .8, .65, label="dormer_roofs")
    for side in (-1, 1):
        for yy in (-3.2, 5.2):
            x = side * 8.5
            structure.box(x, yy, 20.75, 1.45, 1.85, 2.18, label="side_dormer_cheeks")
            window(centre, x + side * .74, yy, 20.0, 1.2, 1.55, side * math.pi / 2)
            frustum(roof, x, yy, 21.93, 1.7, 2.1, .8, .7, .65, label="side_dormer_caps")
    # Rooftop lantern/platform and unflagged pole; no controlled insignia reproduced.
    structure.box(0, 1, 24.35, 4.7, 5.0, .45, label="roof_lantern_base")
    for x in (-2.0, 2.0):
        for y in (-1.1, 3.1):
            columns.box(x, y, 25.0, .25, .25, 1.0, label="roof_lantern_posts")
    for y in (-1.15, 3.15):
        balustrade(rails, (-2.0, y), (2.0, y), 24.62, .7)
    for x in (-2.1, 2.1):
        balustrade(rails, (x, -1.15), (x, 3.15), 24.62, .7)
    columns.lathe(0, 1, 24.55, [(0, .11), (3.7, .055), (3.8, .025)], "metal", "unflagged_rooftop_pole", sides=12)

    # Rear elevation remains fully modelled for aerial exploration.
    for x in (-8.8, -4.4, 0, 4.4, 8.8):
        for zz in (1.65, 7.05):
            window(centre, x, 16.58, zz, 2.5, 3.9, math.pi)
    for z in (1.4, 6.68, 12.4):
        ring_band(trim, 0, 1.8, 22.5, 29.5, z, .22, .16)

    # Small original fittings: downpipes, wall lamps, drain shoes.
    for x in (-50.7, -40.2, -11.3, 11.3, 40.2, 50.7):
        columns.lathe(x, -7.2, 1.25, [(0, .07), (11, .07)], "metal", "rainwater_downpipes", sides=10)
        for z in (2, 5, 8, 11):
            columns.box(x, -7.2, z, .23, .2, .09, "metal", "downpipe_brackets")
    for x in (-9, -3.6, 3.6, 9):
        columns.beam((x, -13.3, 3.8), (x, -14.0, 4.2), .07, "metal", "lantern_brackets")
        columns.box(x, -14.0, 3.8, .33, .33, .60, "glass", "portico_lantern_glazing")
        for dx in (-.18, .18):
            for dy in (-.18, .18):
                columns.box(x + dx, -14 + dy, 3.8, .035, .035, .65, "metal", "lantern_frames")
        frustum(columns, x, -14, 4.1, .48, .48, .09, .09, .24, "metal", "lantern_caps")
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "SourceAssets" / "Architecture")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = build()
    manifest = {"license": "CC0-1.0", "units": "centimetres", "up_axis": "Z", "front_direction": "-Y",
                "origin": "building centre at local ground level", "unreal_import_uniform_scale": 1,
                "uv_scale": "1 UV repeat per metre; projected by face normal",
                "description": "Original public-exterior artistic reconstruction, not a surveyed or as-built model. No interiors, security infrastructure, official insignia, or downloaded assets.",
                "reference_pages": ["https://www.istana.gov.sg/visit-and-explore/buildings/", "https://www.roots.gov.sg/places/places-landing/Places/national-monuments/the-istana-and-sri-temasek"],
                "materials": MATERIALS, "parts": [part.write(args.output) for part in parts.values()]}
    with (args.output / "Istana.mtl").open("w", encoding="utf-8", newline="\n") as f:
        f.write("# CC0 material values; replace with UE PBR materials using architecture_manifest.json\n")
        for name, mat in MATERIALS.items():
            f.write("newmtl " + name + "\nKd %.4f %.4f %.4f\n" % mat["color"])
            f.write("Ka 0.03 0.03 0.03\nKs 0.1 0.1 0.1\nNs %.2f\nd 1.0\nillum 2\n\n" % (1 + (1 - mat["roughness"]) * 100))
    manifest["total_triangles"] = sum(p["triangles"] for p in manifest["parts"])
    manifest["bounds_cm"] = [[min(p["bounds_cm"][0][a] for p in manifest["parts"]) for a in range(3)], [max(p["bounds_cm"][1][a] for p in manifest["parts"]) for a in range(3)]]
    (args.output / "architecture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text("# Istana exterior source geometry\n\nOriginal procedural geometry dedicated to CC0-1.0. Regenerate with `python Tools/generate_istana.py` from the project root.\n\nImport all `Istana_*.obj` files at the same location, rotation zero, uniform scale 1. Units are centimetres, Z up, front -Y. Enable Nanite for these static exterior meshes where supported. Import materials by the six lowercase material names in `architecture_manifest.json`; its values describe the intended PBR appearance. UVs repeat once per metre.\n\nThis is an artistic architectural reconstruction of the publicly visible exterior, not a precise survey, digital twin, interior model, or representation of security arrangements. Official photographic references informed architectural shapes only; no reference photos, textures, logos, crests or presidential standard are included.\n\nReferences: [Istana buildings](https://www.istana.gov.sg/visit-and-explore/buildings/) and [National Heritage Board](https://www.roots.gov.sg/places/places-landing/Places/national-monuments/the-istana-and-sri-temasek).\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parts": len(parts), "triangles": manifest["total_triangles"], "bounds_cm": manifest["bounds_cm"], "bytes": sum(p["size_bytes"] for p in manifest["parts"])}, indent=2))


if __name__ == "__main__":
    main()
