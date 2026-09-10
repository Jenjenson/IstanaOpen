"""Build the original offline landscape and attributed OSM visual context.

Python 3.11+, standard library only. Internal dimensions are metres; OBJ and
scene.json locations are centimetres, Z up. No site imagery is consumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAT, LON, ROTATION = 1.30705, 103.8431, -64.5
RADIUS = 1100.0
SEED = 20260910
MATERIALS = {
    "Lawn": (0.19, 0.30, 0.10), "Asphalt": (0.10, 0.11, 0.12),
    "StonePaving": (0.55, 0.52, 0.45), "Concrete": (0.60, 0.60, 0.57),
    "UrbanIvory": (0.76, 0.73, 0.65), "UrbanWarm": (0.56, 0.48, 0.39),
    "UrbanCool": (0.48, 0.53, 0.55), "UrbanBrick": (0.43, 0.31, 0.24),
    "UrbanRoof": (0.28, 0.30, 0.29), "UrbanGlass": (0.12, 0.24, 0.28),
    "UrbanGlassDark": (0.08, 0.14, 0.17), "Trim": (0.82, 0.80, 0.73),
    "MetalDark": (0.08, 0.09, 0.08), "Water": (0.12, 0.28, 0.26),
    "FountainStone": (0.66, 0.66, 0.59), "Hedge": (0.13, 0.25, 0.08),
    "LampGlow": (1.0, 0.83, 0.58), "RoadMarking": (0.86, 0.85, 0.72),
}


def terrain_height(x, y):
    """Authored hill: flat central 130 m, smooth descent to -23 m at 600 m."""
    t = max(0.0, min(1.0, (math.hypot(x, y) - 130.0) / 470.0))
    return -23.0 * t * t * (3.0 - 2.0 * t)


def to_local(lat, lon):
    east = (lon - LON) * (math.pi / 180.0) * 6378137.0 * math.cos(math.radians(LAT))
    north = (lat - LAT) * (math.pi / 180.0) * 6378137.0
    a = math.radians(ROTATION)
    return east * math.cos(a) - north * math.sin(a), east * math.sin(a) + north * math.cos(a)


def area(poly):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1])) * 0.5


def cross2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def clean_polygon(points):
    out = []
    for p in points:
        if not out or math.dist(out[-1], p) > 0.05:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) < 0.05:
        out.pop()
    if len(out) < 3:
        return []
    # Remove exact straight-line intermediary vertices before ear clipping.
    out = [p for i, p in enumerate(out) if abs(cross2(out[i-1], p, out[(i+1) % len(out)])) > 0.0001]
    if area(out) < 0:
        out.reverse()
    return out


def inside(point, polygon):
    x, y = point
    odd = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            if x < (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1]) + a[0]:
                odd = not odd
    return odd


def triangulate(poly):
    """Ear clipping for simple CCW footprints; never span a concave roof with a fan."""
    idx = list(range(len(poly)))
    tris = []
    while len(idx) > 3:
        found = False
        for j, mid in enumerate(idx):
            prev, nxt = idx[j-1], idx[(j+1) % len(idx)]
            a, b, c = poly[prev], poly[mid], poly[nxt]
            if cross2(a, b, c) <= 1e-8:
                continue
            if any(cross2(a, b, poly[k]) >= -1e-8 and cross2(b, c, poly[k]) >= -1e-8
                   and cross2(c, a, poly[k]) >= -1e-8 for k in idx if k not in (prev, mid, nxt)):
                continue
            tris.append((prev, mid, nxt))
            del idx[j]
            found = True
            break
        if not found:
            return []
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


class Mesh:
    """Streaming OBJ writer with per-face normals and bounded output chunks."""
    def __init__(self, folder, label, triangle_limit=50000):
        self.folder, self.label, self.limit = folder, label, triangle_limit
        self.records, self.stream = [], None
        self.chunk, self.vertices, self.normals, self.triangles = 0, 0, 0, 0
        self.material = None
        self.materials = set()

    def _start(self):
        self.chunk += 1
        name = f"SM_Env_{self.label}_{self.chunk:02d}.obj"
        self.path = self.folder / name
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.stream.write("# Istana Open; centimetres; Z up; original geometry / attributed OSM where applicable\n")
        self.stream.write("mtllib environment.mtl\n")
        self.stream.write(f"o {self.path.stem}\n")
        self.vertices = self.normals = self.triangles = 0
        self.material = None
        self.materials = set()

    def close(self):
        if self.stream:
            self.stream.close()
            self.records.append(dict(path=self.path.name, category=self.label, triangles=self.triangles,
                                     vertices=self.vertices, materials=sorted(self.materials),
                                     bytes=self.path.stat().st_size,
                                     sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()))
            self.stream = None

    def face(self, points, material, uv=None):
        if len(points) < 3:
            return
        a, b, c = points[:3]
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        normal = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        length = math.sqrt(sum(q*q for q in normal))
        if length < 1e-9:
            return
        if any(not math.isfinite(q) for p in points for q in p):
            raise ValueError("Non-finite vertex")
        normal = tuple(q/length for q in normal)
        if self.stream and self.triangles + len(points)-2 > self.limit:
            self.close()
        if not self.stream:
            self._start()
        if self.material != material:
            self.stream.write(f"usemtl {material}\n")
            self.material = material
        self.materials.add(material)
        first = self.vertices + 1
        # World-scale UVs keep PBR texture scale stable across individual surfaces.
        axis = max(range(3), key=lambda i: abs(normal[i]))
        axes = [i for i in range(3) if i != axis]
        for j, p in enumerate(points):
            self.stream.write("v %.4f %.4f %.4f\n" % tuple(q*100.0 for q in p))
            coord = uv[j] if uv else (p[axes[0]]*0.25, p[axes[1]]*0.25)
            self.stream.write("vt %.5f %.5f\n" % coord)
        self.normals += 1
        self.stream.write("vn %.7f %.7f %.7f\n" % normal)
        self.stream.write("f " + " ".join(f"{i}/{i}/{self.normals}" for i in range(first, first+len(points))) + "\n")
        self.vertices += len(points)
        self.triangles += len(points)-2


def box(mesh, x, y, z, sx, sy, sz, material):
    p = [(x-sx/2,y-sy/2,z),(x+sx/2,y-sy/2,z),(x+sx/2,y+sy/2,z),(x-sx/2,y+sy/2,z)]
    top = [(a,b,z+sz) for a,b,_ in p]
    for i in range(4):
        j = (i+1) % 4
        mesh.face([p[i],p[j],top[j],top[i]], material)
    mesh.face(top, material)


def cylinder(mesh, x, y, z, radius, height, material, segments=20, top=True):
    points = [(x+radius*math.cos(i*math.tau/segments), y+radius*math.sin(i*math.tau/segments)) for i in range(segments)]
    for a,b in zip(points,points[1:]+points[:1]):
        mesh.face([(a[0],a[1],z),(b[0],b[1],z),(b[0],b[1],z+height),(a[0],a[1],z+height)],material)
        if top:
            mesh.face([(x,y,z+height),(a[0],a[1],z+height),(b[0],b[1],z+height)],material)


def disk(mesh, cx, cy, rx, ry, material, z_offset=0.02, segments=160):
    for i in range(segments):
        a,b = i*math.tau/segments,(i+1)*math.tau/segments
        p=[(cx,cy),(cx+rx*math.cos(a),cy+ry*math.sin(a)),(cx+rx*math.cos(b),cy+ry*math.sin(b))]
        mesh.face([(x,y,terrain_height(x,y)+z_offset) for x,y in p],material)


def ellipse_ring(mesh,cx,cy,rx,ry,width,material,z_offset=0.025,segments=200):
    for i in range(segments):
        a,b = i*math.tau/segments,(i+1)*math.tau/segments
        p=[(cx+rx*math.cos(a),cy+ry*math.sin(a)),
           (cx+(rx+width)*math.cos(a),cy+(ry+width)*math.sin(a)),
           (cx+(rx+width)*math.cos(b),cy+(ry+width)*math.sin(b)),
           (cx+rx*math.cos(b),cy+ry*math.sin(b))]
        mesh.face([(x,y,terrain_height(x,y)+z_offset) for x,y in p],material)


def ribbon(mesh, a, b, width, material, offset=0.04, spacing=12.0, lateral=0.0):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dy)
    if length < 0.05:
        return
    nx,ny=-dy/length,dx/length
    steps=max(1,math.ceil(length/spacing))
    for i in range(steps):
        s,t=i/steps,(i+1)/steps
        p=(a[0]+dx*s+nx*lateral,a[1]+dy*s+ny*lateral)
        q=(a[0]+dx*t+nx*lateral,a[1]+dy*t+ny*lateral)
        xy=[(p[0]+nx*width/2,p[1]+ny*width/2),(p[0]-nx*width/2,p[1]-ny*width/2),
            (q[0]-nx*width/2,q[1]-ny*width/2),(q[0]+nx*width/2,q[1]+ny*width/2)]
        mesh.face([(x,y,terrain_height(x,y)+offset) for x,y in xy],material)


def distance_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    d=dx*dx+dy*dy
    t=max(0.0,min(1.0,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d)) if d else 0.0
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


class Obstacles:
    def __init__(self):
        self.buildings=defaultdict(list)
        self.roads=defaultdict(list)
        self.trees=defaultdict(list)
        self.cell=30.0

    def keys(self,x0,y0,x1,y1):
        for i in range(math.floor(x0/self.cell),math.floor(x1/self.cell)+1):
            for j in range(math.floor(y0/self.cell),math.floor(y1/self.cell)+1):
                yield i,j

    def building(self,poly):
        for key in self.keys(min(x for x,y in poly)-8,min(y for x,y in poly)-8,
                             max(x for x,y in poly)+8,max(y for x,y in poly)+8):
            self.buildings[key].append(poly)

    def road(self,a,b,width):
        w=width/2+5.0
        for key in self.keys(min(a[0],b[0])-w,min(a[1],b[1])-w,max(a[0],b[0])+w,max(a[1],b[1])+w):
            self.roads[key].append((a,b,w))

    def free(self,x,y,tree_spacing=0):
        key=(math.floor(x/self.cell),math.floor(y/self.cell))
        for poly in self.buildings.get(key,[]):
            if inside((x,y),poly) or any(distance_segment((x,y),a,b)<6 for a,b in zip(poly,poly[1:]+poly[:1])):
                return False
        if any(distance_segment((x,y),a,b)<w for a,b,w in self.roads.get(key,[])):
            return False
        if tree_spacing:
            for near in self.keys(x-tree_spacing,y-tree_spacing,x+tree_spacing,y+tree_spacing):
                if any(math.hypot(x-a,y-b)<tree_spacing for a,b in self.trees.get(near,[])):
                    return False
        return True

    def tree(self,x,y):
        self.trees[(math.floor(x/self.cell),math.floor(y/self.cell))].append((x,y))


def numeric(value):
    if not value:
        return None
    match=re.search(r"[-+]?\d+(?:\.\d+)?",str(value))
    if not match:
        return None
    number=float(match.group())
    if "ft" in str(value) or "feet" in str(value):
        number*=0.3048
    return number


def build_context(data,meshes,obstacles):
    rng=random.Random(SEED+1)
    walls,roofs,glass,trim=(meshes[n] for n in ("ContextWalls","ContextRoofs","ContextGlass","ContextTrim"))
    records=[]
    skipped=Counter()
    for e in data["elements"]:
        tags=e.get("tags",{})
        if e["type"]!="way" or "building" not in tags or tags.get("building") in ("no","construction"):
            continue
        poly=clean_polygon([to_local(p["lat"],p["lon"]) for p in e.get("geometry",[])])
        if len(poly)<3 or abs(area(poly))<20:
            skipped["small_or_invalid"]+=1
            continue
        cx,cy=sum(x for x,y in poly)/len(poly),sum(y for x,y in poly)/len(poly)
        distance=math.hypot(cx,cy)
        if distance>1050 or any(math.hypot(x,y)<130 for x,y in poly):
            skipped["outside_scope_or_central_exclusion"]+=1
            continue
        triangles=triangulate(poly)
        if not triangles:
            skipped["non_simple_footprint"]+=1
            continue
        observed_height=numeric(tags.get("height"))
        levels=numeric(tags.get("building:levels"))
        if observed_height and observed_height>1:
            height=max(3.0,min(observed_height,240.0))
            basis="OSM height tag (mapping-grade, unverified)"
        elif levels and levels>0:
            height=max(3.5,min(levels*3.35,240.0))
            basis="Inferred metres from OSM building:levels at 3.35 m/floor"
        else:
            building_type=tags.get("building","")
            if building_type in ("apartments","hotel","commercial","office"):
                height=rng.uniform(24,72)
            elif abs(area(poly))>1800:
                height=rng.uniform(18,43)
            else:
                height=rng.uniform(7,22)
            basis="Authored plausible height; no OSM height/levels"
        base=max(terrain_height(x,y) for x,y in poly)
        top=base+height
        surface=rng.choices(["UrbanIvory","UrbanWarm","UrbanCool","UrbanBrick"],[5,2,3,1])[0]
        glass_material=rng.choice(["UrbanGlass","UrbanGlass","UrbanGlassDark"])
        floors=max(1,int(height/3.35))
        for a,b in zip(poly,poly[1:]+poly[:1]):
            dx,dy=b[0]-a[0],b[1]-a[1]
            length=math.hypot(dx,dy)
            if length<0.05:
                continue
            ux,uy=dx/length,dy/length
            nx,ny=uy,-ux
            walls.face([(a[0],a[1],terrain_height(*a)-0.4),(b[0],b[1],terrain_height(*b)-0.4),
                        (b[0],b[1],top),(a[0],a[1],top)],surface)
            # Window rectangles are modeled planes set just outside the wall.
            bay=3.7 if distance<650 else 4.8
            cols=min(34,max(0,int(length/bay)))
            for floor in range(floors):
                z=base+floor*(height/floors)+0.8
                window_height=min(2.15,height/floors-1.0)
                for col in range(cols):
                    u0=(col+0.5)*length/cols-min(1.1,length/cols*0.30)
                    u1=(col+0.5)*length/cols+min(1.1,length/cols*0.30)
                    p=(a[0]+ux*u0+nx*0.045,a[1]+uy*u0+ny*0.045)
                    q=(a[0]+ux*u1+nx*0.045,a[1]+uy*u1+ny*0.045)
                    glass.face([(p[0],p[1],z),(q[0],q[1],z),(q[0],q[1],z+window_height),(p[0],p[1],z+window_height)],glass_material)
            # Real geometry catches light at each second storey and on the roofline.
            for level in range(0,floors+1,2):
                z=base+min(level*(height/floors),height)-0.13
                if level==0:
                    z=base+0.25
                p=(a[0]+nx*0.12,a[1]+ny*0.12)
                q=(b[0]+nx*0.12,b[1]+ny*0.12)
                trim.face([(p[0],p[1],z),(q[0],q[1],z),(q[0],q[1],z+0.19),(p[0],p[1],z+0.19)],"Trim")
                trim.face([(p[0],p[1],z+0.19),(q[0],q[1],z+0.19),(b[0],b[1],z+0.19),(a[0],a[1],z+0.19)],"Trim")
        for ia,ib,ic in triangles:
            roofs.face([(poly[j][0],poly[j][1],top) for j in (ia,ib,ic)],"UrbanRoof")
        # A modest original rooftop housing breaks uniformly flat silhouettes.
        if abs(area(poly))>200 and inside((cx,cy),poly):
            box(roofs,cx,cy,top,3.0,3.5,1.5,"UrbanRoof")
        obstacles.building(poly)
        records.append(dict(osm_type="way",osm_id=e["id"],footprint_vertices=len(poly),
                            height_m=round(height,3),height_basis=basis,facade="Original generic window/trim design",
                            roof="Original flat roof approximation",height_tag=tags.get("height"),levels_tag=tags.get("building:levels")))
    return records,dict(skipped)


def build_roads(data,meshes,obstacles):
    records=[]
    excluded=Counter()
    roads,kerbs,marks=meshes["Roads"],meshes["Paving"],meshes["RoadMarkings"]
    widths={"motorway":14,"trunk":12,"primary":10,"secondary":9,"tertiary":7,"residential":6,"unclassified":6}
    for e in data["elements"]:
        tags=e.get("tags",{})
        typ=tags.get("highway")
        if e["type"]!="way" or typ not in widths:
            continue
        if tags.get("access") in ("private","no","permit","customers") or tags.get("tunnel") in ("yes","building_passage") or (numeric(tags.get("layer")) or 0)<0:
            excluded["private_or_underground"]+=1
            continue
        points=[to_local(p["lat"],p["lon"]) for p in e.get("geometry",[])]
        width=numeric(tags.get("width")) or widths[typ]
        width=max(4.0,min(width,20.0))
        segments=0
        for a,b in zip(points,points[1:]):
            # Resampling lets the exclusion boundary and authored hill stay continuous.
            n=max(1,math.ceil(math.dist(a,b)/12.0))
            for i in range(n):
                p=tuple(a[j]+(b[j]-a[j])*i/n for j in range(2))
                q=tuple(a[j]+(b[j]-a[j])*(i+1)/n for j in range(2))
                if not (138<math.hypot(*p)<1085 and 138<math.hypot(*q)<1085):
                    continue
                ribbon(roads,p,q,width,"Asphalt")
                ribbon(kerbs,p,q,1.8,"StonePaving",offset=0.17,lateral=width/2+0.9)
                ribbon(kerbs,p,q,1.8,"StonePaving",offset=0.17,lateral=-width/2-0.9)
                if width>=8 and i%2==0:
                    end=tuple(p[j]+(q[j]-p[j])*0.5 for j in range(2))
                    ribbon(marks,p,end,0.12,"RoadMarking",offset=0.055)
                obstacles.road(p,q,width+3.6)
                segments+=1
        if segments:
            records.append(dict(osm_id=e["id"],highway=typ,width_m=width,width_basis="OSM tag" if numeric(tags.get("width")) else "Authored by road class",segments=segments))
    return records,dict(excluded)


def build_terrain(mesh):
    radii=list(range(10,301,10))+list(range(320,1101,20))
    segments=256
    previous=0
    for radius in radii:
        for i in range(segments):
            a,b=i*math.tau/segments,(i+1)*math.tau/segments
            p=(radius*math.cos(a),radius*math.sin(a))
            q=(radius*math.cos(b),radius*math.sin(b))
            if previous==0:
                points=[(0.0,0.0,0.0),(p[0],p[1],terrain_height(*p)),(q[0],q[1],terrain_height(*q))]
            else:
                u=(previous*math.cos(a),previous*math.sin(a))
                v=(previous*math.cos(b),previous*math.sin(b))
                points=[(u[0],u[1],terrain_height(*u)),(p[0],p[1],terrain_height(*p)),
                        (q[0],q[1],terrain_height(*q)),(v[0],v[1],terrain_height(*v))]
            mesh.face(points,"Lawn")
        previous=radius
    # Downward outer skirt stops the finite landscape showing an exposed paper edge.
    for i in range(segments):
        a,b=i*math.tau/segments,(i+1)*math.tau/segments
        p=(RADIUS*math.cos(a),RADIUS*math.sin(a))
        q=(RADIUS*math.cos(b),RADIUS*math.sin(b))
        mesh.face([(p[0],p[1],-70),(q[0],q[1],-70),(q[0],q[1],-23),(p[0],p[1],-23)],"Lawn")


def build_horizon(mesh):
    """Original distant ground underlay; no geographic content beyond the AOI."""
    segments=256
    for i in range(segments):
        a,b=i*math.tau/segments,(i+1)*math.tau/segments
        mesh.face([(RADIUS*math.cos(a),RADIUS*math.sin(a),-23.0),
                   (5000*math.cos(a),5000*math.sin(a),-23.0),
                   (5000*math.cos(b),5000*math.sin(b),-23.0),
                   (RADIUS*math.cos(b),RADIUS*math.sin(b),-23.0)],"Lawn")


def build_formal_landscape(meshes,obstacles):
    paving,roads,details,water=(meshes[n] for n in ("Paving","Roads","LandscapeDetails","Water"))
    # Main building is centered at the origin with its formal front facing -Y.
    box(paving,0,-30,-0.1,106,16,0.20,"StonePaving")
    ellipse_ring(roads,0,-110,49,39,6,"Asphalt",z_offset=0.05)
    ellipse_ring(paving,0,-110,47.8,37.8,1.2,"StonePaving",z_offset=0.14)
    ellipse_ring(paving,0,-110,55,45,1.8,"StonePaving",z_offset=0.14)
    ribbon(paving,(0,-38),(0,-71),10,"StonePaving",offset=0.12)
    # The east and west garden walks and sweeping approach are original design.
    for sign in (-1,1):
        walk=[(sign*65,-30),(sign*75,-70),(sign*83,-130),(sign*95,-200),(sign*145,-260)]
        for a,b in zip(walk,walk[1:]):
            ribbon(paving,a,b,3.0,"StonePaving",offset=0.1)
            obstacles.road(a,b,3.0)
        drive=[(sign*52,-125),(sign*67,-156),(sign*98,-190),(sign*140,-208),(sign*182,-224)]
        for a,b in zip(drive,drive[1:]):
            ribbon(roads,a,b,6.0,"Asphalt",offset=0.06)
            ribbon(paving,a,b,1.2,"StonePaving",offset=0.14,lateral=3.6)
            ribbon(paving,a,b,1.2,"StonePaving",offset=0.14,lateral=-3.6)
            obstacles.road(a,b,8.4)
        for y in range(-39,-22,4):
            box(details,sign*55,y,0,1.25,3.7,0.80,"Hedge")
        for y in (-55,-88,-128,-168):
            x=sign*(72 if y>-100 else 89)
            ground=terrain_height(x,y)
            cylinder(details,x,y,ground,0.14,3.7,"MetalDark",12)
            cylinder(details,x,y,ground+3.55,0.40,0.16,"MetalDark",16)
            cylinder(details,x,y,ground+3.38,0.22,0.22,"LampGlow",12)
    # Fountain bowl: low stone outer wall, raised rim, shallow water, central tiers.
    fx,fy=0,-105
    ground=terrain_height(fx,fy)
    cylinder(details,fx,fy,ground+0.05,7.1,0.43,"FountainStone",96,top=False)
    ellipse_ring(details,fx,fy,6.55,6.55,0.55,"FountainStone",z_offset=0.54,segments=128)
    disk(water,fx,fy,6.55,6.55,"Water",z_offset=0.38)
    cylinder(details,fx,fy,ground+0.36,1.10,1.25,"FountainStone",40)
    cylinder(details,fx,fy,ground+1.58,2.4,0.22,"FountainStone",64)
    cylinder(details,fx,fy,ground+1.77,0.38,1.15,"FountainStone",28)
    cylinder(details,fx,fy,ground+2.84,1.14,0.17,"FountainStone",48)
    disk(water,fx,fy,2.23,2.23,"Water",z_offset=1.81)
    for i in range(32):
        a=i*math.tau/32
        x,y=57.4*math.cos(a),-110+47.4*math.sin(a)
        z=terrain_height(x,y)+0.12
        cylinder(details,x,y,z,0.11,0.62,"MetalDark",10)
        cylinder(details,x,y,z+0.58,0.13,0.05,"Trim",10)
    # Low benches are deliberate garden furniture, not operational equipment.
    for sign in (-1,1):
        for y in (-91,-130):
            x=sign*64
            z=terrain_height(x,y)
            box(details,x,y,z+0.45,2.7,0.58,0.13,"UrbanWarm")
            for dx in (-0.9,0.9):
                box(details,x+dx,y,z,0.12,0.46,0.48,"MetalDark")


def formal_clear(x,y,trees=True):
    if abs(x)<68 and -185<y<65:
        return False
    if math.hypot(x,y)<72:
        return False
    if ((x/63)**2+((y+110)/53)**2)<1.0:
        return False
    return True


def scatter_vegetation(obstacles,park_polygon):
    rng=random.Random(SEED+2)
    instances=[]
    counts=Counter()
    def add(mesh,x,y,height):
        yaw=rng.uniform(0,360)
        instances.append(dict(mesh=mesh,location=[round(x*100,3),round(y*100,3),round(terrain_height(x,y)*100,3)],
                              yaw=round(yaw,3),target_height_cm=round(height*100,2)))
        counts[mesh]+=1
    # Alternating loose groves preserve a clear arrival view and distant city skyline.
    for _ in range(50000):
        if sum(counts[n] for n in ("island_tree_02","tree_small_02"))>=520:
            break
        maximum=525 if rng.random()<0.82 else 1030
        r=math.sqrt(rng.uniform(90**2,maximum**2))
        a=rng.uniform(0,math.tau)
        x,y=r*math.cos(a),r*math.sin(a)
        if not formal_clear(x,y) or not obstacles.free(x,y,tree_spacing=15.5):
            continue
        mesh="island_tree_02" if rng.random()<0.62 else "tree_small_02"
        add(mesh,x,y,rng.uniform(14,24))
        obstacles.tree(x,y)
    # Added canopy is confined to the mapped Istana park, not empty city parcels.
    # Keep the existing outer-context placement stable; enrich the rear backdrop
    # first, then the remaining estate. Individual tree positions remain artistic.
    x0,x1=min(x for x,y in park_polygon),max(x for x,y in park_polygon)
    y0,y1=min(y for x,y in park_polygon),max(y for x,y in park_polygon)
    extra=0
    for rear_only,target,attempt_limit in [(True,120,18000),(False,430,40000)]:
        for _ in range(attempt_limit):
            if extra>=target:
                break
            if rear_only:
                x,y=rng.uniform(-240,180),rng.uniform(80,y1)
            else:
                x,y=rng.uniform(x0,x1),rng.uniform(y0,y1)
            if math.hypot(x,y)<130 or not inside((x,y),park_polygon):
                continue
            if not formal_clear(x,y) or not obstacles.free(x,y,tree_spacing=9.5):
                continue
            mesh="island_tree_02" if rng.random()<0.70 else "tree_small_02"
            add(mesh,x,y,rng.uniform(16,25))
            obstacles.tree(x,y)
            extra+=1
    for mesh,count,lo,hi in [("pachira_aquatica_01",170,3.2,6.0),("shrub_04",260,0.8,1.6),("grass_medium_01",1500,0.20,0.48)]:
        admitted=0
        for _ in range(count*100):
            if admitted>=count:
                break
            r=math.sqrt(rng.uniform(78**2,(380 if mesh!="grass_medium_01" else 270)**2))
            a=rng.uniform(0,math.tau)
            x,y=r*math.cos(a),r*math.sin(a)
            if not formal_clear(x,y,False) or not obstacles.free(x,y):
                continue
            add(mesh,x,y,rng.uniform(lo,hi))
            admitted+=1
    # Intentional flowering/foliage borders on both sides of the forecourt.
    for sign in (-1,1):
        for i in range(12):
            x,y=sign*(56+rng.uniform(-0.6,0.6)),-23-i*1.4
            add("shrub_04",x,y,rng.uniform(0.55,1.0))
    return instances,dict(counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"SourceAssets/Geodata/osm_context.overpass.json")
    parser.add_argument("--output",type=Path,default=ROOT/"SourceAssets/Environment")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.input.read_text(encoding="utf-8-sig"))
    categories=["Terrain","Horizon","Roads","Paving","RoadMarkings","ContextWalls","ContextRoofs","ContextGlass","ContextTrim","LandscapeDetails","Water"]
    meshes={name:Mesh(args.output,name) for name in categories}
    obstacles=Obstacles()
    print("Creating original terrain and formal landscape",flush=True)
    build_terrain(meshes["Terrain"])
    build_horizon(meshes["Horizon"])
    build_formal_landscape(meshes,obstacles)
    print("Creating attributed surrounding buildings",flush=True)
    buildings,building_exclusions=build_context(data,meshes,obstacles)
    print("Creating attributed roads",flush=True)
    roads,road_exclusions=build_roads(data,meshes,obstacles)
    print("Scattering original landscape vegetation",flush=True)
    park_way=next(e for e in data["elements"] if e["type"]=="way" and e["id"]==41662337)
    park_polygon=clean_polygon([to_local(p["lat"],p["lon"]) for p in park_way["geometry"]])
    vegetation,counts=scatter_vegetation(obstacles,park_polygon)
    for mesh in meshes.values():
        mesh.close()
    records=[record for mesh in meshes.values() for record in mesh.records]
    for record in records:
        if record["category"]=="Horizon":
            record.update(visual_only=True,collision=False,claim="Original horizon underlay; no mapped data")
    mtl=[]
    for name,color in MATERIALS.items():
        mtl.extend([f"newmtl {name}","Kd %.4f %.4f %.4f"%color,"Ks 0.12 0.12 0.12","Ns 32","d 1","illum 2",""])
    (args.output/"environment.mtl").write_text("\n".join(mtl),encoding="utf-8")
    scene=dict(schema="istana-open.environment.v1",units="centimetres",up_axis="Z",front_axis="-Y",seed=SEED,
               georeference=dict(latitude=LAT,longitude=LON,east_north_rotation_degrees=ROTATION),
               terrain=dict(type="Original artistic hill",central_flat_radius_m=130,outer_level_m=-23,outer_level_radius_m=600,radius_m=RADIUS),
               background=dict(type="Original visual horizon underlay",inner_radius_m=RADIUS,outer_radius_m=5000,
                               elevation_m=-23,geographic_data=False,collision=False),
               meshes=records,instances=vegetation,vegetation_counts=counts,
               vegetation_layout=dict(additional_canopy_boundary_osm_way=41662337,
                                      additional_canopy_min_hero_radius_m=130,
                                      added_canopy="Original tree positions confined to mapped Domain of the Istana park; denser rear garden backdrop"),
               material_colors={k:list(v) for k,v in MATERIALS.items()},
               source_accuracy="Public exterior approximation; original landscape, inferred facades/heights where not mapped; not survey-controlled")
    (args.output/"scene.json").write_text(json.dumps(scene,indent=2),encoding="utf-8")
    provenance=dict(schema="istana-open.osm-environment-provenance.v1",licence="ODbL-1.0",
                    attribution="© OpenStreetMap contributors",licence_url="https://www.openstreetmap.org/copyright",
                    input_path=args.input.relative_to(ROOT).as_posix(),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    osm_snapshot_timestamp=data.get("osm3s",{}).get("timestamp_osm_base"),
                    coordinate_transform=scene["georeference"],building_count=len(buildings),road_way_count=len(roads),
                    height_basis_counts=dict(Counter(b["height_basis"] for b in buildings)),buildings=buildings,roads=roads,
                    vegetation_boundary=dict(osm_type="way",osm_id=41662337,name="Domain of the Istana",use="Boundary for additional original tree placement only; no fence geometry or individual-tree location claim"),
                    artistic_horizon=dict(outer_radius_m=5000,mapped_context_radius_m=RADIUS,
                                          claim="Original visual ground underlay only; no mapped buildings or site data beyond context radius"),
                    building_exclusions=building_exclusions,road_exclusions=road_exclusions,
                    caveats=["Mapped footprints and tags are not survey data.","All facades, roof details, untaged heights, road widths without tags, and formal landscape are artistic interpretations.",
                             "Terrain is entirely authored, not a DEM or measured elevation source.","No Google/OneMap imagery, operational information, interior, or access-control geometry used."])
    (args.output/"context_provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding="utf-8")
    readme="""# Offline environment sources

Regenerate with `python Tools/generate_environment.py` from the project root.
The generator uses only the bundled ODbL OpenStreetMap extract and original
geometry. Output OBJ coordinates are standard right-handed, Z-up centimetres.
`scene.json` supplies the same coordinates for vegetation and names every mesh.
The main building sits at (0,0,0), facing -Y. The formal landscape is original.

All buildings beyond the 130 m central exclusion use approximate mapped
footprints. Windows, facade trims, roofs and fallback heights are authored;
`context_provenance.json` records every height decision and retained OSM ID.
The smooth hill is entirely artistic and is not a measured terrain model.
No source images or downloaded commercial map tiles are used.
The separate 512-triangle Horizon mesh extends flat ground from 1.1 km to 5 km
only to hide the finite landscape edge. It contains no geographic data and
must have collision disabled; the mapped context remains limited to 1.1 km.

Geographic data and its adapted database remain ODbL-1.0:
© OpenStreetMap contributors — https://www.openstreetmap.org/copyright
Original environment code and independent designed landscape follow the root
project licence. Nature meshes referenced by scene.json are separately CC0.

OBJ UVs repeat in world-scale metres and include explicit face normals. Each
chunk is limited to 50,000 triangles; Unreal may group/import these separately.
Generic foliage is not a measured botanical inventory.
"""
    (args.output/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps(dict(meshes=len(records),triangles=sum(r["triangles"] for r in records),
                          total_mb=round(sum(r["bytes"] for r in records)/1e6,2),
                          largest_file_mb=round(max(r["bytes"] for r in records)/1e6,2),
                          buildings=len(buildings),roads=len(roads),vegetation=counts),indent=2),flush=True)


if __name__=="__main__":
    main()
