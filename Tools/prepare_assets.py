"""Collect only attributed CC0 scans and ODbL data from the previous local cache.

One-time bootstrap, not needed to run/build the prepared submission.
"""
from pathlib import Path
import shutil, json, hashlib
SRC = Path('D:/triad/TRIAD/SourceAssets')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'SourceAssets'
records=[]
def copy(src,dst,license,url):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    records.append(dict(path=dst.relative_to(ROOT).as_posix(),source=src.relative_to(SRC).as_posix(),license=license,url=url,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),bytes=dst.stat().st_size))
for name,base in [('island_tree_02','IstanaPublicViewExploreV4/Sources/Vegetation/assets/polyhaven'),('tree_small_02','IstanaPublicViewExploreV4/Sources/Vegetation/assets/polyhaven'),('shrub_04','IstanaPublicViewExploreV4/Sources/Vegetation/assets/polyhaven'),('grass_medium_01','IstanaPublicViewExploreV4/Sources/Vegetation/assets/polyhaven'),('pachira_aquatica_01','IstanaPublicViewExploreV5/Sources/Vegetation/polyhaven')]:
    folder=SRC/base/name
    for p in folder.rglob('*'):
        if p.is_file():copy(p,OUT/'Nature'/name/p.relative_to(folder),'CC0-1.0','https://polyhaven.com/a/'+name)
for p in (SRC/'IstanaPublicView/HeroMaterialsV3/Source/PolyHaven').glob('*'):
    if any(c in p.name for c in ['Diffuse','NormalDX','Roughness']):
        name=p.name[3:].rsplit('_',2)[0]
        copy(p,OUT/'Surfaces'/p.name,'CC0-1.0','https://polyhaven.com/a/'+name)
grass=SRC/'IstanaPublicViewExploreV4/Sources/Vegetation/assets/ambientcg/Grass001/extracted'
for p in grass.glob('*'):
    if any(c in p.name for c in ['Color','NormalDX','Roughness']) and p.suffix.lower() in ['.png','.jpg']:
        copy(p,OUT/'Surfaces'/p.name,'CC0-1.0','https://ambientcg.com/view?id=Grass001')
osm=SRC/'IstanaPublicViewExploreV3/Sources/geodata/osm_context_1km_2026-08-25.overpass.json'
copy(osm,OUT/'Geodata/osm_context.overpass.json','ODbL-1.0','https://www.openstreetmap.org/copyright')
(OUT/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Prepared',len(records),'files;',sum(x['bytes'] for x in records)//1000000,'MB')
print('Grass',list((OUT/'Surfaces').glob('Grass*')))
