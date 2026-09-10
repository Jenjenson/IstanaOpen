# Offline environment sources

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
