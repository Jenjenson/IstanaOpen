"""UE 5.5 editor Python: import the open sources and build the offline scene.

Run with -ExecutePythonScript=... -unattended -RenderOffscreen. Reuses completed
imports; the map is deterministically rebuilt. See build.ps1.
"""
from pathlib import Path
import json, math, traceback, shutil, hashlib
import unreal as u
ROOT=Path(u.Paths.project_dir()).resolve()
SRC=ROOT/'SourceAssets'
EAL=u.EditorAssetLibrary
AT=u.AssetToolsHelpers.get_asset_tools()
SML=u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
ACT=u.get_editor_subsystem(u.EditorActorSubsystem)
LEVEL=u.get_editor_subsystem(u.LevelEditorSubsystem)
MEL=u.MaterialEditingLibrary
REPORT={'meshes':{},'materials':{},'instances':{}}
STATE_FILE=ROOT/'Saved/import-state.json'
IMPORT_STATE=json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
def log(s): u.log('ISTANA_BUILD '+str(s))
def node(m,cls,**props):
    n=MEL.create_material_expression(m,cls)
    for k,v in props.items():n.set_editor_property(k,v)
    return n
def bind(n,prop,out=''):
    if not MEL.connect_material_property(n,out,prop):raise RuntimeError('Material connection failed')
def texture(path,normal=False,linear=False):
    dest='/Game/Open/Textures/'+path.stem
    t=u.load_asset(dest)
    if not t:
        task=u.AssetImportTask();task.filename=str(path);task.destination_path='/Game/Open/Textures';task.automated=True;task.save=True
        AT.import_asset_tasks([task]);t=u.load_asset(dest)
    if not t:raise RuntimeError('Texture import failed '+str(path))
    if normal:
        t.set_editor_property('compression_settings',u.TextureCompressionSettings.TC_NORMALMAP)
        t.set_editor_property('srgb',False)
        t.set_editor_property('flip_green_channel','nor_gl' in path.name)
    elif linear:t.set_editor_property('srgb',False)
    EAL.save_loaded_asset(t)
    return t
def material(name,color=(.5,.5,.5),rough=.7,metal=0,files=None,world=False,foliage=False,normal_strength=1.0):
    dest='/Game/Open/Materials/M_'+name
    m=u.load_asset(dest)
    if m:return m
    m=AT.create_asset('M_'+name,'/Game/Open/Materials',u.Material,u.MaterialFactoryNew())
    files=files or {}
    if foliage:
        m.set_editor_property('two_sided',True)
        m.set_editor_property('shading_model',u.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    if 'opacityMask' in files:
        m.set_editor_property('blend_mode',u.BlendMode.BLEND_MASKED)
        m.set_editor_property('opacity_mask_clip_value',.25)
    m.set_editor_property('used_with_instanced_static_meshes',True)
    m.set_editor_property('used_with_nanite',True)
    uv=None
    if world:
        pos=node(m,u.MaterialExpressionWorldPosition)
        pin=u.CustomInput();pin.set_editor_property('input_name','Position')
        uv=node(m,u.MaterialExpressionCustom,code='return Position.xy * 0.002;',output_type=u.CustomMaterialOutputType.CMOT_FLOAT2,inputs=[pin])
        MEL.connect_material_expressions(pos,'',uv,'Position')
    samples={}
    for role,path in files.items():
        if role=='mask':continue
        normal=role in ['normalGL','normalDX']
        n=node(m,u.MaterialExpressionTextureSample,texture=texture(path,normal,role in ['roughness','opacityMask']))
        if normal:n.set_editor_property('sampler_type',u.MaterialSamplerType.SAMPLERTYPE_NORMAL)
        elif role in ['roughness','opacityMask']:n.set_editor_property('sampler_type',u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
        if uv:MEL.connect_material_expressions(uv,'',n,'UVs')
        samples[role]=n
    if 'baseColor' in samples:
        col=samples['baseColor'];out='RGB'
    else:
        col=node(m,u.MaterialExpressionConstant3Vector,constant=u.LinearColor(*color,1));out=''
    bind(col,u.MaterialProperty.MP_BASE_COLOR,out)
    if foliage:
        sub=node(m,u.MaterialExpressionMultiply)
        fac=node(m,u.MaterialExpressionConstant,r=.35)
        MEL.connect_material_expressions(col,out,sub,'A');MEL.connect_material_expressions(fac,'',sub,'B')
        bind(sub,u.MaterialProperty.MP_SUBSURFACE_COLOR)
    if 'roughness' in samples:bind(samples['roughness'],u.MaterialProperty.MP_ROUGHNESS,'R')
    else:bind(node(m,u.MaterialExpressionConstant,r=rough),u.MaterialProperty.MP_ROUGHNESS)
    bind(node(m,u.MaterialExpressionConstant,r=metal),u.MaterialProperty.MP_METALLIC)
    for role,prop,out in [('normalGL',u.MaterialProperty.MP_NORMAL,'RGB'),('normalDX',u.MaterialProperty.MP_NORMAL,'RGB'),('opacityMask',u.MaterialProperty.MP_OPACITY_MASK,'R')]:
        if role in samples:
            if role.startswith('normal') and normal_strength!=1.0:
                pin=u.CustomInput();pin.set_editor_property('input_name','N')
                reduced=node(m,u.MaterialExpressionCustom,code='return normalize(float3(N.xy * '+str(normal_strength)+', max(N.z,0.2)));',inputs=[pin],output_type=u.CustomMaterialOutputType.CMOT_FLOAT3)
                MEL.connect_material_expressions(samples[role],'RGB',reduced,'N');bind(reduced,prop)
            else:bind(samples[role],prop,out)
    MEL.recompile_material(m);EAL.save_loaded_asset(m)
    REPORT['materials'][name]=list(files)
    return m
def surfaces():
    f=SRC/'Surfaces'
    mats={}
    def pbr(key,stem,world=False):
        return material(key,files={'baseColor':f/(stem+'_Diffuse_4k.jpg'),'normalDX':f/(stem+'_NormalDX_4k.jpg'),'roughness':f/(stem+'_Roughness_4k.jpg')},world=world)
    mats['plaster']=material('painted_lime_plaster',(.82,.80,.74),.70,files={'normalDX':f/'PH_white_plaster_02_NormalDX_4k.jpg'},normal_strength=.16)
    mats['stone']=material('pale_architectural_stone',(.55,.56,.51),.72,files={'normalDX':f/'PH_rock_01_NormalDX_4k.jpg'},normal_strength=.23)
    mats['slate']=material('charcoal_roof_slate',(.048,.059,.065),.69,files={'normalDX':f/'PH_roof_slates_03_NormalDX_4k.jpg','roughness':f/'PH_roof_slates_03_Roughness_4k.jpg'},normal_strength=.52)
    mats['asphalt']=pbr('asphalt_scan','PH_asphalt_02',world=True)
    mats['paving']=pbr('paving_scan','PH_rock_01',world=True)
    mats['dark_wood']=material('dark_wood',(.035,.046,.045),.54,files={'normalDX':f/'PH_fine_grained_wood_NormalDX_4k.jpg'})
    mats['grass']=material('grass',files={'baseColor':f/'Grass001_2K-JPG_Color.jpg','normalDX':f/'Grass001_2K-JPG_NormalDX.jpg','roughness':f/'Grass001_2K-JPG_Roughness.jpg'},world=True)
    for k,col,r,me in [('glass',(.065,.15,.19),.16,.65),('metal',(.09,.11,.11),.38,.8),('asphalt',(.055,.065,.067),.86,0),('paving',(.43,.43,.40),.84,0),('concrete',(.48,.49,.46),.82,0),('white',(.8,.79,.71),.75,0),('water',(.025,.16,.18),.12,.5),('soil',(.075,.044,.022),.98,0),('hedge',(.07,.17,.024),.93,0),('gold',(.55,.36,.10),.25,.75)]:
        if k not in mats:mats[k]=material(k,col,r,me)
    for k,col in [('urbanivory',(.62,.59,.51)),('urbanwarm',(.48,.36,.25)),('urbancool',(.35,.41,.43)),('urbanbrick',(.36,.16,.10)),('urbanroof',(.17,.20,.21)),('urbanglassdark',(.035,.08,.10))]:
        mats[k]=material(k,col,.24 if 'glass' in k else .78,.45 if 'glass' in k else 0)
    for alias,key in {'lawn':'grass','stonepaving':'paving','urbanglass':'glass','trim':'white','metaldark':'metal','fountainstone':'stone','lampglow':'white'}.items():mats[alias]=mats[key]
    return mats
def mirror_obj(path):
    """FBX's final RH->LH conversion negates Y even with axis conversion off."""
    out=ROOT/'Saved/ImportStaging'/path.name;out.parent.mkdir(parents=True,exist_ok=True)
    lines=[]
    for line in path.read_text().splitlines():
        p=line.split()
        if p and p[0] in ['v','vn']:
            p[2]=str(-float(p[2]));line=' '.join(p)
        elif p and p[0]=='f':line='f '+' '.join(reversed(p[1:]))
        elif p and p[0]=='mtllib':
            source=path.parent/' '.join(p[1:])
            if source.exists():shutil.copy2(source,out.parent/source.name)
        lines.append(line)
    out.write_text('\n'.join(lines))
    return out
def import_mesh(path,dest,name=None,combine=True):
    name=name or path.stem
    fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()
    source_key=str(path.resolve())
    existing=u.load_asset(dest+'/'+name)
    if existing and IMPORT_STATE.get(source_key,fingerprint)==fingerprint:return [existing]
    task=u.AssetImportTask();task.filename=str(mirror_obj(path) if path.suffix.lower()=='.obj' else path)
    task.destination_path=dest;task.destination_name=name if combine else '';task.automated=True;task.save=True;task.replace_existing=bool(existing)
    opt=u.FbxImportUI();opt.import_mesh=True;opt.import_as_skeletal=False;opt.import_materials=False;opt.import_textures=False;opt.automated_import_should_detect_type=False;opt.mesh_type_to_import=u.FBXImportType.FBXIT_STATIC_MESH
    data=opt.static_mesh_import_data;data.combine_meshes=combine;data.auto_generate_collision=False;data.generate_lightmap_u_vs=False
    data.convert_scene=path.suffix.lower()!='.obj';data.convert_scene_unit=True
    data.normal_import_method=u.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS
    task.options=opt;AT.import_asset_tasks([task])
    assets=[u.load_asset(p) for p in task.imported_object_paths]
    assets=[a for a in assets if isinstance(a,u.StaticMesh)]
    if not assets:raise RuntimeError('No mesh imported '+str(path))
    IMPORT_STATE[source_key]=fingerprint
    STATE_FILE.write_text(json.dumps(IMPORT_STATE,indent=2))
    return assets
def mesh_settings(mesh,nanite=True):
    build=SML.get_lod_build_settings(mesh,0)
    if not build.recompute_tangents:
        build.recompute_tangents=True
        SML.set_lod_build_settings(mesh,0,build)
    ns=mesh.get_editor_property('nanite_settings')
    if ns.enabled==nanite and abs(ns.fallback_percent_triangles-.02)<.0001:
        EAL.save_loaded_asset(mesh)
        return
    ns.enabled=nanite;ns.preserve_area=True
    ns.fallback_percent_triangles=.02;ns.fallback_relative_error=1
    SML.set_nanite_settings(mesh,ns,apply_changes=True)
    EAL.save_loaded_asset(mesh)
def nature():
    bindings=json.loads((SRC/'Nature/material_bindings.json').read_text())
    bindings['pachira_aquatica_01']={key:{role:SRC/'Nature/pachira_aquatica_01/textures'/file for role,file in files.items()} for key,files in { 'pachira_aquatica_01_bark':{'baseColor':'pachira_aquatica_01_bark_diff_2k.png','normalGL':'pachira_aquatica_01_bark_nor_gl_2k.png','roughness':'pachira_aquatica_01_bark_rough_2k.png'},'pachira_aquatica_01_leaves':{'baseColor':'pachira_aquatica_01_leaves_diff_2k.png','normalGL':'pachira_aquatica_01_leaves_nor_gl_2k.png','roughness':'pachira_aquatica_01_leaves_rough_2k.png','opacityMask':'pachira_aquatica_01_leaves_alpha_2k.png'}}.items()}
    results={}
    for key,slots in bindings.items():
        log('nature '+key)
        mats={}
        for slot,files in slots.items():
            paths={r:(p if isinstance(p,Path) else SRC/'Nature'/key/'textures'/p) for r,p in files.items()}
            mats[slot]=material(slot,files=paths,foliage='opacityMask' in paths)
        dest='/Game/Open/Nature/'+key
        existing=EAL.list_assets(dest,recursive=False,include_folder=False)
        assets=[u.load_asset(x) for x in existing];assets=[a for a in assets if isinstance(a,u.StaticMesh)]
        if not assets:assets=import_mesh(next((SRC/'Nature'/key).glob('*.fbx')),dest,combine=False)
        for mesh in assets:
            for i,slot in enumerate(mesh.static_materials):
                sn=str(slot.get_editor_property('imported_material_slot_name'))
                if sn not in mats:
                    raise RuntimeError('Unknown nature material '+key+' '+sn+' expected '+str(list(mats)))
                mesh.set_material(i,mats[sn])
            mesh_settings(mesh,True)
            bounds=mesh.get_bounds()
            REPORT['meshes'][mesh.get_path_name()]={'extent':[bounds.box_extent.x,bounds.box_extent.y,bounds.box_extent.z],'origin':[bounds.origin.x,bounds.origin.y,bounds.origin.z],'slots':[str(s.get_editor_property('imported_material_slot_name')) for s in mesh.static_materials]}
        results[key]=assets
        u.SystemLibrary.collect_garbage()
    return results
def geometry(mats):
    results=[]
    for folder in ['Architecture','Environment']:
        for path in sorted((SRC/folder).rglob('*.obj')):
            log('geometry '+str(path))
            mesh=import_mesh(path,'/Game/Open/'+folder)[0]
            for i,slot in enumerate(mesh.static_materials):
                key=str(slot.get_editor_property('imported_material_slot_name')).lower()
                if key not in mats:
                    if any(s in key for s in ['wall','facade','plaster','trim']):key='plaster'
                    elif any(s in key for s in ['roof','slate']):key='slate'
                    elif any(s in key for s in ['window','glaz','glass']):key='glass'
                    elif any(s in key for s in ['lawn','terrain','grass']):key='grass'
                    elif any(s in key for s in ['road','asphalt']):key='asphalt'
                    elif any(s in key for s in ['stone','step','pav']):key='stone'
                    elif any(s in key for s in ['wood','shutter']):key='dark_wood'
                    else:raise RuntimeError('Unknown material '+str(slot.get_editor_property('imported_material_slot_name'))+' in '+str(path))
                mesh.set_material(i,mats[key])
            mesh_settings(mesh,True)
            REPORT['meshes'][mesh.get_path_name()]={'extent':[mesh.get_bounds().box_extent.x,mesh.get_bounds().box_extent.y,mesh.get_bounds().box_extent.z],'origin':[mesh.get_bounds().origin.x,mesh.get_bounds().origin.y,mesh.get_bounds().origin.z]}
            points=[tuple(map(float,line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith('v ')]
            bounds=mesh.get_bounds()
            actual_min=[bounds.origin.x-bounds.box_extent.x,bounds.origin.y-bounds.box_extent.y,bounds.origin.z-bounds.box_extent.z]
            actual_max=[bounds.origin.x+bounds.box_extent.x,bounds.origin.y+bounds.box_extent.y,bounds.origin.z+bounds.box_extent.z]
            for axis in range(3):
                if abs(min(p[axis] for p in points)-actual_min[axis])>2 or abs(max(p[axis] for p in points)-actual_max[axis])>2:
                    raise RuntimeError('Imported axis/unit bounds mismatch '+str(path)+' '+str((actual_min,actual_max)))
            results.append(mesh)
    return results
def spawn(cls,label,loc=(0,0,0),rot=(0,0,0)):
    log('spawn '+label)
    a=ACT.spawn_actor_from_class(cls,u.Vector(*loc),u.Rotator(pitch=rot[0],yaw=rot[1],roll=rot[2]));a.set_actor_label(label);return a
def make_level(meshes,plants):
    map_path='/Game/Maps/Istana'
    if EAL.does_asset_exist(map_path):
        if not LEVEL.load_level(map_path):raise RuntimeError('Cannot reload generated map')
        for old in ACT.get_all_level_actors():
            if not isinstance(old,u.WorldSettings) and old.get_name()!='DefaultPhysicsVolume_0':ACT.destroy_actor(old)
    elif not LEVEL.new_level(map_path):raise RuntimeError('Cannot create generated map')
    for mesh in meshes:
        a=spawn(u.StaticMeshActor,mesh.get_name());c=a.static_mesh_component;c.set_static_mesh(mesh);c.set_mobility(u.ComponentMobility.STATIC)
    scene=json.loads((SRC/'Environment/scene.json').read_text())
    items=scene if isinstance(scene,list) else scene['instances']
    groups={};expected_instances=0
    for i,item in enumerate(items):
        key=item['mesh'];choices=plants[key]
        pieces=([m for m in choices if m.get_name().endswith('_'+'abcd'[i%4])]
                if key=='pachira_aquatica_01' else [choices[i%len(choices)]])
        if key=='pachira_aquatica_01' and len(pieces)!=2:raise RuntimeError('Pachira bark/leaf pair incomplete')
        bounds=[m.get_bounds() for m in pieces]
        low=min(b.origin.z-b.box_extent.z for b in bounds)
        high=max(b.origin.z+b.box_extent.z for b in bounds)
        scale=item.get('target_height_cm',high-low)/max(high-low,1)
        anchor_x=anchor_y=0
        if key in ['pachira_aquatica_01','shrub_04','grass_medium_01']:
            base=next((m for m in pieces if '_bark_' in m.get_name()),pieces[0]).get_bounds()
            anchor_x,anchor_y=base.origin.x,base.origin.y
        yaw=item.get('rotation',item.get('yaw',0))
        if isinstance(yaw,list):yaw=yaw[1]
        a=math.radians(yaw);loc=item['location']
        loc=[loc[0]-(anchor_x*math.cos(a)-anchor_y*math.sin(a))*scale,
             loc[1]-(anchor_x*math.sin(a)+anchor_y*math.cos(a))*scale,loc[2]-low*scale]
        transform=u.Transform(location=u.Vector(*loc),rotation=u.Rotator(pitch=0,yaw=yaw,roll=0),scale=u.Vector(scale,scale,scale))
        for mesh in pieces:
            group=mesh.get_path_name()
            if group not in groups:
                a=spawn(u.IstanaPlantingGroup,'Planting_'+mesh.get_name())
                c=a.get_editor_property('instances')
                c.set_static_mesh(mesh);c.set_mobility(u.ComponentMobility.STATIC)
                c.set_editor_property('cast_shadow',key!='grass_medium_01')
                c.set_cull_distances(0,9000 if key=='grass_medium_01' else 30000 if key in ['shrub_04','pachira_aquatica_01'] else 100000)
                groups[group]=c
            groups[group].add_instance(transform,world_space=True);expected_instances+=1
    REPORT['instances']={k:c.get_instance_count() for k,c in groups.items()}
    sun=spawn(u.DirectionalLight,'Late afternoon sunlight',rot=(-35,135,0));sun.light_component.set_mobility(u.ComponentMobility.MOVABLE)
    sun.light_component.set_editor_property('intensity',5.0)
    sun.light_component.set_editor_property('light_color',u.Color(255,238,211,255))
    sun.light_component.set_editor_property('atmosphere_sun_light',True)
    sun.light_component.set_editor_property('light_source_angle',1.2)
    spawn(u.SkyAtmosphere,'Singapore tropical atmosphere')
    spawn(u.VolumetricCloud,'Soft tropical clouds')
    sky=spawn(u.SkyLight,'Ambient skylight');sky.light_component.set_mobility(u.ComponentMobility.MOVABLE)
    sky.light_component.set_editor_property('real_time_capture',True)
    sky.light_component.set_editor_property('intensity',1.6)
    fog=spawn(u.ExponentialHeightFog,'Tropical distance haze',loc=(0,0,-2000))
    fog.component.set_editor_property('fog_density',.009)
    fog.component.set_editor_property('fog_height_falloff',.15)
    pp=spawn(u.PostProcessVolume,'Architectural photography')
    pp.set_editor_property('unbound',True);settings=pp.settings
    for k,v in {'override_auto_exposure_method':True,'auto_exposure_method':u.AutoExposureMethod.AEM_MANUAL,'override_auto_exposure_bias':True,'auto_exposure_bias':0.0,'override_auto_exposure_apply_physical_camera_exposure':True,'auto_exposure_apply_physical_camera_exposure':False,'override_vignette_intensity':True,'vignette_intensity':.16,'override_bloom_intensity':True,'bloom_intensity':.25,'override_motion_blur_amount':True,'motion_blur_amount':0.0}.items():settings.set_editor_property(k,v)
    pp.set_editor_property('settings',settings)
    spawn(u.PlayerStart,'Start at the ceremonial lawn',loc=(0,-18000,5500),rot=(-12,90,0))
    LEVEL.save_current_level()
    EAL.save_directory('/Game',only_if_is_dirty=True,recursive=True)
    if not LEVEL.load_level('/Game/Maps/Istana'):raise RuntimeError('Saved level cannot reload')
    persisted=sum(c.get_instance_count() for a in ACT.get_all_level_actors() for c in a.get_components_by_class(u.InstancedStaticMeshComponent))
    if persisted!=expected_instances:raise RuntimeError('Planting did not survive reload: '+str(persisted)+' vs '+str(expected_instances))
    REPORT['plant_positions']=len(items)
    REPORT['paired_pachira_positions']=sum(1 for item in items if item['mesh']=='pachira_aquatica_01')
    REPORT['persisted_instances']=persisted
    REPORT['success']=True
    log('scene saved')
def main():
    error_file=ROOT/'Saved/build-error.txt'
    if error_file.exists():error_file.unlink()
    mats=surfaces();plants=nature();meshes=geometry(mats);make_level(meshes,plants)
    (ROOT/'Saved/build-report.json').write_text(json.dumps(REPORT,indent=2))
    log('SUCCESS')
try:main()
except Exception:
    (ROOT/'Saved/build-error.txt').write_text(traceback.format_exc())
    raise
finally:
    u.SystemLibrary.quit_editor()
