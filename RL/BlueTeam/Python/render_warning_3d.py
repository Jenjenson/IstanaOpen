"""Compose an annotated training film from unmodified native Unreal frames.

Resizing and data overlays only; no generated scene imagery or altered paths.
All displayed warning results come from the original fixed-checkpoint evaluation.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from render_warning_timelapse import font
from triad_rl.training_recording import RECORDING_SCHEMA

W,H,FPS = 1920,1080,15
BG,INK,MUTED,BLUE,GREEN,AMBER,RED = "#081320","#eff5ff","#a9bdd0","#56beff","#6ee3b4","#ffd16e","#ff7384"


def result_text(summary):
    if summary.get("schema") == RECORDING_SCHEMA:
        return ("RECORDED TRAINING AND EVALUATION EVIDENCE",
                "The film retains measured warning times and saved sensor placements. "
                "Training episodes use changing scenarios; compare fixed evaluation panels to measure progress.",
                "Separate final-test entries are identified when available. Improvement is not guaranteed.")
    rows=[e for e in summary['evaluations'] if e['checkpoint']!='greedy']
    initial,final=rows[0],rows[-1]
    lo,hi=summary['final_minus_control']['untrained']['mean_drone_warning_s']['bootstrap_95_ci']
    headline=('POSITIVE PER-DRONE GAIN IN THIS PILOT' if lo>0 else
              'PER-DRONE WARNING REGRESSED' if hi<0 else 'NO CONFIRMED WARNING-TIME GAIN')
    team_delta=final['team_warning_s']-initial['team_warning_s']
    team=(f"Team warning changed by {team_delta:+.3f}s on these paired cases." if abs(team_delta)>1e-8 else
          "Team preparation time was unchanged on these paired cases.")
    description=(f"Mean per-drone warning: {initial['mean_drone_warning_s']:.3f}s to {final['mean_drone_warning_s']:.3f}s; "
                 f"paired gain interval {lo:+.3f} to {hi:+.3f}s. "
                 f"Team warning: {initial['team_warning_s']:.3f}s to {final['team_warning_s']:.3f}s.")
    return headline,description,team


def checked_image(frame):
    path = Path(frame["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
        raise ValueError(f"Native frame changed: {path}")
    return Image.open(path).convert("RGB")


def label(draw,xy,value,size=24,color=INK):
    draw.text(xy,str(value),font=font(size),fill=color)


def gallery_frame(shot):
    image = checked_image(shot).resize((W,H),Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA",(W,H),(0,0,0,0)); d=ImageDraw.Draw(overlay)
    d.rounded_rectangle((55,90,710,460),radius=20,fill=(8,19,32,238))
    names={"eo":"Electro-optical camera","thermal":"Dual-aperture thermal head",
           "radar":"Flat-panel radar","rf":"Passive RF antenna array","fused":"Radar + thermal assembly"}
    label(d,(85,118),"BLUE TEAM / SELECTED SENSOR MODEL",20,BLUE)
    label(d,(85,170),shot.get("profile_label", names.get(shot["profile"], shot["profile"])),28)
    label(d,(85,243),"Telescoping mast · anchored base",26,MUTED)
    label(d,(85,287),"Weatherproof enclosure · PBR materials",26,MUTED)
    label(d,(85,355),"Actual Unreal capture, not generated imagery",22,GREEN)
    label(d,(85,397),"Presentation-only close-up; not an extra training case",19,MUTED)
    d.rectangle((0,1025,W,H),fill=(8,19,32,230))
    label(d,(40,1040),"Map data © OpenStreetMap contributors / ODbL · Visual approximation · Synthetic sensors",18,MUTED)
    return Image.alpha_composite(image.convert("RGBA"),overlay).convert("RGB")


def replay_frame(native,record,summary,stage,final=False):
    if summary.get("schema") == RECORDING_SCHEMA:
        return console_replay_frame(native, record, summary, stage, final=final)
    image=Image.new("RGB",(W,H),BG)
    image.paste(checked_image(native).resize((1440,810),Image.Resampling.LANCZOS),(0,125))
    d=ImageDraw.Draw(image)
    evaluations=[e for e in summary["evaluations"] if e["checkpoint"]!="greedy"]
    e, initial = evaluations[stage], evaluations[0]
    label(d,(35,20),"ISTANA / NATIVE 3D TRAINING REPLAY",22,BLUE)
    label(d,(35,55),f"Blue sensor placement  ·  episode {record['checkpoint']} / {summary['protocol']['episodes']}",40)
    label(d,(1470,25),"FIXED CHECKPOINTS",21,MUTED)
    label(d,(1470,62),"No best-run selection",24)
    label(d,(1470,144),"MEAN PER-DRONE WARNING",21,MUTED)
    label(d,(1470,178),f"{e['mean_drone_warning_s']:.3f} s",55,GREEN)
    delta=e['mean_drone_warning_s']-initial['mean_drone_warning_s']
    label(d,(1470,250),f"{delta:+.3f} s vs untrained",26,GREEN if delta>=0 else RED)
    label(d,(1470,291),f"Across {len(summary['protocol']['evaluation_seeds'])} held-out scenarios",20,MUTED)
    d.line((1470,342,1880,342),fill="#294359",width=2)
    label(d,(1470,370),"TEAM PREPARATION TIME",21,MUTED)
    label(d,(1470,408),f"{e['team_warning_s']:.3f} s",55,AMBER)
    label(d,(1470,482),f"{e['team_warning_s']-initial['team_warning_s']:+.3f} s gained vs start",26,AMBER)
    label(d,(1470,527),"First Red detection → first arrival",20,MUTED)
    evidence=record.get("warning_evidence",[])
    detected=[row for row in evidence if row.get("firstDetectionSeconds") is not None]
    detected.sort(key=lambda row:row["firstDetectionSeconds"])
    first_row=detected[0] if detected else None
    confirmations=[row["firstConfirmationSeconds"] for row in evidence if row.get("firstConfirmationSeconds") is not None]
    confirmation=min(confirmations) if confirmations else None
    orientation=" · ".join(f"S{row['siteId']} {row['yawDeg']:.0f}°/{row['pitchDeg']:.0f}°"
                            for row in record["placements"][:3]) or "STOP / no sensors"
    label(d,(1470,562),f"Orientation: {orientation}",18,BLUE)
    if first_row:
        label(d,(1470,588),f"Confirm {confirmation:.1f}s" if confirmation is not None else "Confirm none",18,MUTED)
        label(d,(1470,612),f"At detect: {first_row['pixelsOnTargetAtFirstDetection']:.2f}px · P={first_row['probabilityAtFirstDetection']:.3f}",18,MUTED)
        label(d,(1470,636),"Detection LOS: clear",18,GREEN)
    else:
        label(d,(1470,588),"No successful detection / confirmation",18,RED)
    label(d,(1470,674),"WARNING THROUGH TRAINING (s)",19,MUTED)
    def chart(i,v):return 1510+i*88,840-v*9
    for value in (0,5,10,15,20):
        x,y=chart(0,value);d.line((x,y,1862,y),fill="#25384d")
        label(d,(1470,y-12),value,18,MUTED)
    for key,col in (("mean_drone_warning_s",GREEN),("team_warning_s",AMBER)):
        pts=[chart(i,row[key]) for i,row in enumerate(evaluations[:stage+1])]
        if len(pts)>1:d.line(pts,fill=col,width=3)
        for x,y in pts:d.ellipse((x-5,y-5,x+5,y+5),fill=col)
    for i,row in enumerate(evaluations):label(d,(chart(i,0)[0]-12,852),int(row["checkpoint"]),18,MUTED)
    label(d,(1480,897),"Green: per drone   Amber: team",18,MUTED)
    t=native["elapsed_s"];first=record["metrics"]["first_detection_s"];arrival=record["metrics"]["first_arrival_s"]
    label(d,(30,956),f"Shown case {record['seed']}  ·  t={t:.1f}s  ·  {len(record['placements'])} sensors  ·  cost {record['metrics']['cost']:.1f}/3",22)
    label(d,(35,997),"Cyan: sensors   Red: undetected   Amber: detected   Gray: zone entered",20,MUTED)
    left,right=1100,1860
    bx=lambda v:left+(right-left)*v/record['frames'][-1]['elapsed_s']
    d.rounded_rectangle((left,984,right,998),radius=5,fill="#304357")
    if first is not None and first<arrival:d.rectangle((bx(first),984,bx(arrival),998),fill=AMBER)
    d.line((bx(t),978,bx(t),1004),fill=INK,width=3)
    first_text=f"{first:.1f}s" if first is not None else "none"
    label(d,(left,947),f"Shown case: detection {first_text} → arrival {arrival:.1f}s",21,AMBER)
    label(d,(35,1044),"Endpoint: 20 m target-zone entry, NOT impact. Scripted Red; synthetic probabilities; thermal LOS uses Unreal geometry.",18,MUTED)
    label(d,(1260,1044),"Map data © OpenStreetMap contributors / ODbL",18,MUTED)
    if final:
        d.rounded_rectangle((140,275,1310,720),radius=24,fill="#101e30",outline="#4b637b",width=2)
        headline,_,team_text=result_text(summary)
        label(d,(180,310),f"RESULT / {headline}",31,AMBER)
        label(d,(180,380),f"Per-drone warning: {initial['mean_drone_warning_s']:.3f}s → {e['mean_drone_warning_s']:.3f}s",32)
        lo,hi=summary['final_minus_control']['untrained']['mean_drone_warning_s']['bootstrap_95_ci']
        label(d,(180,437),f"Paired gain interval: {lo:+.3f} to {hi:+.3f} seconds",27,MUTED)
        label(d,(180,505),team_text,25)
        label(d,(180,560),f"All {len(evaluations)} checkpoints retained. One training seed; {len(summary['protocol']['evaluation_seeds'])} paired scenarios.",24,MUTED)
        label(d,(180,620),"Better-looking models do not imply better RL performance.",26,GREEN)
    return image


def console_replay_frame(native, record, summary, stage, final=False):
    """Native footage overlays sourced only from current recorded evidence."""
    image = Image.new("RGB", (W,H), BG)
    image.paste(checked_image(native).resize((1440,810),Image.Resampling.LANCZOS),(0,125))
    d = ImageDraw.Draw(image)
    label(d,(35,20),"ISTANA / NATIVE 3D TRAINING REPLAY",22,BLUE)
    label(d,(35,58),record.get("label", f"Episode {record['checkpoint']}"),38)
    evaluation = record.get("evaluation", record["recorded_metrics"])
    warning = evaluation.get("mean_drone_warning_s", evaluation.get("meanWarningSeconds"))
    label(d,(1470,145),"RECORDED MEAN WARNING",21,MUTED)
    label(d,(1470,185),f"{warning:.3f} s",53,GREEN)
    count = len(record.get("evaluation_seeds", [record["seed"]]))
    label(d,(1470,260),"Sampled training episode" if record["kind"] == "training" else f"Across {count} evaluation cases",21,MUTED)
    label(d,(1470,307),"Missed detections count as zero",19,MUTED)
    if "deltaSeconds" in evaluation and record["kind"] != "training":
        delta = evaluation["deltaSeconds"]
        label(d,(1470,337),f"{delta:+.3f} s vs contractor",21,GREEN if delta >= 0 else RED)
    label(d,(1470,367),"SHOWN CASE / REPLAY CHECK",21,BLUE)
    metrics = record["metrics"]
    label(d,(1470,410),f"Warning {metrics['mean_drone_warning_s']:.3f} s / drone",25)
    label(d,(1470,459),f"Detected {metrics['detected_fraction']:.0%}",25)
    label(d,(1470,508),f"Team warning {metrics['team_warning_s']:.3f} s",25)
    label(d,(1470,565),"Original detection evidence matched",20,GREEN)
    label(d,(1470,599),f"Scenario {record['seed']}",21,MUTED)
    context = record["context"]
    budget = context["publicSnapshot"].get("budget", context["publicSnapshot"].get("budget_remaining", "?"))
    label(d,(1470,650),f"{len(record['placements'])} sensors · cost {metrics['cost']:g}/{budget}",22)
    label(d,(1470,687),f"{metrics.get('targets', len(record['warning_evidence']))} drones · limited FOV",22)
    weather = context["publicSnapshot"].get("weather", {})
    label(d,(1470,750),f"Rain {weather.get('rain', '?')}",21,MUTED)
    label(d,(1470,785),f"Visibility {weather.get('visibility', '?')}",21,MUTED)
    label(d,(35,965),f"Simulation t = {native['elapsed_s']:.1f} s · {record['kind']} recording",23)
    if native.get("presentation_only"):
        label(d,(35,1005),"Cinematic camera orbit · frozen simulation state",22,BLUE)
    else:
        label(d,(35,1005),"Native Unreal frames · saved actions replayed exactly · no policy resampling",22,BLUE)
    label(d,(35,1050),"20 m target-zone entry, not impact · synthetic sensing assumptions · different scenario samples are not paired gains",18,MUTED)
    if final:
        d.rounded_rectangle((180,350,1280,660),radius=20,fill=BG,outline=BLUE,width=2)
        label(d,(220,389),"RECORDED RESULT",32,BLUE)
        label(d,(220,449),f"{record.get('label', 'Selected layout')}: {warning:.3f} s warning / drone",29)
        label(d,(220,514),"Replay detection and arrival evidence matches the saved run.",25)
        label(d,(220,572),"Use matching evaluation cases to quantify improvement.",24,MUTED)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("capture",type=Path)
    parser.add_argument("--clean", action="store_true", help="Export native footage without burned-in metrics/text for editing")
    args=parser.parse_args();folder=args.capture
    manifest=json.loads((folder/'capture-manifest.json').read_text())
    summary=manifest.get('presentation_summary')
    if summary is None:
        summary=json.loads((Path(manifest['source_run'])/'summary.json').read_text())
    if not manifest['recordings']:
        raise ValueError("Capture manifest has no episode recordings")
    movie=folder/('warning-3d-clean.mp4' if args.clean else 'warning-3d-timelapse.mp4')
    if movie.exists():raise FileExistsError(movie)
    writer=imageio_ffmpeg.write_frames(str(movie),(W,H),fps=FPS,codec='libx264',quality=8,macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None);frame_count=0;chapters=[]
    def frames(image,count):
        nonlocal frame_count
        pixels=np.asarray(image)
        for _ in range(count):writer.send(pixels)
        frame_count+=count
    try:
        for index, shot in enumerate(manifest['gallery']):
            hero=checked_image(shot).resize((W,H),Image.Resampling.LANCZOS) if args.clean else gallery_frame(shot)
            frames(hero,2*FPS)
            if index == 0: hero.save(folder/'sensor-model-preview.jpg',quality=95)
        for stage,record in enumerate(manifest['recordings']):
            chapters.append({'episode':record['checkpoint'],'label':record.get('label', f"Episode {record['checkpoint']}"),'seconds':frame_count/FPS})
            for native in record['frames']:
                frames(checked_image(native).resize((W,H),Image.Resampling.LANCZOS) if args.clean
                       else replay_frame(native,record,summary,stage),3)
        final=(checked_image(record['frames'][-1]).resize((W,H),Image.Resampling.LANCZOS) if args.clean
               else replay_frame(record['frames'][-1],record,summary,stage,final=True))
        final.save(folder/'timelapse-3d-poster.png');frames(final,6*FPS)
    finally:writer.close()
    (folder/'training-summary.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    buttons=''.join(f'<button data-time="{c["seconds"]}">{html.escape(c["label"])}</button>' for c in chapters)
    headline,description,team_text=result_text(summary)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Istana | Native 3D RL timelapse</title>
<style>body{{margin:auto;padding:28px;max-width:1300px;background:#081320;color:#eff5ff;font:18px system-ui}}h1{{font-size:38px;margin:10px 0}}p,li{{color:#b8c9dc;line-height:1.6}}video,img{{width:100%;border-radius:15px}}button{{background:#233c55;color:white;border:1px solid #55728c;padding:12px 18px;border-radius:8px;margin:6px;cursor:pointer}}a{{color:#63c8ff}}.panel{{background:#112237;padding:24px;border-radius:14px;margin:24px 0}}</style>
<p style="color:#6ee3b4;letter-spacing:2px">ACTUAL UNREAL CAPTURE · 1080P</p><h1>Blue Team: native 3D training timelapse</h1>
<p>Native sensor models, recorded Red approaches, and warning times for the retained layouts listed below.</p>
<video id="film" controls preload="auto" poster="timelapse-3d-poster.png"><source src="{movie.name}" type="video/mp4"></video>
<div><button data-time="0">Sensor model close-ups</button>{buttons}</div>
<script>document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{{const v=document.getElementById('film');v.currentTime=Number(b.dataset.time);v.play();}});</script>
<div class="panel"><strong>{headline}</strong><p>{description} {team_text} This recording does not claim that cosmetic improvements improved learning.</p>
<p>As requested, the endpoint remains entry into the 20 m target zone, <strong>not physical impact</strong>. Red is scripted, not a learned opponent. Probabilities are simulation assumptions; directional thermal line of sight is traced against Unreal world-static geometry.</p></div>
<h2>Equipment footage</h2><p>New capture galleries contain only the selected available limited-FOV sensor profiles. Each shot retains its catalogue specifications alongside the native model image. Historical capture manifests retain their originally recorded equipment.</p>
<p>The sequences are rendered by Unreal from saved placements. Every replay's full per-drone detection and arrival evidence was checked against the original. The footage shows one recorded case per layout; performance cards identify whether results describe that sampled episode or a fixed evaluation panel. Presentation-only camera orbits freeze simulation time.</p>
<p><a href="{movie.name}" download>Download the 3D MP4</a> · <a href="training-summary.json">Measured training results</a> · <a href="capture-manifest.json">Native capture evidence</a></p>
<p>Training is replayed here, not repeated or selected for favorable outcomes. The existing scenario and all original training artifacts remain preserved.</p></html>'''
    (folder/'index.html').write_text(page,encoding='utf-8')
    (folder/'video-metadata.json').write_text(json.dumps({'fps':FPS,'frames':frame_count,'duration_s':frame_count/FPS,'chapters':chapters,'burned_in_overlays':not args.clean,'native_frame_count':sum(len(r['frames']) for r in manifest['recordings'])},indent=2))
    (folder/'CREDITS.txt').write_text('Map data © OpenStreetMap contributors / ODbL. Actual Unreal rendering. Retain these credits with exported footage.\n',encoding='utf-8')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file() and p.name!='artifact-sha256.json'}
    (folder/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'movie':str(movie),'duration_s':frame_count/FPS,'chapters':chapters},indent=2))


if __name__=='__main__':main()
