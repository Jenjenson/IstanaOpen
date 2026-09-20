"""Encode native footage with external statistics and an optional non-occluding footer."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from render_warning_3d import checked_image
from PIL import Image, ImageDraw
from render_warning_timelapse import font


def episode_footer(image, segment, recordings, total_episodes):
    """Append labels below the native image; never cover or resize scene pixels."""
    result = Image.new("RGB", (image.width, image.height + 72), "#0b1119")
    result.paste(image, (0, 0))
    draw = ImageDraw.Draw(result)
    if segment["episode"] is None:
        text = segment["label"] + "  |  Model close-up"
    else:
        record = next(r for r in recordings if r["checkpoint"] == segment["episode"])
        warning = record["metrics"]["team_warning_s"]
        text = (f"Episode {segment['episode']} / {total_episodes}  |  "
                f"Episode warning: {warning:.2f} s  |  First detection to 20 m arrival (not impact)")
    draw.text((28, image.height + 17), text, font=font(28), fill="#e8edf4")
    return result


def episode_header(image, segment, recordings, total_episodes):
    """Embed a readable stats band above the scene, away from player controls."""
    result = Image.new("RGB", (image.width, image.height + 120), "#0b1119")
    result.paste(image, (0,120))
    draw = ImageDraw.Draw(result)
    if segment["episode"] is None:
        headline, detail = segment["label"], "Actual sensor model close-up"
    else:
        record = next(r for r in recordings if r["checkpoint"] == segment["episode"])
        m = record["metrics"]
        prefix = "PILOT EPISODE" if total_episodes == 128 else "EPISODE"
        headline = f"{prefix} {segment['episode']} / {total_episodes}     CASE WARNING: {m['team_warning_s']:.2f} s"
        detection = "Not detected" if m["first_detection_s"] is None else f"{m['first_detection_s']:.2f} s"
        detail = (f"Detection: {detection}   |   20 m arrival: {m['first_arrival_s']:.2f} s (not impact)"
                  f"   |   Replay: {segment['simulated_s']:.1f} s   |   {segment.get('view', 'overview')} view")
    draw.text((28,10), headline, font=font(44), fill="#e8edf4")
    draw.text((28,72), detail, font=font(27), fill="#92ddcd")
    return result


def timeline(manifest, fps=20, speed=8):
    if fps <= 0 or speed <= 0: raise ValueError("Positive FPS and speed required")
    cursor, segments = 0, []
    for shot in manifest["gallery"]:
        segments.append({"start_frame": cursor, "frames": 2*fps, "image": shot,
                         "label": f"Sensor close-up: {shot['profile']}", "episode": None, "simulated_s": None})
        cursor += 2*fps
    for record in manifest["recordings"]:
        base = cursor
        for i, native in enumerate(record["frames"]):
            next_t = record["frames"][i+1]["elapsed_s"] if i+1 < len(record["frames"]) else native["elapsed_s"] + 1
            end = base + round(next_t / speed * fps)
            count = max(1, end-cursor)
            segments.append({"start_frame": cursor, "frames": count, "image": native,
                             "label": f"Episode {record['checkpoint']}", "episode": record["checkpoint"],
                             "simulated_s": native["elapsed_s"], "view": native.get("view", "overview")})
            cursor += count
    return segments


def report_html(data):
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Istana | Clean 3D training footage</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b1119;color:#e8edf4;font:16px system-ui;padding:20px}
header,footer{max-width:1920px;margin:auto}h1{font-size:26px;margin:0 0 6px}p{color:#acb9c9;line-height:1.6}
.layout{max-width:1920px;margin:22px auto;display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:24px;align-items:start}
video{display:block;width:100%;max-height:85vh;background:#000;aspect-ratio:16/9}aside{padding:18px;background:#131e2b;border-radius:8px}
aside h2{font-size:20px}.value{font-size:30px;color:#92ddcd;margin:5px 0 15px}.small{font-size:13px;color:#acb9c9}
button,a{color:#9dd6ff}button{background:#203349;border:1px solid #45617d;border-radius:6px;padding:10px 14px;margin:4px;cursor:pointer}
.clean{grid-template-columns:1fr}.clean aside{display:none}#chapters{margin:14px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:left;border-bottom:1px solid #293b50;padding:8px}
details{margin:20px 0}summary{cursor:pointer}@media(max-width:950px){.layout{grid-template-columns:1fr}aside{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
</style><header><h1>Blue Team · clean 3D footage</h1><p>No boxes, labels or charts over the simulation. Timings below and beside the player. Synthetic benchmark; 20 m arrival boundary, not impact.</p><button id="toggle">Hide statistics / enlarge footage</button></header>
<main class="layout" id="layout"><section><video id="film" controls preload="metadata" poster="raw-poster.jpg"><source src="warning-raw.mp4" type="video/mp4"></video><div id="chapters"></div><p id="caption"></p></section>
<aside><div><h2 id="stage">Sensor close-ups</h2><p class="small">Simulation clock (not video time)</p><div class="value" id="clock">—</div></div>
<div><p class="small">Shown case: first detection → first arrival</p><div id="events">—</div><p class="small">Shown case warning</p><div class="value" id="caseWarning">—</div></div>
<div><p class="small">Mean team warning · all held-out cases</p><div class="value" id="warning">—</div><p id="gain"></p><p class="small">First detection of any Red → first 20 m zone entry. Undetected = zero.</p></div></aside></main>
<footer><h2>Measured learning result</h2><p id="result"></p><p id="control"></p><p id="coverage"></p><details open><summary>Every fixed checkpoint and control</summary><table><thead><tr><th>Episode</th><th>Team warning</th><th>Per-drone warning</th><th>Detected</th><th>Mean cost</th><th>First-look saturation</th></tr></thead><tbody id="results"></tbody></table></details>
<p id="method"></p><p><a href="warning-raw.mp4" download>Download clean MP4</a> · <a href="raw-video-metadata.json">Timing sidecar</a> · <a href="training-summary.json">Training results</a> · <a href="CREDITS.txt">Credits to retain when sharing</a></p>
<p class="small">Actual Unreal rendering. Camera cuts are presentation only; replay event evidence must match evaluation. Scripted Red, synthetic sensor probabilities, no sensing occlusion. Map data © OpenStreetMap contributors / ODbL.</p></footer>
<script>const data=PAYLOAD;const video=document.getElementById('film');const rows=data.summary.evaluations.filter(r=>r.checkpoint!=='greedy');const initial=rows[0];
const put=(id,t)=>document.getElementById(id).textContent=t;const seconds=x=>x==null?'Not detected':x.toFixed(2)+' s';
document.getElementById('toggle').onclick=()=>document.getElementById('layout').classList.toggle('clean');
for(const chapter of data.chapters){const b=document.createElement('button');b.textContent=chapter.label;b.onclick=()=>{video.currentTime=chapter.seconds;video.play();video.scrollIntoView({block:'start'})};document.getElementById('chapters').append(b)}
video.addEventListener('timeupdate',()=>{const frame=Math.floor(video.currentTime*data.fps);const s=data.segments.findLast(s=>s.start_frame<=frame)||data.segments[0];put('stage',s.label);put('clock',s.simulated_s==null?'—':seconds(s.simulated_s));put('caption',s.episode==null?'Actual sensor close-up · presentation only':s.label+' · '+s.view+' camera · '+data.speed+'× simulated time');
const r=data.recordings.find(r=>r.checkpoint===s.episode);const e=rows.find(r=>Number(r.checkpoint)===s.episode);
put('events',r?seconds(r.metrics.first_detection_s)+' → '+seconds(r.metrics.first_arrival_s):'—');put('caseWarning',r?seconds(r.metrics.team_warning_s):'—');put('warning',e?seconds(e.team_warning_s):'—');put('gain',e?(e.team_warning_s-initial.team_warning_s).toFixed(2)+' s versus untrained':'');});
for(const r of data.summary.evaluations){const tr=document.createElement('tr');for(const text of [r.checkpoint==='greedy'?'Greedy':Number(r.checkpoint),seconds(r.team_warning_s),seconds(r.mean_drone_warning_s),(100*data.coverage[r.checkpoint]).toFixed(1)+'%',r.mean_cost.toFixed(2),r.first_look_saturated_cases==null?'Not recorded':r.first_look_saturated_cases+' cases']){const td=document.createElement('td');td.textContent=text;tr.append(td)}document.getElementById('results').append(tr)}
const c=data.summary.final_minus_control.untrained.team_warning_s;put('result','Final − untrained team warning: '+c.mean.toFixed(2)+' s; paired 95% interval '+c.bootstrap_95_ci.map(x=>x.toFixed(2)).join(' to ')+' s. '+(c.bootstrap_95_ci[0]>0?'Positive gain in this pilot.':'No confirmed positive gain in this pilot.'));
const g=data.summary.final_minus_control.greedy.team_warning_s;put('control','Final − greedy team warning: '+g.mean.toFixed(2)+' s; paired 95% interval '+g.bootstrap_95_ci.map(x=>x.toFixed(2)).join(' to ')+' s. Check per-drone warning and detection coverage for trade-offs; a better first alert does not necessarily mean better coverage of every drone.');
const final=rows[rows.length-1];put('coverage','Detection coverage: '+(100*data.coverage[initial.checkpoint]).toFixed(1)+'% → '+(100*data.coverage[final.checkpoint]).toFixed(1)+'%. Mean per-drone warning: '+seconds(initial.mean_drone_warning_s)+' → '+seconds(final.mean_drone_warning_s)+'. Training optimizes the initial team alert; these secondary outcomes must be considered separately.');
put('method','Same held-out scenarios at every checkpoint; no best-checkpoint selection. '+data.summary.limitations);
</script></html>'''.replace("PAYLOAD", payload)
    if data.get("observer_markers") is not False:
        page = page.replace("No boxes, labels or charts over the simulation.",
                            "New edit of previously recorded footage. Original in-scene marker boxes remain; no added timing panels or charts cover the simulation.")
        page = page.replace("clean 3D", "full-frame 3D").replace("clean MP4", "full-frame MP4")
    page = page.replace("Training optimizes the initial team alert; these secondary outcomes must be considered separately.",
                        "These outcomes describe the original recorded experiment; this edit does not change training or evaluation.")
    if data.get("episode_footer"):
        page = page.replace("aspect-ratio:16/9", "aspect-ratio:5/3")
        page = page.replace("Timings below and beside the player.",
                            "Episode count and that episode's warning time appear in a narrow video footer, below the scene. Other statistics stay beside the player.")
    if data.get("episode_header"):
        page = page.replace("aspect-ratio:16/9", "aspect-ratio:8/5")
        page = page.replace("Timings below and beside the player.",
                            "Episode count, warning time, detection time and arrival time are embedded at the top of the video itself. Camera close-ups show the existing models; no training results changed.")
        page = page.replace('class="layout" id="layout"', 'class="layout clean" id="layout"')
        page = page.replace('Hide statistics / enlarge footage', 'Toggle additional statistics')
    if data.get("summary", {}).get("protocol", {}).get("episodes") == 128:
        page = page.replace("Blue Team · clean 3D footage", "Original 128-episode pilot · visual replay")
        page = page.replace("<h2>Measured learning result</h2>",
                            "<h2>Original pilot results—not the later 256-episode run</h2>"
                            "<p>This video's fixed case has 12.33 s warning at every checkpoint: detection at 1.00 s, arrival at 13.33 s. "
                            "It does not illustrate the later run's roughly 40-second mean. Case warning in the video and the mean across test cases are different statistics.</p>")
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, help="New output folder; preserve original captures")
    parser.add_argument("--allow-recorded-markers", action="store_true",
                        help="Keep and disclose markers already present in legacy frames")
    parser.add_argument("--episode-footer", action="store_true",
                        help="Append an episode/warning label band without covering the native scene")
    parser.add_argument("--episode-header", action="store_true", help="Large in-video stats above the scene")
    parser.add_argument("--speed", type=float, default=8, help="Playback speed relative to simulated time")
    args = parser.parse_args()
    if args.episode_header and args.episode_footer: parser.error("Choose header or footer, not both")
    manifest = json.loads((args.capture / "capture-manifest.json").read_text())
    if manifest.get("observer_markers") is not False and not args.allow_recorded_markers:
        raise ValueError("Recapture without observer markers; cannot erase boxes from old footage")
    folder = args.output or args.capture
    if args.output: folder.mkdir(parents=True, exist_ok=False)
    summary = json.loads((Path(manifest["source_run"]) / "summary.json").read_text())
    movie = folder / "warning-raw.mp4"
    if movie.exists(): raise FileExistsError(movie)
    fps, speed = 20, args.speed
    segments = timeline(manifest, fps, speed)
    height = 1200 if args.episode_header else 1152 if args.episode_footer else 1080
    writer = imageio_ffmpeg.write_frames(str(movie), (1920,height), fps=fps, codec="libx264", quality=8,
                                        macro_block_size=2, output_params=["-movflags", "+faststart", "-metadata",
                                        "copyright=Map data (c) OpenStreetMap contributors / ODbL; IstanaOpen original art CC0"])
    writer.send(None)
    try:
        for i, segment in enumerate(segments):
            image = checked_image(segment["image"])
            if image.size != (1920,1080): raise ValueError("Raw frames must be native 1920x1080")
            if args.episode_footer:
                image = episode_footer(image, segment, manifest["recordings"], summary["protocol"]["episodes"])
            if args.episode_header:
                image = episode_header(image, segment, manifest["recordings"], summary["protocol"]["episodes"])
            if i == 0: image.save(folder / "raw-poster.jpg", quality=95)
            pixels = np.asarray(image)
            for _ in range(segment["frames"]): writer.send(pixels)
    finally: writer.close()
    chapters, seen = [], set()
    for s in segments:
        if s["label"] not in seen:
            chapters.append({"label": s["label"], "seconds": s["start_frame"]/fps}); seen.add(s["label"])
    metadata = {"fps": fps, "speed": speed, "chapters": chapters,
                "segments": [{k:v for k,v in s.items() if k != "image"} for s in segments],
                "duration_s": sum(s["frames"] for s in segments)/fps,
                "recordings": [{k:v for k,v in r.items() if k != "frames"} for r in manifest["recordings"]],
                "summary": summary, "burned_in_overlays": False,
                "episode_footer": args.episode_footer,
                "episode_header": args.episode_header,
                "header_height_px": 120 if args.episode_header else 0,
                "footer_height_px": 72 if args.episode_footer else 0,
                "observer_markers": manifest.get("observer_markers") is not False,
                "source_capture": str(args.capture.resolve()),
                "presentation_only_edit": True}
    metadata["coverage"] = {
        row["checkpoint"]: float(np.mean([case["metrics"]["detected_fraction"] for case in
            json.loads((Path(manifest["source_run"]) / f"evaluation-{row['checkpoint']}.json").read_text())]))
        for row in summary["evaluations"]}
    (folder / "CREDITS.txt").write_text("Map data © OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright\n"
                                      "IstanaOpen original art: CC0. Actual Unreal Engine rendering.\n"
                                      "Retain these credits with the clean MP4 when sharing it.\n", encoding="utf-8")
    (folder / "raw-video-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (folder / "training-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (folder / "index.html").write_text(report_html(metadata), encoding="utf-8")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()
              if p.is_file() and p.name != "artifact-sha256.json"}
    (folder / "artifact-sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps({"movie": str(movie), "duration_s": metadata["duration_s"], "overlays": False}), flush=True)


if __name__ == "__main__": main()
