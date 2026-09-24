"""Render an auditable MP4 and HTML report from recorded native warning runs.

Top-down data replay, NOT Unreal camera footage. Interpolation affects display
only. All event times, placements and performance values come from saved runs.
Install optional dependencies: pip install pillow imageio-ffmpeg
"""
import argparse
import bisect
import hashlib
import html
import json
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from triad_rl.training_recording import (
    RECORDING_SCHEMA, evaluation_warning, frame_drones, frame_time,
    load_recordings, parse_episodes,
)

W, H = 1440, 900
BG, PANEL, INK, MUTED = "#0a1220", "#111e30", "#edf4ff", "#9caec8"
BLUE, GREEN, RED, AMBER = "#59bcff", "#65e2b4", "#ff6c7c", "#ffcf70"


@lru_cache(maxsize=32)
def font(size):
    for path in ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def interpolate_drones(frames, t):
    index = max(0, min(len(frames) - 1, bisect.bisect_right([f["t"] for f in frames], t) - 1))
    first, last = frames[index], frames[min(index + 1, len(frames) - 1)]
    fraction = (t - first["t"]) / (last["t"] - first["t"]) if last["t"] > first["t"] else 0
    latter = {d["droneId"]: d for d in last["drones"]}
    return [{"id": d["droneId"], "p": {k: d["positionCm"][k] * (1 - fraction) + latter[d["droneId"]]["positionCm"][k] * fraction for k in "xyz"}}
            for d in first["drones"]]


def draw_frame(summary, runs, stage, progress):
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    def text(x, y, value, size=20, fill=INK): d.text((x, y), str(value), font=font(size), fill=fill)
    def panel(box): d.rounded_rectangle(box, radius=18, fill=PANEL)
    stage = min(stage, len(runs) - 1)
    run = runs[stage]
    evaluations = [e for e in summary["evaluations"] if e["checkpoint"] != "greedy"]
    current, initial = evaluations[stage], evaluations[0]
    episode = int(current["checkpoint"])
    t = run["frames"][-1]["t"] * min(1, progress * 1.2)
    text(40, 22, "NATIVE ISTANA  /  BLUE TEAM REINFORCEMENT LEARNING", 16, BLUE)
    text(40, 49, "Learning to place sensors for earlier warning", 34)
    text(1080, 53, f"EPISODE {episode:03d} / {summary['protocol']['episodes']}", 23, GREEN)
    panel((32, 120, 824, 790)); panel((844, 120, 1408, 400)); panel((844, 416, 1408, 790))
    text(54, 138, "RECORDED SIMULATION • TOP-DOWN REPLAY", 16, MUTED)
    text(54, 166, f"Same held-out Red scenario {run['seed']}   |   t = {t:04.1f} s", 18)
    cx, cy, scale = 426, 450, 3.1
    def point(x, y): return cx + x * scale, cy - y * scale
    origin = run["context"]["worldOriginCm"]
    for g in range(-80, 81, 20):
        d.line((*point(g, -80), *point(g, 80)), fill="#1c3048")
        d.line((*point(-80, g), *point(80, g)), fill="#1c3048")
    radius = run["context"]["temporalConfig"]["objective_radius_m"] * scale
    d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill="#433523", outline=AMBER, width=2)
    text(cx-37, cy-12, "TARGET", 18, AMBER)
    text(598, 679, "20 m target zone", 16, AMBER)
    sites = run["context"]["publicSnapshot"]["sites"]
    blocked = run["context"]["publicSnapshot"]["blocked_sites"]
    for i, (x, y) in enumerate(sites):
        if i not in blocked:
            px, py = point(x, y); d.ellipse((px-2, py-2, px+2, py+2), fill="#486480")
    # Actual trajectories, muted so chosen sensor positions remain legible.
    for drone_id in range(run["metrics"]["targets"]):
        trail = []
        for f in run["frames"]:
            if f["t"] > t: break
            drone = next((a for a in f["drones"] if a["droneId"] == drone_id), None)
            if drone:
                p = drone["positionCm"]; trail.append(point((p["x"]-origin["x"])/100, (p["y"]-origin["y"])/100))
        if len(trail) > 1: d.line(trail, fill="#3a3346", width=1)
    evidence = {row["droneId"]: row for row in run["warning_evidence"]}
    for drone in interpolate_drones(run["frames"], t):
        row = evidence[drone["id"]]
        first, arrival = row["firstDetectionSeconds"], row["zoneEntrySeconds"]
        color = "#657189" if arrival is not None and t >= arrival else AMBER if first is not None and t >= first else RED
        p = drone["p"]; x, y = point((p["x"]-origin["x"])/100, (p["y"]-origin["y"])/100)
        d.polygon([(x,y-4),(x+4,y+3),(x-4,y+3)], fill=color)
    for index, placement in enumerate(run["placements"], 1):
        x, y = point(*sites[placement["siteId"]])
        d.ellipse((x-10,y-10,x+10,y+10), fill=BLUE, outline=INK, width=2)
        text(x+13, y-12, f"{index} {placement['profileId']}", 17, BLUE)
    text(56, 717, "● Sensor on approved surface     ▲ Red: undetected / detected / arrived", 16, MUTED)
    text(56, 747, f"Layout cost {run['metrics']['cost']:.1f} / 3.0   •   {len(run['placements'])} sensors   •   60 Red drones", 18)
    text(868, 139, "AVERAGE WARNING PER DRONE", 17, MUTED)
    text(868, 163, f"{current['mean_drone_warning_s']:.2f} s", 54, GREEN)
    delta = current["mean_drone_warning_s"] - initial["mean_drone_warning_s"]
    text(1113, 187, f"{delta:+.2f} s vs start", 21, GREEN if delta >= 0 else RED)
    text(868, 234, f"Mean across {len(summary['protocol']['evaluation_seeds'])} held-out scenarios; missed detections = 0", 17, MUTED)
    d.line((868,274,1382,274), fill="#314259")
    text(868, 290, "TEAM-LEVEL WARNING", 16, MUTED)
    text(868, 315, f"{current['team_warning_s']:.2f} s", 35, AMBER)
    team_delta = current["team_warning_s"] - initial["team_warning_s"]
    text(1095, 328, f"{team_delta:+.2f} s vs start", 21, AMBER)
    text(868, 367, "First detection of any Red → first Red zone entry", 17, MUTED)
    text(868, 433, "FIXED CHECKPOINTS — NO BEST-RUN SELECTION", 16, MUTED)
    left, right, top, bottom = 897, 1373, 488, 649
    maximum = max(20, max(e['mean_drone_warning_s'] for e in evaluations) + 1)
    def chart(i, value): return left+(right-left)*i/max(1, len(runs)-1), bottom-(bottom-top)*value/maximum
    for val in (0, 5, 10, 15, 20):
        yy = chart(0,val)[1]; d.line((left,yy,right,yy), fill="#294059"); text(855, yy-10, str(val), 15, MUTED)
    greedy = next(e for e in summary["evaluations"] if e["checkpoint"] == "greedy")
    gy = chart(0,greedy["mean_drone_warning_s"])[1]
    for xx in range(left,right,14): d.line((xx,gy,min(xx+7,right),gy), fill=MUTED, width=1)
    for metric, color in (("mean_drone_warning_s", GREEN),("team_warning_s", AMBER)):
        points = [chart(i,e[metric]) for i,e in enumerate(evaluations[:stage+1])]
        if len(points)>1: d.line(points, fill=color, width=3)
        for x,y in points: d.ellipse((x-4,y-4,x+4,y+4), fill=color)
    for i, e in enumerate(evaluations): text(chart(i,0)[0]-12, bottom+8, int(e["checkpoint"]), 16, MUTED)
    text(892, 693, "Green: per-drone   Amber: team   Dashed: greedy", 16, MUTED)
    first, arrival = run["metrics"]["first_detection_s"], run["metrics"]["first_arrival_s"]
    first_label = f"{first:.1f}s" if first is not None else "none"
    text(868, 721, f"Shown case: first hit {first_label} → arrival {arrival:.1f}s", 16, MUTED)
    duration = run["frames"][-1]["t"]
    bx = lambda seconds: 868 + 514 * seconds / duration
    d.rounded_rectangle((868, 751, 1382, 761), radius=4, fill="#344259")
    if first is not None and first < arrival:
        d.rectangle((bx(first),751,bx(arrival),761), fill=AMBER)
    d.line((bx(t),746,bx(t),766), fill=INK, width=2)
    text(40, 810, "Surface-mounted sensors  •  Native Red trajectories  •  Seeded scripted swarms, not self-play", 19, MUTED)
    text(40, 841, "Synthetic sensing; no sensing occlusion. One training seed. This pilot does not establish real-world benefit.", 18, MUTED)
    return image


def draw_training_frame(metadata, entries, stage, progress):
    """Current console evidence, including per-episode layouts and log rows."""
    entry = entries[min(stage, len(entries) - 1)]
    run, context = entry["run"], entry["run"]["context"]
    config = metadata["configuration"]
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    def text(x, y, value, size=20, fill=INK):
        d.text((x, y), str(value), font=font(size), fill=fill)
    text(36, 22, "ISTANA / RECORDED TRAINING EVIDENCE", 18, BLUE)
    text(36, 54, entry["label"], 34)
    text(1040, 28, str(config.get("algorithm", "RL")).upper(), 20, GREEN)
    text(1040, 59, f"Episode {entry['checkpoint']} / {config.get('episodes', '?')}", 22)
    d.rounded_rectangle((28, 115, 850, 800), radius=18, fill=PANEL)
    d.rounded_rectangle((870, 115, 1410, 800), radius=18, fill=PANEL)
    frames = run.get("frames", [])
    if not frames:
        raise ValueError("Recorded episode has no trajectory frames")
    t = frame_time(frames[-1]) * min(1., max(0., progress))
    frame = frames[max(0, bisect.bisect_right([frame_time(f) for f in frames], t) - 1)]
    sites = context["publicSnapshot"]["sites"]
    origin = context["worldOriginCm"]
    # Keep placement geometry legible; the inset shows the full approach so
    # distant drone spawns do not shrink the sensor ring to a few pixels.
    positions = [(p[0], p[1]) for p in sites]
    extent = max(50., max(abs(value) for point in positions for value in point)) * 1.35
    route_positions = [tuple((row["positionCm"][axis] - origin[axis]) / 100 for axis in "xy")
                       for f in frames for row in frame_drones(run, f)]
    route_extent = max(extent, max((abs(value) for point in route_positions for value in point), default=extent)) * 1.1
    scale, cx, cy = 270 / extent, 410, 449
    def point(x, y): return cx + x * scale, cy - y * scale
    for fraction in (-1, -.5, 0, .5, 1):
        q = extent * fraction
        d.line((*point(q, -extent), *point(q, extent)), fill="#203249")
        d.line((*point(-extent, q), *point(extent, q)), fill="#203249")
    radius = context["temporalConfig"]["objective_radius_m"] * scale
    d.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=AMBER, width=2)
    text(cx + radius + 5, cy, "Target zone", 15, AMBER)
    for placement in run["placements"]:
        x, y = point(*sites[placement["siteId"]])
        yaw = np.deg2rad(placement.get("yawDeg", 0))
        d.line((x, y, x + 24 * np.cos(yaw), y - 24 * np.sin(yaw)), fill=BLUE, width=3)
        d.ellipse((x-5, y-5, x+5, y+5), fill=BLUE)
        text(x+7, y-20, f"S{placement['siteId']}", 14, BLUE)
    evidence = {row["droneId"]: row for row in run["warning_evidence"]}
    for drone in frame_drones(run, frame):
        row = evidence.get(drone["droneId"], {})
        first, arrival = row.get("firstDetectionSeconds"), row.get("zoneEntrySeconds")
        color = MUTED if arrival is not None and t >= arrival else GREEN if first is not None and t >= first else RED
        p = drone["positionCm"]
        x, y = point((p["x"]-origin["x"])/100, (p["y"]-origin["y"])/100)
        if 50 <= x <= 826 and 208 <= y <= 705:
            d.polygon(((x,y-6),(x+5,y+4),(x-5,y+4)), fill=color)
    d.rounded_rectangle((610, 512, 829, 707), radius=10, fill=BG, outline="#34465e")
    text(624, 523, "FULL APPROACH", 14, MUTED)
    mini_scale = 74 / route_extent
    def mini_point(x, y): return 719 + x * mini_scale, 622 - y * mini_scale
    for drone_id in evidence:
        trail = []
        for f in frames:
            if frame_time(f) > t:
                break
            row = next((item for item in frame_drones(run, f) if item["droneId"] == drone_id), None)
            if row:
                p = row["positionCm"]
                trail.append(mini_point((p["x"]-origin["x"])/100, (p["y"]-origin["y"])/100))
        if len(trail) > 1:
            d.line(trail, fill="#685563", width=1)
    for placement in run["placements"]:
        x, y = mini_point(*sites[placement["siteId"]])
        d.ellipse((x-2, y-2, x+2, y+2), fill=BLUE)
    for drone in frame_drones(run, frame):
        p = drone["positionCm"]
        x, y = mini_point((p["x"]-origin["x"])/100, (p["y"]-origin["y"])/100)
        d.ellipse((x-2, y-2, x+2, y+2), fill=RED)
    text(48, 136, "TOP-DOWN DATA REPLAY / NATIVE RECORDED POSITIONS", 16, MUTED)
    text(48, 165, f"Layout detail · scenario {run['seed']} · t = {t:.1f} s", 18)
    metrics = run["metrics"]
    budget = context["publicSnapshot"].get("budget", context["publicSnapshot"].get("budget_remaining", "?"))
    text(48, 723, f"{len(run['placements'])} sensors · {metrics.get('targets', len(evidence))} drones · cost {metrics['cost']:g} / {budget}", 20)
    text(48, 754, "Sensor heading →   Red: undetected   Green: detected   Gray: arrived", 16, MUTED)
    warning = evaluation_warning(entry)
    text(895, 141, "MEAN WARNING PER DRONE", 19, MUTED)
    text(895, 177, f"{warning:.3f} s", 49, GREEN)
    training = entry["kind"] == "training"
    seeds = entry.get("evaluation_seeds", [run["seed"]])
    text(895, 248, "Sampled training episode" if training else f"{len(seeds)} fixed evaluation scenarios", 18, MUTED)
    text(895, 278, "Missed detections count as zero", 18, MUTED)
    baseline = next((item for item in entries if item["kind"] == "baseline"), None)
    evaluation = entry.get("evaluation", {})
    if not training and ("deltaSeconds" in evaluation or baseline and seeds == baseline.get("evaluation_seeds")):
        delta = evaluation["deltaSeconds"] if "deltaSeconds" in evaluation else warning - evaluation_warning(baseline)
        text(895, 313, f"{delta:+.3f} s vs contractor baseline", 21, GREEN if delta >= 0 else RED)
    else:
        text(895, 313, "Different scenarios are not a paired gain", 18, MUTED)
    text(895, 365, "PER-EPISODE WARNING LOG (s)", 19, MUTED)
    history_end = (max((row["episode"] for row in metadata.get("training_history", [])), default=0)
                   if entry["kind"] in ("best", "best_test") else entry["checkpoint"])
    history = [row for row in metadata.get("training_history", []) if row["episode"] <= history_end]
    if history:
        chart_rows = history[::max(1, len(history)//450)]
        if chart_rows[-1] is not history[-1]: chart_rows.append(history[-1])
        maximum = max(1., max(row["meanWarningSeconds"] for row in chart_rows))
        right_episode = max(1, history_end)
        pts = [(902 + 470 * row["episode"] / right_episode,
                541 - 115 * row["meanWarningSeconds"] / maximum) for row in chart_rows]
        d.line((902, 550, 1380, 550), fill=MUTED)
        if len(pts) > 1: d.line(pts, fill=GREEN, width=2)
        text(905, 558, "Training scenarios vary; spikes are not validation gains", 15, MUTED)
        for index, row in enumerate(history[-5:]):
            text(900, 609 + index * 29,
                 f"ep {row['episode']:>5}   warning {row['meanWarningSeconds']:>7.3f}s   "
                 f"detected {row.get('detectedFraction', 0):.0%}", 18)
    else:
        text(895, 417, "Initial layout, before any optimizer update", 18, MUTED)
    text(36, 824, "Every displayed layout and event comes from saved evidence. Timing improvement is not guaranteed.", 19, MUTED)
    weather = context["publicSnapshot"].get("weather", {})
    text(36, 855, f"Synthetic sensing · limited FOV · weather retained: rain {weather.get('rain', '?')}, visibility {weather.get('visibility', '?')}", 18, MUTED)
    return image


def render_console_run(folder, metadata, entries, *, preview_only=False, seconds_per_stage=3., fps=15):
    final = draw_training_frame(metadata, entries, len(entries)-1, 1.)
    final.save(folder / "timelapse-poster.png")
    if preview_only:
        return
    movie = folder / "warning-timelapse.mp4"
    if movie.exists(): raise FileExistsError(movie)
    writer = imageio_ffmpeg.write_frames(str(movie), (W,H), fps=fps, codec="libx264", quality=8,
                                        macro_block_size=2, output_params=["-movflags", "+faststart"])
    writer.send(None)
    count = max(1, round(seconds_per_stage * fps))
    chapters = []
    try:
        for stage, entry in enumerate(entries):
            chapters.append({"label": entry["label"], "episode": entry["checkpoint"], "seconds": stage*count/fps})
            for index in range(count):
                writer.send(np.asarray(draw_training_frame(metadata, entries, stage, index/max(1,count-1))))
    finally:
        writer.close()
    (folder / "timelapse-chapters.json").write_text(json.dumps(chapters, indent=2), encoding="utf-8")
    rows = "".join(f"<tr><td>{html.escape(entry['label'])}</td><td>{entry['kind']}</td>"
                   f"<td>{evaluation_warning(entry):.3f} s</td><td>{len(entry.get('evaluation_seeds', [entry['run']['seed']]))}</td></tr>"
                   for entry in entries)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Training evidence</title>
<style>body{{background:#0a1220;color:#edf4ff;max-width:1200px;margin:32px auto;font:18px system-ui}}video{{width:100%}}td,th{{padding:12px;text-align:left}}a{{color:#59bcff}}</style>
<h1>Recorded training layouts and warning times</h1><video controls poster="timelapse-poster.png" src="warning-timelapse.mp4"></video>
<p>Top-down data replay, not Unreal camera footage. Sampled training episodes use different scenarios; only evaluations on matching seeds measure paired improvement. All outcomes, including regressions, are retained.</p>
<table><tr><th>Layout</th><th>Evidence</th><th>Mean warning / drone</th><th>Cases</th></tr>{rows}</table>
<p><a href="training.jsonl">Per-episode logs</a> · <a href="recording-manifest.json">Recording manifest</a> · <a href="timelapse-chapters.json">Video chapters</a></p></html>'''
    (folder / "index.html").write_text(page, encoding="utf-8")
    print(f"Rendered {movie}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--episodes", help="all or comma-separated sampled training episodes; default: baseline and checkpoint evaluations")
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--seconds-per-stage", type=float, default=3.)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not 0 < args.seconds_per_stage <= 60 or not 1 <= args.fps <= 60:
        parser.error("seconds-per-stage must be in (0,60] and fps in 1..60")
    summary, entries = load_recordings(args.run, episodes=parse_episodes(args.episodes), best_only=args.best_only)
    if summary.get("schema") == RECORDING_SCHEMA:
        return render_console_run(args.run, summary, entries, preview_only=args.preview_only,
                                  seconds_per_stage=args.seconds_per_stage, fps=args.fps)
    checkpoints = summary["protocol"]["checkpoints"]
    runs = [json.loads((args.run / f"evaluation-{i:04d}.json").read_text())[0] for i in checkpoints]
    final = draw_frame(summary, runs, len(runs)-1, 1)
    final.save(args.run / "timelapse-poster.png")
    if args.preview_only: return
    movie = args.run / "warning-timelapse.mp4"
    if movie.exists(): raise FileExistsError(movie)
    writer = imageio_ffmpeg.write_frames(str(movie), (W,H), fps=15, codec="libx264", quality=8,
                                        macro_block_size=2, output_params=["-movflags", "+faststart"])
    writer.send(None)
    try:
        for stage in range(len(runs)):
            for i in range(90):
                writer.send(np.asarray(draw_frame(summary, runs, stage, i/89)))
        for _ in range(60): writer.send(np.asarray(final))
    finally:
        writer.close()
    rows = "".join(f"<tr><td>{html.escape(e['checkpoint'])}</td><td>{e['mean_drone_warning_s']:.3f} s</td><td>{e['team_warning_s']:.3f} s</td><td>{e['mean_cost']:.2f}</td></tr>" for e in summary['evaluations'])
    comparisons = []
    for name, values in summary["final_minus_control"].items():
        v = values["mean_drone_warning_s"]; lo, hi = v["bootstrap_95_ci"]
        comparisons.append(f"<li>Final vs {name}: {v['mean']:+.3f} s per drone (paired 95% bootstrap interval {lo:+.3f} to {hi:+.3f} s). Team warning: {values['team_warning_s']['mean']:+.3f} s.</li>")
    change = summary["final_minus_control"]["untrained"]["mean_drone_warning_s"]
    team_change = summary["final_minus_control"]["untrained"]["team_warning_s"]["mean"]
    conclusion = ("A positive per-drone gain was observed in this small paired evaluation."
                  if change["bootstrap_95_ci"][0] > 0 else
                  "No clear per-drone improvement was established in this pilot; the video preserves the measured fluctuations.")
    if abs(team_change) < 1e-8:
        conclusion += " Team-level preparation time did not improve: detection already happened on the first sensing look."
    buttons = "".join(f'<button data-seconds="{i*6}">Episode {n}</button>' for i,n in enumerate(checkpoints))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Istana | Warning-time training</title><style>body{{background:#0a1220;color:#edf4ff;font:17px system-ui;margin:auto;padding:28px;max-width:1200px}}h1{{font-size:36px;margin:8px 0}}p,li{{line-height:1.65;color:#b9c9df}}video{{width:100%;border-radius:16px;background:#111e30}}a{{color:#65c8ff}}button{{background:#213752;color:#fff;border:1px solid #456482;border-radius:8px;padding:10px 16px;margin:6px 5px 15px 0;cursor:pointer}}table{{border-collapse:collapse;width:100%;max-width:800px}}td,th{{text-align:left;border-bottom:1px solid #314259;padding:12px}}.eyebrow{{color:#65e2b4;font-size:14px;letter-spacing:2px}}.box{{background:#111e30;border-radius:12px;padding:20px;margin:22px 0}}</style>
<div class="eyebrow">NATIVE ISTANA · MEASURED TRAINING REPLAY</div><h1>Blue Team: warning-time timelapse</h1>
<p>{summary['protocol']['episodes']} training episodes · {len(checkpoints)} fixed checkpoints · {len(summary['protocol']['evaluation_seeds'])} paired held-out scenarios · one training seed.</p>
<video id="movie" controls preload="metadata" poster="timelapse-poster.png"><source src="warning-timelapse.mp4" type="video/mp4"></video>
<div>{buttons}</div><script>document.querySelectorAll('button[data-seconds]').forEach(b=>b.onclick=()=>{{let v=document.getElementById('movie');v.currentTime=Number(b.dataset.seconds);v.play();}});</script>
<div class="box"><strong>What the warning times mean</strong><p>Team warning is the interval from first detection of any Red drone until the first drone enters the target zone. Per-drone warning is each drone’s arrival minus its own first detection, averaged over all drones; undetected arrivals receive zero. Training optimizes the latter. Unresolved episodes fail the run instead of being excluded.</p>
<p>The target endpoint is entry into a 20 m zone, not collision with the building. This is a top-down replay of recorded native trajectories, not 3D camera footage. Sensor markers are actual accepted, surface-mounted placements. The map shows one case selected before training; numerical checkpoint results average all held-out cases.</p></div>
<h2>Measured results</h2><p><strong>{conclusion}</strong></p><table><thead><tr><th>Training episode</th><th>Per-drone warning</th><th>Team warning</th><th>Mean cost / 3</th></tr></thead><tbody>{rows}</tbody></table>
<ul>{''.join(comparisons)}</ul><p>Intervals reflect variation across eight default paired scenarios, not variation across independent training runs. The untrained policy is a random legal-layout policy, not the earlier published temporal checkpoint. Greedy is the existing non-RL temporal-public control. No checkpoint was selected for favorable results.</p>
<p>Red uses seeded rotations of scripted swarms, not a trained adversarial policy. Synthetic sensing has no occlusion model. First-look saturation can leave team-level warning unchanged even when individual detections become earlier. Improvement is not guaranteed.</p>
<p><a href="warning-timelapse.mp4" download>Download MP4</a> · <a href="summary.json">Full measured summary</a> · <a href="protocol.json">Predeclared protocol</a> · <a href="training.jsonl">All training episodes</a> · <a href="policy-{checkpoints[-1]:04d}.json">Trained policy</a></p></html>'''
    (args.run / "index.html").write_text(page, encoding="utf-8")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.run.iterdir() if p.is_file() and p.name != "artifact-sha256.json"}
    (args.run / "artifact-sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {movie}")


if __name__ == "__main__": main()
