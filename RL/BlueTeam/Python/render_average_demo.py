"""Presentation-only composition of native visuals and separately saved averages.

No simulation, checkpoint execution, scenario selection, or evaluation occurs.
The identical background sequence is used for both result sections.
"""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from render_warning_3d import checked_image
from render_warning_timelapse import font


def compose(scene, section, early, final, count, fresh=False):
    out = Image.new("RGB", (1920,1280), "#0b1119")
    out.paste(scene, (0,200))
    d = ImageDraw.Draw(out)
    if section == "summary":
        title = "SAVED SIMULATION RESULTS | AVERAGE WARNING TIME"
        values = f"Episode 1: {early:.2f} s    Episode 256: {final:.2f} s    Gained: {final-early:+.2f} s"
    else:
        number, value = (1, early) if section == "early" else (256, final)
        title = f"EPISODE {number} | SAVED SIMULATION RESULT"
        values = f"Average Warning Time: {value:.2f} s"
    d.text((28,10), title, font=font(39), fill="#e8edf4")
    d.text((28,62), values, font=font(43), fill="#92ddcd")
    disclosure = "Fresh model showcase — NOT a checkpoint replay or the source of these averages." if fresh else "Illustrative archived footage — NOT the replay that produced these averages."
    d.text((28,122), disclosure, font=font(27), fill="#ffcf70")
    d.text((28,163), f"Mean across {count} saved simulated cases | Detection to 20 m target-zone arrival, not impact | Not real-site security performance", font=font(24), fill="#bdc9d9")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    manifest = json.loads((args.capture / "capture-manifest.json").read_text())
    summary = json.loads((args.results / "summary.json").read_text())
    early = next(r for r in summary["evaluations"] if r["checkpoint"] == "0001")["team_warning_s"]
    final = next(r for r in summary["evaluations"] if r["checkpoint"] == "0256")["team_warning_s"]
    count = len(summary["protocol"]["evaluation_seeds"])
    # One fixed pre-existing sequence, repeated identically. No outcome-based
    # selection and no implication that the placements represent Episode 256.
    background = manifest["recordings"][0]
    frames = background["frames"]
    fresh = manifest.get("fresh_neutral_showcase", False)
    args.output.mkdir(parents=True, exist_ok=False)
    movie = args.output / "combined-average-warning-demo.mp4"
    fps, speed, cursor = 20, (1 if fresh else 2), 0
    time_key = "presentation_s" if fresh else "elapsed_s"
    chapters = []
    writer = imageio_ffmpeg.write_frames(str(movie), (1920,1280), fps=fps,
        codec="libx264", quality=8, macro_block_size=2,
        output_params=["-movflags", "+faststart", "-metadata", "comment=Illustrative native visuals; displayed averages are separate saved simulation results, not timings produced by this footage.",
                       "-metadata", "copyright=Map data (c) OpenStreetMap contributors / ODbL; IstanaOpen original art CC0"])
    writer.send(None)
    try:
        for section in ("early", "final"):
            chapters.append({"label": "Episode 1 averages" if section == "early" else "Episode 256 averages", "seconds": cursor/fps})
            base = cursor
            for i, native in enumerate(frames):
                scene = checked_image(native)
                if scene.size != (1920,1080): raise ValueError("Unexpected native resolution")
                composed = compose(scene, section, early, final, count, fresh)
                if section == "early" and i == 0: composed.save(args.output / "poster.jpg", quality=95)
                next_time = frames[i+1][time_key] if i+1 < len(frames) else native[time_key] + (1/fps if fresh else .5)
                end = base + round(next_time / speed * fps)
                for _ in range(max(1, end-cursor)):
                    writer.send(np.asarray(composed)); cursor += 1
        chapters.append({"label": "Final comparison", "seconds": cursor/fps})
        final_frame = compose(checked_image(frames[-1]), "summary", early, final, count, fresh)
        final_frame.save(args.output / "summary-preview.jpg", quality=95)
        for _ in range(5*fps): writer.send(np.asarray(final_frame)); cursor += 1
    finally:
        writer.close()
    metadata = {"early_episode": 1, "final_episode": 256, "early_average_warning_s": early,
        "final_average_warning_s": final, "average_warning_gained_s": final-early,
        "averaged_cases": count, "fps": fps, "duration_s": cursor/fps, "chapters": chapters,
        "results_source": str(args.results.resolve()), "footage_source": str(args.capture.resolve()),
        "background_checkpoint": background["checkpoint"], "identical_background_repeated": True,
        "footage_did_not_produce_displayed_averages": True, "no_new_simulation_or_evaluation": True,
        "fresh_neutral_showcase": fresh,
        "gain_bootstrap_95_ci": summary["final_minus_control"]["untrained"]["team_warning_s"]["bootstrap_95_ci"]}
    (args.output / "measured-values.json").write_text(json.dumps(metadata,indent=2), encoding="utf-8")
    (args.output / "CREDITS.txt").write_text("Map data © OpenStreetMap contributors / ODbL: https://www.openstreetmap.org/copyright\nIstanaOpen original art CC0. Actual Unreal frames, not a replay of the displayed checkpoints.\n", encoding="utf-8")
    payload = json.dumps(chapters)
    page = '''<!doctype html><html><meta charset="utf-8"><title>Combined average-warning demo</title>
<style>body{margin:20px;background:#0b1119;color:#e8edf4;font:18px system-ui}video{width:100%;max-height:80vh;background:black}button{padding:12px;margin:5px;background:#203349;color:white;border:0;border-radius:5px}a{color:#92ddcd}p{max-width:1100px}</style>
<h1>Combined average-warning demo</h1><p>One MP4. Saved Episode 1 and Episode 256 averages over illustrative archived footage. All labels are embedded in the video.</p>
<video id="film" controls preload="metadata" poster="poster.jpg" src="combined-average-warning-demo.mp4"></video><div id="chapters"></div>
<p><a href="combined-average-warning-demo.mp4" download>Download combined MP4</a> · <a href="measured-values.json">Measured values and provenance</a> · <a href="CREDITS.txt">Credits</a></p>
<p>The +4.75 s difference is descriptive: its paired 95% interval includes zero (−0.50 to +9.88 s). Coverage fell from 100% to 52.9%; this is not evidence of overall improvement. No real-site security claims.</p>
<script>for(const c of CHAPTERS){const b=document.createElement('button');b.textContent=c.label;b.onclick=()=>{const v=document.getElementById('film');v.currentTime=c.seconds;v.play();v.scrollIntoView({block:'start'})};document.getElementById('chapters').append(b)}</script></html>'''.replace("CHAPTERS", payload)
    if fresh:
        page = page.replace("illustrative archived footage", "fresh native model-showcase footage (camera motion; simulation frozen)")
    (args.output / "index.html").write_text(page, encoding="utf-8")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file()}
    (args.output / "artifact-sha256.json").write_text(json.dumps(hashes,indent=2), encoding="utf-8")
    print(json.dumps({"mp4": str(movie.resolve()), **metadata}), flush=True)


if __name__ == "__main__": main()
