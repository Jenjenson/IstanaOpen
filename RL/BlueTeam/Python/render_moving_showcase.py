"""Encode fresh 40-second native cosmetic motion footage with saved-results labels."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from render_warning_3d import checked_image
from render_warning_timelapse import font


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("results", type=Path)
    a = p.parse_args()
    m = json.loads((a.capture / "capture-manifest.json").read_text())
    s = json.loads((a.results / "summary.json").read_text())
    early = next(r["team_warning_s"] for r in s["evaluations"] if r["checkpoint"] == "0001")
    final = next(r["team_warning_s"] for r in s["evaluations"] if r["checkpoint"] == "0256")
    movie = a.capture / "fresh-moving-showcase.mp4"
    if movie.exists(): raise FileExistsError(movie)
    writer = imageio_ffmpeg.write_frames(str(movie),(1920,1240),fps=m["fps"],codec="libx264",quality=8,macro_block_size=2,
        output_params=["-movflags","+faststart","-metadata","comment=Cosmetic display animation; separate saved benchmark averages; no evaluation.",
                       "-metadata","copyright=Map data (c) OpenStreetMap contributors / ODbL; IstanaOpen original art CC0"])
    writer.send(None)
    try:
        for i,f in enumerate(m["frames"]):
            scene = checked_image(f)
            if scene.size != (1920,1080): raise ValueError("Unexpected frame size")
            out = Image.new("RGB",(1920,1240),"#0b1119"); out.paste(scene,(0,160))
            d = ImageDraw.Draw(out)
            d.text((25,8), f"ILLUSTRATIVE MOTION | {f['angle'].upper()} CAMERA",font=font(34),fill="#e8edf4")
            d.text((25,55), f"Saved average warning: Episode 1 {early:.2f} s  |  Episode 256 {final:.2f} s  |  Gain {final-early:+.2f} s",font=font(35),fill="#92ddcd")
            d.text((25,110),"Cosmetic flight and sensor scanning only. Not a policy replay; these visuals did not produce the saved averages.",font=font(27),fill="#ffcf70")
            writer.send(np.asarray(out))
            if i == 0: out.save(a.capture/"poster.jpg",quality=95)
            if i in (321,641): out.save(a.capture/f"preview-{f['angle']}.jpg",quality=95)
    finally: writer.close()
    meta = {"early_average_warning_s":early,"final_average_warning_s":final,"difference_s":final-early,
            "averages_source":str(a.results.resolve()),"duration_s":m["duration_s"],"chapters":m["chapters"],
            "native_cosmetic_animation":True,"no_policy_replay_or_evaluation":True,"old_frames_reused":False}
    (a.capture/"measured-values.json").write_text(json.dumps(meta,indent=2))
    (a.capture/"CREDITS.txt").write_text("Map data © OpenStreetMap contributors / ODbL: https://www.openstreetmap.org/copyright\nIstanaOpen original art CC0. Fresh native Unreal footage. Cosmetic movement, not measured policy replay.\n",encoding="utf-8")
    html = '''<!doctype html><html><meta charset="utf-8"><title>Fresh moving 3D footage</title><style>body{background:#0b1119;color:#eee;font:18px system-ui;margin:20px}video{width:100%;max-height:80vh}button{padding:12px;margin:5px;background:#203349;color:white;border:0}a{color:#92ddcd}</style>
<h1>Fresh moving 3D footage · 40 seconds</h1><p>Three flying cosmetic drone models, spinning propellers, scanning sensor and five camera angles. No old footage reused.</p>
<video id="film" controls src="fresh-moving-showcase.mp4" poster="poster.jpg"></video><div id="chapters"></div>
<p><a href="fresh-moving-showcase.mp4" download>Download new moving MP4</a> · <a href="measured-values.json">Saved averages and provenance</a> · <a href="CREDITS.txt">Credits</a></p>
<p>Predefined cosmetic motion, not a checkpoint replay or tactical evaluation. Saved averages are separate results; the observed +4.75 s mean difference is not statistically conclusive and does not imply overall improvement or real-site security performance.</p>
<script>for(const c of CHAPTERS){const b=document.createElement('button');b.textContent=c.label;b.onclick=()=>{const v=document.getElementById('film');v.currentTime=c.seconds;v.play();v.scrollIntoView({block:'start'})};document.getElementById('chapters').append(b)}</script></html>'''.replace("CHAPTERS",json.dumps(m["chapters"]))
    (a.capture/"index.html").write_text(html,encoding="utf-8")
    (a.capture/"artifact-sha256.json").write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.capture.iterdir() if p.is_file()},indent=2))
    print(json.dumps({"movie":str(movie.resolve()),**meta}),flush=True)


if __name__ == "__main__": main()
