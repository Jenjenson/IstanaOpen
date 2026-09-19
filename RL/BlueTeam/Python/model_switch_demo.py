"""Local saved-layout selector with actual native deployment and canvas recording.

No policy inference, Red deployment, stepping, or performance evaluation occurs.
"""
import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import subprocess
import threading
import time

import imageio_ffmpeg
from capture_warning_3d import capture
from triad_rl.istana_live import IstanaLiveClient

ROOT = Path(__file__).resolve().parents[3]
LAYOUT_SOURCE = ROOT / "RL/BlueTeam/Results/model-switch-demo"
MODELS = (("rl", "RL Policy", "0128"), ("greedy", "Greedy Algorithm", "greedy"),
          ("initial", "Initial Policy", "0000"))


def load_layouts(source):
    layouts = {}
    for key, label, checkpoint in MODELS:
        path = source / f"evaluation-{checkpoint}.json"
        row = json.loads(path.read_text())[0]
        layouts[key] = dict(label=label, seed=row["seed"], placements=row["placements"],
                            source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if len({r["seed"] for r in layouts.values()}) != 1:
        raise ValueError("Saved layouts must use the same scenario seed")
    return layouts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=9051)
    p.add_argument("--bridge-port", type=int, default=8766)
    p.add_argument("--output", type=Path, default=ROOT / "Saved/ModelSwitchDemo")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    layouts = load_layouts(LAYOUT_SOURCE)
    token, lock, events, assets = secrets.token_urlsafe(32), threading.Lock(), [], {}
    client = IstanaLiveClient(port=a.bridge_port)
    origin = f"http://127.0.0.1:{a.port}"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, content_type="application/json", code=200):
            if not isinstance(data, bytes): data = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.headers.get("Host") != f"127.0.0.1:{a.port}":
                return self.reply(dict(error="Local host required"), code=403)
            if self.path == "/":
                return self.reply(Path(__file__).with_name("model_switch_demo.html").read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/session":
                return self.reply(dict(token=token, models=[dict(id=k, label=v["label"]) for k,v in layouts.items()]))
            if self.path in assets:
                return self.reply(assets[self.path].read_bytes(), "image/png")
            if self.path == "/model-switching.mp4" and (a.output/"model-switching.mp4").exists():
                return self.reply((a.output/"model-switching.mp4").read_bytes(), "video/mp4")
            self.reply(dict(error="Not found"), code=404)

        def do_POST(self):
            if self.headers.get("Host") != f"127.0.0.1:{a.port}" or self.headers.get("Origin") != origin or self.headers.get("X-Demo-Token") != token:
                return self.reply(dict(error="Local session required"), code=403)
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > (100_000_000 if self.path == "/api/recording" else 4096):
                return self.reply(dict(error="Invalid request size"), code=413)
            body = self.rfile.read(length)
            try:
                with lock:
                    if self.path == "/api/apply":
                        key = json.loads(body)["model"]
                        if key not in layouts: raise ValueError("Unknown saved model")
                        row = layouts[key]
                        client.reset(row["seed"])
                        context = client.get_blue_context()
                        accepted = client.deploy(row["placements"])
                        shots = {}
                        for view in ("sensor", "layout"):
                            site = row["placements"][0]["siteId"] if view == "sensor" else None
                            capture(client, ROOT/"Saved", view, site)
                            time.sleep(.3)
                            frame = capture(client, ROOT/"Saved", view, site)
                            if view == "layout":
                                anchors = frame["sensor_screen_anchors"]
                                if len(anchors) != len(row["placements"]) or not all(
                                    30 < point["x"] < 1810 and 100 < point["y"] < 1040 for point in anchors):
                                    raise ValueError("Fixed camera does not show every sensor clearly")
                            url = f"/frame/{len(events)}-{view}.png"
                            assets[url] = Path(frame["path"])
                            shots[view] = dict(url=url, **frame)
                        event = dict(model=key, **row, accepted=accepted, shots=shots,
                                     completed_steps=client.completed_steps,
                                     constraints=dict(budget=context["publicSnapshot"].get("budget"),
                                                      max_sites=context["publicSnapshot"].get("max_sites")))
                        events.append(event)
                        (a.output/"deployment-log.json").write_text(json.dumps(events, indent=2))
                        return self.reply(dict(model=key, label=row["label"], shots=shots))
                    if self.path == "/api/recording":
                        webm = a.output/"model-switching.webm"
                        mp4 = a.output/"model-switching.mp4"
                        if mp4.exists(): raise ValueError("Recording exists; choose a new output directory")
                        webm.write_bytes(body)
                        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-i", str(webm),
                            "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
                            "-movflags", "+faststart", str(mp4)], check=True)
                        (a.output/"CREDITS.txt").write_text("Native Unreal preview of archived layouts; no model inference or evaluation.\nMap data (c) OpenStreetMap contributors / ODbL. IstanaOpen original art CC0.\n")
                        return self.reply(dict(url="/model-switching.mp4"))
                self.reply(dict(error="Not found"), code=404)
            except Exception as exc:
                self.reply(dict(error=str(exc)), code=400)

    print(f"Saved-layout interface: {origin}", flush=True)
    try: ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
    finally: client.close()


if __name__ == "__main__": main()
