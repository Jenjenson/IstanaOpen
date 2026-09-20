"""Fresh native cosmetic flying props and scanning sensor; no evaluation."""
import argparse
import json
from pathlib import Path
from capture_warning_3d import capture
from triad_rl.istana_live import IstanaLiveClient


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = Path(__file__).resolve().parents[3] / "Saved"
    fps, frames, chapters = 20, [], []
    with IstanaLiveClient(port=args.port) as client:
        client.reset(4900200)
        context = client.get_blue_context()
        site = next(i for i in (8,0,16,24,2) if i not in context["publicSnapshot"]["blocked_sites"])
        client.deploy([{"profileId":"eo", "siteId":site}])
        for angle in ("wide", "overhead", "tracking", "sensor", "together"):
            chapters.append({"label":angle.title(), "seconds":len(frames)/fps})
            capture(client, saved, "showcase", presentation_seconds=len(frames)/fps, showcase_angle=angle)
            for i in range(8*fps):
                t = len(frames)/fps
                shot = capture(client, saved, "showcase", presentation_seconds=t, showcase_angle=angle)
                assert shot["step"] == 0 and shot["elapsed_s"] == 0 and shot["cosmetic_motion_only"]
                assert len(shot["cosmetic_prop_locations_cm"]) == 3
                frames.append({**shot, "presentation_s":t, "angle":angle})
            (args.output / f"capture-{angle}.json").write_text(json.dumps(frames[-8*fps:],indent=2))
            print(json.dumps({"angle":angle,"frames":len(frames),"cosmetic_drones":3,"solver_steps":0}),flush=True)
    assert frames[0]["cosmetic_prop_locations_cm"] != frames[1]["cosmetic_prop_locations_cm"]
    manifest = {"fps":fps,"duration_s":len(frames)/fps,"chapters":chapters,"frames":frames,
                "native_frames":True,"cosmetic_motion_only":True,"simulation_steps":0,
                "policy_loaded":False,"evaluation_performed":False,"reused_old_frames":False}
    (args.output / "capture-manifest.json").write_text(json.dumps(manifest,indent=2))


if __name__ == "__main__": main()
