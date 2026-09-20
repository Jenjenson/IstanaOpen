"""Fresh native camera footage of frozen models; no policy or simulation steps."""
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
    frames = []
    with IstanaLiveClient(port=args.port) as client:
        client.reset(4900100)
        context = client.get_blue_context()
        # A single prop on the first existing supported display mount. No policy
        # loaded, placement search, opponent commands, step call or evaluation.
        blocked = context["publicSnapshot"]["blocked_sites"]
        site = next(i for i in (8,0,16,24,2) if i not in blocked)
        client.deploy([{"profileId": "eo", "siteId": site}])
        for view in ("overview", "sensor", "drone_model"):
            site_id = site if view == "sensor" else None
            capture(client, saved, view, site_id, -12)
            for i in range(60):
                shot = capture(client, saved, view, site_id, -12+24*i/59)
                assert shot["step"] == 0 and shot["elapsed_s"] == 0
                frames.append({**shot, "presentation_s": len(frames)/20, "view": view})
            print(json.dumps({"completed_view": view, "frames": len(frames), "simulation_steps": 0}), flush=True)
    manifest = {"fresh_neutral_showcase": True, "observer_markers": False,
                "simulation_steps": 0, "policy_loaded": False, "evaluation_performed": False,
                "recordings": [{"checkpoint": "neutral model showcase", "frames": frames}]}
    (args.output / "capture-manifest.json").write_text(json.dumps(manifest,indent=2), encoding="utf-8")


if __name__ == "__main__": main()
