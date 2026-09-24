"""Record native Unreal frames for fixed warning-policy checkpoints.

Requires a rendered IstanaBlueLive scene with -IstanaAllowCapture. Frames come
from Unreal's screenshot pipeline, never reconstructed or image-generated.
The captured replay must reproduce the selected evaluation's event evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from triad_rl.istana_live import IstanaLiveClient
from triad_rl.warning_policy import warning_metrics
from triad_rl.training_recording import load_recordings, parse_episodes, RECORDING_SCHEMA
from train_warning_live import red_centers, write_json
from triad_rl.warning_scenario import approach_centers, scenario_contract, digest


def capture(client, saved, view="overview", site_id=None, orbit_degrees=None, presentation_seconds=None, showcase_angle=None):
    fields = {"runId": client.red_context["runId"], "expectedStep": client.completed_steps, "view": view}
    if site_id is not None: fields["siteId"] = site_id
    if orbit_degrees is not None: fields["orbitDegrees"] = orbit_degrees
    if presentation_seconds is not None: fields["presentationSeconds"] = presentation_seconds
    if showcase_angle is not None: fields["showcaseAngle"] = showcase_angle
    response = client.request("blue_capture", **fields)
    path = (saved / response["captureRelativeToSaved"]).resolve()
    if not path.is_relative_to(saved.resolve()) or path.suffix.lower() != ".png":
        raise ValueError("Invalid native capture destination")
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        try:
            with Image.open(path) as image:
                image.load()
                if image.size[0] < 1000: raise ValueError("Capture viewport too small")
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "elapsed_s": client.elapsed_seconds, "step": client.completed_steps,
                    "camera_location_cm": response.get("cameraLocationCm"),
                    "camera_target_cm": response.get("cameraTargetCm"),
                    "camera_rotation": response.get("cameraRotation"),
                    "sensor_screen_anchors": response.get("sensorScreenAnchors", []),
                    "nearest_visible_drone_cm": response.get("nearestVisibleDroneActorToTargetCm"),
                    "cosmetic_prop_locations_cm": response.get("cosmeticPropLocationsCm"),
                    "cosmetic_motion_only": response.get("cosmeticMotionOnly", False)}
        except (OSError, SyntaxError):
            time.sleep(.05)
    raise TimeoutError(f"Native PNG was not completed: {path}")


def prepare_replay(client, expected, protocol):
    """Deploy recorded actions directly; never resample a saved actor."""
    red = client.reset(expected["seed"])
    context = client.get_blue_context()
    if "red_context" in expected:
        if digest(scenario_contract(red, context)) != digest(
                scenario_contract(expected["red_context"], expected["context"])):
            raise ValueError("Capture scene differs from the recorded physical/weather contract")
    elif protocol.get("schema") == RECORDING_SCHEMA:
        raise ValueError("Console recording has no frozen Red context; exact capture cannot be verified")
    is_approach = protocol.get("schema") == "istana.native_warning_approach.v2"
    if is_approach and digest(scenario_contract(red, context)) != expected["physical_contract_sha256"]:
        raise ValueError("Capture scene differs from the frozen training scenario")
    placements = expected["placements"]
    client.deploy(placements)
    if "red_decision" in expected:
        centers = expected["red_decision"]["centers"]
    else:
        centers = (approach_centers if is_approach else red_centers)(red, expected["seed"])
    client.place_red(centers)
    return placements


def gallery_profiles(context, selected_ids=None):
    """Show only selected, available limited-FOV equipment in new captures."""
    available = context["publicSnapshot"]["available_sensor_ids"]
    directional = {row["id"] for row in context["catalogue"]
                   if row.get("directional") and row["id"] in available}
    if selected_ids is None:
        return [row["id"] for row in context["catalogue"] if row["id"] in directional]
    if not selected_ids or not set(selected_ids) <= directional:
        raise ValueError("The recorded sensor selection must contain only available limited-FOV profiles")
    return list(selected_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--step-batch", type=int, default=10)
    parser.add_argument("--gallery-only", action="store_true")
    parser.add_argument("--detail-views", action="store_true", help="Presentation-only drone/sensor close-ups during the unchanged replay")
    parser.add_argument("--episodes", help="Recorded sampled episodes: all or comma-separated numbers (e.g. 500,1000,1500)")
    parser.add_argument("--best-only", action="store_true", help="Capture the selected best policy's separate test case")
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--cinematic-orbit", action="store_true", help="Append an overview orbit of the final frozen layout")
    parser.add_argument("--orbit-frames", type=int, default=120, help="Presentation frames in the optional frozen-scene orbit")
    args = parser.parse_args()
    if not 1 <= args.step_batch <= 250: parser.error("step-batch must be 1..250")
    if not 2 <= args.orbit_frames <= 600: parser.error("orbit-frames must be 2..600")
    args.output.mkdir(parents=True, exist_ok=False)
    saved = Path(__file__).resolve().parents[3] / "Saved"
    if args.gallery_only:
        path = args.run / "recording-manifest.json"
        summary, entries = (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}), []
    else:
        summary, entries = load_recordings(
            args.run, episodes=parse_episodes(args.episodes), best_only=args.best_only)
    protocol = summary.get("protocol", summary)
    shots, recordings = [], []
    with IstanaLiveClient(port=args.port) as client:
        # Product close-ups are presentation-only resets, not extra training or
        # selected evaluation episodes. One supported roof site for all heads.
        profiles = []
        if not args.skip_gallery:
            client.reset(4900000)
            profiles = gallery_profiles(client.get_blue_context(),
                summary.get("configuration", {}).get("sensorIds"))
        for index, profile in enumerate(profiles):
            client.reset(4900000 + index)
            context = client.get_blue_context()
            blocked = context["publicSnapshot"]["blocked_sites"]
            site = next(i for i in (8,0,16,24,2) if i not in blocked)
            sensor = next(row for row in context["catalogue"] if row["id"] == profile)
            yaw = sensor.get("yaw_bins_deg", [0.])[0]
            pitch = sensor.get("pitch_bins_deg", [0.])[0]
            client.deploy([{"profileId": profile, "siteId": site, "yawDeg": yaw, "pitchDeg": pitch}])
            # Warm the renderer at the new camera; first frame is retained but
            # not used for the hero image (temporal exposure/AA convergence).
            capture(client, saved, "sensor", site)
            time.sleep(.4)
            image = capture(client, saved, "sensor", site)
            shots.append({"profile": profile, "profile_label": sensor.get("label", profile),
                          "profile_specifications": sensor, "site_id": site, **image})
            print(json.dumps({"gallery": profile, **image}), flush=True)
        write_json(args.output / "gallery.json", shots)
        if args.gallery_only: return
        for entry in entries:
            number, expected = entry["checkpoint"], entry["run"]
            placements = prepare_replay(client, expected, protocol)
            # A close tracking view makes the real drone mesh visible during
            # the distant approach; the original overview shows deployment.
            switch_time = max(0., (expected["metrics"]["first_detection_s"] or 0.) - 3.)
            def view():
                if args.detail_views:
                    if client.elapsed_seconds < 8: return "drone"
                    if client.elapsed_seconds < 16 and placements: return "sensor"
                    return "overview"
                return "drone" if client.elapsed_seconds < switch_time else "overview"
            def shot():
                return capture(client, saved, view(), placements[0]["siteId"] if view() == "sensor" else None)
            shot()
            time.sleep(.4)
            frames = [{**shot(), "view": view()}]
            previous_view = view()
            for _ in range(4000):
                response = client.step(args.step_batch)
                blue = response["blueObservation"]
                if view() != previous_view:
                    shot()
                    time.sleep(.2)
                    previous_view = view()
                frames.append({**shot(), "view": view()})
                if blue["terminated"] or blue["truncated"]: break
            else: raise RuntimeError("Capture episode exceeded step bound")
            metrics = warning_metrics(blue)
            if blue["warningEvidenceForEvaluationOnly"] != expected["warning_evidence"]:
                raise ValueError("Presentation change altered native detection/arrival evidence")
            if args.cinematic_orbit:
                # Orbit frozen simulation state. These are presentation frames,
                # not extra successful detections or additional evaluation time.
                for index in range(args.orbit_frames):
                    frames.append({**capture(client, saved, "overview", orbit_degrees=-30 + 60 * index / (args.orbit_frames-1)),
                                   "view": "overview", "presentation_only": True})
            record = {"checkpoint": number, "seed": expected["seed"], "placements": placements,
                      "frames": frames, "metrics": metrics,
                      "label": entry["label"], "kind": entry["kind"],
                      "recorded_metrics": expected["metrics"],
                      "evaluation": entry.get("evaluation", expected["metrics"]),
                      "evaluation_seeds": entry.get("evaluation_seeds", [expected["seed"]]),
                      "context": expected["context"],
                      "warning_evidence": blue["warningEvidenceForEvaluationOnly"],
                      "matched_original_evidence": True}
            recordings.append(record)
            write_json(args.output / f"capture-{entry['kind']}-{number:04d}.json", record)
            print(json.dumps({"checkpoint": number, "frames": len(frames), "metrics": metrics}), flush=True)
    write_json(args.output / "capture-manifest.json", {"schema": "istana.native_3d_capture.v2",
               "source_run": str(args.run.resolve()), "gallery": shots, "recordings": recordings,
               "presentation_summary": summary,
               "endpoint": "20m target-zone arrival, NOT impact", "native_frames": True,
               "warning_evidence_unchanged": True, "rendering_only_changes": True,
               "observer_markers": False})


if __name__ == "__main__": main()
