"""Read paired native evaluations without recomputing sensing in the browser.

The capture tool invokes the current live Blue planner and native Red simulation.
This module only validates the saved evidence and derives display statistics.
"""
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from triad_rl.training_workbench import BALANCED_LANE_YAWS
from triad_rl.trained_models import TrainedModelRegistry


ROOT = Path(__file__).resolve().parents[1] / "Results/native-placement-comparison"
WORKBENCH_ROOT = Path(__file__).resolve().parents[1] / "Results/workbench-placement-comparison"
POLICY_LABELS = {"406": "Temporal RL · policy A", "407": "Temporal RL · policy B",
                 "408": "Temporal RL · policy C"}
COMPARISON_LAYOUTS = (
    {"id": "matched_common_sense", "label": "Matched common sense · 3 sensors",
     "kind": "native_results"},
    {"id": "directional_balanced_8", "label": "Eight directional sensors · measured workbench",
     "kind": "native_results"},
)


def policy_label(identifier):
    return POLICY_LABELS.get(str(identifier), f"RL policy {identifier}")


def _average(values):
    return mean(values) if values else None


def _targets(view):
    rows = view["metrics"]["target_results"]
    result = {row["id"]: row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("Native comparison must contain unique target evidence")
    return result


def summarize(view):
    targets = list(_targets(view).values())
    detected = [row["first_detection"] for row in targets if row["first_detection"] is not None]
    confirmed = [row["first_confirmation"] for row in targets if row["first_confirmation"] is not None]
    metrics = view["metrics"]
    return {"target_count": len(targets), "detected_count": len(detected),
            "confirmed_count": len(confirmed),
            "timely_confirmed_count": sum(bool(row["timely_confirmed"]) for row in targets),
            "sensors_placed": len(view["placements"]), "cost": metrics["cost"],
            "first_detection_s": min(detected) if detected else None,
            "first_confirmation_s": min(confirmed) if confirmed else None,
            "mean_warning_s": metrics["mean_drone_warning_seconds_lower_bound"],
            "detection_rate": len(detected) / len(targets),
            "return": metrics["return"]}


def paired_timing(rl, baseline):
    left, right = _targets(rl), _targets(baseline)
    if left.keys() != right.keys():
        raise ValueError("Both native layouts must face the same target IDs")
    result = {}
    for label, field, count in (("detection", "first_detection", "detected"),
                                ("confirmation", "first_confirmation", "confirmed")):
        ids = sorted(key for key in left if left[key][field] is not None and right[key][field] is not None)
        result[f"shared_{count}_count"] = len(ids)
        result[f"mean_rl_{label}_s"] = _average([left[key][field] for key in ids])
        result[f"mean_baseline_{label}_s"] = _average([right[key][field] for key in ids])
        result[f"mean_{label}_delta_s"] = _average([left[key][field] - right[key][field] for key in ids])
    return result


def comparison_result(episode, protocol=None):
    rl, baseline = episode["rl"], episode["baseline"]
    summaries = {"rl": summarize(rl), "baseline": summarize(baseline)}
    delta_keys = ("detected_count", "confirmed_count", "timely_confirmed_count", "cost",
                  "sensors_placed", "mean_warning_s")
    return {"schema": "istana.native_sensor_comparison.v1", "episodeId": episode["id"],
            "label": episode["label"],
            "policyLabel": episode.get("policyLabel", policy_label(episode["policy"])),
            "method": "common_sense", "rl": deepcopy(rl), "baseline": deepcopy(baseline),
            "metrics": summaries, "timing": paired_timing(rl, baseline),
            "deltas": {key: summaries["rl"][key] - summaries["baseline"][key] for key in delta_keys},
            "deltaDirection": "RL minus common sense; negative timing differences mean earlier RL detection",
            "selection": {"label": "Common-sense layout", "kind": "fixed_common_sense",
                "explanation": "Use affordable sensors to add coverage over likely approaches, discount overlapping coverage, and stop at the shared budget or sensor limit. The rule uses public information and stays fixed across episodes."},
            "audit": {**deepcopy(episode.get("audit", {})), "recorded": True,
                "native_unreal_capture": True, "live_unreal": False, "policy_truth_access": False,
                "red_policy": "Scripted radial swarm placement and native objective-following flight",
                "blue_policy": "Existing temporal RL checkpoints with the current public-prior orientation adapter; orientations were not learned by these checkpoints",
                "warning_definition": "Mean max(0, zone entry minus first detection), missed targets zero",
                "protocol": deepcopy(protocol or {})}}


def _angle_distance(left, right):
    return abs((left - right + 180.) % 360. - 180.)


def _empty_layout_view(view):
    """Retain a placement contract while removing unrelated episode evidence."""
    result = deepcopy(view)
    result.update({"frames": [{"time": 0., "completedSteps": 0, "threats": [],
                               "detections": [], "tracks": []}],
                   "metrics": None, "warningEvidence": None, "outcome": "layout_preview",
                   "ended": True})
    return result


def _balanced_directional_placements(view):
    """Recreate the public full-circle coverage preset on the saved site grid."""
    catalogue = view["catalogue"]
    sensor_index, sensor = next(
        ((index, row) for index, row in enumerate(catalogue)
         if row.get("directional") and row.get("cost", float("inf")) <= 1.),
        (None, None))
    if sensor is None:
        raise ValueError("The comparison catalogue has no affordable directional sensor")
    blocked, used, rows = set(view.get("blockedSites", [])), set(), []
    for desired_yaw in BALANCED_LANE_YAWS:
        candidates = []
        for site_index, position in enumerate(view["sites"]):
            if site_index in blocked or site_index in used:
                continue
            x, y = position
            bearing = math.degrees(math.atan2(y, x)) % 360.
            candidates.append(((-_angle_distance(bearing, desired_yaw), math.hypot(x, y),
                                -site_index), site_index, position))
        if not candidates:
            raise ValueError("The comparison site grid cannot support eight directional sensors")
        _, site_index, position = max(candidates)
        used.add(site_index)
        rows.append({"sensor_id": sensor["id"], "sensor_index": sensor_index,
                     "cost": sensor["cost"], "position": deepcopy(position),
                     "yaw_deg": desired_yaw, "pitch_deg": 20.})
    return rows


def layout_preview_result(episode, protocol=None):
    """Show the new workbench preset without inventing cross-contract metrics."""
    rl = _empty_layout_view(episode["rl"])
    baseline = _empty_layout_view(episode["rl"])
    baseline.update({"label": "Eight directional sensors · balanced sectors",
                     "policy": "Fixed public-only workbench starting placement",
                     "placements": _balanced_directional_placements(baseline),
                     "budget": 8, "maxSensors": 8,
                     "selection": {"kind": "fixed_directional_workbench_start",
                         "rule": "outward perimeter cameras spread across representative coverage bearings"}})
    return {"schema": "istana.placement_layout_preview.v1", "episodeId": episode["id"],
            "label": f"{episode['label']} · placement preview",
            "policyLabel": policy_label(episode["policy"]),
            "method": "directional_balanced_8", "layoutOnly": True,
            "rl": rl, "baseline": baseline, "metrics": None, "timing": None, "deltas": None,
            "selection": {"label": "Eight directional sensors · workbench start",
                "kind": "fixed_directional_workbench_start",
                "explanation": (
                    "Place eight limited-FOV thermal cameras on supported perimeter sites, "
                    "centered on the benchmark sectors. This preview shows "
                    "placement geometry only; it does not reuse the archived sensing metrics.")},
            "fairness": {"matched": False,
                "description": (
                    "Layout preview only. The archived RL layout uses its original three-sensor "
                    "contract; the workbench start uses eight sensors. No performance comparison is made.")},
            "audit": {"recorded": True, "native_unreal_capture": False,
                "layout_preview_only": True, "policy_truth_access": False,
                "protocol": deepcopy(protocol or {})}}


def _validate_episode(episode):
    if episode.get("policy") not in POLICY_LABELS or not isinstance(episode.get("id"), str):
        raise ValueError("Unknown native comparison policy or episode")
    left, right = episode["rl"], episode["baseline"]
    for key in ("catalogue", "sites", "budget", "objectiveRadius", "seed"):
        if left[key] != right[key]:
            raise ValueError(f"Native comparison differs in shared {key}")
    if len(left["frames"]) != len(right["frames"]) or not left["frames"]:
        raise ValueError("Native comparison timelines do not match")
    for a, b in zip(left["frames"], right["frames"]):
        if a["time"] != b["time"]:
            raise ValueError("Native comparison sample times do not match")
        truth = lambda frame: [(row["id"], row["position"], row["active"]) for row in frame["threats"]]
        if truth(a) != truth(b):
            raise ValueError("Native comparison drone trajectories do not match")
    for view in (left, right):
        if view["metrics"]["cost"] > view["budget"] + 1e-8:
            raise ValueError("Native comparison exceeds its budget")
    paired_timing(left, right)


@lru_cache(maxsize=4)
def _read_bundle(root, manifest_stamp):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    name = manifest["bundle"]
    if not isinstance(name, str) or Path(name).name != name or name != "bundle.json.gz":
        raise ValueError("Native comparison bundle path is invalid")
    raw = (root / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["sha256"]:
        raise ValueError("Native comparison bundle checksum differs from its manifest")
    bundle = json.loads(gzip.decompress(raw), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid number: {value}")))
    if bundle.get("schema") not in {
            "istana.native_comparison_capture.v1",
            "istana.workbench_layout_comparison_capture.v1"}:
        raise ValueError("Unsupported native comparison capture")
    episodes = bundle["episodes"]
    if len({row["id"] for row in episodes}) != len(episodes):
        raise ValueError("Duplicate native comparison episode")
    for episode in episodes:
        _validate_episode(episode)
    return bundle


class NativeComparisons:
    def __init__(self, root=ROOT, workbench_root=None, model_root=None, registry=None):
        self.root = Path(root)
        self.workbench_root = Path(
            workbench_root if workbench_root is not None else
            WORKBENCH_ROOT if self.root.resolve() == ROOT.resolve() else self.root / "workbench")
        default_models = (Path(__file__).resolve().parents[3] / "Saved/WarningTraining/models"
                          if self.root.resolve() == ROOT.resolve() else self.root / "models")
        self.registry = registry or TrainedModelRegistry(
            model_root if model_root is not None else default_models)

    def _bundle(self):
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise ValueError("Native comparison recordings are unavailable. Generate them with build_native_comparison.py.")
        return _read_bundle(str(self.root.resolve()), manifest.stat().st_mtime_ns)

    def _workbench_bundle(self):
        manifest = self.workbench_root / "manifest.json"
        if not manifest.exists():
            raise ValueError(
                "Measured eight-sensor comparisons are unavailable. Generate them with "
                "build_workbench_comparison.py.")
        return _read_bundle(str(self.workbench_root.resolve()), manifest.stat().st_mtime_ns)

    def list(self):
        rows = ([] if not (self.root / "manifest.json").exists() else
            [{"id": row["id"], "policy": row["policy"],
              "policyLabel": policy_label(row["policy"]), "case": row["case"],
              "label": row["label"]} for row in self._bundle()["episodes"]])
        rows.extend({"id": model["id"], "policy": model["id"],
            "policyLabel": f"{model['name']} · {model.get('algorithmLabel', 'REINFORCE')}", "case": 1,
            "label": f"{model['name']} · held-out episode",
            "defaultLayout": "directional_balanced_8", "trainedModel": True}
            for model in self.registry.list())
        return rows

    def layouts(self):
        return deepcopy(COMPARISON_LAYOUTS)

    def get(self, identifier, *, scenario=False, layout_id="matched_common_sense"):
        if not isinstance(identifier, str):
            raise ValueError("Choose an available native comparison episode")
        if layout_id not in {row["id"] for row in COMPARISON_LAYOUTS}:
            raise ValueError("Choose an available fixed comparison placement")
        if identifier.startswith("trained-"):
            if layout_id != "directional_balanced_8":
                raise ValueError("Named warning models use their matched selected-count evaluation")
            episode = self.registry.comparison(identifier)
            result = comparison_result(episode)
            sensor_count = result["baseline"].get(
                "maxSensors", len(result["baseline"].get("placements", [])))
            result.update({"method": "trained_directional_warning", "trainedModel": True,
                "label": episode["label"], "selection": deepcopy(
                    episode["baseline"].get("selection", {})),
                "fairness": {"matched": True, "description": (
                    f"The saved best checkpoint and fixed {sensor_count}-directional-sensor placement face "
                    "the same held-out native episode, paths, speeds, sensing draws and limits.")},
                "description": (
                    "Automatically registered held-out evaluation for this completed named "
                    "training run. This is one matched episode, not an aggregate claim.")})
            result["audit"].update({"red_policy": "Five seeded approaches selected from eight synthetic sectors",
                "blue_policy": "Named best warning-time checkpoint selected during training",
                "exact_sensor_count": sensor_count,
                "exact_count_rl_trained": True})
            if scenario:
                return {"episodeId": identifier, "label": result["label"],
                        "selection": result["selection"], "layoutId": layout_id,
                        "layoutOnly": False,
                        **{key: deepcopy(result["baseline"][key])
                           for key in ("catalogue", "sites", "budget", "objectiveRadius")}}
            return result
        bundle = self._bundle()
        episode = next((row for row in bundle["episodes"] if row["id"] == identifier), None)
        if episode is None:
            raise ValueError("Choose an available native comparison episode")
        if layout_id == "matched_common_sense":
            result = comparison_result(episode, bundle.get("protocol"))
        else:
            workbench = self._workbench_bundle()
            measured = next((row for row in workbench["episodes"]
                             if row["policy"] == episode["policy"]
                             and row["case"] == episode["case"]), None)
            if measured is None:
                raise ValueError("Choose an available measured eight-sensor comparison")
            result = comparison_result(measured, workbench.get("protocol"))
            result.update({"method": "directional_balanced_8", "layoutOnly": False,
                "label": measured["label"],
                "selection": deepcopy(measured["baseline"].get("selection", {})),
                "fairness": {"matched": True,
                    "description": (
                        "Both layouts face the same five native drones, paths, speeds, sensor "
                        "capabilities, sites and episode seed. The archived RL layout was trained "
                        "for the earlier three-sensor contract and was not retrained for eight sensors.")},
                "description": (
                    "Matched native eight-sector workbench replay. Warning values come from the "
                    "captured episode; the result is not a like-for-like trained-policy comparison.")})
            result["audit"].update({"red_policy": "Five seeded approaches selected from eight synthetic sectors",
                "blue_policy": (
                    "Archived temporal RL placement replayed unchanged versus the fixed "
                    "eight-camera balanced workbench start"),
                "eight_sensor_rl_trained": False})
        if scenario:
            return {"episodeId": identifier, "label": result["label"], "selection": result["selection"],
                    "layoutId": layout_id, "layoutOnly": result.get("layoutOnly", False),
                    **{key: deepcopy(result["baseline"][key])
                       for key in ("catalogue", "sites", "budget", "objectiveRadius")}}
        return result
