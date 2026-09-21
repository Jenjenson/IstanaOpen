"""Read paired native evaluations without recomputing sensing in the browser.

The capture tool invokes the current live Blue planner and native Red simulation.
This module only validates the saved evidence and derives display statistics.
"""
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1] / "Results/native-placement-comparison"
POLICY_LABELS = {"406": "Temporal RL · policy A", "407": "Temporal RL · policy B",
                 "408": "Temporal RL · policy C"}


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
            "label": episode["label"], "policyLabel": policy_label(episode["policy"]),
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
    if bundle.get("schema") != "istana.native_comparison_capture.v1":
        raise ValueError("Unsupported native comparison capture")
    episodes = bundle["episodes"]
    if len({row["id"] for row in episodes}) != len(episodes):
        raise ValueError("Duplicate native comparison episode")
    for episode in episodes:
        _validate_episode(episode)
    return bundle


class NativeComparisons:
    def __init__(self, root=ROOT):
        self.root = Path(root)

    def _bundle(self):
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise ValueError("Native comparison recordings are unavailable. Generate them with build_native_comparison.py.")
        return _read_bundle(str(self.root.resolve()), manifest.stat().st_mtime_ns)

    def list(self):
        if not (self.root / "manifest.json").exists():
            return []
        return [{"id": row["id"], "policy": row["policy"], "policyLabel": policy_label(row["policy"]),
                 "case": row["case"], "label": row["label"]} for row in self._bundle()["episodes"]]

    def get(self, identifier, *, scenario=False):
        if not isinstance(identifier, str):
            raise ValueError("Choose an available native comparison episode")
        bundle = self._bundle()
        episode = next((row for row in bundle["episodes"] if row["id"] == identifier), None)
        if episode is None:
            raise ValueError("Choose an available native comparison episode")
        result = comparison_result(episode, bundle.get("protocol"))
        if scenario:
            return {"episodeId": identifier, "label": result["label"], "selection": result["selection"],
                    **{key: deepcopy(episode["rl"][key]) for key in ("catalogue", "sites", "budget", "objectiveRadius")}}
        return result
