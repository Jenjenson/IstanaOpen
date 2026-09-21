"""Optional generated eight- and sixty-drone comparisons, outside the archive.

The temporal actors trained on 1..8 drones. Eight is their upper training
population; sixty is an explicitly out-of-training-population stress test.
Neither option is a native Unreal warning-policy replay. Frozen training,
scenario generators, checkpoints and archived evidence are never modified.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

import placement_comparison as comparison
from recommend_temporal import recommend_layout
from triad_rl.adaptive_env import _validate_scenario
from triad_rl.adaptive_inputs import validate_catalogue
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.temporal_policy import TemporalPolicy


SCHEMA = "istana.generated_swarm_comparison.v1"
COUNTS = (8, 60)
CHECKPOINT_ROOT = Path(__file__).resolve().parent.parent / "Results/temporal-v6-pilot/training"
_STREAM = 0x5A4A2601


def _count(value):
    if type(value) is not int or value not in COUNTS:
        raise ValueError("Generated comparison drone count must be 8 or 60")
    return value


def _population(count):
    within = count == 8
    return {
        "schema": SCHEMA, "droneCount": count, "modelTrainingDroneRange": [1, 8],
        "withinTrainingPopulationRange": within, "generated": True, "nativeReplay": False,
        "label": ("8 drones · upper training population" if within else
                  "60 drones · larger-swarm stress test"),
        "description": ("New synthetic case at the maximum drone count used in temporal training. "
            "Training used 1–5 drones normally and 3–8 under stress; this is not an archived case."
            if within else "The temporal models trained on 1–8 drones. This generated 60-drone "
            "case tests a larger population; it is not their training setup or the separate "
            "native 60-drone warning-policy experiment."),
    }


def make_swarm_case(base_replay, drone_count):
    """Generate truth plus imperfect public reports, without policy inference.

    Truth follows the original adaptive generator's profile-dependent path,
    speed, altitude and emission distributions, with an explicit population.
    Existing public resource/site/catalogue/weather constraints are retained.
    Dedicated named streams distinguish these demos from published evidence.
    No archived target, action, frame or measured outcome is used as input.
    """
    count = _count(drone_count)
    policy_seed = base_replay["temporal_seed"]
    if type(policy_seed) is not int or policy_seed not in (406, 407, 408):
        raise ValueError("Generated comparison requires temporal checkpoint 406, 407 or 408")
    template = base_replay["scenario"]
    seed = template["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("Scenario seed must be a nonnegative integer")
    split = template["split"]
    if split not in ("train", "stress"):
        raise ValueError("Generated comparison requires the normal or stress threat family")
    stress = split == "stress"
    sequence = np.random.SeedSequence([seed, _STREAM, count])
    truth_stream, report_stream, sensing_stream = sequence.spawn(3)
    rng, reports = np.random.default_rng(truth_stream), np.random.default_rng(report_stream)
    # Keep the scenario identity exact when the JSON reaches a browser Number.
    sensing_seed = int(sensing_stream.generate_state(1, dtype=np.uint64)[0]) & ((1 << 53) - 1)
    catalogue = validate_catalogue(base_replay["catalogue"])
    public = deepcopy(template["public"])
    # An explicit public mission is required by the saved temporal policy.
    mission = TemporalConfig()
    for key, field in (("objective_radius", "objective_radius_m"), ("dt", "look_interval_s"),
                       ("required_confirmations", "required_confirmations"),
                       ("confirmation_window", "confirmation_window"),
                       ("defence_lead_time", "lead_time_s")):
        if template[key] != getattr(mission, field):
            raise ValueError("Scenario mission differs from the temporal checkpoint configuration")

    approach = float(rng.uniform(-np.pi, np.pi))
    multi_direction = bool(rng.random() < (.65 if stress else .3))
    spread = float(rng.uniform(.2, 1.2 if stress else .65))
    base_altitude = float(rng.uniform(65, 160) if stress else rng.uniform(8, 92))
    base_speed = float(rng.uniform(17, 32) if stress else rng.uniform(6, 22))
    emitter_duty = float(rng.choice([.0, .15, .45, .8, 1.]))
    targets, tracks, public_bearings = [], [], []
    for index in range(count):
        bearing = float(approach + rng.normal(0, spread))
        if multi_direction and index % 2:
            bearing += float(rng.uniform(1.4, 3.2))
        target = {"id": f"drone-{index + 1}", "bearing": bearing,
            "spawn_radius": float(rng.uniform(270, 370)),
            "altitude": float(np.clip(base_altitude + rng.normal(0, 9), 5, 185)),
            "speed": float(np.clip(base_speed + rng.normal(0, 2), 3, 38)),
            "path": str(rng.choice(("direct", "curved", "weaving"))),
            "curvature": float(rng.uniform(-.8, .8)),
            "weave_amplitude": float(rng.uniform(.10, .5 if stress else .32)),
            "weave_phase": float(rng.uniform(0, 2 * np.pi)),
            "altitude_amplitude": float(rng.uniform(0, 15 if stress else 6)),
            "emitter_duty": float(np.clip(emitter_duty + rng.normal(0, .05), 0, 1)),
            "emitter_period": int(rng.integers(5, 14)), "emitter_phase": float(rng.random())}
        targets.append(target)
        public_bearing = bearing + float(reports.normal(0, .38 if stress else .23))
        public_bearings.append(public_bearing)
        if reports.random() > (.3 if stress else .16):
            radius = target["spawn_radius"] + float(reports.normal(0, 20))
            tracks.append({"id": target["id"],
                "position": [float(radius * np.cos(public_bearing)), float(radius * np.sin(public_bearing)),
                             float(max(0., target["altitude"] + reports.normal(0, 10)))],
                "velocity": [float(-base_speed * np.cos(public_bearing) + reports.normal(0, 2)),
                             float(-base_speed * np.sin(public_bearing) + reports.normal(0, 2)), 0.],
                "confidence": float(reports.uniform(.45, .9)), "timestamp": 0., "confirmed": False,
                "emitter_probability": float(np.clip(target["emitter_duty"] + reports.normal(0, .18), 0, 1))})
    angles = np.arange(8) * np.pi / 4
    weights = .04 + np.sum(np.exp(3.8 * (np.cos(angles[:, None] - np.asarray(public_bearings)[None, :]) - 1.)), axis=1)
    weights /= weights.sum()
    public.update(timestamp=0., episode_id=f"swarm-{sensing_seed}-{count}", source="simulation",
                  placements=[], budget_remaining=public["budget_total"], tracks=tracks, done=False,
                  forecast={"approach_weights": weights.tolist(),
                    "altitude": float(max(1., base_altitude + reports.normal(0, 9))),
                    "speed": float(max(1., base_speed + reports.normal(0, 2))),
                    "emitter_probability": float(np.clip(emitter_duty + reports.normal(0, .13), 0, 1)),
                    "swarm_size": float(count), "angular_uncertainty": .65 if stress else .4})
    public.pop("fresh_track_fraction", None)
    scenario = {key: deepcopy(template[key]) for key in ("schema", "split", "objective_radius", "dt",
        "required_confirmations", "confirmation_window", "defence_lead_time", "max_invalid_actions")}
    scenario.update(seed=sensing_seed, public=public, targets=targets)
    scenario = _validate_scenario(scenario, catalogue)
    return {"schema": SCHEMA, "seed": sensing_seed, "profile": base_replay["profile"],
        "case_index": base_replay["case_index"], "temporal_seed": policy_seed,
        "scenario": scenario, "catalogue": catalogue, "swarm": _population(count),
        "generation": {"schema": SCHEMA, "template_scenario_seed": seed, "stream": _STREAM,
            "drone_count": count, "threat_family": split, "scenario_filtered": False,
            "selection": "One fixed generated case per template and count; no outcome selection",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def _label(replay):
    return (f"Temporal {replay['temporal_seed']} · {replay['profile']} · case {replay['case_index']}"
            f" · {replay['swarm']['droneCount']} drones (generated)")


def swarm_scenario(base_replay, drone_count):
    """Public editor context only: no RL inference, sensing or future paths."""
    replay = make_swarm_case(base_replay, drone_count)
    result = comparison.comparison_scenario(replay)
    result.update(label=_label(replay), swarm=deepcopy(replay["swarm"]))
    result["audit"].update(scenario_generation_performed=True, fresh_rl_inference=False)
    return result


def make_swarm_replay(base_replay, count, checkpoint_root=None):
    """Infer this checkpoint's fresh layout from only the new public snapshot."""
    replay = make_swarm_case(base_replay, count)
    checkpoint = Path(checkpoint_root or CHECKPOINT_ROOT) / f"seed-{replay['temporal_seed']}/last"
    config = TemporalConfig()
    policy = TemporalPolicy.load(checkpoint, config=config)
    before = {"weights_sha256": policy.weights_fingerprint(), "rng_sha256": policy.rng_fingerprint()}
    plan = recommend_layout(policy, deepcopy(replay["scenario"]["public"]),
                            config=config, catalogue=deepcopy(replay["catalogue"]))
    after = {"weights_sha256": policy.weights_fingerprint(), "rng_sha256": policy.rng_fingerprint()}
    if before != after:
        raise RuntimeError("Deterministic comparison inference changed policy state")
    view = comparison._score(replay, plan["decisions"], label=f"RL checkpoint {replay['temporal_seed']}",
                             policy="Fresh deterministic temporal RL inference")
    replay.update({key: view[key] for key in ("placements", "decisions", "frames", "metrics")})
    replay["generation"].update(checkpoint_weights_sha256=before["weights_sha256"],
        policy_state_before=before, policy_state_after=after, actor_updates=policy.actor_update_count)
    return replay


def compare_swarm_placements(base_replay, drone_count, *, baseline="common_sense", placements=None,
                             checkpoint_root=None):
    """Pair new RL inference with the baseline on identical generated truth."""
    replay = make_swarm_replay(base_replay, drone_count, checkpoint_root)
    result = comparison.compare_placements(replay, baseline=baseline, placements=placements)
    result.update(label=_label(replay), swarm=deepcopy(replay["swarm"]))
    audit = result["audit"]
    reproduced = audit.pop("archived_rl_result_reproduced")
    audit.update(fresh_rl_inference=True, scenario_generation_performed=True,
        generated_rl_result_reproduced=reproduced, generation=deepcopy(replay["generation"]),
        model_training_drone_range=[1, 8], generated_drone_count=drone_count,
        within_training_population_range=drone_count == 8,
        interpretation=replay["swarm"]["description"] + " Both layouts use identical generated paths, "
            "weather, resources and sensing draws. No training or native simulation is performed.")
    result["rl"]["policy"] = f"Fresh deterministic temporal {replay['temporal_seed']} inference"
    for view in (result["rl"], result["baseline"]):
        view["swarm"] = deepcopy(replay["swarm"])
        view["audit"].update(scenario_generation_performed=True, generated_drone_count=drone_count)
    json.dumps(result, allow_nan=False)
    return result
