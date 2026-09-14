"""Offline temporal-v6 replays of the first two scored cases per fixed endpoint.

The stored cases are supplied directly to the unchanged scoring core: no case
generator, training, endpoint selection, device commands or policy promotion.
Private truth is used only by the scorer and recorded viewer, never the actor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
from pathlib import Path
import tempfile

import numpy as np

import evaluate_temporal as evaluator
from demo_balanced import export_balanced_html, implementation_fingerprints as viewer_sources
from triad_rl.adaptive_evaluation import canonical_hash, json_safe
from triad_rl.ranking_policy import _strict_json
from triad_rl.temporal_env import TemporalPlacementEnv
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.temporal_policy import TemporalPolicy, POLICY_SCHEMA

SCHEMA = "triad.temporal_replay.v1"
RELEASE = "temporal-v6-pilot"
CASES_PER_PROFILE = 2


def implementation_fingerprints():
    return {**viewer_sources(), **evaluator.source_provenance(),
            "demo_temporal.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _policy_state(policy):
    return {"weights_sha256": policy.weights_fingerprint(), "rng_sha256": policy.rng_fingerprint()}


def _file_hash(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Replay evidence must not traverse symlinks")
    return evaluator.trainer._sha(path)


def _metrics(replay):
    info = replay["metrics"]
    result = {key: value for key, value in info.items()
              if (isinstance(value, (int, float, bool)) or value is None)
              and key not in {"scenario_seed", "objective_radius"}}
    result["sensors_placed"] = len(replay["placements"])
    result["blind_spot_fraction"] = 1. - info["coverage"]
    result["budget_fraction_spent"] = info["cost"] / replay["scenario"]["public"]["budget_total"]
    result["mean_confirmation_time_censored"] = float(np.mean([
        target["time_to_zone"] if target["first_confirmation"] is None
        else min(target["first_confirmation"], target["time_to_zone"])
        for target in info["target_results"]]))
    return result


def _verify_replay(replay, row, state):
    stored = {**replay["scenario"], "evaluation_catalogue": replay["catalogue"],
              "curriculum_metadata": replay["case_metadata"]}
    if (canonical_hash(stored) != row["scenario_sha256"]
            or not evaluator._same(stored, row["scenario"])
            or not evaluator._same([item["action_index"] for item in replay["decisions"]], row["actions"])
            or not evaluator._same(replay["placements"], row["placements"])
            or not evaluator.derived_same(_metrics(replay), row["metrics"])
            or not evaluator.derived_same(replay["metrics"]["reward_components"], row["reward_components"])
            or not evaluator.derived_same(replay["metrics"]["target_results"], row["target_results"])
            or replay["metrics"]["outcome"] != row["outcome"]
            or state["weights_sha256"] != row["weights_sha256_before"]
            or state["weights_sha256"] != row["weights_sha256_after"]):
        raise ValueError("Recreated temporal replay differs from its scored actions/scenario/metrics")


def collect_stored_replay(policy, row, *, config, profile):
    """Re-score exactly one archived case, without generating or resetting another."""
    stored = deepcopy(row["scenario"])
    if (profile not in evaluator.PROFILES or canonical_hash(stored) != row["scenario_sha256"]
            or type(row["seed"]) is not int or stored.get("seed") != row["seed"]):
        raise ValueError("Stored temporal scenario identity differs")
    catalogue, metadata = stored.pop("evaluation_catalogue"), stored.pop("curriculum_metadata")
    if (metadata.get("profile") != profile or metadata.get("requested_profile") != profile
            or metadata.get("scenario_seed") != row["seed"] or metadata.get("scenario_filtered") is not False):
        raise ValueError("Stored temporal curriculum identity differs")
    before, input_hash = _policy_state(policy), canonical_hash(row)
    # No seed constructor or subsequent reset: the only physical input is the
    # exact archived case, catalogue and explicitly declared public mission.
    env = TemporalPlacementEnv(scenario=deepcopy(stored), catalogue=deepcopy(catalogue), config=config)
    observation, decisions, total_return = env.observe(), [], 0.
    for _ in range(32):
        public_before = deepcopy(observation["state"])
        action = policy.act(deepcopy(observation), deterministic=True)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < len(observation["action_mask"]) or not observation["action_mask"][action]):
            raise ValueError("Temporal replay policy returned an invalid or masked action")
        action = int(action)
        option = deepcopy(observation["options"][action])
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward): raise ValueError("Temporal replay reward is nonfinite")
        decisions.append({"action_index": action, "action": option, "reward": float(reward), "public_before": public_before})
        total_return += float(reward)
        if done: break
    else:
        raise RuntimeError("Temporal replay exceeded the placement decision bound")
    after = _policy_state(policy)
    if before != after or canonical_hash(row) != input_hash:
        raise RuntimeError("Frozen temporal replay weights, RNG or input changed")
    if not info.get("frames"): raise ValueError("Scored temporal replay did not contain frames")
    # The core validates public numbers again. Only derived numeric roundoff is
    # allowed there; the viewer keeps the byte-identical stored input identity.
    if (not evaluator.derived_same(json_safe(env.scenario), stored)
            or not evaluator.derived_same(json_safe(env.catalogue), catalogue)):
        raise ValueError("Scoring core changed the stored temporal case or catalogue")
    replay = json_safe({"schema": SCHEMA, "release": RELEASE, "seed": row["seed"], "stage": "validation",
        "profile": profile, "requested_profile": profile, "scenario": stored, "catalogue": catalogue,
        "case_metadata": metadata, "placements": deepcopy(env.placements), "decisions": decisions,
        "metrics": {**{key: deepcopy(value) for key, value in info.items() if key != "frames"}, "return": total_return},
        "frames": deepcopy(info["frames"]), "audit": {
            "schema": "triad.temporal_replay_audit.v1", "policy_schema": POLICY_SCHEMA,
            "recorded_simulator_replay": True, "stored_case_only": True, "scenario_generation_performed": False,
            "browser_inference": False, "training_performed": False, "independent_final_test_evidence": False,
            "ground_truth_in_policy_observation": False, "policy_state_before": before, "policy_state_after": after,
            "interpretation": "Experimental offline initial placement followed by scored sensing; not live control, sensor relocation, physical interception or real-world validation"}})
    _verify_replay(replay, row, before)
    return replay


def run_demo(*, runs, evaluation, protocol, output):
    """Verify all endpoints/evidence before replay; create one new offline file."""
    output, evaluation, protocol = Path(output).absolute(), Path(evaluation), Path(protocol)
    if output.exists() or output.is_symlink(): raise FileExistsError("Temporal demo output already exists")
    if any(parent.is_symlink() for parent in output.parents): raise ValueError("Temporal output must not traverse symlinks")
    runs = {seed: Path(path) for seed, path in runs.items()}
    sources = implementation_fingerprints()
    paths = [protocol, *(evaluation / name for name in evaluator.EVALUATION_FILES),
             *(path / name for path in runs.values() for name in evaluator.trainer.RUN_FILES)]
    # Stream large training-record files, retaining only their hashes.
    snapshots = {path: _file_hash(path) for path in paths}
    evaluator.verify_completed_evaluation(runs, protocol_path=protocol, output=evaluation)
    declaration = _strict_json(evaluator._file(protocol))
    if set(runs) != set(declaration["training"]["policy_seeds"]): raise ValueError("Every fixed endpoint is required")
    config = TemporalConfig(**declaration["temporal_config"])
    policies = {seed: TemporalPolicy.load(path / "last", config=config) for seed, path in runs.items()}
    states = {seed: _policy_state(policy) for seed, policy in policies.items()}

    def stable():
        if (sources != implementation_fingerprints() or any(_file_hash(path) != digest for path, digest in snapshots.items())
                or any(_policy_state(policy) != states[seed] for seed, policy in policies.items())):
            raise RuntimeError("Temporal replay source/checkpoint/evidence/RNG changed; no output written")

    plan = []
    for profile in evaluator.PROFILES:
        with gzip.GzipFile(fileobj=io.BytesIO(evaluator._file(evaluation / f"validation-{profile}.json.gz"))) as handle:
            raw = handle.read(evaluator.MAX_BYTES + 1)
        if len(raw) > evaluator.MAX_BYTES: raise ValueError("Temporal replay report exceeds size bound")
        report = _strict_json(raw)
        start = declaration["validation"]["scenario_seed_ranges"][profile]["start"]
        for seed in sorted(policies):
            rows = report["methods"][f"temporal_{seed}"]["episodes"][:CASES_PER_PROFILE]
            if len(rows) != CASES_PER_PROFILE or not evaluator._same([row["seed"] for row in rows], [start, start + 1]):
                raise ValueError("Replay requires the first two declared scored cases of every profile and endpoint")
            plan.extend((seed, profile, index + 1, row) for index, row in enumerate(rows))
    stable()
    replays = []
    for seed, profile, case, row in plan:
        replay = collect_stored_replay(policies[seed], row, config=config, profile=profile)
        _verify_replay(replay, row, states[seed])
        replay.update(temporal_seed=seed, case_index=case,
                      split=f"EXPERIMENTAL temporal {seed} / {profile} / case {case} of 2 / offline replay")
        files = {name: snapshots[runs[seed] / "last" / name] for name in ("checkpoint.json", "arrays.npz")}
        replay["audit"].update(implementation_sha256_before=deepcopy(sources), implementation_sha256_after=deepcopy(sources),
            evaluation_protocol_sha256=snapshots[protocol], scored_scenario_sha256=row["scenario_sha256"],
            scored_actions_metrics_verified=True, checkpoint_files_sha256_before=deepcopy(files), checkpoint_files_sha256_after=deepcopy(files),
            example_rule="First two scored cases per profile for every fixed endpoint; no winner selection",
            experimental=True, default_policy_changed=False, physical_commands=False, live_feed=False)
        replays.append(replay)
    stable()
    with tempfile.TemporaryDirectory(prefix="temporal-offline-demo-") as temporary:
        rendered = Path(temporary) / "replay.html"
        export_balanced_html(replays, rendered,
            checkpoint="EXPERIMENTAL temporal-v6 / ALL fixed endpoints / OFFLINE, NOT LIVE / no promotion or physical commands / checkpoint-set digest",
            weights_sha256=canonical_hash({str(seed): state["weights_sha256"] for seed, state in states.items()}))
        data = rendered.read_bytes()
        stable()
        if any(parent.is_symlink() for parent in output.parents): raise ValueError("Temporal output parent changed")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle: handle.write(data)
    return replays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    for name in ("evaluation", "protocol", "output"): parser.add_argument("--" + name, type=Path, required=True)
    args, runs = vars(parser.parse_args(argv)), {}
    for item in args.pop("run"):
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdecimal() or int(seed) in runs or not path:
            parser.error("Use one unique integer SEED=RUN_DIRECTORY per endpoint")
        runs[int(seed)] = Path(path)
    print(f"Saved {len(run_demo(runs=runs, **args))} verified offline temporal replays; no policy promoted.")


if __name__ == "__main__":
    main()
