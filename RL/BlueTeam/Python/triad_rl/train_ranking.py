"""Fixed-budget rollout-guided sensor/site ranking from public observations.

Every candidate branch completes with frozen public greedy. Outcomes supervise
pairwise preferences; they never choose the live action or enter actor inputs.
This is a rollout-guided policy improvement experiment, not an unbiased policy
gradient or a guarantee of monotonic approximate policy iteration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .adaptive_inputs import FEATURE_NAMES
from .adaptive_evaluation import canonical_hash
from .balanced_policy import BalancedPolicy
from .robust_scenarios import RobustPlacementEnv, curriculum_manifest
from . import ranking_rollout as rollout
from .ranking_policy import RankPolicy
from .ranking_lineage import load_published_lineage, assert_disjoint
from . import train_credit_pilot as shared

BLUE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "triad.ranking_training.v1"
PROTOCOL_SCHEMA = "triad.ranking_pilot_protocol.v1"
V3_PATH = BLUE_ROOT / "Checkpoints/balanced-v3-candidate"
V3_WEIGHTS = "a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0"
RUN_FILES = ("config.json", "protocol.json", "training.jsonl", "comparisons.jsonl.gz", "summary.json",
             "initialized/checkpoint.json", "initialized/arrays.npz", "last/checkpoint.json", "last/arrays.npz")
SOURCE_FILES = (*shared.SOURCE_FILES, "triad_rl/adaptive_evaluation.py", "triad_rl/placement_policy.py",
                "triad_rl/anchored_lineage.py", "triad_rl/ranking_lineage.py",
                "triad_rl/ranking_policy.py", "triad_rl/ranking_rollout.py", "triad_rl/train_ranking.py")


def source_provenance():
    return {name: shared._file_hash(BLUE_ROOT / "Python" / name) for name in SOURCE_FILES}


def _bytes(value):
    return (shared._json(value) + "\n").encode()


def _read(path):
    return shared._strict_json(Path(path).read_bytes())


def load_protocol(path):
    raw = Path(path).read_bytes()
    if len(raw) > 4_000_000:
        raise ValueError("Ranking protocol exceeds size bound")
    protocol = shared._strict_json(raw)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported ranking protocol")
    train = protocol.get("training", {})
    if (not isinstance(train.get("policy_seeds"), list) or not train["policy_seeds"]
            or len(set(train["policy_seeds"])) != len(train["policy_seeds"])):
        raise ValueError("Training seeds must be a nonempty unique list")
    for seed in train["policy_seeds"]:
        shared._integer(seed, "policy seed", 0, 999999)
    for key, low, high in (("episodes", 1, 4096), ("batch_size", 1, 16), ("passes_per_batch", 1, 4)):
        shared._integer(train.get(key), key, low, high)
    if train["episodes"] % train["batch_size"]:
        raise ValueError("Ranking experiment requires full batches")
    fixed = {"profile": "mixed", "hidden_size": 64, "max_actions": 6, "learning_rate": .002,
             "temperature": .1, "regularization": .01, "max_grad_norm": 1.,
             "continuation": "frozen_public_greedy", "behavior": "current_ranker_deterministic_before_branch_labels",
             "initialization": "exact_public_greedy_prior_plus_zero_residual", "endpoint": "fixed_no_selection_or_resume"}
    for key, expected in fixed.items():
        if type(train.get(key)) is not type(expected) or train[key] != expected:
            raise ValueError(f"Ranking experiment fixes {key}={expected}")
    ranges = {str(seed): {"start": seed * 1_000_000, "count": train["episodes"]} for seed in train["policy_seeds"]}
    if shared._json(train.get("scenario_seed_ranges")) != shared._json(ranges):
        raise ValueError("Ranking training scenario slots differ")
    if protocol.get("training_implementation_sha256") != source_provenance():
        raise ValueError("Ranking training source differs from predeclared hashes")
    if protocol.get("proposal_v3_weights_sha256") != V3_WEIGHTS:
        raise ValueError("V3 proposal checkpoint differs")
    return protocol, raw


def _configuration(protocol, raw, seed, lineage, initial, runtime):
    if source_provenance() != protocol["training_implementation_sha256"]:
        raise ValueError("Training source changed after reading the protocol")
    declaration = protocol.get("prior_publication", {})
    if (declaration.get("path") != lineage["prior_pilot"]["publication_manifest"]["path"]
            or declaration.get("sha256") != lineage["prior_pilot"]["publication_manifest"]["sha256"]):
        raise ValueError("Ranking prior publication differs")
    return {"schema": SCHEMA, "policy_seed": seed, "training": protocol["training"],
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "source_sha256": protocol["training_implementation_sha256"],
            "initial_weights_sha256": initial.weights_fingerprint(), "inherited_lineage": lineage,
            "proposal_checkpoint": {"path": "Checkpoints/balanced-v3-candidate", "weights_sha256": V3_WEIGHTS,
                                    "files_sha256": shared._checkpoint_files(V3_PATH),
                                    "role": "public-input slate proposal only; not initializer or continuation"},
            "curriculum": curriculum_manifest(), "runtime": runtime,
            "comparison_priority": "timely fraction; then detection fraction; then original episode return",
            "comparison_weights": "abs timely delta; .1*abs detection delta; .01*min(abs return delta/20,1)",
            "labels": "One paired simulator draw per alternative; not expected values, causal subgroup estimates or optimal-action labels",
            "continuation": "Frozen public greedy completes every branch; learned live trajectory may differ",
            "extra_simulation": "Each slate action consumes one separate terminal branch; no first-action exception",
            "checkpoint_rule": "fixed_endpoint_only"}


def _stable(protocol_path, raw, config):
    if (Path(protocol_path).read_bytes() != raw or source_provenance() != config["source_sha256"]
            or shared._checkpoint_files(V3_PATH) != config["proposal_checkpoint"]["files_sha256"]
            or any(shared._file_hash(BLUE_ROOT / row["path"]) != row["sha256"]
                   for row in config["inherited_lineage"]["publication_manifests"])):
        raise ValueError("Ranking inputs or implementation changed during run")


def collect_episode(env, policy, v3, seed):
    """Collect all visited states; only non-tie pairs enter optimization."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 10**15:
        raise ValueError("Ranking collection requires a training seed, never validation or final test")
    seed = int(seed)
    observation = env.reset(seed=seed)
    slate_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB10E5001]))
    prefix, states, comparisons_count, branches = [], [], 0, 0
    while True:
        # Commit behavior before any private branch outcome exists.
        action = policy.act(observation, deterministic=True)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < len(observation["action_mask"]) or not observation["action_mask"][action]):
            raise ValueError("Ranker proposed illegal live action")
        action = int(action)
        slate = rollout.propose_slate(observation, policy, v3, rng=slate_rng, max_actions=6)
        if action not in slate:
            raise ValueError("Live action must appear in the declared six-action slate")
        outcomes = rollout.evaluate_slate(seed=seed, profile=env.profile, prior_actions=prefix,
                                          actions=slate, expected_observation=observation)
        pairs = rollout.comparisons(outcomes)
        features = np.asarray(observation["option_features"], dtype=np.float64)[slate].copy()
        states.append({"seed": seed, "prefix": list(prefix), "slate_actions": slate, "live_action": action,
                       "public_observation_sha256": canonical_hash(observation), "features": features.tolist(),
                       "comparisons": pairs, "outcomes": outcomes})
        branches += len(slate)
        comparisons_count += len(pairs)
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward):
            raise ValueError("Live reward must be finite")
        if done:
            break
        prefix.append(action)
        if len(states) >= 32:
            raise ValueError("Live ranking trajectory exceeded bounded horizon")
    return states, {"seed": seed, "profile": env.case_metadata["profile"], "decisions": len(states),
                    "branches": branches, "comparisons": comparisons_count,
                    "training_states": sum(bool(row["comparisons"]) for row in states),
                    "timely_fraction": 1. - info["breached_fraction"], "detection_fraction": info["detected_fraction"],
                    "episode_return": info["return"], "cost": info["cost"], "success": info["success"],
                    "invalid_actions": info["invalid_actions"], "placements": info["placements"]}


def run_training(protocol_path, policy_seed, output):
    started = time.perf_counter()
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    shared._empty_output(output)
    protocol, raw = load_protocol(protocol_path)
    train = protocol["training"]
    if type(policy_seed) is not int or policy_seed not in train["policy_seeds"]:
        raise ValueError("Run seed is not declared")
    lineage = load_published_lineage()
    for row in train["scenario_seed_ranges"].values():
        assert_disjoint(row["start"], row["count"], lineage, label="ranking training")
    actor = RankPolicy(FEATURE_NAMES, seed=policy_seed, hidden_size=train["hidden_size"])
    v3 = BalancedPolicy.load(V3_PATH, feature_names=FEATURE_NAMES)
    if v3.weights_fingerprint() != V3_WEIGHTS:
        raise ValueError("Frozen V3 proposal fingerprint differs")
    config = _configuration(protocol, raw, policy_seed, lineage, actor,
                            {"python": platform.python_version(), "numpy": np.__version__})
    _stable(protocol_path, raw, config)
    shared._empty_output(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_bytes(raw)
    (output / "config.json").write_bytes(_bytes(config))
    seed_start = policy_seed * 1_000_000
    def state(completed, files):
        return {"schema": SCHEMA, "config": config, "completed_episodes": completed,
                "evidence_files_sha256": files,
                "seed_provenance": {"training": {"start": seed_start, "count": completed},
                                    "inherited": lineage, "validation": "never accessed", "final_test": "never accessed"}}
    actor.save(output / "initialized", state(0, {}))
    initialized_files = shared._checkpoint_files(output / "initialized")
    env = RobustPlacementEnv(seed=seed_start, profile="mixed")
    records_all, batch_rows = [], []
    data_path, log_path = output / "comparisons.jsonl.gz", output / "training.jsonl"
    # mtime=0 and no embedded filename keep the evidence portable. A failed run
    # remains incomplete; this fixed pilot never resumes or overwrites it.
    with data_path.open("xb") as data_file, gzip.GzipFile(fileobj=data_file, mode="wb", mtime=0, filename="") as data:
        with log_path.open("xb") as log:
            for begin in range(0, train["episodes"], train["batch_size"]):
                batch_started = time.perf_counter()
                batch, records = [], []
                for offset in range(begin, begin + train["batch_size"]):
                    states, record = collect_episode(env, actor, v3, seed_start + offset)
                    for row in states:
                        data.write(_bytes(row))
                        if row["comparisons"]:
                            batch.append({"features": np.asarray(row["features"], dtype=np.float64),
                                          "comparisons": row["comparisons"]})
                    records.append(record)
                collection_seconds = time.perf_counter() - batch_started
                update_started = time.perf_counter()
                metrics = []
                if batch:
                    for _ in range(train["passes_per_batch"]):
                        metrics.append(actor.update_comparisons(batch, **{key: train[key] for key in
                            ("learning_rate", "temperature", "regularization", "max_grad_norm")}))
                _stable(protocol_path, raw, config)
                row = {"event": "training_batch", "episode_start": begin, "episode": begin + train["batch_size"],
                       "policy_seed": policy_seed, "scenario_seed_range": {"start": seed_start + begin, "count": len(records)},
                       "optimizer_updates": actor.update_count, "update_passes": len(metrics),
                       "training_states": len(batch), "all_visited_states": sum(r["decisions"] for r in records),
                       "pair_comparisons": sum(r["comparisons"] for r in records),
                       "branch_rollouts": sum(r["branches"] for r in records),
                       "case_profile_counts": dict(Counter(r["profile"] for r in records)),
                       "sensor_deployment_counts": dict(Counter(p["sensor_id"] for r in records for p in r["placements"])),
                       **{f"mean_{key}": float(np.mean([r[key] for r in records])) for key in
                          ("timely_fraction", "detection_fraction", "episode_return", "cost", "success", "invalid_actions")},
                       "episodes": records, "updates": metrics,
                       "mean_update_metrics": {key: float(np.mean([item[key] for item in metrics]))
                                               for key in metrics[0]} if metrics else {},
                       "weights_sha256": actor.weights_fingerprint(),
                       "collection_wall_seconds": collection_seconds, "update_wall_seconds": time.perf_counter() - update_started}
                data.flush()
                log.write(_bytes(row))
                log.flush()
                records_all.extend(records)
                batch_rows.append(row)
                print(shared._json({"seed": policy_seed, "episode": row["episode"], "comparisons": row["pair_comparisons"],
                                    "mean_return": row["mean_episode_return"]}), flush=True)
    _stable(protocol_path, raw, config)
    if load_published_lineage() != lineage or shared._checkpoint_files(output / "initialized") != initialized_files:
        raise ValueError("Published lineage or initialized checkpoint changed")
    if (output / "protocol.json").read_bytes() != raw or (output / "config.json").read_bytes() != _bytes(config):
        raise ValueError("Archived protocol or run configuration changed")
    evidence = {name: shared._file_hash(output / name) for name in ("protocol.json", "config.json", "training.jsonl", "comparisons.jsonl.gz")}
    actor.save(output / "last", state(train["episodes"], evidence))
    summary = {"schema": SCHEMA, "policy_seed": policy_seed, "completed_episodes": train["episodes"],
               "optimizer_updates": actor.update_count, "checkpoint_rule": "fixed_endpoint_only",
               "initial_weights_sha256": config["initial_weights_sha256"], "last_weights_sha256": actor.weights_fingerprint(),
               "initialized_files_sha256": initialized_files, "last_files_sha256": shared._checkpoint_files(output / "last"),
               "evidence_files_sha256": evidence, "source_sha256": config["source_sha256"], "protocol_sha256": config["protocol_sha256"],
               "seed_provenance": state(train["episodes"], evidence)["seed_provenance"],
               "profile_counts": dict(Counter(r["profile"] for r in records_all)),
               "sensor_deployment_counts": dict(Counter(p["sensor_id"] for r in records_all for p in r["placements"])),
               **{key: sum(r[key] for r in records_all) for key in ("decisions", "branches", "comparisons", "training_states")},
               **{f"mean_{key}": float(np.mean([r[key] for r in records_all])) for key in
                  ("timely_fraction", "detection_fraction", "episode_return", "cost", "success", "invalid_actions")},
               "collection_wall_seconds": sum(r["collection_wall_seconds"] for r in batch_rows),
               "update_wall_seconds": sum(r["update_wall_seconds"] for r in batch_rows),
               "total_trainer_wall_seconds": time.perf_counter() - started}
    (output / "summary.json").write_bytes(_bytes(summary))
    return summary


def _ranking_same(left, right):
    """Tolerance is for derived arithmetic only, never artifact/hash identities."""
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-12, atol=1e-12))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_ranking_same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_ranking_same(a, b) for a, b in zip(left, right))
    return left == right


def validate_run(run_dir, protocol_path, *, lineage=None):
    """Validate a fixed endpoint and replay archived comparisons without sampling.

    No environment, actor action sampler, scenario generator or filesystem writer
    is called. Replay changes only a newly constructed in-memory ranker. Optional
    lineage must be freshly verified by the caller; the default loads it here.
    Hashes bind exact archived bytes; 1e-12 tolerance applies only to recomputed
    floating arithmetic. Slate rows cannot reconstruct the hashed full public
    observation, nor can saved outcomes independently prove a simulation run.
    """
    import io

    protocol, raw = load_protocol(protocol_path)
    run = Path(run_dir).resolve()
    paths = {name: (run / name).resolve() for name in RUN_FILES}
    if (len(set(paths.values())) != len(paths)
            or any(not path.is_relative_to(run) or not path.is_file() for path in paths.values())
            or {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()} != set(RUN_FILES)):
        raise ValueError("Ranking run requires exactly nine independent in-directory artifacts")
    bounds = {name: (256_000_000 if name == "comparisons.jsonl.gz" else
                    128_000_000 if name == "training.jsonl" else
                    64_000_000 if name.endswith(".npz") else 4_000_000) for name in RUN_FILES}
    blobs = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            blobs[name] = handle.read(bounds[name] + 1)
        if len(blobs[name]) > bounds[name]:
            raise ValueError("Ranking artifact exceeds bounded size")
    hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in blobs.items()}
    if blobs["protocol.json"] != raw:
        raise ValueError("Run protocol differs from exact predeclared bytes")

    def exact(left, right, label):
        if shared._json(left) != shared._json(right):
            raise ValueError(f"Ranking {label} differs")

    def integer(value, label, low=0, high=2**63 - 1):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid ranking {label}")
        return value

    def number(value, label, low=None, high=None):
        if (type(value) is not float or not np.isfinite(value)
                or (low is not None and value < low) or (high is not None and value > high)):
            raise ValueError(f"Invalid finite ranking {label}")
        return value

    def digest(value, label):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid ranking {label} digest")

    def canonical(blob, label):
        value = shared._strict_json(blob)
        if _bytes(value) != blob:
            raise ValueError(f"Ranking {label} is not exact canonical JSONL")
        return value

    config, summary = (canonical(blobs[name], name) for name in ("config.json", "summary.json"))
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("Ranking config and summary must be JSON objects")
    seed = integer(config.get("policy_seed"), "policy seed", 0, 999999)
    train = protocol["training"]
    if seed not in train["policy_seeds"]:
        raise ValueError("Ranking run seed is not predeclared")
    count, batch_size = train["episodes"], train["batch_size"]
    seed_start = seed * 1_000_000
    lineage = load_published_lineage() if lineage is None else lineage
    for interval in train["scenario_seed_ranges"].values():
        assert_disjoint(interval["start"], interval["count"], lineage, label="ranking recorded training")
    replay = RankPolicy(FEATURE_NAMES, seed=seed, hidden_size=train["hidden_size"])
    runtime = config.get("runtime")
    if (not isinstance(runtime, dict) or set(runtime) != {"python", "numpy"}
            or any(not isinstance(v, str) or not v for v in runtime.values())):
        raise ValueError("Ranking recorded runtime differs")
    expected_config = _configuration(protocol, raw, seed, lineage, replay, runtime)
    exact(config, expected_config, "configuration/source/lineage")
    _stable(protocol_path, raw, config)
    initial = RankPolicy.load(run / "initialized", feature_names=FEATURE_NAMES)
    endpoint = RankPolicy.load(run / "last", feature_names=FEATURE_NAMES)
    for group in ("parameters", "adam_m", "adam_v"):
        for key, value in getattr(replay, group).items():
            if getattr(initial, group)[key].tobytes() != value.tobytes():
                raise ValueError("Ranking initialization is not exact seeded greedy-prior/zero-residual/zero-Adam")
    if initial.update_count != 0:
        raise ValueError("Ranking initializer optimizer count must be zero")
    exact(initial.rng.bit_generator.state, replay.rng.bit_generator.state, "initializer RNG")
    exact(endpoint.rng.bit_generator.state, replay.rng.bit_generator.state, "deterministic-training RNG")
    initial_files = {name: hashes[f"initialized/{name}"] for name in ("checkpoint.json", "arrays.npz")}
    last_files = {name: hashes[f"last/{name}"] for name in ("checkpoint.json", "arrays.npz")}
    evidence_files = {name: hashes[name] for name in ("protocol.json", "config.json", "training.jsonl", "comparisons.jsonl.gz")}

    def checkpoint_state(completed, evidence):
        return {"schema": SCHEMA, "config": config, "completed_episodes": completed,
                "evidence_files_sha256": evidence,
                "seed_provenance": {"training": {"start": seed_start, "count": completed},
                                    "inherited": lineage, "validation": "never accessed", "final_test": "never accessed"}}

    for policy, completed, evidence in ((initial, 0, {}), (endpoint, count, evidence_files)):
        expected = checkpoint_state(completed, evidence)
        exact(policy.training_state, expected, "checkpoint training state")
        exact(policy.metadata["seed_provenance"], expected["seed_provenance"], "checkpoint exposure")
    lines = blobs["training.jsonl"].splitlines(keepends=True)
    if len(lines) != count // batch_size:
        raise ValueError("Ranking batch log does not reach the exact fixed endpoint")
    log = [canonical(line, "batch log") for line in lines]
    scalar_fields = ("timely_fraction", "detection_fraction", "episode_return", "cost", "success", "invalid_actions")
    episode_fields = {"seed", "profile", "decisions", "branches", "comparisons", "training_states",
                      *scalar_fields, "placements"}
    state_fields = {"seed", "prefix", "slate_actions", "live_action", "public_observation_sha256",
                    "features", "comparisons", "outcomes"}
    outcome_fields = {"action", "timely_fraction", "detection_fraction", "episode_return", "cost",
                      "placements", "steps", "continuation_steps"}
    all_episodes = []
    visited = branches = pair_count = informed = 0

    def placements(value):
        if not isinstance(value, list) or len(value) > 32:
            raise ValueError("Invalid ranking placement list")
        positions = set()
        for row in value:
            if not isinstance(row, dict) or set(row) != {"sensor_id", "sensor_index", "position", "cost"}:
                raise ValueError("Invalid ranking placement fields")
            if not isinstance(row["sensor_id"], str) or not row["sensor_id"]:
                raise ValueError("Ranking placement sensor ID missing")
            integer(row["sensor_index"], "sensor index", 0, 31)
            number(row["cost"], "placement cost", 0.)
            if not isinstance(row["position"], list) or len(row["position"]) != 2:
                raise ValueError("Ranking placement coordinates differ")
            for v in row["position"]:
                number(v, "placement coordinate")
            position = tuple(row["position"])
            if position in positions:
                raise ValueError("Ranking layout repeats a site")
            positions.add(position)
        return value

    data = gzip.GzipFile(fileobj=io.BytesIO(blobs["comparisons.jsonl.gz"]), mode="rb")
    expanded = 0
    try:
        for batch_index, row in enumerate(log):
            begin = batch_index * batch_size
            batch_fields = {"event", "episode_start", "episode", "policy_seed", "scenario_seed_range",
                "optimizer_updates", "update_passes", "training_states", "all_visited_states", "pair_comparisons",
                "branch_rollouts", "case_profile_counts", "sensor_deployment_counts", "episodes", "updates",
                "mean_update_metrics", "weights_sha256", "collection_wall_seconds", "update_wall_seconds",
                *("mean_" + key for key in scalar_fields)}
            if not isinstance(row, dict) or set(row) != batch_fields or row["event"] != "training_batch":
                raise ValueError("Ranking batch schema differs")
            exact([row["episode_start"], row["episode"], row["policy_seed"], row["scenario_seed_range"]],
                  [begin, begin + batch_size, seed, {"start": seed_start + begin, "count": batch_size}], "batch cadence")
            if not isinstance(row["episodes"], list) or len(row["episodes"]) != batch_size:
                raise ValueError("Ranking batch episode records incomplete")
            batch, batch_visited, batch_branches, batch_pairs = [], 0, 0, 0
            for offset, record in enumerate(row["episodes"]):
                expected_seed = seed_start + begin + offset
                if not isinstance(record, dict) or set(record) != episode_fields:
                    raise ValueError("Ranking live episode schema differs")
                exact(record["seed"], expected_seed, "live seed")
                if record["profile"] not in ("normal", "stress", "capability"):
                    raise ValueError("Invalid recorded ranking profile")
                decisions = integer(record["decisions"], "live decision count", 1, 32)
                for key in ("branches", "comparisons", "training_states"):
                    integer(record[key], key)
                for key in ("timely_fraction", "detection_fraction"):
                    number(record[key], key, 0., 1.)
                number(record["episode_return"], "live return")
                # The frozen environment's sum([]) is integer zero for an
                # initial STOP; branch exports explicitly cast that cost.
                if not (type(record["cost"]) is int and record["cost"] == 0):
                    number(record["cost"], "live cost", 0.)
                if (type(record["success"]) is not bool or record["success"] != (record["timely_fraction"] == 1.)
                        or record["timely_fraction"] > record["detection_fraction"] + 1e-12):
                    raise ValueError("Live sensing metrics are inconsistent")
                integer(record["invalid_actions"], "invalid actions", 0, 0)
                placements(record["placements"])
                episode_branches = episode_pairs = episode_informed = 0
                prefix, committed, stop_action = [], [], None
                final_outcome = None
                for depth in range(decisions):
                    line = data.readline(150_001)
                    expanded += len(line)
                    if not line or len(line) > 150_000 or expanded > 1_000_000_000:
                        raise ValueError("Ranking dataset missing records or exceeds bounded expanded size")
                    item = canonical(line, "comparison dataset")
                    if not isinstance(item, dict) or set(item) != state_fields:
                        raise ValueError("Ranking comparison state schema differs")
                    exact(item["seed"], expected_seed, "dataset scenario seed")
                    exact(item["prefix"], prefix, "committed live prefix")
                    digest(item["public_observation_sha256"], "public observation")
                    actions = item["slate_actions"]
                    if not isinstance(actions, list) or not 1 <= len(actions) <= train["max_actions"]:
                        raise ValueError("Invalid ranking slate length")
                    for action in actions:
                        integer(action, "slate action", 0, rollout.MAX_OPTIONS - 1)
                    if len(set(actions)) != len(actions):
                        raise ValueError("Ranking slate actions repeat")
                    if stop_action is None:
                        stop_action = actions[0]
                    if actions[0] != stop_action or any(a >= stop_action for a in actions[1:]) or any(a in prefix for a in actions[1:]):
                        raise ValueError("Ranking slate STOP/order/committed action mismatch")
                    integer(item["live_action"], "live action", 0, rollout.MAX_OPTIONS - 1)
                    if item["live_action"] not in actions:
                        raise ValueError("Recorded live action is absent from slate")
                    features = item["features"]
                    if (not isinstance(features, list) or len(features) != len(actions)
                            or any(not isinstance(f, list) or len(f) != len(FEATURE_NAMES)
                                   or any(type(v) is not float or not np.isfinite(v) for v in f) for f in features)):
                        raise ValueError("Ranking slate features must be complete finite float rows")
                    x = replay._features(features, complete=False)
                    if (not np.array_equal(x[:, FEATURE_NAMES.index("stop")], [1.] + [0.] * (len(actions) - 1))
                            or not np.all(x[:, FEATURE_NAMES.index("bias")] == 1.)
                            or not np.all(x[:, FEATURE_NAMES.index("placed_count")] == depth / 4.)):
                        raise ValueError("Ranking slate feature STOP/bias/prefix semantics differ")
                    # Full-matrix and small-slate BLAS reductions can differ in
                    # their last bits; reject meaningful deviations, not ULPs.
                    scores = replay._forward(x)[2]
                    chosen_index = actions.index(item["live_action"])
                    if scores[chosen_index] + 1e-12 * max(1., abs(float(scores.max()))) < scores.max():
                        raise ValueError("Recorded live choice contradicts its public slate scores")
                    outcomes = item["outcomes"]
                    if not isinstance(outcomes, list) or len(outcomes) != len(actions):
                        raise ValueError("Ranking outcome slate incomplete")
                    for index, outcome in enumerate(outcomes):
                        if not isinstance(outcome, dict) or set(outcome) != outcome_fields:
                            raise ValueError("Ranking outcome schema differs")
                        exact(outcome["action"], actions[index], "outcome action order")
                        for key in ("timely_fraction", "detection_fraction"):
                            number(outcome[key], key, 0., 1.)
                        if outcome["timely_fraction"] > outcome["detection_fraction"] + 1e-12:
                            raise ValueError("Ranking timely fraction exceeds detection")
                        number(outcome["episode_return"], "branch return")
                        number(outcome["cost"], "branch cost", 0.)
                        layout = placements(outcome["placements"])
                        exact(layout[:depth], committed, "branch committed layout")
                        steps = integer(outcome["steps"], "branch steps", depth + 1, 32)
                        integer(outcome["continuation_steps"], "continuation steps", 0, 31)
                        if (outcome["continuation_steps"] != steps - depth - 1
                                or not len(layout) <= steps <= len(layout) + 1
                                or not _ranking_same(outcome["cost"], float(sum(p["cost"] for p in layout)))):
                            raise ValueError("Ranking branch horizon or cost differs")
                        if index == 0:
                            if steps != depth + 1 or len(layout) != depth:
                                raise ValueError("STOP branch must end with exactly the committed layout")
                        else:
                            if len(layout) <= depth:
                                raise ValueError("Deployment branch must add its proposed placement")
                            placement = layout[depth]
                            expected_features = {"cost": placement["cost"] / 4., "east": placement["position"][0] / 150.,
                                "north": placement["position"][1] / 150.,
                                "radius": float(np.linalg.norm(placement["position"])) / 150.}
                            if any((not np.isclose(x[index, FEATURE_NAMES.index(key)], float(np.float32(value)), rtol=2e-7, atol=2e-7)
                                    if key == "radius" else x[index, FEATURE_NAMES.index(key)] != float(np.float32(value)))
                                   for key, value in expected_features.items()):
                                raise ValueError("Ranking proposed placement differs from public slate geometry/cost")
                    exact(item["comparisons"], rollout.comparisons(outcomes), "paired preference labels")
                    if item["comparisons"]:
                        batch.append({"features": x, "comparisons": item["comparisons"]})
                        episode_informed += 1
                    episode_branches += len(actions)
                    episode_pairs += len(item["comparisons"])
                    final_outcome = outcomes[chosen_index]
                    if depth < decisions - 1:
                        if item["live_action"] == stop_action:
                            raise ValueError("Live trajectory continued after STOP")
                        prefix.append(item["live_action"])
                        committed = final_outcome["placements"][:depth + 1]
                exact([record["branches"], record["comparisons"], record["training_states"]],
                      [episode_branches, episode_pairs, episode_informed], "episode dataset counts")
                if final_outcome["continuation_steps"] != 0 or final_outcome["steps"] != decisions:
                    raise ValueError("Final live action did not terminate its archived branch")
                for key in ("timely_fraction", "detection_fraction", "episode_return", "cost", "placements"):
                    exact(float(record[key]) if key == "cost" else record[key], final_outcome[key], "terminal live outcome")
                batch_visited += decisions
                batch_branches += episode_branches
                batch_pairs += episode_pairs
                all_episodes.append(record)
            expected_updates = []
            if batch:
                for _ in range(train["passes_per_batch"]):
                    expected_updates.append(replay.update_comparisons(batch, **{key: train[key] for key in
                        ("learning_rate", "temperature", "regularization", "max_grad_norm")}))
            if not _ranking_same(row["updates"], expected_updates):
                raise ValueError("Ranking optimizer metrics differ from archived-data replay")
            means = {key: float(np.mean([m[key] for m in expected_updates])) for key in expected_updates[0]} if expected_updates else {}
            if not _ranking_same(row["mean_update_metrics"], means):
                raise ValueError("Ranking flattened update metrics differ")
            exact([row["update_passes"], row["optimizer_updates"], row["training_states"], row["all_visited_states"],
                   row["pair_comparisons"], row["branch_rollouts"]],
                  [len(expected_updates), replay.update_count, len(batch), batch_visited, batch_pairs, batch_branches], "batch totals")
            exact(row["case_profile_counts"], dict(Counter(r["profile"] for r in row["episodes"])), "batch profiles")
            exact(row["sensor_deployment_counts"], dict(Counter(p["sensor_id"] for r in row["episodes"] for p in r["placements"])), "batch sensors")
            for key in scalar_fields:
                if not _ranking_same(row["mean_" + key], float(np.mean([r[key] for r in row["episodes"]]))):
                    raise ValueError("Ranking live batch means differ")
            for key in ("collection_wall_seconds", "update_wall_seconds"):
                number(row[key], key, 0.)
            digest(row["weights_sha256"], "batch weights")
            # Intermediate fingerprints are byte identities from the producer;
            # cross-platform replay is checked numerically against the endpoint.
            visited += batch_visited
            branches += batch_branches
            pair_count += batch_pairs
            informed += len(batch)
        if data.read(1):
            raise ValueError("Ranking dataset has unconsumed records")
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise ValueError("Invalid or truncated ranking gzip dataset") from error
    finally:
        data.close()
    if endpoint.update_count != replay.update_count:
        raise ValueError("Ranking endpoint optimizer count differs from replay")
    for group in ("parameters", "adam_m", "adam_v"):
        for key, expected in getattr(replay, group).items():
            if not np.allclose(getattr(endpoint, group)[key], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Ranking endpoint arrays differ from complete optimizer replay")
    exact(log[-1]["weights_sha256"], endpoint.weights_fingerprint(), "last batch endpoint fingerprint")
    expected_summary = {"schema": SCHEMA, "policy_seed": seed, "completed_episodes": count,
        "optimizer_updates": replay.update_count, "checkpoint_rule": "fixed_endpoint_only",
        "initial_weights_sha256": initial.weights_fingerprint(), "last_weights_sha256": endpoint.weights_fingerprint(),
        "initialized_files_sha256": initial_files, "last_files_sha256": last_files, "evidence_files_sha256": evidence_files,
        "source_sha256": config["source_sha256"], "protocol_sha256": config["protocol_sha256"],
        "seed_provenance": checkpoint_state(count, evidence_files)["seed_provenance"],
        "profile_counts": dict(Counter(r["profile"] for r in all_episodes)),
        "sensor_deployment_counts": dict(Counter(p["sensor_id"] for r in all_episodes for p in r["placements"])),
        "decisions": visited, "branches": branches, "comparisons": pair_count, "training_states": informed,
        **{"mean_" + key: float(np.mean([r[key] for r in all_episodes])) for key in scalar_fields},
        "collection_wall_seconds": sum(row["collection_wall_seconds"] for row in log),
        "update_wall_seconds": sum(row["update_wall_seconds"] for row in log),
        "total_trainer_wall_seconds": summary.get("total_trainer_wall_seconds")}
    number(summary.get("total_trainer_wall_seconds"), "total trainer seconds", 0.)
    if not _ranking_same(summary, expected_summary):
        raise ValueError("Ranking summary differs from complete archived evidence")
    if summary["total_trainer_wall_seconds"] + 1e-9 < summary["collection_wall_seconds"] + summary["update_wall_seconds"]:
        raise ValueError("Ranking component times exceed total trainer elapsed time")
    _stable(protocol_path, raw, config)
    if any(shared._file_hash(paths[name]) != digest_value for name, digest_value in hashes.items()):
        raise ValueError("Ranking evidence changed during validation")
    return endpoint, {"weights_sha256": endpoint.weights_fingerprint(), "initialized_weights_sha256": initial.weights_fingerprint(),
        "run_files_sha256": hashes, "summary": summary,
        "verification_scope": {"preference_labels_recomputed": True, "optimizer_updates_replayed": replay.update_count,
            "runtime_numeric_tolerance": 1e-12, "scenario_rollouts": 0, "filesystem_writes": 0,
            "full_public_observations_reconstructed": False,
            "limitation": "Exact hashes bind saved slate rows and outcomes; no full public snapshot or independent simulator replay is claimed."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", dest="protocol_path", type=Path, required=True)
    parser.add_argument("--seed", dest="policy_seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run_training(**vars(parser.parse_args()))
    print(shared._json({key: result[key] for key in ("policy_seed", "completed_episodes", "optimizer_updates", "last_weights_sha256")}))


if __name__ == "__main__":
    main()
