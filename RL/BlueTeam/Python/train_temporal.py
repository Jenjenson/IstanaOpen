"""Fixed-endpoint temporal REINFORCE pilot; no selection, resume or filtering."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from triad_rl.adaptive_policy import _json
from triad_rl.ranking_policy import _strict_json
from triad_rl.robust_scenarios import curriculum_manifest
from triad_rl.temporal_env import TemporalPlacementEnv
from triad_rl.temporal_inputs import TemporalConfig
from triad_rl.temporal_lineage import load_published_lineage, assert_disjoint
from triad_rl.temporal_policy import TemporalPolicy

BLUE_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "triad.temporal_training.v1"
PROTOCOL_SCHEMA = "triad.temporal_pilot_protocol.v1"
FIXED_TRAINING = {"policy_seeds": [406, 407, 408], "episodes": 512, "batch_size": 16,
    "profile": "mixed", "hidden_size": 64, "temperature": .05, "learning_rate": .001,
    "critic_learning_rate": .001, "gamma": 1., "entropy_coef": 0., "value_coef": 1.,
    "normalize_advantages": False, "max_grad_norm": 1., "endpoint": "fixed_no_selection_or_resume"}
RUN_FILES = ("config.json", "protocol.json", "training.jsonl", "episodes.jsonl.gz", "summary.json",
             "initialized/checkpoint.json", "initialized/arrays.npz", "last/checkpoint.json", "last/arrays.npz")
SOURCE_FILES = ("train_temporal.py", "triad_rl/temporal_policy.py", "triad_rl/temporal_env.py",
    "triad_rl/temporal_inputs.py", "triad_rl/temporal_confirmation.py", "triad_rl/temporal_lineage.py",
    "triad_rl/adaptive_env.py", "triad_rl/adaptive_inputs.py", "triad_rl/robust_scenarios.py",
    "triad_rl/adaptive_policy.py", "triad_rl/ranking_policy.py", "triad_rl/credit_lineage.py")
TERMINAL_KEYS = ("return", "cost", "success", "detected_fraction", "confirmed_fraction", "breached_fraction",
                 "early_detection", "invalid_actions", "placements", "reward_components")
UPDATE_KEYS = ("learning_rate", "critic_learning_rate", "gamma", "entropy_coef", "value_coef",
               "normalize_advantages", "max_grad_norm")
METRIC_KEYS = {"actor_loss", "critic_loss", "entropy", "joint_entropy", "gate_entropy", "stop_probability",
    "deployment_probability", "actor_gradient_norm", "actor_gradient_scale", "critic_gradient_norm",
    "critic_gradient_scale", "loss", "mean_return_to_go", "raw_advantage_mean", "raw_advantage_std",
    "advantages_normalized", "transitions", "updates"}


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_plain(item) for item in value]
    return value


def _bytes(value):
    return (_json(_plain(value)) + "\n").encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    with Path(path).open("rb") as handle: raw = handle.read(4_000_001)
    if len(raw) > 4_000_000: raise ValueError("Temporal JSON document exceeds bound")
    return _strict_json(raw), raw


def _same(left, right):
    return _bytes(left) == _bytes(right)


def source_provenance():
    return {name: _sha(BLUE_ROOT / "Python" / name) for name in SOURCE_FILES}


def load_protocol(path):
    protocol, raw = _read(path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported temporal pilot protocol")
    training = protocol.get("training", {})
    ranges = {str(seed): {"start": seed * 1_000_000, "count": FIXED_TRAINING["episodes"]}
              for seed in FIXED_TRAINING["policy_seeds"]}
    if not _same(training, {**FIXED_TRAINING, "scenario_seed_ranges": ranges}):
        raise ValueError("Temporal pilot training settings or scenario slots differ")
    mission = protocol.get("temporal_config")
    if not isinstance(mission, dict) or not _same(mission, asdict(TemporalConfig(**mission))):
        raise ValueError("Explicit complete temporal mission configuration is required")
    if not _same(protocol.get("training_implementation_sha256"), source_provenance()):
        raise ValueError("Temporal source differs from predeclared hashes")
    return protocol, raw


def _new_output(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or path.exists():
        raise ValueError("Temporal output must be a new non-symlink directory")
    return path


def _lineage(protocol, lineage):
    if not _same(protocol.get("prior_publication"), lineage["prior_pilot"]["publication_manifest"]):
        raise ValueError("Temporal prior publication binding differs")
    for interval in protocol["training"]["scenario_seed_ranges"].values():
        assert_disjoint(interval["start"], interval["count"], lineage, label="temporal training")


def _configuration(protocol, raw, seed, lineage, initial, runtime):
    return {"schema": SCHEMA, "policy_seed": seed, "training": protocol["training"],
        "temporal_config": protocol["temporal_config"], "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": protocol["training_implementation_sha256"], "inherited_lineage": lineage,
        "initial_weights_sha256": initial.weights_fingerprint(), "initial_rng_sha256": initial.rng_fingerprint(),
        "initialization": "Fresh seeded temporal prior; zero actor residual and separate critic; constructor RNG retained",
        "checkpoint_rule": "fixed_endpoint_only", "curriculum": curriculum_manifest(), "runtime": runtime}


def _state(config, completed, evidence=None):
    return {"schema": SCHEMA, "policy_seed": config["policy_seed"], "completed_episodes": completed,
            "protocol_sha256": config["protocol_sha256"], "source_sha256": config["source_sha256"],
            "config_sha256": hashlib.sha256(_bytes(config)).hexdigest(), "evidence_files_sha256": evidence or {},
            "seed_provenance": {"inherited": config["inherited_lineage"],
                "training": {"start": config["policy_seed"] * 1_000_000, "count": completed}}}


def _stable(path, raw, config, output=None):
    if (Path(path).read_bytes() != raw or not _same(source_provenance(), config["source_sha256"])
            or any(_sha(BLUE_ROOT / row["path"]) != row["sha256"] for row in config["inherited_lineage"]["publication_manifests"])):
        raise ValueError("Temporal protocol, sources or inherited manifests changed")
    if output is not None and ((output / "config.json").read_bytes() != _bytes(config)
                              or (output / "protocol.json").read_bytes() != raw):
        raise ValueError("Temporal archived configuration changed")


def _checkpoint_files(path):
    return {name: _sha(path / name) for name in ("checkpoint.json", "arrays.npz")}


def _episode_metrics(episodes):
    records = [row["terminal"] for row in episodes]
    return {"profile_counts": dict(Counter(row["profile"] for row in episodes)),
            "sensor_deployment_counts": dict(Counter(p["sensor_id"] for row in records for p in row["placements"])),
            "transitions": sum(row["transition_count"] if "transition_count" in row else len(row["transitions"]) for row in episodes),
            **{f"mean_{key}": float(np.mean([row[key] for row in records])) for key in
               ("return", "cost", "success", "detected_fraction", "confirmed_fraction", "breached_fraction", "invalid_actions")},
            "mean_timely_fraction": float(np.mean([1. - row["breached_fraction"] for row in records]))}


def collect_episode(policy, scenario_seed, profile, mission, index):
    # Exactly one explicitly requested case; never construct then reset another.
    env = TemporalPlacementEnv(seed=scenario_seed, profile=profile, config=mission)
    observation, transitions = env.observe(), []
    before_rng, before_weights = policy.rng_fingerprint(), policy.weights_fingerprint()
    for _ in range(32):
        action, record = policy.sample(observation)
        observation, reward, done, info = env.step(action)
        if not np.isfinite(reward): raise ValueError("Nonfinite original environment reward")
        record["reward"] = float(reward)
        transitions.append(record)
        if done: break
    else:
        raise ValueError("Temporal episode exceeded the fixed placement horizon")
    if info["invalid_actions"] != 0 or not np.isclose(sum(row["reward"] for row in transitions), info["return"], rtol=0., atol=1e-10):
        raise ValueError("Temporal live trajectory must preserve original legal episode return")
    terminal = {key: info[key] for key in TERMINAL_KEYS}
    row = {"schema": SCHEMA, "episode": index, "seed": scenario_seed, "profile": env.case_metadata["profile"],
           "weights_sha256": before_weights, "rng_before_sha256": before_rng, "rng_after_sha256": policy.rng_fingerprint(),
           "transitions": transitions, "terminal": terminal}
    _bytes(row)  # Validate finite, serializable audit data before retaining it.
    return transitions, row


def run_training(protocol_path, policy_seed, output):
    started, output = time.perf_counter(), _new_output(output)
    protocol, raw = load_protocol(protocol_path)
    train = protocol["training"]
    if type(policy_seed) is not int or policy_seed not in train["policy_seeds"]:
        raise ValueError("Temporal policy seed was not declared")
    lineage = load_published_lineage()
    _lineage(protocol, lineage)
    mission = TemporalConfig(**protocol["temporal_config"])
    actor = TemporalPolicy(mission, seed=policy_seed, hidden_size=train["hidden_size"], temperature=train["temperature"])
    config = _configuration(protocol, raw, policy_seed, lineage, actor, {"python": platform.python_version(), "numpy": np.__version__})
    _stable(protocol_path, raw, config)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("protocol.json", raw), ("config.json", _bytes(config))):
        with (output / name).open("xb") as handle: handle.write(data)
    actor.save(output / "initialized", _state(config, 0))
    initialized_files = _checkpoint_files(output / "initialized")
    all_rows, batch_rows = [], []
    with (output / "training.jsonl").open("xb") as log, (output / "episodes.jsonl.gz").open("xb") as archive:
        with gzip.GzipFile(fileobj=archive, mode="wb", filename="", mtime=0) as records:
            for first in range(0, train["episodes"], train["batch_size"]):
                _stable(protocol_path, raw, config, output)
                if _checkpoint_files(output / "initialized") != initialized_files:
                    raise ValueError("Temporal initialized checkpoint changed")
                batch_started, batch, rows = time.perf_counter(), [], []
                before = actor.weights_fingerprint()
                for index in range(first, first + train["batch_size"]):
                    transitions, row = collect_episode(actor, policy_seed * 1_000_000 + index, train["profile"], mission, index)
                    batch.append(transitions); rows.append(row)
                    records.write(_bytes(row))
                collection_seconds, update_started = time.perf_counter() - batch_started, time.perf_counter()
                metrics = actor.update(batch, **{key: train[key] for key in UPDATE_KEYS})
                row = {"schema": SCHEMA, "event": "training_batch", "policy_seed": policy_seed,
                    "episode_start": first, "episode": first + len(rows), "weights_before_sha256": before,
                    "weights_sha256": actor.weights_fingerprint(), "rng_sha256": actor.rng_fingerprint(),
                    "episode_records_sha256": hashlib.sha256(b"".join(_bytes(item) for item in rows)).hexdigest(),
                    "update_metrics": metrics, **_episode_metrics(rows),
                    "collection_wall_seconds": collection_seconds, "update_wall_seconds": time.perf_counter() - update_started}
                log.write(_bytes(row)); log.flush(); records.flush()
                # The full matrices are already in gzip. Retain only compact
                # terminal accounting, not every training tensor for the run.
                all_rows.extend({"profile": item["profile"], "terminal": item["terminal"],
                                 "transition_count": len(item["transitions"])} for item in rows)
                batch_rows.append(row)
                print(_json({"event": "training_batch", "policy_seed": policy_seed, "episode": row["episode"],
                             "updates": metrics["updates"], "mean_return": row["mean_return"], "loss": metrics["loss"]}), flush=True)
    _stable(protocol_path, raw, config, output)
    if not _same(load_published_lineage(), lineage): raise ValueError("Temporal inherited publication changed")
    evidence = {name: _sha(output / name) for name in ("config.json", "protocol.json", "training.jsonl", "episodes.jsonl.gz")}
    actor.save(output / "last", _state(config, train["episodes"], evidence))
    summary = {"schema": SCHEMA, "policy_seed": policy_seed, "completed_episodes": len(all_rows),
        "optimizer_updates": actor.update_count, "initial_weights_sha256": config["initial_weights_sha256"],
        "last_weights_sha256": actor.weights_fingerprint(), "initial_rng_sha256": config["initial_rng_sha256"],
        "last_rng_sha256": actor.rng_fingerprint(), "checkpoint_rule": "fixed_endpoint_only",
        "initialized_files_sha256": initialized_files, "last_files_sha256": _checkpoint_files(output / "last"),
        "evidence_files_sha256": evidence, "seed_provenance": _state(config, len(all_rows))["seed_provenance"],
        **_episode_metrics(all_rows), "collection_wall_seconds": sum(row["collection_wall_seconds"] for row in batch_rows),
        "update_wall_seconds": sum(row["update_wall_seconds"] for row in batch_rows),
        "total_trainer_wall_seconds": time.perf_counter() - started}
    with (output / "summary.json").open("xb") as handle: handle.write(_bytes(summary))
    return summary


def _derived(left, right):
    if type(left) is not type(right): return False
    if isinstance(left, dict): return left.keys() == right.keys() and all(_derived(left[key], right[key]) for key in left)
    if isinstance(left, list): return len(left) == len(right) and all(_derived(a, b) for a, b in zip(left, right))
    if isinstance(left, float): return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-12, atol=1e-12))
    return left == right


def validate_run(path, protocol_path, *, lineage=None):
    """Read-only binding/record validation; no historical update replay or inference."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir(): raise ValueError("Invalid temporal run path")
    if {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()} != set(RUN_FILES):
        raise ValueError("Temporal run requires exactly its complete fixed-endpoint artifacts")
    if any((path / name).is_symlink() for name in RUN_FILES): raise ValueError("Temporal artifacts cannot be symlinks")
    hashes = {name: _sha(path / name) for name in RUN_FILES}
    protocol, raw = load_protocol(protocol_path)
    train, config, summary = protocol["training"], _read(path / "config.json")[0], _read(path / "summary.json")[0]
    lineage = load_published_lineage() if lineage is None else lineage
    _lineage(protocol, lineage)
    seed = config.get("policy_seed")
    if type(seed) is not int or seed not in train["policy_seeds"]: raise ValueError("Unexpected temporal run seed")
    mission = TemporalConfig(**protocol["temporal_config"])
    initial, actor = (TemporalPolicy.load(path / name, config=mission) for name in ("initialized", "last"))
    fresh = TemporalPolicy(mission, seed=seed, hidden_size=train["hidden_size"], temperature=train["temperature"])
    if (initial.weights_fingerprint() != fresh.weights_fingerprint() or initial.rng_fingerprint() != fresh.rng_fingerprint()
            or any(np.any(value) for group in (initial.adam_m, initial.adam_v) for value in group.values())
            or initial.update_count != 0 or initial.critic_update_count != 0):
        raise ValueError("Temporal initialization differs from declared fresh prior")
    if not _same(config, _configuration(protocol, raw, seed, lineage, fresh, config.get("runtime"))):
        raise ValueError("Temporal run configuration binding differs")
    evidence = {name: hashes[name] for name in ("config.json", "protocol.json", "training.jsonl", "episodes.jsonl.gz")}
    if (not _same(initial.training_state, _state(config, 0)) or not _same(actor.training_state, _state(config, train["episodes"], evidence))
            or actor.update_count != train["episodes"] // train["batch_size"] or actor.critic_update_count != actor.update_count):
        raise ValueError("Temporal endpoint or checkpoint evidence binding differs")
    with (path / "training.jsonl").open("rb") as handle: batches = [_strict_json(line) for line in handle]
    rows, expanded = [], 0
    with gzip.open(path / "episodes.jsonl.gz", "rb") as handle:
        for index in range(train["episodes"]):
            line = handle.readline(32_000_001); expanded += len(line)
            if not line or len(line) > 32_000_000 or expanded > 512_000_000: raise ValueError("Invalid bounded temporal episode archive")
            row = _strict_json(line)
            if not _same(row.get("schema"), SCHEMA) or type(row.get("episode")) is not int or row["episode"] != index or type(row.get("seed")) is not int or row["seed"] != seed * 1_000_000 + index:
                raise ValueError("Temporal episode sequence differs")
            if row.get("profile") not in ("normal", "stress", "capability"): raise ValueError("Unexpected temporal profile")
            transitions, terminal = row["transitions"], row["terminal"]
            if not 1 <= len(transitions) <= 32 or set(terminal) != set(TERMINAL_KEYS): raise ValueError("Incomplete temporal episode")
            for record in transitions:
                if set(record) != {"features", "mask", "action", "value", "reward", "temperature", "feature_schema", "temporal_config"}:
                    raise ValueError("Unexpected temporal transition fields")
                actor._record(record)  # Matrix, legality and semantic validation only.
                for key in ("reward", "value"):
                    if type(record.get(key)) not in (int, float) or not np.isfinite(record[key]): raise ValueError("Invalid sampled numeric record")
            for key in ("detected_fraction", "confirmed_fraction", "breached_fraction", "early_detection"):
                if type(terminal[key]) not in (int, float) or not 0 <= terminal[key] <= 1: raise ValueError("Invalid sensing fraction")
            if (type(terminal["invalid_actions"]) is not int or terminal["invalid_actions"] != 0
                    or type(terminal["return"]) not in (int, float) or type(terminal["cost"]) not in (int, float)
                    or type(terminal["success"]) is not bool or terminal["success"] != (terminal["breached_fraction"] == 0.)
                    or terminal["confirmed_fraction"] > terminal["detected_fraction"] + 1e-12
                    or 1. - terminal["breached_fraction"] > terminal["confirmed_fraction"] + 1e-12
                    or not np.isclose(sum(r["reward"] for r in transitions), terminal["return"], rtol=0., atol=1e-10)
                    or terminal["cost"] < 0 or not np.isclose(sum(p["cost"] for p in terminal["placements"]), terminal["cost"], rtol=0., atol=1e-10)
                    or not np.isclose(sum(terminal["reward_components"].values()), terminal["return"], rtol=0., atol=1e-10)):
                raise ValueError("Temporal original return, cost or outcome differs")
            rows.append(row)
        if handle.read(1): raise ValueError("Unexpected extra temporal episodes")
    if len(batches) != actor.update_count: raise ValueError("Incomplete temporal batch log")
    previous_weights, previous_rng = initial.weights_fingerprint(), initial.rng_fingerprint()
    for index, batch in enumerate(batches):
        first = index * train["batch_size"]; group = rows[first:first + train["batch_size"]]
        required = {"schema": SCHEMA, "event": "training_batch", "policy_seed": seed, "episode_start": first,
                    "episode": first + len(group), "weights_before_sha256": previous_weights,
                    "episode_records_sha256": hashlib.sha256(b"".join(_bytes(row) for row in group)).hexdigest(), **_episode_metrics(group)}
        if any(not _derived(batch.get(key), value) for key, value in required.items()): raise ValueError("Temporal batch aggregate or record digest differs")
        for row in group:
            if row["weights_sha256"] != previous_weights or row["rng_before_sha256"] != previous_rng: raise ValueError("Temporal within-batch policy/RNG binding differs")
            previous_rng = row["rng_after_sha256"]
        metrics = batch["update_metrics"]
        if (set(metrics) != METRIC_KEYS or type(metrics.get("updates")) is not int or metrics["updates"] != index + 1
                or type(metrics.get("transitions")) is not int or metrics["transitions"] != sum(len(r["transitions"]) for r in group)
                or metrics.get("advantages_normalized") is not False or batch["rng_sha256"] != previous_rng
                or any(type(value) not in (int, float) or not np.isfinite(value) for key, value in metrics.items() if key != "advantages_normalized")
                or any(not 0. <= metrics[key] <= 1. for key in ("actor_gradient_scale", "critic_gradient_scale", "stop_probability", "deployment_probability"))
                or any(metrics[key] < 0. for key in ("actor_gradient_norm", "critic_gradient_norm", "raw_advantage_std"))):
            raise ValueError("Temporal update metrics or cadence differs")
        if any(type(batch.get(key)) is not float or not np.isfinite(batch[key]) or batch[key] < 0.
               for key in ("collection_wall_seconds", "update_wall_seconds")):
            raise ValueError("Temporal batch timing must be finite and nonnegative")
        previous_weights = batch["weights_sha256"]
    if previous_weights != actor.weights_fingerprint() or previous_rng != actor.rng_fingerprint(): raise ValueError("Temporal final log/checkpoint differs")
    expected = {"schema": SCHEMA, "policy_seed": seed, "completed_episodes": len(rows), "optimizer_updates": actor.update_count,
        "initial_weights_sha256": initial.weights_fingerprint(), "last_weights_sha256": actor.weights_fingerprint(),
        "initial_rng_sha256": initial.rng_fingerprint(), "last_rng_sha256": actor.rng_fingerprint(), "checkpoint_rule": "fixed_endpoint_only",
        "initialized_files_sha256": _checkpoint_files(path / "initialized"), "last_files_sha256": _checkpoint_files(path / "last"),
        "evidence_files_sha256": evidence, "seed_provenance": _state(config, len(rows))["seed_provenance"], **_episode_metrics(rows),
        "collection_wall_seconds": sum(row["collection_wall_seconds"] for row in batches),
        "update_wall_seconds": sum(row["update_wall_seconds"] for row in batches), "total_trainer_wall_seconds": summary.get("total_trainer_wall_seconds")}
    if not _derived(summary, expected) or not isinstance(summary["total_trainer_wall_seconds"], float) or summary["total_trainer_wall_seconds"] < 0:
        raise ValueError("Temporal completed summary differs")
    if summary["total_trainer_wall_seconds"] + 1e-9 < summary["collection_wall_seconds"] + summary["update_wall_seconds"]:
        raise ValueError("Temporal component times exceed elapsed trainer time")
    _stable(protocol_path, raw, config, path)
    if hashes != {name: _sha(path / name) for name in RUN_FILES}: raise ValueError("Temporal run changed during verification")
    return actor, {"weights_sha256": actor.weights_fingerprint(), "initialized_weights_sha256": initial.weights_fingerprint(),
        "run_files_sha256": hashes, "summary": summary, "verification_scope": {"scenario_rollouts": 0, "optimizer_replays": 0,
            "inference_calls": 0, "filesystem_writes": 0, "scope": "Exact artifact/endpoint bindings and complete recorded rollout/batch accounting; not independent physical replay"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_training(args.protocol, args.seed, args.output)
    print(_json({key: summary[key] for key in ("policy_seed", "completed_episodes", "optimizer_updates", "last_weights_sha256")}))


if __name__ == "__main__": main()
