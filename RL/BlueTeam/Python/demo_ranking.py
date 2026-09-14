"""Offline replay of every fixed ranking endpoint on already-scored cases only.

Exactly the first two completed validation cases per profile are shown, without
winner/example selection. Experimental synthetic sensing, not live control,
policy promotion, physical commands or independent final-test evidence.
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
import evaluate_ranking as evaluator
from demo_robust import collect_replay, _policy_state
from demo_balanced import export_balanced_html, implementation_fingerprints as viewer_sources
from triad_rl.adaptive_evaluation import canonical_hash
from triad_rl.ranking_policy import RankPolicy, POLICY_SCHEMA

SCHEMA = "triad.ranking_replay.v1"
RELEASE = "ranking-v5-pilot"
CASES_PER_PROFILE = 2


def implementation_fingerprints():
    return {**viewer_sources(), **evaluator.source_provenance(),
            "demo_ranking.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _metrics(replay):
    info = replay["metrics"]
    result = {key: value for key, value in info.items()
              if (isinstance(value, (int, float, bool)) or value is None)
              and key not in {"scenario_seed", "objective_radius"}}
    result["sensors_placed"] = len(replay["placements"])
    result["blind_spot_fraction"] = 1. - info["coverage"]
    result["budget_fraction_spent"] = info["cost"] / replay["scenario"]["public"]["budget_total"]
    result["mean_confirmation_time_censored"] = float(np.mean([
        row["time_to_zone"] if row["first_confirmation"] is None else min(row["first_confirmation"], row["time_to_zone"])
        for row in info["target_results"]]))
    return result


def run_demo(*, runs, evaluation, protocol, output):
    """Verify completion, recreate only the fixed scored subset, then export once."""
    output, evaluation, protocol = Path(output), Path(evaluation), Path(protocol)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Ranking demo output already exists; nothing will be overwritten")
    runs = {seed: Path(path) for seed, path in runs.items()}
    sources = implementation_fingerprints()
    paths = [protocol, *(evaluation / name for name in evaluator.EVALUATION_FILES),
             *(path / name for path in runs.values() for name in evaluator.trainer.RUN_FILES)]
    snapshots = {path: evaluator._file(path) for path in paths}
    evaluator.verify_completed_evaluation(runs, protocol_path=protocol, output=evaluation)
    declaration = evaluator.trainer.shared._strict_json(snapshots[protocol])
    policies = {seed: RankPolicy.load(path / "last", feature_names=evaluator.FEATURE_NAMES) for seed, path in runs.items()}
    states = {seed: _policy_state(policy) for seed, policy in policies.items()}

    def stable():
        if (sources != implementation_fingerprints() or any(evaluator._file(path) != data for path, data in snapshots.items())
                or any(_policy_state(policy) != states[seed] for seed, policy in policies.items())):
            raise RuntimeError("Ranking replay source/checkpoint/evidence/RNG changed; no output written")

    plan = []
    for profile in evaluator.PROFILES:
        data = snapshots[evaluation / f"validation-{profile}.json.gz"]
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
            expanded = handle.read(evaluator.MAX_BYTES + 1)
        if len(expanded) > evaluator.MAX_BYTES:
            raise ValueError("Ranking replay report exceeds size bound")
        report = evaluator.trainer.shared._strict_json(expanded)
        start = declaration["validation"]["scenario_seed_ranges"][profile]["start"]
        for seed in sorted(policies):
            rows = report["methods"][f"ranker_{seed}"]["episodes"][:CASES_PER_PROFILE]
            if len(rows) != CASES_PER_PROFILE or [row["seed"] for row in rows] != [start, start + 1]:
                raise ValueError("Replay requires the first two declared scored cases of every profile")
            plan.extend((seed, profile, index + 1, row) for index, row in enumerate(rows))
    stable()
    replays = []
    for seed, profile, case, row in plan:
        replay = collect_replay(policies[seed], seed=row["seed"], profile=profile)
        scenario = {**replay["scenario"], "evaluation_catalogue": replay["catalogue"],
                    "curriculum_metadata": replay["case_metadata"]}
        if (canonical_hash(scenario) != row["scenario_sha256"]
                or not evaluator._same([item["action_index"] for item in replay["decisions"]], row["actions"])
                or not evaluator._same(replay["placements"], row["placements"])
                or not evaluator.derived_same(_metrics(replay), row["metrics"])
                or not evaluator.derived_same(replay["metrics"]["reward_components"], row["reward_components"])
                or not evaluator.derived_same(replay["metrics"]["target_results"], row["target_results"])
                or replay["metrics"]["outcome"] != row["outcome"]
                or states[seed]["weights_sha256"] != row["weights_sha256_before"]
                or states[seed]["weights_sha256"] != row["weights_sha256_after"]):
            raise ValueError("Recreated ranking replay differs from its scored actions/scenario/metrics")
        replay.update(schema=SCHEMA, release=RELEASE, ranker_seed=seed, case_index=case,
                      split=f"EXPERIMENTAL ranker {seed} / {profile} / case {case} of 2 / offline replay")
        replay["audit"].update(schema="triad.ranking_replay_audit.v1", policy_schema=POLICY_SCHEMA, ranking_implementation_sha256=deepcopy(sources),
            evaluation_protocol_sha256=hashlib.sha256(snapshots[protocol]).hexdigest(),
            scored_scenario_sha256=row["scenario_sha256"], scored_actions_metrics_verified=True,
            checkpoint_files_sha256_before=evaluator._checkpoint_files(runs[seed] / "last"),
            checkpoint_files_sha256_after=evaluator._checkpoint_files(runs[seed] / "last"),
            example_rule="First two scored cases per profile for every fixed endpoint; no winner selection",
            experimental=True, default_policy_changed=False, physical_commands=False, live_feed=False)
        replays.append(replay)
    stable()
    with tempfile.TemporaryDirectory(prefix="ranking-offline-demo-") as temporary:
        rendered = Path(temporary) / "replay.html"
        export_balanced_html(replays, rendered,
            checkpoint="EXPERIMENTAL ranking-v5 / ALL fixed endpoints / OFFLINE, NOT LIVE / no promotion or physical commands / checkpoint-set digest",
            weights_sha256=canonical_hash({str(seed): state["weights_sha256"] for seed, state in states.items()}))
        data = rendered.read_bytes()
        stable()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(data)
    return replays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED=RUN_DIRECTORY")
    for key in ("evaluation", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args, runs = vars(parser.parse_args(argv)), {}
    for item in args.pop("run"):
        seed, separator, path = item.partition("=")
        if not separator or not seed.isdecimal() or int(seed) in runs or not path:
            parser.error("Use one unique integer SEED=RUN_DIRECTORY per endpoint")
        runs[int(seed)] = Path(path)
    print(f"Saved {len(run_demo(runs=runs, **args))} verified offline ranking replays; no policy promoted.")


if __name__ == "__main__":
    main()
