from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from triad_rl.adaptive_evaluation import (
    LEGACY_PARAMETER_SHA256, GreedyPublicCoverage, LegacyToy210, RandomLegal,
    UniformFixedRFAndRadar, assert_disjoint_seeds, canonical_hash,
    checkpoint_provenance, evaluate_methods, nearest_same_sensor, paired_bootstrap,
)


CHECKPOINT = Path(__file__).resolve().parents[2] / "Checkpoints" / "toy-210"


def public_observation():
    sensors = [
        {"id": "rf", "cost": .8, "ranges": {"rf": 130}, "height_m": 4},
        {"id": "radar", "cost": 1.2, "ranges": {"radar": 100}, "height_m": 4},
        {"id": "eo", "cost": .7, "ranges": {"eo": 100}, "height_m": 4},
        {"id": "thermal", "cost": 1, "ranges": {"thermal": 115}, "height_m": 4},
        {"id": "fused", "cost": 2, "ranges": {"radar": 100, "thermal": 125}, "height_m": 4},
    ]
    options = [{"sensor_id": s["id"], "sensor_index": i, "position": [0., y]}
               for i, s in enumerate(sensors) for y in (100., -100.)]
    options.append({"sensor_id": None, "position": [0., 0.]})
    return {
        "options": options, "action_mask": np.ones(len(options), dtype=bool),
        "option_features": np.zeros((len(options), 4)),
        "feature_names": ["marginal_coverage", "marginal_early_coverage", "cost", "overlap"],
        "catalogue": sensors,
        "state": {"budget_remaining": 2.4, "budget_total": 2.4,
                  "max_sites": 2, "deployment_max_radius": 150., "deployment_min_radius": 30.,
                  "placements": [], "forecast": {}, "weather": {}, "tracks": []},
    }


class OneStepEnv:
    def __init__(self, *, seed, split):
        self.seed = seed
        self.scenario = {"seed": seed, "split": split,
                         "weather": {"name": "clear"},
                         "threats": [{"emitter": "on", "path": "straight"}]}

    def reset(self, *, seed):
        return public_observation()

    def step(self, action):
        # Explicit keyed stochastic outcome: actor RNG consumption is irrelevant.
        result = float(np.random.default_rng(self.seed).random())
        return public_observation(), result, True, {"win": result > .5, "coverage": result,
                                                   "placements": [], "frames": [{"t": 0}]}


def test_exact_paired_seeds_and_independent_policy_rng():
    report = evaluate_methods({"adaptive": RandomLegal, "random": RandomLegal},
                              episodes=5, seed=30, replay_count=1,
                              env_factory=OneStepEnv, bootstrap_samples=20)
    a, b = [report["methods"][name]["episodes"] for name in ("adaptive", "random")]
    assert [row["seed"] for row in a] == list(range(30, 35))
    assert [row["scenario_sha256"] for row in a] == [row["scenario_sha256"] for row in b]
    assert a[0]["policy_seed"] != b[0]["policy_seed"]
    assert [r["metrics"] for r in a] == [r["metrics"] for r in b]
    assert report["paired_differences"]["adaptive_minus_random"]["return"]["upper95"] == 0
    assert "replay" in a[0] and "replay" not in a[1]
    assert report["methods"]["random"]["subgroups"]["swarm_size"]["1"]["episodes"] == 5


def test_method_cannot_mutate_shared_public_observation_or_see_truth():
    class Mutator:
        def __init__(self, seed):
            pass

        def act(self, observation, deterministic=True):
            assert "scenario" not in observation and "threats" not in observation
            observation["state"]["placements"].append({"sensor_id": "bogus"})
            return 0

    result = evaluate_methods({"adaptive": Mutator}, episodes=1, seed=3,
                              env_factory=OneStepEnv, bootstrap_samples=5)
    assert result["methods"]["adaptive"]["episodes"][0]["placements"] == []


def test_bootstrap_is_paired_reproducible_and_rejects_missing():
    result = paired_bootstrap([5, 10, -3], [3, 8, -5], seed=1, samples=50)
    assert result["difference"] == result["lower95"] == result["upper95"] == 2
    assert result == paired_bootstrap([5, 10, -3], [3, 8, -5], seed=1, samples=50)
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
    with pytest.raises(ValueError):
        paired_bootstrap([np.nan], [1])


def test_declared_training_and_validation_seed_overlap_rejected():
    provenance = {"training": {"start": 100, "count": 20},
                  "validation": {"start": 1000, "count": 10}}
    for start, count in ((90, 11), (110, 1), (999, 2), (1009, 1)):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint_seeds(start, count, provenance)
    assert_disjoint_seeds(120, 880, provenance)


def test_actual_adaptive_checkpoint_provenance_rejects_overlap(tmp_path):
    from triad_rl.adaptive_policy import AdaptivePolicy
    checkpoint = tmp_path / "policy"
    policy = AdaptivePolicy(public_observation()["feature_names"])
    policy.save(checkpoint, training_state={"seed_provenance": {
        "training": {"start": 100, "count": 20}, "validation": {"start": 1000, "count": 10}}})
    loaded = AdaptivePolicy.load(checkpoint)
    provenance = checkpoint_provenance(loaded.metadata)
    with pytest.raises(ValueError, match="overlap validation"):
        evaluate_methods({"adaptive": lambda seed: loaded}, episodes=2, seed=1000,
                         seed_provenance=provenance, env_factory=OneStepEnv)


def test_method_rng_is_independent_of_method_order():
    args = dict(episodes=2, seed=5, env_factory=OneStepEnv, bootstrap_samples=5)
    first = evaluate_methods({"adaptive": RandomLegal, "random": RandomLegal}, **args)
    second = evaluate_methods({"random": RandomLegal, "adaptive": RandomLegal}, **args)
    for method in ("adaptive", "random"):
        a = first["methods"][method]["episodes"]
        b = second["methods"][method]["episodes"]
        assert [r["policy_seed"] for r in a] == [r["policy_seed"] for r in b]
        assert [r["actions"] for r in a] == [r["actions"] for r in b]


def test_evaluation_rejects_actor_that_mutates_its_weights():
    class BadActor(RandomLegal):
        def __init__(self, seed):
            super().__init__(seed)
            self.version = 0

        def weights_fingerprint(self):
            return str(self.version)

        def act(self, observation, deterministic=True):
            self.version += 1
            return 0

    with pytest.raises(RuntimeError, match="changed weights"):
        evaluate_methods({"adaptive": BadActor}, episodes=1, seed=5,
                         env_factory=OneStepEnv, bootstrap_samples=5)


def test_projection_selects_nearest_legal_same_type_never_substitutes():
    obs = public_observation()
    index, distance = nearest_same_sensor(obs, "rf", [0, 99])
    assert index == 0 and distance == 1
    obs["action_mask"][:2] = False
    index, distance = nearest_same_sensor(obs, "rf", [0, 99])
    assert index == len(obs["options"]) - 1 and distance is None
    index, _ = nearest_same_sensor(obs, "radar", [0, -99])
    assert index == 3


def test_fixed_layout_is_not_scenario_conditioned():
    obs = public_observation()
    a = UniformFixedRFAndRadar()
    b = UniformFixedRFAndRadar()
    changed = deepcopy(obs)
    changed["state"]["weather"] = {"visibility": .01}
    changed["state"]["tracks"] = [{"east": -500, "north": 500}]
    assert [a.act(obs) for _ in range(3)] == [b.act(changed) for _ in range(3)] == [0, 3, 10]


def test_fixed_layout_skips_unavailable_rf_but_still_uses_planned_radar():
    obs = public_observation()
    obs["action_mask"][:2] = False
    policy = UniformFixedRFAndRadar()
    assert policy.act(obs) == 3
    assert policy.projections[0]["skipped_unavailable"]


def test_greedy_uses_public_marginal_features_and_honors_mask():
    obs = public_observation()
    obs["option_features"][1, 0] = .8
    obs["option_features"][2, 0] = .9
    obs["action_mask"][2] = False
    assert GreedyPublicCoverage().act(obs) == 1
    obs["option_features"][:, 2] = 1
    obs["option_features"][:, 0] = 0
    assert GreedyPublicCoverage().act(obs) == 10


def test_preserved_checkpoint_loaded_and_projected_with_original_contract():
    actor = LegacyToy210(CHECKPOINT)
    obs = public_observation()
    features = actor.features(obs)
    assert features.option_features.shape == (6, 114)
    assert actor.policy.parameter_hash() == LEGACY_PARAMETER_SHA256
    action = actor.act(obs)
    assert obs["action_mask"][action]
    assert actor.metadata["projection"]["lossy"] is True
    assert actor.policy.parameter_hash() == LEGACY_PARAMETER_SHA256


def test_legacy_projection_cannot_consume_new_weather_tracks_or_option_features():
    class Explodes:
        def __array__(self, *args, **kwargs):
            raise AssertionError("New option features must not be read by old policy")

    actor = LegacyToy210(CHECKPOINT)
    obs = public_observation()
    original = actor.features(obs).option_features.copy()
    obs["option_features"] = Explodes()
    obs["state"]["weather"] = {"truth": object()}
    obs["state"]["tracks"] = object()
    obs["state"]["coverage"] = object()
    np.testing.assert_array_equal(original, actor.features(obs).option_features)


def test_scenario_hash_stable_for_numpy_values():
    assert canonical_hash({"x": np.array([1, 2])}) == canonical_hash({"x": [1, 2]})


def test_actual_env_benchmark_smoke():
    pytest.importorskip("triad_rl.adaptive_env")
    report = evaluate_methods({"adaptive": RandomLegal,
                               "legacy_toy210_projected": lambda seed: LegacyToy210(CHECKPOINT, seed=seed)},
                              episodes=2, seed=2_000_000_000_000_000,
                              bootstrap_samples=20)
    assert report["methods"]["legacy_toy210_projected"]["summary"]["episodes"] == 2
    assert report["methods"]["adaptive"]["episodes"][0]["replay"]


def load_evaluator_cli():
    path = Path(__file__).resolve().parents[1] / "evaluate_adaptive.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_adaptive_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_selection_provenance_merges_all_candidates_and_binds_hash(tmp_path):
    cli = load_evaluator_cli()
    path = tmp_path / "selection.json"
    selection = {"schema": "triad.adaptive_model_selection.v1",
                 "selected": {"weights_sha256": "verified-weights"},
                 "seed_provenance": {"selection_validation": {"start": 500, "count": 300},
                                     "training": [{"start": 100, "count": 10},
                                                  {"start": 200, "count": 10}]}}
    path.write_text(json.dumps(selection), encoding="utf-8")
    provenance, evidence = cli.merge_selection_provenance(
        {"training": {"start": 100, "count": 10}}, path, "verified-weights")
    assert provenance["training"] == selection["seed_provenance"]["training"]
    assert evidence["selected"]["weights_sha256"] == "verified-weights"
    assert len(evidence["sha256"]) == 64
    for seed in (105, 205, 505):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint_seeds(seed, 1, provenance)
    with pytest.raises(ValueError, match="weights"):
        cli.merge_selection_provenance({}, path, "different-weights")


def test_cli_rejects_actual_checkpoint_selection_validation_overlap(tmp_path):
    from triad_rl.adaptive_inputs import FEATURE_NAMES
    from triad_rl.adaptive_policy import AdaptivePolicy
    cli = load_evaluator_cli()
    checkpoint = tmp_path / "policy"
    policy = AdaptivePolicy(FEATURE_NAMES)
    policy.save(checkpoint, training_state={"seed_provenance": {
        "training": {"start": 100, "count": 20}, "validation": {"start": 200, "count": 10}}})
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"schema": "triad.adaptive_model_selection.v1",
                                    "selected": {"weights_sha256": policy.weights_fingerprint()},
                                    "seed_provenance": {"selection_validation": {"start": 500, "count": 10}}}), encoding="utf-8")
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="overlap selection_validation"):
        cli.main(["--checkpoint", str(checkpoint), "--selection-report", str(selection),
                  "--seed", "505", "--episodes", "1", "--output", str(output)])
    assert not output.exists()


def test_selection_report_requires_actual_validation_range(tmp_path):
    cli = load_evaluator_cli()
    path = tmp_path / "selection.json"
    for used in ({}, {"selection_validation": {"start": 0, "count": -1}},
                {"selection_validation": {"start": 0, "count": 0}}):
        path.write_text(json.dumps({"schema": "triad.adaptive_model_selection.v1",
                                    "selected": {"weights_sha256": "hash"},
                                    "seed_provenance": used}), encoding="utf-8")
        with pytest.raises(ValueError):
            cli.merge_selection_provenance({}, path, "hash")


def test_common_selection_report_validation_alias_is_reserved(tmp_path):
    cli = load_evaluator_cli()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"schema": "triad.adaptive_model_selection.v1",
                                "selection_split": "validation",
                                "selected": {"weights_sha256": "hash"},
                                "seed_provenance": {"validation": {"start": 500, "count": 300},
                                                    "candidate_validation": [{"start": 100, "count": 10},
                                                                             {"start": 200, "count": 10}]}}), encoding="utf-8")
    provenance, _ = cli.merge_selection_provenance({}, path, "hash")
    assert provenance["selection_validation"] == {"start": 500, "count": 300}
    for seed in (105, 205, 505):
        with pytest.raises(ValueError, match="overlap"):
            assert_disjoint_seeds(seed, 1, provenance)
