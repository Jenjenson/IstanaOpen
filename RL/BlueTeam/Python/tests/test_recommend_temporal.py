"""Handcrafted/public fixture parity; no new scenario generation."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import recommend_temporal
from triad_rl.adaptive_inputs import LiveObservationAdapter
from triad_rl.temporal_env import TemporalPlacementEnv
from triad_rl.temporal_policy import TemporalPolicy
from test_temporal_env import scenario, forbidden
from test_temporal_inputs import catalogue, config


@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("changed_sensor", [False, True])
def test_external_public_plan_matches_exact_supplied_simulation(monkeypatch, learned, changed_sensor):
    import triad_rl.adaptive_env as frozen
    import triad_rl.robust_scenarios as robust
    monkeypatch.setattr(frozen, "generate_scenario", forbidden)
    monkeypatch.setattr(robust, "make_case", forbidden)
    case, sensors, rules = scenario(), catalogue(), config()
    if changed_sensor:
        sensors = sensors[::-1]
        sensors[0]["cost"] *= .7
    env = TemporalPlacementEnv(scenario=case, catalogue=sensors, config=rules)
    policy = TemporalPolicy(rules, seed=499)
    if learned:
        policy.parameters["wa"][:] = np.linspace(-.05, .05, policy.hidden_size)
    before = deepcopy((env.public_state, sensors))
    rng = policy.rng_fingerprint()
    plan = recommend_temporal.recommend_layout(policy, env.public_state, config=rules, catalogue=sensors)
    assert (env.public_state, sensors) == before
    expected = []
    while not env.done:
        observation = env.observe()
        action = policy.act(observation)
        expected.append(LiveObservationAdapter.recommendation(observation, action))
        env.step(action)
    assert plan["decisions"][:len(expected)] == expected
    assert all(row["stop"] for row in plan["decisions"][len(expected):])
    assert plan["new_placements"] == [row for row in expected if not row["stop"]]
    assert plan["physical_commands_sent"] is False
    assert sensors == before[1]
    assert policy.rng_fingerprint() == rng


def test_cli_schema_explicit_config_and_no_overwrite(tmp_path):
    rules = config()
    policy = TemporalPolicy(rules, seed=499)
    policy.save(tmp_path / "checkpoint")
    for name, value in (("config", asdict(rules)), ("input", scenario()["public"]), ("catalogue", catalogue())):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    output = tmp_path / "plan.json"
    args = ["--checkpoint", str(tmp_path / "checkpoint"), "--output", str(output)]
    for name in ("config", "input", "catalogue"):
        args += ["--" + name, str(tmp_path / f"{name}.json")]
    assert recommend_temporal.main(args) == 0
    raw = output.read_bytes()
    assert json.loads(raw)["checkpoint_weights_sha256"] == policy.weights_fingerprint()
    with pytest.raises(ValueError, match="never overwritten"):
        recommend_temporal.main(args)
    assert raw == output.read_bytes()
    with pytest.raises(ValueError, match="configuration"):
        recommend_temporal.recommend_layout(policy, scenario()["public"], config=replace(rules, lead_time_s=5))


def test_done_snapshot_is_noop():
    state = scenario()["public"]
    state["done"] = True
    plan = recommend_temporal.recommend_layout(TemporalPolicy(config()), state, config=config(), catalogue=catalogue())
    assert plan["decisions"] == [] and plan["final_public_state"] == state


def test_json_snapshot_has_identical_plan():
    state, rules = scenario()["public"], config()
    policy = TemporalPolicy(rules)
    expected = recommend_temporal.recommend_layout(policy, state, config=rules, catalogue=catalogue())
    assert recommend_temporal.recommend_layout(policy, json.dumps(state), config=rules, catalogue=catalogue()) == expected


def test_entrypoint_imports_no_simulator_or_training():
    result = subprocess.run([sys.executable, "-c", "import recommend_temporal,sys; "
        "assert not any(n in sys.modules for n in ('triad_rl.adaptive_env','triad_rl.robust_scenarios',"
        "'triad_rl.temporal_env','train_temporal','evaluate_temporal'))"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
