"""Export a portable, offline replay of an adaptive policy on fresh scenarios."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from triad_rl.adaptive_env import AdaptivePlacementEnv
from triad_rl.adaptive_policy import AdaptivePolicy


def collect_replay(policy, *, seed: int, split: str = "heldout", scenario=None) -> dict:
    env = AdaptivePlacementEnv(seed=seed, split=split)
    observation = env.reset(seed=seed, scenario=scenario)
    decisions = []
    done = False
    while not done:
        action = policy.act(observation, deterministic=True)
        option = copy.deepcopy(observation["options"][action])
        observation, reward, done, info = env.step(action)
        decisions.append({"action": option, "reward": float(reward)})
        if len(decisions) > 64:
            raise RuntimeError("Replay exceeded the placement decision limit")
    return {
        "seed": seed, "split": split, "scenario": copy.deepcopy(env.scenario),
        "catalogue": copy.deepcopy(env.catalogue),
        "placements": copy.deepcopy(env.placements), "decisions": decisions,
        "metrics": {key: value for key, value in info.items() if key != "frames"},
        "frames": copy.deepcopy(info.get("frames", [])),
    }


def export_html(replays: list[dict], output: Path, *, checkpoint: str = "",
                weights_sha256: str = "") -> None:
    """Inline data is escaped against script termination; no network is needed."""
    if not replays or any(not replay["frames"] for replay in replays):
        raise ValueError("At least one replay with frames is required")
    template = Path(__file__).with_name("adaptive_demo.html").read_text(encoding="utf-8")
    payload = json.dumps({"schema": "triad.adaptive_demo.v1", "checkpoint": checkpoint,
                          "weights_sha256": weights_sha256,
                          "replays": replays}, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__REPLAY_DATA__", payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3_000_000_000)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--split", choices=("train", "heldout", "stress"), default="heldout")
    parser.add_argument("--scenario-json", type=Path,
                        help="One saved simulator scenario (also fixes the number of replays to one)")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("--episodes must be between 1 and 50")
    policy = AdaptivePolicy.load(args.checkpoint)
    scenario = json.loads(args.scenario_json.read_text(encoding="utf-8")) if args.scenario_json else None
    episodes = 1 if scenario is not None else args.episodes
    replays = [collect_replay(policy, seed=args.seed + index, split=args.split, scenario=scenario)
               for index in range(episodes)]
    export_html(replays, args.output, checkpoint=args.checkpoint.name,
                weights_sha256=policy.weights_fingerprint())
    print(f"Saved {episodes} recorded scenarios to {args.output}")


if __name__ == "__main__":
    main()
