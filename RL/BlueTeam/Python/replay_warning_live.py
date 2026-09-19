"""Show a saved warning-training checkpoint in the real Unreal 3D window.

Start Tools/start_blue_live.ps1 first and disconnect the browser console.
This launches no devices; it places only synthetic sensors in local Unreal.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from triad_rl.istana_live import IstanaLiveClient
from triad_rl.warning_policy import WarningPolicy, warning_metrics
from train_warning_live import red_centers
from triad_rl.warning_scenario import approach_centers, scenario_contract, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--red-seed", type=int)
    parser.add_argument("--action-seed", type=int)
    args = parser.parse_args()
    protocol_path = args.checkpoint.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    approach = protocol.get("schema") == "istana.native_warning_approach.v2"
    if args.red_seed is None: args.red_seed = protocol.get("evaluation_seeds", [2700000])[0]
    if args.action_seed is None: args.action_seed = protocol.get("evaluation_action_seeds", [3700000])[0]
    with IstanaLiveClient(port=args.port) as client:
        red = client.reset(args.red_seed)
        context = client.get_blue_context()
        if approach:
            frozen = json.loads((args.checkpoint.parent / "physical-contract.json").read_text())
            if digest(scenario_contract(red, context)) != digest(frozen):
                raise ValueError("Use the matching -WarningApproachV2 scene; replay contract differs")
        policy = WarningPolicy.load(args.checkpoint, context)
        placements, _ = policy.plan(context, rng=np.random.default_rng(args.action_seed))
        client.deploy(placements)
        client.place_red((approach_centers if approach else red_centers)(red, args.red_seed))
        print(json.dumps({"placements": placements, "actor_updates": policy.updates}), flush=True)
        start = time.monotonic()
        for _ in range(800):
            blue = client.step(10)["blueObservation"]
            remaining = blue["elapsedSeconds"] - (time.monotonic() - start)
            if remaining > 0: time.sleep(min(remaining, 1.))
            if blue["terminated"] or blue["truncated"]:
                print(json.dumps(warning_metrics(blue), indent=2))
                return
        client.cancel()
        raise RuntimeError("Replay exceeded step bound")


if __name__ == "__main__": main()
