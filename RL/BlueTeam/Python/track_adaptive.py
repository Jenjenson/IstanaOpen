"""Import portable training events into an optional local Trackio dashboard.

The training process does not depend on Trackio. Import completed JSONL runs
afterward so dashboard dependencies cannot change training or checkpoint RNG.
No Hugging Face Space is created and no cloud sync is requested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def numeric_events(path: Path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        event = json.loads(line)
        kind = event.get("event")
        if kind not in ("validation", "training_batch"):
            continue
        prefix = "validation" if kind == "validation" else "training"
        metrics = {f"{prefix}/{key}": value for key, value in event.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and key not in ("episode", "episode_start")}
        metrics["episode"] = event["episode"]
        # Strict finite output, with an actionable source location on failure.
        try:
            json.dumps(metrics, allow_nan=False)
        except ValueError as error:
            raise ValueError(f"Nonfinite metric at {path.name}:{line_number}") from error
        yield metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="blue-team-adaptive")
    parser.add_argument("--name", required=True, help="Use a distinct import name to avoid duplicate dashboard runs")
    args = parser.parse_args()
    try:
        import trackio
    except ImportError:
        parser.error("Optional dashboard requires: python -m pip install trackio")
    events = list(numeric_events(args.run_dir / "training.jsonl"))
    if not events:
        parser.error("No training or validation events found")
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    trackio.init(project=args.project, name=args.name, config=config, space_id=None)
    try:
        for event in events:
            trackio.log(event)
    finally:
        trackio.finish()
    print(f"Imported {len(events)} local events; dashboard: trackio show --project {args.project}")


if __name__ == "__main__":
    main()
