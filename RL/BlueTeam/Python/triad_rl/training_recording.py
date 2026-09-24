"""Read saved training evidence without loading or sampling a policy.

Console recordings retain actual sampled placements, so replay is independent
of optimizer choice and later policy changes. Legacy CLI recordings remain
supported. Recorded results and newly captured measurements stay separate.
"""
from copy import deepcopy
import json
from pathlib import Path


RECORDING_SCHEMA = "istana.console_training_recording.v1"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_run(folder, value):
    if isinstance(value, str):
        path = (folder / value).resolve()
        if not path.is_relative_to(folder.resolve()) or path.suffix != ".json":
            raise ValueError("Recording paths must name JSON files inside the run directory")
        value = _json(path)
    if not isinstance(value, dict):
        raise ValueError("Recording entry has no saved run")
    run = deepcopy(value.get("run", value))
    for key in ("placements", "seed", "metrics", "warning_evidence", "context"):
        if key not in run:
            raise ValueError(f"Recording is missing {key}; older console runs cannot be reconstructed exactly")
    return run


def parse_episodes(value):
    if value is None or value == "all":
        return value
    try:
        episodes = [int(part.strip()) for part in value.split(",")]
    except (ValueError, AttributeError) as error:
        raise ValueError("Episodes must be 'all' or comma-separated positive episode numbers") from error
    if not episodes or any(number < 1 for number in episodes) or len(set(episodes)) != len(episodes):
        raise ValueError("Episode numbers must be positive and unique")
    return episodes


def load_recordings(folder, *, episodes=None, best_only=False):
    """Return (presentation metadata, entries with exact saved runs).

    Defaults to the contractor baseline, periodic checkpoint evaluations and
    selected final policy. ``episodes`` explicitly selects sampled training
    episodes; those different scenarios must not be called a paired gain.
    """
    folder = Path(folder)
    manifest_path = folder / "recording-manifest.json"
    if not manifest_path.exists():
        summary = _json(folder / "summary.json")
        if "protocol" not in summary:
            raise ValueError("This older console run has no recording manifest. Start a new run to retain every layout.")
        if episodes is not None or best_only:
            raise ValueError("Episode/best selection requires a console recording manifest")
        entries = []
        for number in summary["protocol"]["checkpoints"]:
            run = _json(folder / f"evaluation-{number:04d}.json")[0]
            entries.append({"checkpoint": number, "label": f"Episode {number}",
                            "kind": "checkpoint", "run": run})
        return summary, entries
    manifest = _json(manifest_path)
    if manifest.get("schema") != RECORDING_SCHEMA:
        raise ValueError("Unsupported training recording manifest")
    if episodes is not None and best_only:
        raise ValueError("Choose either training episodes or the selected best policy")
    selected = deepcopy(manifest["entries"])
    if best_only:
        kind = "best_test" if any(entry["kind"] == "best_test" for entry in selected) else "best"
        selected = [entry for entry in selected if entry["kind"] == kind]
    elif episodes is not None:
        pattern = manifest.get("episodePattern", "episodes/episode-{episode:06d}.json")
        if len(Path(pattern).parts) == 1:
            pattern = str(Path(manifest.get("episodeDirectory", "episodes")) / pattern)
        if episodes == "all":
            # Enumerate retained evidence, not requested episode count: stopped
            # runs may contain fewer episodes than the configuration requests.
            directory = (folder / manifest.get("episodeDirectory", "episodes")).resolve()
            if not directory.is_relative_to(folder.resolve()):
                raise ValueError("Episode directory must stay inside the run directory")
            episodes = sorted(_json(path)["episode"] for path in directory.glob("episode-*.json"))
        selected = [{"checkpoint": number, "label": f"Training episode {number}",
                     "kind": "training", "run": pattern.format(episode=number)}
                    for number in episodes]
    if not selected:
        raise ValueError("No retained recordings match this selection")
    if episodes is None:
        rank = {"baseline": 0, "initial": 1, "checkpoint": 2, "best": 3, "best_test": 4}
        selected.sort(key=lambda row: (rank.get(row["kind"], 2), row["checkpoint"]))
    for entry in selected:
        entry["run"] = _read_run(folder, entry["run"])
        entry.setdefault("label", f"Episode {entry['checkpoint']}")
        entry.setdefault("evaluation", entry["run"]["metrics"])
        entry.setdefault("evaluation_seeds", [entry["run"]["seed"]])
    metadata = {"schema": RECORDING_SCHEMA, "configuration": manifest["configuration"],
                "context": manifest.get("context", selected[0]["run"]["context"]),
                "entries": [{key: value for key, value in entry.items() if key != "run"}
                            for entry in selected], "training_history": []}
    log_path = folder / "training.jsonl"
    if log_path.exists():
        # Complete rows are flushed individually. Ignore an unfinished last row
        # only, making snapshots during an active run safe to read.
        lines = log_path.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or line.endswith("\n"):
                    raise
                continue
            metadata["training_history"].append(row)
    return metadata, selected


def evaluation_warning(entry):
    values = entry.get("evaluation", entry["run"]["metrics"])
    return values.get("mean_drone_warning_s", values.get("meanWarningSeconds"))


def frame_drones(run, frame):
    """Native positions in world cm for either legacy or current frame schema."""
    if "drones" in frame:
        return frame["drones"]
    origin = run["context"]["worldOriginCm"]
    return [{"droneId": int(row["id"].removeprefix("drone-")),
             "positionCm": {key: origin[key] + row["position"][index] * 100
                            for index, key in enumerate("xyz")}}
            for row in frame["threats"]]


def frame_time(frame):
    return frame.get("time", frame.get("t", 0.))
