"""Local tracking imports must retain both old and per-profile metrics."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from track_adaptive import numeric_events


def test_import_retains_flat_and_nested_metrics_without_boolean_flags(tmp_path):
    path = tmp_path / "training.jsonl"
    events = [
        {"event": "resume", "episode": 16},
        {"event": "training_batch", "episode": 32, "episode_start": 16,
         "loss": 1.25, "updates": 2},
        {"event": "validation", "episode": 32, "selected": True,
         "mean_return": 2.0,
         "profiles": {"normal": {"success_rate": .7, "mean_cost": 2.3},
                      "stress": {"success_rate": .05, "mean_cost": 1.2}}},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    training, validation = list(numeric_events(path))
    assert training == {"episode": 32, "training/loss": 1.25, "training/updates": 2}
    assert validation["validation/profiles/normal/success_rate"] == .7
    assert validation["validation/profiles/stress/mean_cost"] == 1.2
    assert validation["validation/mean_return"] == 2.0
    assert "validation/selected" not in validation


def test_nested_nonfinite_metric_is_rejected_with_source_line(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps({"event": "validation", "episode": 0,
                               "profiles": {"stress": {"cost": float("nan")}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="training.jsonl:1"):
        list(numeric_events(path))
