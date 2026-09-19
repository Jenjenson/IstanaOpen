import json
import pytest
from model_switch_demo import load_layouts, MODELS


def write_cases(tmp_path, different_seed=False):
    for i, (_, _, checkpoint) in enumerate(MODELS):
        (tmp_path / f"evaluation-{checkpoint}.json").write_text(json.dumps([
            {"seed": 8 + (i if different_seed else 0),
             "placements": [{"profileId": "eo", "siteId": i}],
             "metrics": {"must_not_reach_ui": 123}}]))


def test_preserves_saved_layouts_without_metrics(tmp_path):
    write_cases(tmp_path)
    layouts = load_layouts(tmp_path)
    assert list(layouts) == ["rl", "greedy", "initial"]
    for i, row in enumerate(layouts.values()):
        assert row["placements"] == [{"profileId": "eo", "siteId": i}]
        assert "metrics" not in row
        assert len(row["source_sha256"]) == 64


def test_requires_matched_seed(tmp_path):
    write_cases(tmp_path, different_seed=True)
    with pytest.raises(ValueError, match="same scenario seed"):
        load_layouts(tmp_path)
