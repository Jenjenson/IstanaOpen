"""Offline replay presentation uses recorded evidence and safely embeds JSON."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo_adaptive import export_html


def test_replay_embeds_json_without_script_injection(tmp_path):
    target = tmp_path / "demo.html"
    replay = {"frames": [{"time": 0, "threats": []}],
              "label": '</script><script>alert("x")</script>&'}
    export_html([replay], target, checkpoint="test")
    html = target.read_text(encoding="utf-8")
    payload = html.split('<script id="replay-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)["replays"] == [replay]
    assert "<script>alert" not in html
    assert "__REPLAY_DATA__" not in html
    assert 'aria-label="Replay frame"' in html
    assert 'id="placements"' in html
    assert 'id="components"' in html
    assert "fetch(" not in html


def test_empty_demo_fails_instead_of_publishing_blank_replay(tmp_path):
    with pytest.raises(ValueError, match="frames"):
        export_html([], tmp_path / "demo.html")


def test_demo_refuses_nonfinite_metrics(tmp_path):
    with pytest.raises(ValueError):
        export_html([{"frames": [{"time": 0}], "reward": float("nan")}], tmp_path / "demo.html")
