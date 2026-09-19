import pytest

pytest.importorskip("PIL")
pytest.importorskip("imageio_ffmpeg")
from render_warning_timelapse import interpolate_drones


def test_replay_interpolation_preserves_recorded_endpoints():
    def frame(t, x):
        return {"t": t, "drones": [{"droneId": 8, "positionCm": {"x": x, "y": 50, "z": 100}}]}
    frames = [frame(0, 0), frame(.5, 100), frame(1, 200)]
    assert interpolate_drones(frames, 0)[0]["p"]["x"] == 0
    assert interpolate_drones(frames, .25)[0]["p"]["x"] == 50
    assert interpolate_drones(frames, .5)[0]["p"]["x"] == 100
    assert interpolate_drones(frames, 1)[0]["p"]["x"] == 200
    assert interpolate_drones(frames, 10)[0]["p"]["x"] == 200
