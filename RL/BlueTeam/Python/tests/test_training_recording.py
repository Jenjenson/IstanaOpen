"""Recording compatibility and truth-preserving native replay contracts."""
from copy import deepcopy
import hashlib
import json

import pytest

from triad_rl.training_recording import RECORDING_SCHEMA, load_recordings, parse_episodes, frame_drones


def recorded_run():
    context = {
        "worldOriginCm": {"x": 100, "y": 200, "z": 300},
        "publicSnapshot": {"sites": [[10, 0]], "blocked_sites": [], "budget": 8,
                           "weather": {"rain": 0, "visibility": 1}, "tracks": []},
        "catalogue": [{"id": "thermal"}], "siteSurfacesWorldCm": [[1100, 200, 300]],
        "temporalConfig": {"objective_radius_m": 20}, "placementRule": "supported",
        "sensorModel": "directional",
    }
    return {
        "seed": 1700000, "placements": [{"profileId": "thermal", "siteId": 0, "yawDeg": 0, "pitchDeg": 20}],
        "metrics": {"mean_drone_warning_s": 5., "team_warning_s": 5., "cost": 1., "targets": 1,
                    "detected_fraction": 1., "first_detection_s": 10., "first_arrival_s": 15.},
        "warning_evidence": [{"droneId": 0, "firstDetectionSeconds": 10., "zoneEntrySeconds": 15.}],
        "context": context, "red_context": {"runId": "old", "memberSeed": 17, "groupCount": 1},
        "red_decision": {"centers": [[28000., 0., 12000.]]},
        "frames": [{"time": 0., "threats": [{"id": "drone-0", "position": [100, 0, 120]}]},
                   {"time": 15., "threats": [{"id": "drone-0", "position": [10, 0, 120]}]}],
    }


def write_manifest(folder, algorithm="ppo"):
    run = recorded_run()
    (folder / "episodes").mkdir()
    for number in (1, 500):
        (folder / "episodes" / f"episode-{number:06d}.json").write_text(
            json.dumps({"episode": number, "run": run}), encoding="utf-8")
    manifest = {"schema": RECORDING_SCHEMA, "configuration": {"algorithm": algorithm, "episodes": 1500},
                "entries": [{"checkpoint": 0, "kind": "baseline", "label": "Contractor baseline", "run": {"run": run}},
                            {"checkpoint": 500, "kind": "best", "label": "Selected best", "run": "episodes/episode-000500.json"}]}
    (folder / "recording-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("algorithm", ["reinforce", "ppo", "a2c", "local_ppo"])
def test_console_recording_reads_exact_actions_without_policy_files(tmp_path, algorithm):
    write_manifest(tmp_path, algorithm)
    metadata, entries = load_recordings(tmp_path)
    assert metadata["configuration"]["algorithm"] == algorithm
    assert entries[0]["kind"] == "baseline"
    assert entries[1]["run"]["placements"] == recorded_run()["placements"]
    assert load_recordings(tmp_path, best_only=True)[1][0]["label"] == "Selected best"
    assert [entry["checkpoint"] for entry in load_recordings(tmp_path, episodes="all")[1]] == [1, 500]
    assert load_recordings(tmp_path, episodes=[500])[1][0]["kind"] == "training"


def test_loader_rejects_missing_episode_and_paths_outside_run(tmp_path):
    manifest = write_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_recordings(tmp_path, episodes=[1500])
    manifest["entries"][0]["run"] = "../other-run.json"
    (tmp_path / "recording-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the run"):
        load_recordings(tmp_path)


def test_loader_does_not_pretend_old_console_logs_are_replayable(tmp_path):
    (tmp_path / "summary.json").write_text('{"schema":"istana.console_warning_training_summary.v1"}')
    with pytest.raises(ValueError, match="older console run"):
        load_recordings(tmp_path)


def test_active_log_snapshot_ignores_only_last_incomplete_row(tmp_path):
    write_manifest(tmp_path)
    (tmp_path / "training.jsonl").write_text('{"episode":1}\n{"episode":')
    assert load_recordings(tmp_path)[0]["training_history"] == [{"episode": 1}]
    (tmp_path / "training.jsonl").write_text('broken\n{"episode":1}\n')
    with pytest.raises(json.JSONDecodeError):
        load_recordings(tmp_path)
    (tmp_path / "training.jsonl").write_text('broken\n')
    with pytest.raises(json.JSONDecodeError):
        load_recordings(tmp_path)


def test_episode_selection_and_frame_coordinate_conversion():
    assert parse_episodes("500,1000,1500") == [500, 1000, 1500]
    for value in ("0", "1,1", "nope"):
        with pytest.raises(ValueError): parse_episodes(value)
    run = recorded_run()
    assert frame_drones(run, run["frames"][0]) == [
        {"droneId": 0, "positionCm": {"x": 10100, "y": 200, "z": 12300}}]


def test_native_replay_preserves_layout_order_and_recorded_red_centers():
    pytest.importorskip("PIL")
    from capture_warning_3d import prepare_replay
    run = recorded_run()
    class Client:
        def reset(self, seed):
            assert seed == run["seed"]
            return {**run["red_context"], "runId": "new"}
        def get_blue_context(self): return deepcopy(run["context"])
        def deploy(self, placements): self.placements = deepcopy(placements)
        def place_red(self, centers): self.centers = deepcopy(centers)
    client = Client()
    assert prepare_replay(client, run, {"schema": RECORDING_SCHEMA}) == run["placements"]
    assert client.placements == run["placements"]
    assert client.centers == run["red_decision"]["centers"]
    changed = deepcopy(run)
    changed["context"]["publicSnapshot"]["weather"]["rain"] = 1
    with pytest.raises(ValueError, match="physical/weather"):
        prepare_replay(client, changed, {"schema": RECORDING_SCHEMA})


def test_console_timelapse_uses_current_episode_schema(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("imageio_ffmpeg")
    from render_warning_timelapse import draw_training_frame
    write_manifest(tmp_path)
    (tmp_path / "training.jsonl").write_text(json.dumps(
        {"episode": 500, "meanWarningSeconds": 5, "detectedFraction": 1}) + "\n")
    metadata, entries = load_recordings(tmp_path)
    assert draw_training_frame(metadata, entries, 1, .5).size == (1440, 900)


def test_console_loader_accepts_backend_episode_pattern_and_prefers_final_test(tmp_path):
    manifest = write_manifest(tmp_path)
    manifest.update(episodeDirectory="episodes", episodePattern="episode-{episode:06d}.json")
    manifest["entries"].append({**manifest["entries"][-1], "kind": "best_test", "label": "Independent test"})
    (tmp_path / "recording-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_recordings(tmp_path, episodes=[500])[1][0]["run"]["seed"] == 1700000
    assert load_recordings(tmp_path, best_only=True)[1][0]["kind"] == "best_test"


def test_legacy_cli_recordings_still_load_without_policy_resampling(tmp_path):
    summary = {"protocol": {"checkpoints": [0, 500]}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for number in (0, 500):
        (tmp_path / f"evaluation-{number:04d}.json").write_text(json.dumps([recorded_run()]), encoding="utf-8")
    metadata, entries = load_recordings(tmp_path)
    assert metadata == summary
    assert [entry["checkpoint"] for entry in entries] == [0, 500]


def test_native_overlay_renders_console_metrics_without_legacy_summary(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    Image = pytest.importorskip("PIL.Image")
    from render_warning_3d import replay_frame
    path = tmp_path / "native.png"
    Image.new("RGB", (1920, 1080)).save(path)
    run = recorded_run()
    native = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "elapsed_s": 15.}
    record = {**run, "recorded_metrics": run["metrics"], "frames": [native],
              "kind": "best_test", "checkpoint": 500, "label": "Best policy · independent test"}
    summary = {"schema": RECORDING_SCHEMA, "configuration": {"algorithm": "ppo"}}
    assert replay_frame(native, record, summary, 0, final=True).size == (1920, 1080)


def test_new_gallery_excludes_omnidirectional_and_unselected_profiles():
    pytest.importorskip("PIL")
    from capture_warning_3d import gallery_profiles
    context = {"publicSnapshot": {"available_sensor_ids": ["thermal", "eo", "rf"]},
               "catalogue": [{"id": "thermal", "directional": True},
                             {"id": "eo", "directional": True},
                             {"id": "rf", "directional": False},
                             {"id": "unavailable", "directional": True}]}
    assert gallery_profiles(context, ["thermal"]) == ["thermal"]
    assert gallery_profiles(context) == ["thermal", "eo"]
    for selected in (["rf"], ["thermal", "rf"], ["unavailable"], []):
        with pytest.raises(ValueError, match="limited-FOV"):
            gallery_profiles(context, selected)


def test_refused_console_render_preserves_existing_movie_and_poster(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("imageio_ffmpeg")
    from render_warning_timelapse import render_console_run
    write_manifest(tmp_path)
    metadata, entries = load_recordings(tmp_path, best_only=True)
    movie = tmp_path / "warning-timelapse.mp4"
    poster = tmp_path / "timelapse-poster.png"
    movie.write_bytes(b"existing movie")
    poster.write_bytes(b"existing poster from original selection")
    with pytest.raises(FileExistsError):
        render_console_run(tmp_path, metadata, entries)
    assert movie.read_bytes() == b"existing movie"
    assert poster.read_bytes() == b"existing poster from original selection"
