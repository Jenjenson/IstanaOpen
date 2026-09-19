from render_warning_raw import timeline, report_html, episode_footer, episode_header
from PIL import Image
import pytest


def test_raw_timeline_preserves_checkpoint_order_and_clock():
    manifest = {"gallery": [{"profile": "eo", "path": "test"}],
                "recordings": [{"checkpoint": n, "frames": [{"elapsed_s": t} for t in (0.,1.,2.)]}
                               for n in (0,1,64,128)]}
    segments = timeline(manifest)
    assert segments[0]["frames"] == 40
    assert [s["episode"] for s in segments if s["simulated_s"] == 0.] == [0,1,64,128]
    assert all(b["start_frame"] == a["start_frame"]+a["frames"] for a,b in zip(segments,segments[1:]))
    assert all(s["frames"] > 0 for s in segments)
    with pytest.raises(ValueError): timeline(manifest, speed=0)


def test_report_metrics_outside_video_and_safe_embedded_data():
    page = report_html({"user_text": "</script><script>bad()</script>"})
    assert '<video id="film"' in page and '<aside>' in page
    assert page.index('</video>') < page.index('<aside>')
    assert 'burned' not in page  # no burnt-in overlay rendering in report
    assert '\\u003c/script>' in page
    assert 'Hide statistics / enlarge footage' in page
    assert 'position:absolute' not in page and '<canvas' not in page


def test_legacy_markers_are_disclosed_without_claiming_removal():
    page = report_html({"observer_markers": True})
    assert "Original in-scene marker boxes remain" in page
    assert "No boxes, labels or charts" not in page
    assert "full-frame MP4" in page
    clean = report_html({"observer_markers": False})
    assert "No boxes, labels or charts" in clean


def test_episode_footer_preserves_every_scene_pixel():
    scene = Image.new("RGB", (1920,1080), "#334455")
    result = episode_footer(scene, {"episode": 32},
                            [{"checkpoint": 32, "metrics": {"team_warning_s": 12.33}}], 128)
    assert result.size == (1920,1152)
    assert result.crop((0,0,1920,1080)).tobytes() == scene.tobytes()
    page = report_html({"observer_markers": False, "episode_footer": True})
    assert "aspect-ratio:5/3" in page
    assert "narrow video footer" in page


def test_stats_header_is_embedded_above_unchanged_scene():
    scene = Image.new("RGB", (1920,1080), "#334455")
    record = {"checkpoint": 32, "metrics": {"team_warning_s": 12.33, "first_detection_s": 1, "first_arrival_s": 13.33}}
    result = episode_header(scene, {"episode": 32, "simulated_s": 5, "view": "drone"}, [record], 128)
    assert result.size == (1920,1200)
    assert result.crop((0,120,1920,1200)).tobytes() == scene.tobytes()
    assert len(result.crop((0,0,1920,120)).getcolors(100000)) > 1
    page = report_html({"observer_markers": False, "episode_header": True})
    assert 'aspect-ratio:8/5' in page
    assert 'class="layout clean"' in page
    assert 'embedded at the top of the video itself' in page


def test_original_pilot_is_not_presented_as_later_run():
    page = report_html({"observer_markers": False, "summary": {"protocol": {"episodes": 128}}})
    assert "Original 128-episode pilot" in page
    assert "not the later 256-episode run" in page
    assert "Case warning in the video and the mean across test cases are different" in page
