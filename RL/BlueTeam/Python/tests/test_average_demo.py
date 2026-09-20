from PIL import Image
from render_average_demo import compose


def test_average_band_does_not_cover_scene():
    scene = Image.new("RGB", (1920,1080), "#234567")
    for section in ("early", "final", "summary"):
        result = compose(scene, section, 36.41547539889028, 41.16547539889028, 16, fresh=True)
        assert result.size == (1920,1280)
        assert result.crop((0,200,1920,1280)).tobytes() == scene.tobytes()
        assert len(result.crop((0,0,1920,200)).getcolors(100000)) > 1
