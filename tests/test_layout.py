from types import SimpleNamespace

from PIL import Image

from blockpali.layout import write_page_blocks


def test_crop_margin_and_relative_paths(tmp_path):
    (tmp_path / "page_images").mkdir()
    (tmp_path / "block_images").mkdir()
    detector = lambda image: SimpleNamespace(boxes=[[20, 20, 80, 80]], class_names=["text"], scores=[0.9])
    result = write_page_blocks(Image.new("RGB", (100, 100)), "p0", "0", tmp_path, detector)
    block = result["blocks"][0]
    assert block["block_image_path"] == "block_images/0_0.jpg"
    assert block["label"] == "text"
    with Image.open(tmp_path / block["block_image_path"]) as image:
        assert image.size == (66, 66)


def test_no_detections_retains_page(tmp_path):
    (tmp_path / "page_images").mkdir()
    (tmp_path / "block_images").mkdir()
    detector = lambda image: SimpleNamespace(boxes=[], class_names=[], scores=[])
    result = write_page_blocks(Image.new("RGB", (100, 120)), "p0", "0", tmp_path, detector)
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["label"] == "full_page"
