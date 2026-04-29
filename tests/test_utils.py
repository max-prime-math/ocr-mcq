"""Tests for utils.py."""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import materialise_figures


def test_materialise_figures_refines_full_page_box(tmp_path):
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((320, 420, 700, 860), fill="black")

    figures = [
        {
            "page": 1,
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
            "caption": "Figure",
        }
    ]

    saved = materialise_figures(figures, [image], str(tmp_path), "sample")
    assert len(saved) == 1

    out_path = tmp_path / "sample_fig_1.png"
    assert out_path.exists()

    cropped = Image.open(out_path)
    assert cropped.size[0] < image.size[0]
    assert cropped.size[1] < image.size[1]
    assert cropped.size[0] > 300
    assert cropped.size[1] > 350
