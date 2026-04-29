"""Tests for utils.py."""

import tempfile
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import build_zip_bundle, materialise_figures


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


def test_materialise_figures_drops_caption_and_choices_below_graph(tmp_path):
    image = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(image)

    # Main figure: a large graph near the top-middle.
    draw.rectangle((220, 180, 760, 620), outline="black", width=10)
    draw.line((240, 580, 720, 220), fill="black", width=8)

    # Nearby axis labels should remain with the figure.
    draw.rectangle((180, 570, 205, 600), fill="black")
    draw.rectangle((735, 630, 765, 655), fill="black")

    # Caption/stem text below the figure should be excluded.
    for row in range(5):
        y = 760 + row * 28
        for col in range(18):
            x = 80 + col * 46
            draw.rectangle((x, y, x + 22, y + 12), fill="black")

    # Answer choice text lower on the page should also be excluded.
    for row in range(2):
        y = 1080 + row * 40
        for col in range(10):
            x = 120 + col * 70
            draw.rectangle((x, y, x + 26, y + 14), fill="black")

    figures = [{"page": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}]
    saved = materialise_figures(figures, [image], str(tmp_path), "sample")
    assert len(saved) == 1

    cropped = Image.open(tmp_path / "sample_fig_1.png")
    # The graph plus local labels should remain.
    assert cropped.size[0] >= 560
    assert cropped.size[1] >= 470
    # The crop should stop well before the caption/choices block.
    assert cropped.size[1] < 620


def test_build_zip_bundle_without_figures_dir():
    bundle = build_zip_bundle(r"\documentclass{article}", None)
    assert bundle


def test_build_zip_bundle_includes_extra_files_and_manifest(tmp_path):
    extra = tmp_path / "sample.pdf"
    extra.write_bytes(b"pdf-bytes")
    bundle = build_zip_bundle(
        r"\documentclass{article}",
        None,
        extra_files=[(str(extra), "tricky_pdfs/sample.pdf")],
        extra_text_files=[("tricky_pdfs/manifest.txt", "sample.pdf: reason")],
    )

    archive = Path(tempfile.mkstemp(suffix=".zip")[1])
    archive.write_bytes(bundle)
    with zipfile.ZipFile(archive) as zf:
        assert "output.tex" in zf.namelist()
        assert "tricky_pdfs/sample.pdf" in zf.namelist()
        assert zf.read("tricky_pdfs/manifest.txt").decode("utf-8") == "sample.pdf: reason"
