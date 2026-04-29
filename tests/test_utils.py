"""Tests for utils.py."""

import tempfile
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import _FITZ_AVAILABLE, build_zip_bundle, materialise_figures, render_page_to_image


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


def test_materialise_figures_trims_dense_caption_band_below_figure(tmp_path):
    image = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(image)

    # Figure content.
    draw.line((120, 430, 790, 430), fill="black", width=10)
    draw.line((260, 70, 260, 640), fill="black", width=10)
    draw.arc((150, 150, 720, 560), start=200, end=340, fill="black", width=10)
    draw.arc((190, 170, 680, 520), start=200, end=340, fill="black", width=10)
    for x in range(180, 760, 80):
        draw.line((x, 418, x, 442), fill="black", width=4)
    for y in range(140, 620, 90):
        draw.line((248, y, 272, y), fill="black", width=4)

    # Light labels near the axis should remain.
    draw.rectangle((730, 398, 760, 438), fill="black")
    draw.rectangle((236, 44, 274, 84), fill="black")

    # Dense caption line close below the figure should be removed.
    for col in range(24):
        x = 20 + col * 34
        draw.rectangle((x, 760, x + 18, 778), fill="black")
        draw.rectangle((x + 20, 760, x + 28, 778), fill="black")

    figures = [{"page": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}]
    saved = materialise_figures(figures, [image], str(tmp_path), "sample")
    assert len(saved) == 1

    cropped = Image.open(tmp_path / "sample_fig_1.png")
    assert cropped.size[1] < 700


def test_materialise_figures_prefers_embedded_pdf_image_objects(tmp_path):
    if not _FITZ_AVAILABLE:
        return

    import fitz

    pdf_path = tmp_path / "embedded.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)

    figure = Image.new("RGB", (180, 110), "white")
    draw = ImageDraw.Draw(figure)
    draw.rectangle((12, 12, 168, 98), outline="black", width=6)
    draw.line((25, 90, 155, 24), fill="black", width=5)
    figure_bytes = BytesIO()
    figure.save(figure_bytes, format="PNG")
    page.insert_image(fitz.Rect(70, 35, 230, 145), stream=figure_bytes.getvalue())
    page.insert_textbox(
        fitz.Rect(30, 165, 270, 260),
        "Caption text and answer choices should stay out of the extracted figure crop.",
        fontsize=16,
    )
    doc.save(pdf_path)
    doc.close()

    page_image = render_page_to_image(str(pdf_path), 0, dpi=200)
    figures = [{"page": 1, "x": 0.0, "y": 0.0, "width": 1.0, "height": 0.8}]
    saved = materialise_figures(
        figures,
        [page_image],
        str(tmp_path),
        "embedded",
        pdf_path=str(pdf_path),
        page_numbers=[0],
    )

    assert len(saved) == 1
    cropped = Image.open(tmp_path / "embedded_fig_1.png")
    doc = fitz.open(pdf_path)
    block = next(block for block in doc[0].get_text("dict")["blocks"] if block.get("type") == 1)
    x0, y0, x1, y1 = block["bbox"]
    doc.close()
    scale_x = page_image.size[0] / 300.0
    scale_y = page_image.size[1] / 400.0
    expected_width = round((x1 - x0) * scale_x)
    expected_height = round((y1 - y0) * scale_y)
    assert abs(cropped.size[0] - expected_width) <= 8
    assert abs(cropped.size[1] - expected_height) <= 8


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
