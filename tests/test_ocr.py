"""Tests for ocr.py — mocks the Anthropic client to avoid real API calls."""

import json
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr import extract_page, should_retry_with_next_page
from cache import MathpixCache as VisionCache


def _make_client(answer="B", question="What is 1+1?"):
    """Return a mock Anthropic client whose messages.create returns structured JSON."""
    payload = {
        "question": question,
        "choices": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "correct_answer": answer,
        "pages_used": 1,
        "solution": None,
        "figures": [],
    }
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0

    response = MagicMock()
    response.content = [block]
    response.usage = usage

    client = MagicMock()
    client.messages.create.return_value = response
    return client, payload


def _tmp_png() -> str:
    """Create a tiny valid-ish PNG temp file."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    # Write a minimal 1×1 white PNG
    import struct, zlib
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(png)
    return path


def test_extract_page_returns_structured_data(tmp_path):
    client, expected = _make_client(answer="C")
    img = _tmp_png()
    try:
        result = extract_page(img, client=client)
    finally:
        Path(img).unlink(missing_ok=True)

    assert result["question"] == expected["question"]
    assert result["correct_answer"] == "C"
    assert set(result["choices"].keys()) == {"A", "B", "C", "D", "E"}


def test_extract_page_uses_cache(tmp_path):
    client, _ = _make_client()
    cache = VisionCache(str(tmp_path))
    img = _tmp_png()
    try:
        # First call — hits API and writes cache.
        extract_page(img, client=client, cache=cache)
        # Second call — should read from cache, not call API again.
        extract_page(img, client=client, cache=cache)
    finally:
        Path(img).unlink(missing_ok=True)

    assert client.messages.create.call_count == 1


def test_extract_page_force_bypasses_cache(tmp_path):
    client, _ = _make_client()
    cache = VisionCache(str(tmp_path))
    img = _tmp_png()
    try:
        extract_page(img, client=client, cache=cache)
        extract_page(img, client=client, cache=cache, force=True)
    finally:
        Path(img).unlink(missing_ok=True)

    assert client.messages.create.call_count == 2


def test_extract_page_null_answer(tmp_path):
    client, _ = _make_client(answer=None)
    img = _tmp_png()
    try:
        result = extract_page(img, client=client)
    finally:
        Path(img).unlink(missing_ok=True)

    assert result["correct_answer"] is None


def test_extract_page_supports_second_image(tmp_path):
    client, _ = _make_client()
    img1 = _tmp_png()
    img2 = _tmp_png()
    try:
        result = extract_page(img1, client=client, second_image_path=img2)
    finally:
        Path(img1).unlink(missing_ok=True)
        Path(img2).unlink(missing_ok=True)

    assert result["pages_used"] == 1


def test_retry_with_next_page_when_choices_are_incomplete():
    result = {
        "question": "Large stem",
        "choices": {"A": "1", "B": "2"},
        "correct_answer": None,
    }
    assert should_retry_with_next_page(result) is True


def test_do_not_retry_when_single_page_result_is_complete():
    result = {
        "question": "What is 1+1?",
        "choices": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"},
        "correct_answer": "B",
    }
    assert should_retry_with_next_page(result) is False
