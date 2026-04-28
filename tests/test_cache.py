"""Tests for cache.py."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cache import MathpixCache


def _write_tmp(content: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    import os; os.close(fd)
    with open(path, "wb") as fh:
        fh.write(content)
    return path


def test_cache_miss_returns_none(tmp_path):
    cache = MathpixCache(str(tmp_path))
    path = _write_tmp(b"fake-image-data")
    assert cache.get(path) is None
    Path(path).unlink()


def test_put_and_get_round_trip(tmp_path):
    cache = MathpixCache(str(tmp_path))
    path = _write_tmp(b"fake-image-data-2")
    data = {"text": "hello", "latex_styled": r"\alpha"}
    cache.put(path, data)
    result = cache.get(path)
    assert result == data
    Path(path).unlink()


def test_invalidate_removes_entry(tmp_path):
    cache = MathpixCache(str(tmp_path))
    path = _write_tmp(b"image-bytes")
    cache.put(path, {"text": "x"})
    cache.invalidate(path)
    assert cache.get(path) is None
    Path(path).unlink()


def test_different_images_get_different_keys(tmp_path):
    cache = MathpixCache(str(tmp_path))
    p1 = _write_tmp(b"image-one")
    p2 = _write_tmp(b"image-two")
    cache.put(p1, {"text": "one"})
    cache.put(p2, {"text": "two"})
    assert cache.get(p1)["text"] == "one"
    assert cache.get(p2)["text"] == "two"
    Path(p1).unlink(); Path(p2).unlink()
