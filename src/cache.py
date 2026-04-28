"""
cache.py — Filesystem cache for Mathpix API responses.

Stores raw JSON responses keyed by a hash of the image content so that
identical crops never hit the API twice.  Pass --force-ocr to bypass.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _image_hash(image_path: str) -> str:
    """Return a stable SHA-256 hex digest for the file at *image_path*."""
    h = hashlib.sha256()
    with open(image_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MathpixCache:
    """Read/write Mathpix JSON responses to *cache_dir*."""

    def __init__(self, cache_dir: str = "cache/mathpix"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, image_path: str) -> Path:
        key = _image_hash(image_path)
        return self.cache_dir / f"{key}.json"

    def get(self, image_path: str):
        """Return cached dict if present, else None."""
        path = self._cache_path(image_path)
        if path.exists():
            logger.debug("Cache hit: %s", path.name)
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def put(self, image_path: str, data: dict) -> None:
        """Persist *data* to the cache."""
        path = self._cache_path(image_path)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.debug("Cached response → %s", path.name)

    def invalidate(self, image_path: str) -> None:
        """Remove cache entry for *image_path* (used with --force-ocr)."""
        path = self._cache_path(image_path)
        if path.exists():
            path.unlink()
            logger.debug("Invalidated cache entry: %s", path.name)
