"""
ocr.py — Mathpix OCR integration.

Sends a cropped page image to the Mathpix API and returns the raw JSON
response.  Caching is handled by MathpixCache; this module is only
responsible for the HTTP call itself.

Environment variables required:
    MATHPIX_APP_ID  — your Mathpix application ID
    MATHPIX_APP_KEY — your Mathpix application key
"""

import base64
import logging
import os

import requests

from cache import MathpixCache

logger = logging.getLogger(__name__)

MATHPIX_API_URL = "https://api.mathpix.com/v3/text"

# Request options sent to Mathpix.  math_inline_delimiters wraps inline math
# in \( … \) so downstream LaTeX is immediately usable.
_DEFAULT_OPTIONS = {
    "math_inline_delimiters": ["\\(", "\\)"],
    "math_display_delimiters": ["\\[", "\\]"],
    "rm_spaces": True,
}


def _read_credentials() -> tuple[str, str]:
    app_id = os.environ.get("MATHPIX_APP_ID", "")
    app_key = os.environ.get("MATHPIX_APP_KEY", "")
    if not app_id or not app_key:
        raise EnvironmentError(
            "MATHPIX_APP_ID and MATHPIX_APP_KEY must be set as environment variables."
        )
    return app_id, app_key


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def call_mathpix(image_path: str) -> dict:
    """
    Send *image_path* to the Mathpix API and return the raw response dict.

    Raises:
        EnvironmentError: if credentials are missing.
        requests.HTTPError: if the API returns a non-2xx status.
    """
    app_id, app_key = _read_credentials()
    b64 = _encode_image(image_path)

    payload = {
        "src": f"data:image/png;base64,{b64}",
        "formats": ["text", "latex_styled"],
        "data_options": _DEFAULT_OPTIONS,
    }
    headers = {
        "app_id": app_id,
        "app_key": app_key,
        "Content-type": "application/json",
    }

    logger.debug("Sending request to Mathpix for %s", image_path)
    resp = requests.post(MATHPIX_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_ocr_text(
    image_path: str,
    cache: MathpixCache | None = None,
    force: bool = False,
) -> dict:
    """
    Return Mathpix response for *image_path*, using cache when available.

    Args:
        image_path: Path to the PNG crop to OCR.
        cache:      MathpixCache instance; pass None to skip caching.
        force:      Bypass cache and re-call the API even if cached.

    Returns:
        Raw Mathpix JSON dict (keys include "text", "latex_styled", etc.).
    """
    if cache is not None:
        if force:
            cache.invalidate(image_path)
        else:
            cached = cache.get(image_path)
            if cached is not None:
                return cached

    result = call_mathpix(image_path)

    if cache is not None:
        cache.put(image_path, result)

    return result
