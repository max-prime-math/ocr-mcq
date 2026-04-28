"""
ocr.py — Claude Vision extraction for multiple-choice exam pages.

Sends a full-page image to Claude and returns the question stem, answer
choices A–E, and the marked correct answer in a single API call.

Requires ANTHROPIC_API_KEY to be set in the environment.

Prompt caching is applied to the system prompt so repeated calls within
the cache TTL only pay for the image + question tokens, not the system
prompt tokens.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

import anthropic

from cache import MathpixCache as VisionCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt & schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are processing scanned multiple-choice exam pages. Each page contains
exactly one question with answer choices labelled A through E. One choice
has a visible mark — a filled bubble, circled letter, checkmark, tick, or
similar — indicating the correct answer. Some pages also include a written
solution or explanation below the answer choices.

Your job:
1. Extract the full question stem (preserve all LaTeX math exactly).
2. Extract the text of each answer choice A–E (preserve LaTeX math).
3. Identify which choice is marked as correct.
4. If there is a written solution or explanation on the page (equations,
   working, or explanatory text beyond the answer choices), extract it as
   the solution. If there is no solution text, set solution to null.

LaTeX conventions: inline math as \\(...\\), display math as \\[...\\].

If the correct-answer mark is absent, ambiguous, or not legible, set
correct_answer to null — never guess silently.\
"""

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "Full question stem with LaTeX math preserved.",
        },
        "choices": {
            "type": "object",
            "description": "Answer choices A through E.",
            "properties": {
                "A": {"type": "string"},
                "B": {"type": "string"},
                "C": {"type": "string"},
                "D": {"type": "string"},
                "E": {"type": "string"},
            },
            "required": ["A", "B", "C", "D", "E"],
            "additionalProperties": False,
        },
        "correct_answer": {
            "description": "Letter of the marked choice, or null if unclear.",
            "anyOf": [
                {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                {"type": "null"},
            ],
        },
        "solution": {
            "description": "Written solution or explanation if present on the page, or null.",
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
    },
    "required": ["question", "choices", "correct_answer", "solution"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_page(
    image_path: str,
    client: anthropic.Anthropic,
    cache: Optional[VisionCache] = None,
    force: bool = False,
    model: str = "claude-haiku-4-5",
    usage_out: Optional[list] = None,
) -> dict:
    """
    Extract question data from a full-page exam image.

    Returns a dict with keys:
        question       (str)
        choices        (dict[str, str])  — keys A–E
        correct_answer (str | None)      — letter, or None if unclear

    Args:
        image_path: Path to a PNG of the full page.
        client:     Anthropic client instance.
        cache:      Optional VisionCache; skipped when None.
        force:      Bypass cache and call the API even if cached.
        model:      Claude model ID to use.
        usage_out:  Optional list; a usage dict is appended for each live
                    API call (cache hits are not counted — no tokens used).
    """
    if cache is not None:
        if force:
            cache.invalidate(image_path)
        else:
            cached = cache.get(image_path)
            if cached is not None:
                logger.debug("Cache hit for %s", image_path)
                return cached

    with open(image_path, "rb") as fh:
        image_b64 = base64.standard_b64encode(fh.read()).decode("ascii")

    logger.debug("Calling Claude Vision (%s) for %s", model, image_path)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                # Cache the system prompt — same for every page.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract the question, all five answer choices (A–E), "
                            "and the marked correct answer from this exam page."
                        ),
                    },
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _OUTPUT_SCHEMA,
            }
        },
    )

    raw = next(b.text for b in response.content if b.type == "text")
    result = json.loads(raw)

    u = response.usage
    usage = {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
    }
    logger.debug(
        "Tokens — input: %d  output: %d  cache_read: %d  cache_create: %d",
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_input_tokens"],
        usage["cache_creation_input_tokens"],
    )

    if usage_out is not None:
        usage_out.append(usage)

    if cache is not None:
        cache.put(image_path, result)

    return result
