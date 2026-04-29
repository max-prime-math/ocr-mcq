"""
ocr.py — Claude Vision extraction for multiple-choice exam pages.

Sends one or two page images to Claude and returns the question stem,
answer choices A–E, any extracted figure locations, and the marked
correct answer in a single API call.

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
You are processing scanned multiple-choice exam pages. The input will
contain either:
- one page image, or
- two consecutive page images from the same PDF.

Usually there is one question per page, but sometimes a large question
continues onto the next page and the answer choices / marked answer appear
on the second page.

Some questions also contain diagrams, graphs, tables, or other figures that
must be preserved in the final LaTeX output.

Your job:
1. Determine whether the question is fully contained on the first image or
   whether it continues onto the second image. Set pages_used to 1 or 2.
2. Extract the full question stem across the used pages (preserve LaTeX math).
3. Extract the text of each answer choice A–E (preserve LaTeX math).
4. Identify which choice is marked as correct.
5. If there is a written solution or explanation on the used page(s)
   (equations, working, or explanatory text beyond the answer choices),
   extract it as the solution. If there is no solution text, set solution
   to null.
6. For every meaningful figure that belongs in the question, return a
   bounding box on the page where the figure appears. Include diagrams,
   graphs, charts, geometry figures, and tables when they are part of the
   problem statement. Do not include decorative page furniture or choice
   bubbles.

LaTeX conventions: inline math as \\(...\\), display math as \\[...\\].

If the correct-answer mark is absent, ambiguous, or not legible, set
correct_answer to null — never guess silently.

Bounding boxes:
- page is 1 for the first image, 2 for the second image
- x, y, width, and height are normalized to the range [0, 1]
- x and y are the top-left corner of the figure box
- boxes should be tight enough to crop the actual figure content

If there is no second image, pages_used must be 1 and figures may still
refer only to page 1.\
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
        "pages_used": {
            "type": "integer",
            "enum": [1, 2],
            "description": "Whether the question uses only the first page image or spans both page images.",
        },
        "solution": {
            "description": "Written solution or explanation if present on the page, or null.",
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
        "figures": {
            "type": "array",
            "description": "Figures that should be cropped from the page image(s) and included in the LaTeX output.",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer", "enum": [1, 2]},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "caption": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
                "required": ["page", "x", "y", "width", "height", "caption"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["question", "choices", "correct_answer", "pages_used", "solution", "figures"],
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
    second_image_path: Optional[str] = None,
) -> dict:
    """
    Extract question data from a full-page exam image.

    Returns a dict with keys:
        question       (str)
        choices        (dict[str, str])  — keys A–E
        correct_answer (str | None)      — letter, or None if unclear

    Args:
        image_path: Path to a PNG of the first page.
        client:     Anthropic client instance.
        cache:      Optional VisionCache; skipped when None.
        force:      Bypass cache and call the API even if cached.
        model:      Claude model ID to use.
        usage_out:  Optional list; a usage dict is appended for each live
                    API call (cache hits are not counted — no tokens used).
        second_image_path:
                    Optional path to the next page when a question may span
                    two pages.
    """
    image_paths = [image_path]
    if second_image_path is not None:
        image_paths.append(second_image_path)

    if cache is not None:
        if force:
            cache.invalidate(image_paths)
        else:
            cached = cache.get(image_paths)
            if cached is not None:
                logger.debug("Cache hit for %s", image_paths)
                return cached

    content_blocks = []
    for idx, path in enumerate(image_paths, start=1):
        with open(path, "rb") as fh:
            image_b64 = base64.standard_b64encode(fh.read()).decode("ascii")
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            }
        )
        content_blocks.append(
            {
                "type": "text",
                "text": f"Image {idx} of {len(image_paths)}.",
            }
        )

    logger.debug("Calling Claude Vision (%s) for %s", model, image_paths)

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
                "content": content_blocks
                + [
                    {
                        "type": "text",
                        "text": (
                            "Extract the question, all five answer choices (A–E), "
                            "the marked correct answer, and any figure bounding boxes. "
                            "If the question continues onto image 2, combine both pages "
                            "and set pages_used to 2; otherwise set pages_used to 1."
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
        cache.put(image_paths, result)

    return result
