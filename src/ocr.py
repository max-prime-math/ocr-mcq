"""
ocr.py — Claude Vision extraction for multiple-choice exam pages.

Sends one or two page images to Claude and returns the question stem,
answer choices A–E, extracted non-table figure locations, extracted LaTeX
tables, and the marked correct answer in a single API call.

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

_SYSTEM_PROMPT_SINGLE = """\
Process one scanned multiple-choice exam page.

Extract:
1. The full question stem.
2. Choices A through E.
3. The marked correct answer, or null if absent or unclear.
4. Any written solution or explanation below the choices, or null if none.

Preserve math as LaTeX using \\(...\\) for inline math and \\[...\\] for display math.
Do not guess silently.\
"""

_SYSTEM_PROMPT_EXTENDED = """\
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
6. For every meaningful non-table figure, return a bounding box on the page
   where the figure appears. Include diagrams, graphs, charts, and geometry
   figures when they are part of the problem statement or written solution.
   Do not include tables as figures. Do not include decorative page
   furniture or choice bubbles.
7. Extract any actual tables as LaTeX table/tabular content instead of image
   crops. Put each table in the tables array with a section label.
8. Every figure and every table must be labelled with:
   - section: question or solution
   - placement:
     - stem if it belongs to the main question/solution body
     - A/B/C/D/E if it belongs to a specific answer choice
   Never place a solution-only asset in the question section.
9. When an answer choice has its own diagram, graph, or table, attach that
   asset to the corresponding choice letter instead of to the overall stem.
10. If image 2 repeats the same question from image 1 but adds a boxed
    answer or worked solution, treat both images as one question and set
    pages_used to 2. Do not emit a second separate question.
11. If image 1 says "solution on the next page" or otherwise indicates that
    the answer/solution is on image 2, combine them into one question.
12. Figure bounding boxes must be tight crops of the figure itself. Do not
   return the entire page unless the figure literally occupies nearly the
   entire page. Exclude surrounding question text, answer choices, headers,
   and margins whenever possible.

LaTeX conventions: inline math as \\(...\\), display math as \\[...\\].

If the correct-answer mark is absent, ambiguous, or not legible, set
correct_answer to null — never guess silently.

Bounding boxes:
- page is 1 for the first image, 2 for the second image
- x, y, width, and height are normalized to the range [0, 1]
- x and y are the top-left corner of the figure box
- boxes should be tight enough to crop the actual figure content
- do not use a full-page bounding box for a small or medium figure

If there is no second image, pages_used must be 1 and figures may still
refer only to page 1.\
"""

_OUTPUT_SCHEMA_SINGLE = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "choices": {
            "type": "object",
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
            "anyOf": [
                {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
                {"type": "null"},
            ],
        },
        "solution": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
    },
    "required": ["question", "choices", "correct_answer", "solution"],
    "additionalProperties": False,
}

_OUTPUT_SCHEMA_EXTENDED = {
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
                    "section": {"type": "string", "enum": ["question", "solution"]},
                    "placement": {"type": "string", "enum": ["stem", "A", "B", "C", "D", "E"]},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                },
                "required": ["page", "section", "placement", "x", "y", "width", "height"],
                "additionalProperties": False,
            },
        },
        "tables": {
            "type": "array",
            "description": "Tables extracted as LaTeX rather than image crops.",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "enum": ["question", "solution"]},
                    "placement": {"type": "string", "enum": ["stem", "A", "B", "C", "D", "E"]},
                    "latex": {"type": "string"},
                },
                "required": ["section", "placement", "latex"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["question", "choices", "correct_answer", "pages_used", "solution", "figures", "tables"],
    "additionalProperties": False,
}


def _normalise_single_page_result(result: dict) -> dict:
    """Add fixed fields omitted from the economical single-page schema."""
    normalised = dict(result)
    normalised["pages_used"] = 1
    normalised["figures"] = []
    normalised["tables"] = []
    return normalised


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
    include_figures: bool = False,
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
    extended_mode = second_image_path is not None or include_figures

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
        if extended_mode:
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
                "text": _SYSTEM_PROMPT_EXTENDED if extended_mode else _SYSTEM_PROMPT_SINGLE,
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
                            (
                                "Extract the question, choices A-E, marked answer, any tight non-table figure boxes, "
                                "and any tables as LaTeX with section and placement labels. "
                                "If image 2 continues or repeats the same question with the answer or solution, combine both pages."
                            )
                            if extended_mode
                            else (
                                "Extract the question, choices A-E, marked answer, and any written solution text from this page."
                            )
                        ),
                    },
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": _OUTPUT_SCHEMA_EXTENDED if extended_mode else _OUTPUT_SCHEMA_SINGLE,
            }
        },
    )

    raw = next(b.text for b in response.content if b.type == "text")
    result = json.loads(raw)
    if not extended_mode:
        result = _normalise_single_page_result(result)

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


def should_retry_with_next_page(result: dict) -> bool:
    """
    Return True when a single-page extraction looks incomplete enough that
    retrying with the next PDF page is worthwhile.

    This is intentionally conservative: it retries when the model failed to
    recover the expected A–E structure, which is the common signature of a
    question whose answer choices spill onto the following page.
    """
    choices = result.get("choices") or {}
    if not isinstance(choices, dict):
        return True

    present = sum(1 for key in ("A", "B", "C", "D", "E") if choices.get(key))
    question = (result.get("question") or "").strip()
    solution = (result.get("solution") or "").strip()
    answer = result.get("correct_answer")
    haystack = " ".join([question, solution]).lower()

    if not question:
        return True
    if present < 5:
        return True
    if "solution on the next page" in haystack:
        return True
    if answer is None and not solution:
        return True
    return False


def should_extract_figures(result: dict) -> bool:
    """
    Return True when the extracted text strongly suggests that a visual
    figure is required for the question.

    This is a cheap heuristic used by figure mode ``auto`` so we only pay
    for the heavier figure-aware extraction on pages that likely need it.
    """
    texts: list[str] = []
    question = result.get("question")
    if isinstance(question, str):
        texts.append(question)

    choices = result.get("choices") or {}
    if isinstance(choices, dict):
        for value in choices.values():
            if isinstance(value, str):
                texts.append(value)

    haystack = " ".join(texts).lower()
    if not haystack:
        return False

    strong_keywords = (
        "figure",
        "fig.",
        "diagram",
        "graph",
        "pictured",
        "sketch",
        "plot",
        "scatterplot",
        "histogram",
        "table above",
        "table below",
    )
    if any(token in haystack for token in strong_keywords):
        return True

    weak_hits = 0
    for token in ("shown", "below", "above", "following"):
        if token in haystack:
            weak_hits += 1
    return weak_hits >= 2
