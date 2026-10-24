"""Gemini API wrapper for structured grading."""
from __future__ import annotations

import os
import random
import re
import time

from google import genai
from google.genai import types

from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .schema import GradingResponse

# Retry on transient server errors / rate limiting.
_RETRY_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 6
_BASE_DELAY = 5.0  # seconds; 429 per-minute quotas need ≥60s total backoff
_MAX_BACKOFF = 90.0

_RETRY_DELAY_RE = re.compile(r"""['"]retryDelay['"]\s*:\s*['"](\d+)s['"]""")


def _server_retry_delay(exc: Exception) -> float | None:
    """Extract Google's server-suggested retryDelay (in seconds) from the error, if present."""
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else None


def _normalize_coords(result: GradingResponse) -> GradingResponse:
    """Rescale coordinates Gemini returns in 0-1000 space into 0-1.

    Gemini 2.5's default spatial output is normalized to 0-1000, not 0-1, and
    it often mixes scales within one response (e.g. x in 0-1, y in 0-1000).
    Rescale each axis independently: if the max absolute value on that axis
    exceeds 1, divide by 1000.
    """
    for q in result.questions:
        x_vals = [q.answer_bbox[0], q.answer_bbox[2], q.mark_anchor[0]]
        y_vals = [q.answer_bbox[1], q.answer_bbox[3], q.mark_anchor[1]]
        if q.error_bbox is not None and len(q.error_bbox) == 4:
            x_vals.extend([q.error_bbox[0], q.error_bbox[2]])
            y_vals.extend([q.error_bbox[1], q.error_bbox[3]])
        x_scale = 1000.0 if max(abs(v) for v in x_vals) > 1.0 else 1.0
        y_scale = 1000.0 if max(abs(v) for v in y_vals) > 1.0 else 1.0
        if x_scale != 1.0 or y_scale != 1.0:
            q.answer_bbox = [
                q.answer_bbox[0] / x_scale,
                q.answer_bbox[1] / y_scale,
                q.answer_bbox[2] / x_scale,
                q.answer_bbox[3] / y_scale,
            ]
            q.mark_anchor = [
                q.mark_anchor[0] / x_scale,
                q.mark_anchor[1] / y_scale,
            ]
            if q.error_bbox is not None and len(q.error_bbox) == 4:
                q.error_bbox = [
                    q.error_bbox[0] / x_scale,
                    q.error_bbox[1] / y_scale,
                    q.error_bbox[2] / x_scale,
                    q.error_bbox[3] / y_scale,
                ]
    return result


def remap_strip_response(
    result: GradingResponse,
    source_page_index: int,
    y_start_norm: float,
    y_end_norm: float,
) -> GradingResponse:
    """Convert strip-local 0..1 coords back to full-page 0..1 coords.

    Gemini grades a strip on its own and returns coords normalized to the
    strip image. Strips are full-width horizontal slices, so x is unchanged
    and y must be mapped from [0, 1] strip-space into [y_start_norm, y_end_norm]
    on the original page. Also overwrites page_index to point at the source
    page, since Gemini saw a single image and would otherwise return 0.
    """
    strip_h = y_end_norm - y_start_norm
    for q in result.questions:
        q.page_index = source_page_index
        q.answer_bbox = [
            q.answer_bbox[0],
            y_start_norm + q.answer_bbox[1] * strip_h,
            q.answer_bbox[2],
            y_start_norm + q.answer_bbox[3] * strip_h,
        ]
        q.mark_anchor = [
            q.mark_anchor[0],
            y_start_norm + q.mark_anchor[1] * strip_h,
        ]
        if q.error_bbox is not None and len(q.error_bbox) == 4:
            q.error_bbox = [
                q.error_bbox[0],
                y_start_norm + q.error_bbox[1] * strip_h,
                q.error_bbox[2],
                y_start_norm + q.error_bbox[3] * strip_h,
            ]
    return result


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return genai.Client(api_key=api_key)


def grade_submission(
    rubric_text: str,
    answer_key_images: list[bytes],
    submission_images: list[bytes],
    model: str = "gemini-2.5-pro",
    thinking_budget: int | None = 0,
) -> GradingResponse:
    """Send rubric + answer key + submission to Gemini, return parsed JSON."""
    client = _client()

    parts: list[types.Part] = [
        types.Part.from_text(text=USER_PROMPT_TEMPLATE.format(rubric_text=rubric_text)),
    ]
    if answer_key_images:
        parts.append(types.Part.from_text(text="--- ANSWER KEY ---"))
        parts.extend(
            types.Part.from_bytes(data=img, mime_type="image/png") for img in answer_key_images
        )
    else:
        parts.append(
            types.Part.from_text(
                text="--- ANSWER KEY --- (none provided; grade using the rubric alone)"
            )
        )
    parts.append(types.Part.from_text(text="--- STUDENT SUBMISSION ---"))
    parts.extend(
        types.Part.from_bytes(data=img, mime_type="image/png") for img in submission_images
    )

    config_kwargs: dict = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "response_schema": GradingResponse,
        "temperature": 0.1,
    }
    if thinking_budget is not None:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        except (AttributeError, TypeError):
            pass

    config = types.GenerateContentConfig(**config_kwargs)

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=config,
            )
            break
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            if code not in _RETRY_CODES or attempt == _MAX_RETRIES - 1:
                # Surface the full error on final failure so quota/rate issues are debuggable.
                print(f"  [final error] {type(exc).__name__} code={code}: {exc}")
                raise
            server_hint = _server_retry_delay(exc)
            if server_hint is not None:
                delay = min(server_hint + random.uniform(0, 1), _MAX_BACKOFF)
                hint_note = f" (server retryDelay={server_hint:.0f}s)"
            else:
                delay = min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_BACKOFF)
                hint_note = ""
            msg_snippet = str(exc).replace("\n", " ")[:300]
            print(
                f"  [retry {attempt + 1}/{_MAX_RETRIES - 1}] {type(exc).__name__} "
                f"code={code}; sleeping {delay:.1f}s{hint_note}\n"
                f"    -> {msg_snippet}"
            )
            time.sleep(delay)
            last_exc = exc
    else:  # pragma: no cover — exhausted without break
        if last_exc is not None:
            raise last_exc

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, GradingResponse):
        return _normalize_coords(parsed)
    return _normalize_coords(GradingResponse.model_validate_json(response.text))
