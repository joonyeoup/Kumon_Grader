"""Draw correctness marks on top of original page PNGs."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schema import QuestionResult

# Teacher-mark colors.
GREEN = (30, 160, 60, 255)    # correct → green circle
RED = (220, 30, 30, 255)      # incorrect → red X
ORANGE = (240, 150, 40, 255)  # partial → orange triangle

# Debug overlay (only drawn when draw_bboxes=True).
DEBUG_BOX = (30, 120, 220, 200)
DEBUG_FILL = (30, 120, 220, 40)
DEBUG_ERROR_BOX = (220, 30, 120, 220)
DEBUG_ERROR_FILL = (220, 30, 120, 50)

# Marks are placed just to the LEFT of the relevant bbox, at its vertical
# center. For correct verdicts the bbox is `answer_bbox`; for partial/
# incorrect it's `error_bbox` (the specific offending characters), falling
# back to `answer_bbox` if Gemini didn't return one. Gemini's own
# `mark_anchor` is ignored because it drifts.
MARK_GAP = 0.015  # horizontal gap between bbox edge and mark center, 0-1
MARK_X_MIN = 0.03  # never draw past this x on the left, to stay on the page


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("Arial.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


def _anchor_left_of_bbox(bbox: list[float]) -> tuple[float, float]:
    """Place the mark just to the left of the bbox, at its vertical center."""
    x_min, y_min, _, y_max = bbox
    ax = max(x_min - MARK_GAP, MARK_X_MIN)
    ay = (y_min + y_max) / 2.0
    return ax, ay


def _draw_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, width: int) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)


def _draw_x(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, width: int) -> None:
    draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill=color, width=width)
    draw.line([(cx - r, cy + r), (cx + r, cy - r)], fill=color, width=width)


def _draw_triangle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, width: int) -> None:
    """Equilateral triangle inscribed in a circle of radius r, apex up."""
    side_dx = r * math.sqrt(3) / 2
    apex = (cx, cy - r)
    bottom_left = (cx - side_dx, cy + r / 2)
    bottom_right = (cx + side_dx, cy + r / 2)
    draw.polygon([apex, bottom_left, bottom_right], outline=color, width=width)


def draw_marks(
    original_png_path: str | Path,
    page_results: list[QuestionResult],
    output_path: str | Path,
    draw_bboxes: bool = False,
) -> Path:
    """Composite teacher marks on top of the original page PNG.

    - correct   → green circle
    - incorrect → red X
    - partial   → orange triangle

    If draw_bboxes is True, also outline the Gemini-reported `answer_bbox`
    in translucent blue and label each box with its `question_id`. In normal
    mode only the colored circles/Xs are drawn — no text labels.
    """
    img = Image.open(original_png_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    radius = int(min(W, H) * 0.025)
    line_width = max(3, radius // 6)
    font = _font(max(14, radius // 2))

    for q in page_results:
        if draw_bboxes:
            x0, y0, x1, y1 = q.answer_bbox
            rect = [int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)]
            draw.rectangle(rect, outline=DEBUG_BOX, fill=DEBUG_FILL, width=2)
            draw.text(
                (rect[0] + 4, max(0, rect[1] - max(14, radius // 2) - 2)),
                f"Q{q.question_id}",
                fill=DEBUG_BOX,
                font=font,
            )
            if q.error_bbox and len(q.error_bbox) == 4:
                ex0, ey0, ex1, ey1 = q.error_bbox
                erect = [int(ex0 * W), int(ey0 * H), int(ex1 * W), int(ey1 * H)]
                draw.rectangle(
                    erect, outline=DEBUG_ERROR_BOX, fill=DEBUG_ERROR_FILL, width=2
                )

        if q.verdict != "correct" and q.error_bbox and len(q.error_bbox) == 4:
            target_bbox = q.error_bbox
        else:
            target_bbox = q.answer_bbox
        ax, ay = _anchor_left_of_bbox(target_bbox)
        cx, cy = int(ax * W), int(ay * H)

        if q.verdict == "correct":
            _draw_circle(draw, cx, cy, radius, GREEN, line_width)
        elif q.verdict == "incorrect":
            _draw_x(draw, cx, cy, radius, RED, line_width)
        else:  # partial
            _draw_triangle(draw, cx, cy, radius, ORANGE, line_width)

    composed = Image.alpha_composite(img, overlay)
    out = Path(output_path)
    composed.convert("RGB").save(out, "PNG")
    return out
