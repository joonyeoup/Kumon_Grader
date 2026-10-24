"""Re-draw marked PNGs from existing response.json files, without re-calling Gemini.

Usage:
    python rerender.py results/reading_hw1/
    python rerender.py results/reading_hw1/ --debug-boxes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from grader.overlay import draw_marks
from grader.schema import GradingResponse, QuestionResult


def _group_by_page(qs: list[QuestionResult]) -> dict[int, list[QuestionResult]]:
    out: dict[int, list[QuestionResult]] = {}
    for q in qs:
        out.setdefault(q.page_index, []).append(q)
    return out


def rerender_student(student_dir: Path, draw_bboxes: bool) -> None:
    response_path = student_dir / "response.json"
    if not response_path.exists():
        print(f"  skip (no response.json): {student_dir.name}")
        return
    result = GradingResponse.model_validate_json(response_path.read_text())
    by_page = _group_by_page(result.questions)
    for orig in sorted(student_dir.glob("page_*_original.png")):
        try:
            page_num = int(orig.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        marked = student_dir / f"page_{page_num}_marked.png"
        page_qs = by_page.get(page_num - 1, [])
        if page_qs:
            draw_marks(orig, page_qs, marked, draw_bboxes=draw_bboxes)
        else:
            # No questions on this page — copy original.
            from PIL import Image
            Image.open(orig).save(marked, "PNG")
    print(f"  rerendered: {student_dir.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder")
    parser.add_argument("--debug-boxes", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_folder).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    student_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not student_dirs:
        sys.exit(f"No student folders in {root}")

    for d in sorted(student_dirs):
        rerender_student(d, args.debug_boxes)


if __name__ == "__main__":
    main()
