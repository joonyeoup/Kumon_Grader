"""Main CLI entry point.

Usage: python grade.py <assignment_folder> [--model gemini-2.5-pro] [--workers 3]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from grader.gemini_client import grade_submission, remap_strip_response
from grader.overlay import draw_marks
from grader.pdf_utils import (
    SUPPORTED_SUFFIXES,
    PageImage,
    PageStrip,
    load_any,
    pages_to_png_bytes,
    partition_page,
    save_pages,
)
from grader.schema import GradingResponse, QuestionResult

PROGRESS_LOCK = threading.Lock()


def _validate_assignment(folder: Path) -> tuple[Path, Path | None, list[Path]]:
    rubric = folder / "rubric.txt"
    submissions_dir = folder / "submissions"
    if not rubric.exists():
        sys.exit(f"Missing {rubric}")
    if not submissions_dir.is_dir():
        sys.exit(f"Missing {submissions_dir}")

    answer_key: Path | None = None
    for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
        candidate = folder / f"answer_key{suffix}"
        if candidate.exists():
            answer_key = candidate
            break

    submissions = sorted(
        p for p in submissions_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not submissions:
        sys.exit(
            f"No PDF/PNG/JPG submissions found in {submissions_dir}"
        )
    return rubric, answer_key, submissions


def _load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_progress(path: Path, data: dict) -> None:
    with PROGRESS_LOCK:
        path.write_text(json.dumps(data, indent=2))


def _write_feedback(
    txt_path: Path, result: GradingResponse, show_transcription: bool = False
) -> None:
    lines: list[str] = []
    lines.append(f"Student: {result.student_identifier}")
    lines.append(
        f"Score: {result.overall_score}/{result.overall_possible}"
        f"  ({100.0 * result.overall_score / result.overall_possible:.1f}%)"
        if result.overall_possible
        else f"Score: {result.overall_score}/{result.overall_possible}"
    )
    lines.append("")
    lines.append("Overall feedback:")
    lines.append(result.overall_feedback)
    lines.append("")
    lines.append("Per-question feedback:")
    for q in result.questions:
        parent = f" (part of Q{q.parent_id})" if q.parent_id else ""
        lines.append(
            f"  Q{q.question_id}{parent} [{q.verdict}] "
            f"{q.points_earned}/{q.points_possible} — {q.feedback}"
        )
        if show_transcription:
            transcript = q.transcribed_answer if q.transcribed_answer else "(blank)"
            lines.append(f"      read: {transcript!r}")
    txt_path.write_text("\n".join(lines))


def _group_by_page(qs: list[QuestionResult]) -> dict[int, list[QuestionResult]]:
    out: dict[int, list[QuestionResult]] = {}
    for q in qs:
        out.setdefault(q.page_index, []).append(q)
    return out


@contextmanager
def _timed(timings: dict | None, key: str):
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = round(time.perf_counter() - start, 3)


def _grade_partitioned(
    pages: list[PageImage],
    rubric_text: str,
    answer_key_pngs: list[bytes],
    model: str,
    partitions: int,
    partition_workers: int,
    thinking_budget: int | None,
) -> GradingResponse:
    """Grade by splitting each page into N strips and calling Gemini on each
    strip in parallel, then merging the per-strip responses.

    A question that lands in the overlap zone may show up in two adjacent
    strips; we dedup by (page_index, question_id), keeping the strip whose
    center is closest to the answer's vertical center (i.e. the strip where
    the answer is most fully visible).
    """
    strips: list[PageStrip] = []
    for page in pages:
        strips.extend(partition_page(page, partitions))

    def _grade_one_strip(strip: PageStrip) -> tuple[PageStrip, GradingResponse]:
        result = grade_submission(
            rubric_text=rubric_text,
            answer_key_images=answer_key_pngs,
            submission_images=[strip.png_bytes],
            model=model,
            thinking_budget=thinking_budget,
        )
        result = remap_strip_response(
            result, strip.page_index, strip.y_start_norm, strip.y_end_norm
        )
        return strip, result

    strip_results: list[tuple[PageStrip, GradingResponse]] = []
    with ThreadPoolExecutor(max_workers=partition_workers) as pool:
        futures = [pool.submit(_grade_one_strip, s) for s in strips]
        for fut in as_completed(futures):
            strip_results.append(fut.result())

    best: dict[tuple[int, str], tuple[float, QuestionResult]] = {}
    for strip, result in strip_results:
        strip_center = (strip.y_start_norm + strip.y_end_norm) / 2.0
        for q in result.questions:
            key = (q.page_index, q.question_id)
            q_center_y = (q.answer_bbox[1] + q.answer_bbox[3]) / 2.0
            dist = abs(q_center_y - strip_center)
            if key not in best or dist < best[key][0]:
                best[key] = (dist, q)

    questions = [q for _, q in best.values()]
    questions.sort(key=lambda q: (q.page_index, q.question_id))

    overall_score = sum(q.points_earned for q in questions)
    overall_possible = sum(q.points_possible for q in questions)

    student_id = ""
    feedback_first = ""
    for _, result in strip_results:
        if not student_id and result.student_identifier:
            student_id = result.student_identifier
        if not feedback_first and result.overall_feedback:
            feedback_first = result.overall_feedback

    return GradingResponse(
        student_identifier=student_id or "unknown",
        overall_score=overall_score,
        overall_possible=overall_possible,
        overall_feedback=feedback_first,
        questions=questions,
    )


def _grade_one(
    submission_pdf: Path,
    rubric_text: str,
    answer_key_pngs: list[bytes],
    results_root: Path,
    model: str,
    draw_bboxes: bool = False,
    thinking_budget: int | None = 0,
    show_transcription: bool = False,
    timing: bool = False,
    partitions: int = 1,
    partition_workers: int = 1,
) -> tuple[str, dict]:
    student = submission_pdf.stem
    student_dir = results_root / student
    student_dir.mkdir(parents=True, exist_ok=True)
    timings: dict | None = {} if timing else None
    t_start = time.perf_counter() if timing else None
    if timing:
        print(f"[start ] {student}")

    try:
        with _timed(timings, "load_pdf"):
            pages = load_any(submission_pdf)
            save_pages(pages, student_dir)
            submission_pngs = pages_to_png_bytes(pages)

        with _timed(timings, "gemini_call"):
            if partitions > 1:
                result = _grade_partitioned(
                    pages=pages,
                    rubric_text=rubric_text,
                    answer_key_pngs=answer_key_pngs,
                    model=model,
                    partitions=partitions,
                    partition_workers=partition_workers,
                    thinking_budget=thinking_budget,
                )
            else:
                result = grade_submission(
                    rubric_text=rubric_text,
                    answer_key_images=answer_key_pngs,
                    submission_images=submission_pngs,
                    model=model,
                    thinking_budget=thinking_budget,
                )

        with _timed(timings, "overlay"):
            (student_dir / "response.json").write_text(result.model_dump_json(indent=2))
            by_page = _group_by_page(result.questions)
            for page in pages:
                original = student_dir / f"page_{page.index + 1}_original.png"
                marked = student_dir / f"page_{page.index + 1}_marked.png"
                page_qs = by_page.get(page.index, [])
                if page_qs:
                    draw_marks(original, page_qs, marked, draw_bboxes=draw_bboxes)
                else:
                    # No questions flagged on this page — copy original so the viewer has it.
                    page.pil_image.save(marked, "PNG")

        with _timed(timings, "feedback_write"):
            _write_feedback(
                student_dir / "feedback.txt", result, show_transcription=show_transcription
            )

        pct = (
            100.0 * result.overall_score / result.overall_possible
            if result.overall_possible
            else 0.0
        )
        info = {
            "status": "complete",
            "score": result.overall_score,
            "possible": result.overall_possible,
            "percentage": round(pct, 1),
        }
        if timings is not None:
            timings["total"] = round(time.perf_counter() - t_start, 3)
            info["timings"] = timings
            (student_dir / "timings.json").write_text(json.dumps(timings, indent=2))
            print(
                f"[timing] {student}  "
                + "  ".join(f"{k}={v}s" for k, v in timings.items())
            )
        return student, info
    except Exception as exc:  # noqa: BLE001
        info = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if timings is not None:
            timings["total"] = round(time.perf_counter() - t_start, 3)
            info["timings"] = timings
        return student, info


def _write_summary_csv(csv_path: Path, progress: dict) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student", "score", "possible", "percentage", "status"])
        for student, info in sorted(progress.items()):
            if info.get("status") == "complete":
                w.writerow(
                    [
                        student,
                        info.get("score", ""),
                        info.get("possible", ""),
                        info.get("percentage", ""),
                        "complete",
                    ]
                )
            else:
                w.writerow([student, "", "", "", info.get("status", "failed")])


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Grade elementary homework PDFs with Gemini.")
    parser.add_argument("assignment_folder", type=str)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Tokens Gemini reasons internally before answering. 0 disables (fastest, Flash only). Try 256–1024 for Pro if quality drops.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--debug-boxes",
        action="store_true",
        help="Also draw Gemini's answer_bbox as a translucent blue rectangle on each marked PNG.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="In feedback.txt, also print Gemini's verbatim transcription of each answer. Use this to tell whether wrong grades are caused by misreading the handwriting vs. faulty reasoning.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Measure per-step wall time (load_pdf, gemini_call, overlay, feedback_write, total). Prints a line per student and writes timings.json into each student folder; also stored under 'timings' in progress.json.",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=1,
        help="Split each page into N horizontal strips and grade each strip in a separate Gemini call (in parallel). 1 = current behavior. Try 2–3 for dense math worksheets where the per-call latency dominates. Cost scales ~Nx since the rubric+answer key are re-sent per strip.",
    )
    parser.add_argument(
        "--partition-workers",
        type=int,
        default=0,
        help="Max concurrent strip calls per submission. 0 (default) means use --partitions. Only used when --partitions > 1.",
    )
    args = parser.parse_args()

    if args.partitions < 1:
        sys.exit("--partitions must be >= 1")
    partition_workers = args.partition_workers or args.partitions

    assignment = Path(args.assignment_folder).resolve()
    rubric_path, answer_key_path, submissions = _validate_assignment(assignment)

    rubric_text = rubric_path.read_text()
    if answer_key_path is not None:
        print(f"Loading answer key: {answer_key_path.name}")
        answer_key_pngs = pages_to_png_bytes(load_any(answer_key_path))
    else:
        print("No answer key found — grading with rubric alone.")
        answer_key_pngs = []

    results_root = Path(args.results_root) / assignment.name
    results_root.mkdir(parents=True, exist_ok=True)
    progress_path = results_root / "progress.json"
    progress = _load_progress(progress_path)

    todo = [p for p in submissions if progress.get(p.stem, {}).get("status") != "complete"]
    print(f"{len(todo)} submissions to grade ({len(submissions) - len(todo)} already complete)")

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _grade_one,
                    pdf,
                    rubric_text,
                    answer_key_pngs,
                    results_root,
                    args.model,
                    args.debug_boxes,
                    args.thinking_budget,
                    args.repeat,
                    args.timing,
                    args.partitions,
                    partition_workers,
                ): pdf
                for pdf in todo
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Grading"):
                student, info = fut.result()
                progress[student] = info
                _save_progress(progress_path, progress)

    _write_summary_csv(results_root / "summary.csv", progress)

    done = sum(1 for v in progress.values() if v.get("status") == "complete")
    failed = sum(1 for v in progress.values() if v.get("status") != "complete")
    print(f"\nDone. complete={done} failed={failed}")
    print(f"Results: {results_root}")
    print(f"Run the viewer:  python viewer.py {results_root}")


if __name__ == "__main__":
    main()
