"""Minimal Flask viewer for graded results.

Usage: python viewer.py results/<assignment>/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import webbrowser
from pathlib import Path

from flask import Flask, abort, render_template, send_from_directory


app = Flask(__name__, template_folder="viewer_static", static_folder=None)
RESULTS_ROOT: Path = Path(".")

PAGE_RE = re.compile(r"page_(\d+)_(original|marked)\.png")


def _load_summary() -> list[dict]:
    csv_path = RESULTS_ROOT / "summary.csv"
    rows: list[dict] = []
    if csv_path.exists():
        import csv as _csv

        with csv_path.open() as f:
            reader = _csv.DictReader(f)
            rows.extend(reader)
    else:
        # Fall back to progress.json
        prog_path = RESULTS_ROOT / "progress.json"
        if prog_path.exists():
            prog = json.loads(prog_path.read_text())
            for name, info in sorted(prog.items()):
                rows.append(
                    {
                        "student": name,
                        "score": str(info.get("score", "")),
                        "possible": str(info.get("possible", "")),
                        "percentage": str(info.get("percentage", "")),
                        "status": info.get("status", ""),
                    }
                )
    return rows


def _student_pages(student_dir: Path) -> list[dict]:
    pages: dict[int, dict] = {}
    for p in student_dir.iterdir():
        m = PAGE_RE.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        kind = m.group(2)
        pages.setdefault(n, {"page": n, "original": None, "marked": None})
        pages[n][kind] = p.name
    return [pages[k] for k in sorted(pages)]


@app.route("/")
def index() -> str:
    rows = _load_summary()
    return render_template("index.html", rows=rows, assignment=RESULTS_ROOT.name)


@app.route("/student/<name>")
def student(name: str) -> str:
    student_dir = RESULTS_ROOT / name
    if not student_dir.is_dir():
        abort(404)

    pages = _student_pages(student_dir)

    response_path = student_dir / "response.json"
    response_data = None
    if response_path.exists():
        response_data = json.loads(response_path.read_text())

    feedback_path = student_dir / "feedback.txt"
    feedback_text = feedback_path.read_text() if feedback_path.exists() else ""

    return render_template(
        "student.html",
        student=name,
        pages=pages,
        response=response_data,
        feedback=feedback_text,
        assignment=RESULTS_ROOT.name,
    )


@app.route("/image/<student>/<path:filename>")
def image(student: str, filename: str):
    student_dir = RESULTS_ROOT / student
    if not student_dir.is_dir():
        abort(404)
    return send_from_directory(student_dir, filename)


def main() -> None:
    global RESULTS_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    RESULTS_ROOT = Path(args.results_folder).resolve()
    if not RESULTS_ROOT.is_dir():
        sys.exit(f"Not a directory: {RESULTS_ROOT}")

    url = f"http://localhost:{args.port}"
    print(f"Serving {RESULTS_ROOT} at {url}")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
