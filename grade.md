# Elementary HW Grader — Local Demo Build Spec

This document specifies a **local prototype** that grades 1st–6th grade reading/essay homework PDFs using Google Gemini, then displays the PDF with ⭕ (correct) and ─ (incorrect) marks overlaid at the right positions on each answer.

**Target environment:** macOS or Linux laptop. No app, no server, no deployment. Just a Python script + a simple local viewer.

**You are pivoting from:** an existing repo (`HW_grader/`) that used GPT-4 Vision + Gradescope integration for college math grading. Most of that repo is not reusable — only the architectural ideas (parallel grading, resumable progress, rubric-grounded prompts) carry over.

---

## 0. What Success Looks Like

A teacher runs:

```bash
python grade.py assignments/reading_hw1/
```

The script:
1. Finds the rubric, answer key, and student PDFs in that folder.
2. Sends each student PDF to Gemini with the rubric + answer key.
3. Gets back structured JSON: per-question correctness + bounding boxes + feedback.
4. Writes an annotated output PNG per student with ⭕ / ─ marks drawn in the right margin next to each answer.
5. Saves a summary CSV and a per-student feedback report.

The teacher then runs:

```bash
python viewer.py results/reading_hw1/
```

A local web page opens showing each student's original PDF side-by-side with the marked-up version, plus the feedback text. That's the demo.

---

## 1. Critical Design Principle — Do Not Ask Gemini To Draw

**The #1 mistake to avoid:** do NOT ask Gemini to return an annotated image. Vision-language models that output images regenerate pixels from scratch, which hallucinates and blurs the original. Text gets mangled, handwriting turns to gibberish.

**The correct pattern, used throughout this project:**

1. Gemini returns **JSON only** — coordinates, correctness, feedback.
2. Python opens the **original** page PNG.
3. Python draws circles and dashes on top of it using Pillow / PIL.
4. The original pixels are never touched by the model.

This is non-negotiable. Every part of the design below assumes coordinates-only responses from Gemini.

---

## 2. Directory Layout To Create

```
elementary_grader/
├── README.md
├── requirements.txt
├── .env.example                  ← template for GEMINI_API_KEY
├── .gitignore
│
├── grade.py                      ← main CLI entry point
├── viewer.py                     ← local Flask viewer for results
│
├── grader/
│   ├── __init__.py
│   ├── gemini_client.py          ← wraps google-genai calls
│   ├── pdf_utils.py              ← PDF → PNG conversion
│   ├── overlay.py                ← draws ⭕ / ─ on PNGs
│   ├── schema.py                 ← Pydantic models for Gemini response
│   └── prompts.py                ← system + user prompt templates
│
├── assignments/
│   └── reading_hw1/              ← example assignment (you will drop files here)
│       ├── rubric.txt            ← grading criteria in plain text
│       ├── answer_key.pdf        ← teacher's answer key
│       └── submissions/
│           ├── student_alice.pdf
│           ├── student_bob.pdf
│           └── ...
│
├── results/
│   └── reading_hw1/              ← auto-created by grade.py
│       ├── progress.json         ← resumable state
│       ├── summary.csv           ← one row per student
│       ├── student_alice/
│       │   ├── page_1_original.png
│       │   ├── page_1_marked.png
│       │   ├── response.json     ← raw Gemini JSON
│       │   └── feedback.txt      ← human-readable feedback
│       └── student_bob/
│           └── ...
│
└── viewer_static/                ← Flask templates & static files
    ├── index.html
    └── style.css
```

---

## 3. Dependencies

Create `requirements.txt` exactly:

```
google-genai>=0.3.0
pdf2image>=1.17.0
Pillow>=10.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
flask>=3.0.0
tqdm>=4.66.0
```

**System dependency:** `pdf2image` needs Poppler.
- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt-get install poppler-utils`

Document this clearly in the README.

---

## 4. The Gemini Call — Exact Contract

Use the `google-genai` SDK (the new one, not the deprecated `google-generativeai`). Model: `gemini-2.5-pro` for the reasoning quality, or `gemini-2.5-flash` for speed/cost during dev.

### 4.1 System Prompt

In `grader/prompts.py`:

```python
SYSTEM_PROMPT = """You are an experienced elementary school teacher grading homework for students in grades 1 through 6. You grade reading comprehension and short essay responses.

Your grading principles:
- Be kind and age-appropriate. These are young children.
- Give credit for understanding, not perfect spelling or grammar (unless the rubric specifically requires it).
- For reading comprehension, accept answers that demonstrate the student understood the text, even if worded differently from the answer key.
- For essays, focus on whether the student addressed the prompt and showed effort.
- When a student is wrong, the feedback should be encouraging and point them toward the right idea, not just say "wrong."

Your output is STRICTLY structured JSON that conforms to the provided schema. You do not output any other text, explanation, or image. You only output coordinates and judgments; a separate program will draw marks on the page.

Coordinate system:
- All bounding boxes and points are NORMALIZED to [0.0, 1.0] where (0, 0) is the top-left of the page image and (1, 1) is the bottom-right.
- For each question, provide `answer_bbox` tightly around the student's written answer.
- Also provide `mark_anchor` — a single point in the RIGHT MARGIN (x around 0.92–0.96) at the same vertical center as the answer. This is where the teacher's mark (⭕ or ─) will be drawn.
"""
```

### 4.2 User Prompt Template

```python
USER_PROMPT_TEMPLATE = """Grade this student's homework.

RUBRIC:
{rubric_text}

ANSWER KEY: attached as image(s) after this message.

STUDENT SUBMISSION: attached as image(s) after the answer key.

For each question on the student's submission:
1. Identify the question number.
2. Locate the student's answer on the page.
3. Compare against the rubric and answer key.
4. Decide: correct, partially correct, or incorrect.
5. Give one sentence of kind, constructive feedback.
6. Provide the bounding box of the answer and the mark anchor point in the right margin.

Return JSON only."""
```

### 4.3 Response Schema (Pydantic)

In `grader/schema.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class QuestionResult(BaseModel):
    question_number: int
    page_index: int = Field(description="0-indexed page the answer is on")
    verdict: Literal["correct", "partial", "incorrect"]
    points_earned: float
    points_possible: float
    feedback: str = Field(description="One sentence, kind and age-appropriate")
    answer_bbox: list[float] = Field(
        description="[x_min, y_min, x_max, y_max], normalized 0-1",
        min_length=4, max_length=4,
    )
    mark_anchor: list[float] = Field(
        description="[x, y] in right margin, normalized 0-1",
        min_length=2, max_length=2,
    )

class GradingResponse(BaseModel):
    student_identifier: str = Field(description="Name/ID if visible on page, else filename")
    overall_score: float
    overall_possible: float
    overall_feedback: str
    questions: list[QuestionResult]
```

Pass this schema to Gemini via `response_schema=GradingResponse` and `response_mime_type="application/json"` so you get validated structured output, not prose to regex.

### 4.4 Building The Request

Pseudocode for `grader/gemini_client.py`:

```python
from google import genai
from google.genai import types

def grade_submission(rubric_text, answer_key_images, submission_images):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    parts = [
        types.Part.from_text(USER_PROMPT_TEMPLATE.format(rubric_text=rubric_text)),
        types.Part.from_text("--- ANSWER KEY ---"),
        *[types.Part.from_bytes(data=img, mime_type="image/png") for img in answer_key_images],
        types.Part.from_text("--- STUDENT SUBMISSION ---"),
        *[types.Part.from_bytes(data=img, mime_type="image/png") for img in submission_images],
    ]

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GradingResponse,
            temperature=0.1,  # deterministic grading
            thinking_config=types.ThinkingConfig(thinking_budget=2048),
        ),
    )
    return GradingResponse.model_validate_json(response.text)
```

Note: check the current `google-genai` SDK docs while implementing — the API surface has changed a few times. The spirit above is correct; the exact method names may need adjusting. If `thinking_config` isn't supported in the version you install, drop it.

---

## 5. PDF → PNG Conversion

In `grader/pdf_utils.py`:

- Use `pdf2image.convert_from_path(pdf_path, dpi=200)` → list of PIL Images.
- Save each page as PNG in the student's results folder: `page_1_original.png`, `page_2_original.png`, etc.
- Also return the raw bytes (for the Gemini call) and the PIL Image objects (for drawing overlays).
- Keep track of each page's `(width, height)` in pixels for the overlay step.

---

## 6. Overlay Drawing

In `grader/overlay.py`:

```python
from PIL import Image, ImageDraw, ImageFont

def draw_marks(original_png_path, page_results, page_index, output_path):
    """
    page_results: list of QuestionResult for THIS page only.
    """
    img = Image.open(original_png_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W, H = img.size

    for q in page_results:
        ax, ay = q.mark_anchor
        cx, cy = int(ax * W), int(ay * H)
        radius = int(min(W, H) * 0.025)  # ~2.5% of smaller dim

        if q.verdict == "correct":
            # Red circle (teacher style)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=(220, 30, 30, 255),
                width=4,
            )
        elif q.verdict == "incorrect":
            # Red dash
            draw.line(
                [(cx - radius, cy), (cx + radius, cy)],
                fill=(220, 30, 30, 255),
                width=4,
            )
        else:  # partial
            # Half circle / check-minus style — draw a partial arc
            draw.arc(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                start=0, end=180,
                fill=(220, 30, 30, 255),
                width=4,
            )

        # Small question number label next to the mark
        draw.text(
            (cx + radius + 4, cy - radius),
            f"Q{q.question_number}",
            fill=(220, 30, 30, 255),
        )

    composed = Image.alpha_composite(img, overlay)
    composed.convert("RGB").save(output_path, "PNG")
```

Key points:
- Always draw on a transparent overlay, then composite. Never modify the original pixels in place.
- Mark size is proportional to page size so it looks consistent at different DPIs.
- If `mark_anchor` coordinates from Gemini look off (e.g., outside [0,1] or clearly in the wrong place), clamp to a sensible right-margin position: `x = 0.94, y = (bbox_y_min + bbox_y_max) / 2`. Implement this fallback.

---

## 7. Main Driver (`grade.py`)

```
Usage: python grade.py <assignment_folder> [--model gemini-2.5-pro] [--workers 3]
```

Logic:

1. Validate folder has `rubric.txt`, `answer_key.pdf`, `submissions/*.pdf`.
2. Read `rubric.txt`. Convert `answer_key.pdf` → list of PNG bytes (cached, done once).
3. Create `results/<assignment_name>/`.
4. Load or create `progress.json`. Skip already-completed students.
5. For each remaining student PDF:
   a. Convert PDF → PNGs, save originals to student's folder.
   b. Call Gemini with rubric + answer key images + submission images.
   c. Save raw JSON response to `response.json`.
   d. For each page, draw overlays → `page_N_marked.png`.
   e. Write `feedback.txt` (human-readable summary).
   f. Update `progress.json`.
6. When all done, write `summary.csv`:
   ```
   student,score,possible,percentage,status
   alice,18,20,90.0,complete
   bob,14,20,70.0,complete
   ```

Use `concurrent.futures.ThreadPoolExecutor` with `max_workers=3` (keep low — Gemini rate limits are generous but not infinite). Wrap each student in try/except so one failure doesn't kill the batch; mark failed students in `progress.json` with the error message.

Use `tqdm` for a progress bar.

---

## 8. Local Viewer (`viewer.py`)

A minimal Flask app. One page, no build step.

```
Usage: python viewer.py results/reading_hw1/
```

Routes:
- `/` → lists all students with their scores. Click a name to see their submission.
- `/student/<name>` → shows, for each page:
  - Left column: `page_N_original.png`
  - Right column: `page_N_marked.png`
  - Below: per-question feedback table
- `/image/<student>/<filename>` → serves PNGs from the results folder.

Single HTML template with inline CSS is fine. Keep it under 200 lines total. The goal is "open browser, see the graded work," not a polished UI.

Auto-open the browser on start: `webbrowser.open("http://localhost:5000")`.

---

## 9. Example Rubric Format

`assignments/reading_hw1/rubric.txt`:

```
Assignment: Reading Comprehension - "The Lost Kitten"
Grade Level: 3rd grade
Total Points: 20

Question 1 (4 points): Who is the main character?
- Full credit: Names the main character correctly.
- Partial: Names a character from the story but not the main one.
- No credit: Blank, or names someone not in the story.

Question 2 (4 points): Where does the story take place?
- Full credit: Identifies the setting (the park / the neighborhood).
- Partial: Vague but related ("outside", "a place with trees").
- No credit: Wrong or blank.

Question 3 (6 points): What problem did the kitten face? Answer in 1-2 sentences.
- Full credit: Clearly describes the problem (kitten was lost / couldn't find its way home).
- Partial: Mentions the kitten was sad or alone but doesn't identify the core problem.
- No credit: Off-topic or blank.

Question 4 (6 points): How would you feel if you found a lost kitten? Why?
- Full credit: Expresses a feeling AND gives a reason. Any reasonable answer counts.
- Partial: Expresses a feeling but no reason, or vice versa.
- No credit: Blank or unrelated.
- Note: Accept invented spellings at grade level. Do not penalize minor spelling errors.
```

Keep the rubric as plain text — it's easier to edit and Gemini handles it fine.

---

## 10. `.gitignore`

```
.env
__pycache__/
*.pyc
results/
venv/
.venv/
*.egg-info/
```

Don't commit API keys or student data.

---

## 11. README.md Content

The README should cover, in this order:
1. One-paragraph overview.
2. Install steps (Python ≥ 3.10, Poppler, `pip install -r requirements.txt`).
3. Get a Gemini API key from https://aistudio.google.com/apikey, put it in `.env`.
4. How to structure an assignment folder.
5. How to run `grade.py`.
6. How to run `viewer.py`.
7. Troubleshooting: what to do if PDFs don't convert (Poppler missing), if the schema validation fails (usually means the model ignored format — retry), if coordinates are off (the fallback clamping should catch most cases).

---

## 12. Order Of Implementation (For Claude Code)

Build in this order so each step is testable:

1. **`requirements.txt` + `.env.example` + folder structure + README skeleton.**
2. **`grader/pdf_utils.py`** — prove PDF → PNG works on a sample file before anything else.
3. **`grader/schema.py`** — define Pydantic models.
4. **`grader/prompts.py`** — drop in the prompt strings.
5. **`grader/gemini_client.py`** — single function, test it on ONE hand-crafted submission and `print(response.model_dump_json(indent=2))`. Confirm the JSON shape is right before moving on.
6. **`grader/overlay.py`** — test with a fake `QuestionResult` list on a real page image. Confirm marks appear in the right margin.
7. **`grade.py`** — wire it all together, no concurrency yet. Grade one student end-to-end.
8. **Add `ThreadPoolExecutor` + `progress.json` + `tqdm`.**
9. **`viewer.py`** + templates.
10. **Polish the README.**

Do not skip step 5's sanity check. The Gemini response shape is the biggest unknown; verify it with one real call before building the rest of the pipeline on top.

---

## 13. What To Tell Me When You're Done

After Claude Code finishes, report:
- Any deviations from this spec (SDK method name changes, etc.).
- A screenshot or description of one graded sample.
- Whether the `mark_anchor` coordinates from Gemini landed in the right places or needed heavy clamping. This tells me if we need to invest in a coordinate-calibration step later.

---

## 14. Out Of Scope For This Demo

Explicitly do NOT build:
- Authentication, accounts, or multi-user support.
- A real web app or iPad app.
- Database storage (JSON + CSV is fine).
- Handwriting OCR preprocessing (Gemini Vision does this natively).
- Chrome extension, Gradescope integration, or any of the old repo's peripheral code.
- Cost tracking / API usage metering.
- Anything that requires deploying to a server.

This is a laptop demo. Keep the surface area small.