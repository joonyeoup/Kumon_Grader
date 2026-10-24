"""Pydantic models for Gemini structured output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuestionResult(BaseModel):
    question_id: str = Field(
        description='Identifier for this question or subquestion, e.g. "1", "1a", "2.3".'
    )
    parent_id: str | None = Field(
        default=None,
        description='If this is a subquestion, the parent question id (e.g. "1" for "1a"). Null for top-level.',
    )
    page_index: int = Field(description="0-indexed page the answer is on")
    verdict: Literal["correct", "partial", "incorrect"]
    points_earned: float
    points_possible: float
    feedback: str = Field(description="One sentence, kind and age-appropriate")
    transcribed_answer: str = Field(
        default="",
        description=(
            "Verbatim transcription of what the student actually wrote in the answer "
            "region — exactly the characters/words/numbers you see in their handwriting. "
            "Used for debugging misreads. Empty string if the student wrote nothing."
        ),
    )
    answer_bbox: list[float] = Field(
        description="[x_min, y_min, x_max, y_max], normalized 0-1",
        min_length=4,
        max_length=4,
    )
    error_bbox: list[float] | None = Field(
        default=None,
        description=(
            "When verdict is 'partial' or 'incorrect', a TIGHT [x_min, y_min, x_max, y_max] "
            "(normalized 0-1) around the SPECIFIC erroneous text — e.g. just the word that "
            "wasn't capitalized, the single wrong number, the missing punctuation. The mark "
            "is drawn directly above this region, so make it tight to the offending characters "
            "only. Null when verdict is 'correct'."
        ),
    )
    mark_anchor: list[float] = Field(
        description="[x, y] in right margin, normalized 0-1",
        min_length=2,
        max_length=2,
    )


class GradingResponse(BaseModel):
    student_identifier: str = Field(description="Name/ID if visible on page, else filename")
    overall_score: float
    overall_possible: float
    overall_feedback: str
    questions: list[QuestionResult]
