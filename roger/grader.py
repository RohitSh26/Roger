"""MCQ grading logic — pure functions, no I/O."""

from __future__ import annotations

from typing import Optional

from roger.models import Question, QuizAnswer


def grade_answer(question: Question, user_answer: Optional[str]) -> bool:
    """True if the user's letter matches the correct option (case-insensitive)."""
    if user_answer is None:
        return False
    return user_answer.strip().upper() == question.correct.strip().upper()


def score_answers(answers: list[QuizAnswer]) -> int:
    """Number of correct answers."""
    return sum(1 for a in answers if a.is_correct)


def has_passed(score: int, total: int, pass_threshold: int) -> bool:
    """A quiz passes when the score meets the configured threshold."""
    if total == 0:
        return True
    return score >= min(pass_threshold, total)
