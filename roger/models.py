"""Data models shared across Roger modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    node_id: str
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct: str             # "A" | "B" | "C" | "D"
    explanation: str
    difficulty: str          # "simple" | "medium" | "hard"
    tier: int                # 0 = template, 1 = local LLM
    snippet: str = ""        # source excerpt shown with the question (may be blanked for cloze)


@dataclass
class QuizAnswer:
    question: Question
    user_answer: Optional[str]
    is_correct: bool
    time_taken_secs: float


@dataclass
class QuizResult:
    session_type: str        # "guard" | "quiz" | "ask"
    answers: list[QuizAnswer] = field(default_factory=list)
    score: int = 0
    total: int = 0
    passed: bool = False
    commit_hash: Optional[str] = None
    module_scope: Optional[str] = None
    duration_secs: float = 0.0

    @property
    def weak_nodes(self) -> list[str]:
        return [a.question.node_id for a in self.answers if not a.is_correct]
