"""One quiz-session blueprint shared by the terminal quiz and the Roger app.

A session is: instant doc questions opening, streamed code questions in the
middle, one system-design question closing (when the session is big enough).
Both surfaces consume the same iterator so question quality and category
mix can never drift between them.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import networkx as nx

from roger.config import Config
from roger.docs import doc_questions
from roger.generator import interleave_questions, iter_questions
from roger.llm.router import DESIGN_NODE_ID, get_design_questions
from roger.models import Question
from roger.quiz import node_display_names


def quiz_blueprint(
    graph: nx.DiGraph,
    config: Config,
    count: int,
    node_picker,
) -> tuple[Iterator[Question], dict[str, str], int]:
    """(question stream, node display names, planned total) for one session.

    node_picker(graph, code_count) chooses which nodes to quiz — the CLI
    passes its god-node-weighted picker; tests can pass a stub.
    """
    difficulty = config.quiz.default_difficulty
    doc_qs = (
        doc_questions(count=max(1, count // 3), difficulty=difficulty, paths=config.docs.paths)
        if config.docs.enabled
        else []
    )
    design_share = 1 if count >= 4 else 0
    code_count = max(1, count - len(doc_qs) - design_share)
    node_ids = node_picker(graph, code_count)
    names = node_display_names(graph, node_ids)
    names[DESIGN_NODE_ID] = "system design (module map)"

    def _design_tail() -> Iterator[Question]:
        if design_share:
            yield from get_design_questions(graph, difficulty, design_share, config)

    stream = itertools.chain(
        interleave_questions(
            iter_questions(node_ids, graph, difficulty=difficulty, count=code_count, config=config),
            doc_qs,
        ),
        _design_tail(),
    )
    return stream, names, count
