"""Shared fixtures: synthetic graphify output, no external dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from roger.graph import load_graph
from roger.models import Question

# Synthetic node-link graph mimicking graphify output: 11 nodes across four
# Leiden communities, with one deliberate "surprise" edge
# (payments.process_payment -> auth.check_token).
GRAPH_DATA = {
    "directed": True,
    "multigraph": False,
    "graph": {},
    "nodes": [
        {"id": "payments.process_payment", "description": "Processes a payment end to end", "file": "src/payments/processor.py", "community": "payments", "returns": "Receipt"},
        {"id": "payments.validate_card", "description": "Validates card number and expiry", "file": "src/payments/validate.py", "community": "payments", "returns": "bool"},
        {"id": "payments.charge", "description": "Charges the card via the gateway", "file": "src/payments/charge.py", "community": "payments", "returns": "ChargeResult"},
        {"id": "payments.refund", "description": "Refunds a completed charge", "file": "src/payments/refund.py", "community": "payments", "returns": "Refund"},
        {"id": "payments.notify", "description": "Sends a payment notification email", "file": "src/payments/notify.py", "community": "payments", "returns": "None"},
        {"id": "auth.login", "description": "Authenticates a user and opens a session", "file": "src/auth/login.py", "community": "auth", "returns": "Session"},
        {"id": "auth.logout", "description": "Closes the user session", "file": "src/auth/logout.py", "community": "auth", "returns": "bool"},
        {"id": "auth.hash_password", "description": "Hashes a password with bcrypt", "file": "src/auth/crypto.py", "community": "auth", "returns": "str"},
        {"id": "auth.check_token", "description": "Verifies a session token", "file": "src/auth/token.py", "community": "auth", "returns": "bool"},
        {"id": "db.connect", "description": "Opens a database connection", "file": "src/db/conn.py", "community": "db", "returns": "Connection"},
        {"id": "api.gateway", "description": "HTTP entry point routing requests", "file": "src/api/gateway.py", "community": "api", "returns": "Response"},
    ],
    "links": [
        {"source": "payments.process_payment", "target": "payments.validate_card"},
        {"source": "payments.process_payment", "target": "payments.charge"},
        {"source": "payments.process_payment", "target": "payments.notify"},
        {"source": "payments.process_payment", "target": "auth.check_token"},
        {"source": "payments.charge", "target": "db.connect"},
        {"source": "payments.refund", "target": "payments.charge"},
        {"source": "auth.login", "target": "auth.hash_password"},
        {"source": "auth.login", "target": "auth.check_token"},
        {"source": "auth.login", "target": "db.connect"},
        {"source": "auth.logout", "target": "auth.check_token"},
        {"source": "api.gateway", "target": "payments.process_payment"},
    ],
}

GRAPH_REPORT_MD = """\
# Graph Report

## God Nodes

- `payments.process_payment` (degree 5)
- `auth.check_token` (degree 3)

## Surprise Edges

- `payments.process_payment` -> `auth.check_token`: payments should not touch auth internals
"""


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    path = tmp_path / "graphify-out" / "graph.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(GRAPH_DATA), encoding="utf-8")
    return path


@pytest.fixture
def graph(graph_file: Path) -> nx.DiGraph:
    return load_graph(str(graph_file))


@pytest.fixture
def report_file(tmp_path: Path) -> Path:
    path = tmp_path / "graphify-out" / "GRAPH_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GRAPH_REPORT_MD, encoding="utf-8")
    return path


@pytest.fixture
def in_tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the test with cwd set to an empty tmp dir (so .roger/ lands there)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_question(
    node_id: str = "payments.process_payment",
    text: str = "What does `payments.process_payment()` return?",
    correct: str = "B",
    difficulty: str = "medium",
    tier: int = 1,
) -> Question:
    return Question(
        node_id=node_id,
        question=text,
        options={"A": "bool", "B": "Receipt", "C": "None", "D": "ChargeResult"},
        correct=correct,
        explanation="It returns a Receipt.",
        difficulty=difficulty,
        tier=tier,
    )
