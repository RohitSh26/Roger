"""Tests for generation: hashing, LLM client parsing, routing, caching, selection."""

from __future__ import annotations

import json

import networkx as nx
import pytest
import requests

from roger import generator
from roger.exceptions import ModelNotRegisteredError, OllamaNotRunningError
from roger.graph import get_node, get_subgraph, load_graph
from roger.llm import local, router
from tests.conftest import GRAPH_DATA, make_question

# --- hash_node ---------------------------------------------------------------


def _node_and_subgraph(graph: nx.DiGraph, node_id: str):
    return get_node(graph, node_id), get_subgraph(graph, node_id, hops=1)


def test_hash_is_stable_across_loads(graph_file) -> None:
    graph_a = load_graph(str(graph_file))
    graph_b = load_graph(str(graph_file))
    hash_a = generator.hash_node(*_node_and_subgraph(graph_a, "payments.charge"))
    hash_b = generator.hash_node(*_node_and_subgraph(graph_b, "payments.charge"))
    assert hash_a == hash_b
    assert len(hash_a) == 64  # sha-256 hex


def test_hash_changes_when_neighbor_changes(graph: nx.DiGraph) -> None:
    before = generator.hash_node(*_node_and_subgraph(graph, "payments.charge"))
    graph.nodes["db.connect"]["description"] = "Opens a pooled database connection"
    after = generator.hash_node(*_node_and_subgraph(graph, "payments.charge"))
    assert before != after


def test_hash_ignores_unrelated_changes(graph: nx.DiGraph) -> None:
    before = generator.hash_node(*_node_and_subgraph(graph, "auth.hash_password"))
    graph.nodes["payments.notify"]["description"] = "totally different"
    after = generator.hash_node(*_node_and_subgraph(graph, "auth.hash_password"))
    assert before == after


# --- strip_thinking ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),  # no thinking block
        ('<think>hmm</think>{"a": 1}', '{"a": 1}'),
        ('<think>line one\nline two</think>\n{"a": 1}', '{"a": 1}'),  # multiline
        ('<think>x</think>{"a": 1}<think>y</think>', '{"a": 1}'),  # multiple blocks
        ("<think>only thoughts</think>", ""),  # nothing left
        ('  <think>a</think>   {"a": 1}  ', '{"a": 1}'),  # whitespace trimmed
        ("no tags at all", "no tags at all"),
    ],
)
def test_strip_thinking(raw: str, expected: str) -> None:
    assert local.strip_thinking(raw) == expected


# --- call_local (mocked Ollama) ------------------------------------------------


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_call_local_strips_thinking_and_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    content = '<think>let me reason</think>{"questions": []}'
    payloads: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        payloads.append(json)
        return FakeResponse({"message": {"content": content}})

    monkeypatch.setattr(local.requests, "post", fake_post)
    assert local.call_local("prompt") == {"questions": []}
    # Thinking must be disabled or the model truncates its JSON answer.
    assert payloads[0]["think"] is False


def test_call_local_invalid_json_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    content = "<think>hmm</think>not json at all"
    monkeypatch.setattr(
        local.requests,
        "post",
        lambda *a, **k: FakeResponse({"message": {"content": content}}),
    )
    with pytest.raises(ValueError):
        local.call_local("prompt")


def test_call_local_connection_error_raises_ollama_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(local.requests, "post", boom)
    with pytest.raises(OllamaNotRunningError):
        local.call_local("prompt")


def test_call_local_404_raises_model_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local.requests, "post", lambda *a, **k: FakeResponse({}, status_code=404)
    )
    with pytest.raises(ModelNotRegisteredError):
        local.call_local("prompt")


# --- router ------------------------------------------------------------------


VALID_RAW = {
    "questions": [
        {
            "question": "What does charge() do?",
            "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "correct": "B",
            "explanation": "Because.",
        }
    ]
}


def test_parse_questions_valid() -> None:
    questions = router.parse_questions(VALID_RAW, node_id="n1", difficulty="medium", tier=1)
    assert len(questions) == 1
    q = questions[0]
    assert q.node_id == "n1"
    assert q.correct == "B"
    assert q.difficulty == "medium"
    assert q.tier == 1


def test_parse_questions_skips_malformed_items() -> None:
    raw = {
        "questions": [
            {"question": "missing options", "correct": "A"},
            {"question": "bad correct", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "correct": "E"},
            VALID_RAW["questions"][0],
        ]
    }
    questions = router.parse_questions(raw, node_id="n1", difficulty="hard", tier=1)
    assert len(questions) == 1


def test_parse_questions_no_valid_items_raises() -> None:
    with pytest.raises(ValueError):
        router.parse_questions({"questions": []}, node_id="n1", difficulty="medium", tier=1)
    with pytest.raises(ValueError):
        router.parse_questions({"nope": True}, node_id="n1", difficulty="medium", tier=1)


def test_router_simple_uses_templates_without_llm(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_network(*args, **kwargs):
        raise AssertionError("Tier 0 must not touch Ollama")

    monkeypatch.setattr(local, "is_ollama_running", no_network)
    monkeypatch.setattr(local, "call_local", no_network)

    node = get_node(graph, "payments.process_payment")
    questions = router.get_questions(node, graph, difficulty="simple", count=5)
    assert questions
    assert all(q.tier == 0 for q in questions)


def test_router_medium_raises_when_ollama_down(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local, "is_ollama_running", lambda *a, **k: False)
    node = get_node(graph, "payments.process_payment")
    with pytest.raises(OllamaNotRunningError) as excinfo:
        router.get_questions(node, graph, difficulty="medium", count=5)
    assert "ollama serve" in str(excinfo.value)


def test_router_medium_calls_local_llm(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts: list[str] = []

    def fake_call_local(prompt: str, **kwargs) -> dict:
        prompts.append(prompt)
        return VALID_RAW

    monkeypatch.setattr(local, "is_ollama_running", lambda *a, **k: True)
    monkeypatch.setattr(local, "call_local", fake_call_local)

    node = get_node(graph, "payments.charge")
    questions = router.get_questions(node, graph, difficulty="medium", count=3)
    assert len(questions) == 1
    assert questions[0].tier == 1
    # The prompt must carry the node context and its neighborhood.
    assert "payments.charge" in prompts[0]
    assert "db.connect" in prompts[0]
    assert "medium" in prompts[0]


def test_router_rejects_unknown_difficulty(graph: nx.DiGraph) -> None:
    node = get_node(graph, "payments.charge")
    with pytest.raises(ValueError):
        router.get_questions(node, graph, difficulty="impossible", count=5)


# --- generate_questions: caching ----------------------------------------------


@pytest.fixture
def graph_in_repo(in_tmp_repo, monkeypatch: pytest.MonkeyPatch) -> nx.DiGraph:
    """A graph whose .roger/ cache lives in an isolated tmp cwd."""
    path = in_tmp_repo / "graphify-out" / "graph.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(GRAPH_DATA), encoding="utf-8")
    return load_graph(str(path))


def test_generate_questions_caches_llm_output(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_llm(node, graph, difficulty, count, config=None):
        calls["count"] += 1
        return [make_question(node_id=node["id"], text=f"Q about {node['id']}?")]

    monkeypatch.setattr(generator, "get_questions_from_llm", fake_llm)

    node_ids = ["payments.charge", "db.connect"]
    first = generator.generate_questions(node_ids, graph_in_repo, "medium", count=2)
    assert calls["count"] == 2
    assert len(first) == 2

    second = generator.generate_questions(node_ids, graph_in_repo, "medium", count=2)
    assert calls["count"] == 2  # cache hit: no new LLM calls
    assert {q.question for q in second} == {q.question for q in first}


def test_generate_questions_regenerates_for_new_difficulty(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_llm(node, graph, difficulty, count, config=None):
        calls["count"] += 1
        return [make_question(node_id=node["id"], difficulty=difficulty)]

    monkeypatch.setattr(generator, "get_questions_from_llm", fake_llm)

    generator.generate_questions(["payments.charge"], graph_in_repo, "medium", count=1)
    generator.generate_questions(["payments.charge"], graph_in_repo, "hard", count=1)
    assert calls["count"] == 2  # difficulty mismatch is a cache miss


def test_generate_questions_regenerates_when_code_changes(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"count": 0}

    def fake_llm(node, graph, difficulty, count, config=None):
        calls["count"] += 1
        return [make_question(node_id=node["id"])]

    monkeypatch.setattr(generator, "get_questions_from_llm", fake_llm)

    generator.generate_questions(["payments.charge"], graph_in_repo, "medium", count=1)
    graph_in_repo.nodes["payments.charge"]["description"] = "changed implementation"
    generator.generate_questions(["payments.charge"], graph_in_repo, "medium", count=1)
    assert calls["count"] == 2  # new hash → regenerated


# --- select_questions ----------------------------------------------------------


def test_select_questions_prefers_god_nodes() -> None:
    pool = [
        make_question(node_id="minor.node", text="minor q1?"),
        make_question(node_id="minor.node", text="minor q2?"),
        make_question(node_id="god.node", text="god q1?"),
        make_question(node_id="god.node", text="god q2?"),
    ]
    selected = generator.select_questions(pool, count=2, god_node_ids=["god.node"])
    assert len(selected) == 2
    assert selected[0].node_id == "god.node"
    # Variety: round-robin means the second pick comes from the other node.
    assert selected[1].node_id == "minor.node"


def test_select_questions_dedupes_and_respects_count() -> None:
    pool = [
        make_question(node_id="a", text="same text?"),
        make_question(node_id="a", text="same text?"),
        make_question(node_id="b", text="other text?"),
    ]
    selected = generator.select_questions(pool, count=5, god_node_ids=[])
    texts = [q.question for q in selected]
    assert len(texts) == len(set(texts)) == 2


def test_select_questions_empty_pool() -> None:
    assert generator.select_questions([], count=5, god_node_ids=[]) == []


def test_call_local_surfaces_ollama_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"error": "request (53240 tokens) exceeds the available context size (8192 tokens)"}
    monkeypatch.setattr(
        local.requests, "post", lambda *a, **k: FakeResponse(body, status_code=400)
    )
    with pytest.raises(ValueError) as excinfo:
        local.call_local("prompt")
    assert "exceeds the available context size" in str(excinfo.value)


def test_build_prompt_caps_huge_neighborhoods(graph: nx.DiGraph) -> None:
    # Give payments.charge a god-node-sized neighborhood.
    for i in range(500):
        node_id = f"generated.caller_{i:03d}"
        graph.add_node(
            node_id,
            description="x" * 100,
            file=f"src/generated/caller_{i:03d}.py",
            community="payments",
        )
        graph.add_edge(node_id, "payments.charge")

    node = get_node(graph, "payments.charge")
    assert len(node["callers"]) > 400
    prompt = router.build_prompt(node, graph, "medium", 5)
    budget = router._subgraph_char_budget(router.DEFAULT_NUM_CTX)
    assert len(prompt) < budget + 4_000  # subgraph budget + template overhead
    assert "omitted" in prompt
    assert "more)" in prompt  # capped caller list

    # A larger context window buys a larger neighborhood budget.
    roomier = router.build_prompt(node, graph, "medium", 5, num_ctx=32_768)
    assert len(roomier) > len(prompt)


# --- lenient JSON parsing (small-model output salvage) --------------------------


def test_parse_json_lenient_handles_markdown_fences() -> None:
    content = 'Here you go:\n```json\n{"questions": [{"q": 1}]}\n```\nHope that helps!'
    assert local._parse_json_lenient(content) == {"questions": [{"q": 1}]}


def test_parse_json_lenient_repairs_truncated_array() -> None:
    # Simulates hitting num_predict mid-generation: second object cut off.
    content = (
        '{"questions": [{"question": "Q1?", "options": {"A": "a", "B": "b", '
        '"C": "c", "D": "d"}, "correct": "A", "explanation": "e"}, '
        '{"question": "Q2?", "options": {"A": "a", "B'
    )
    parsed = local._parse_json_lenient(content)
    assert len(parsed["questions"]) == 1
    assert parsed["questions"][0]["question"] == "Q1?"


def test_parse_json_lenient_hopeless_input_raises() -> None:
    with pytest.raises(ValueError):
        local._parse_json_lenient("total garbage with no braces")


def test_generate_questions_asks_small_per_node_batches(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[int] = []

    def fake_llm(node, graph, difficulty, count, config=None):
        asked.append(count)
        return [make_question(node_id=node["id"], text=f"Q {node['id']}?")]

    monkeypatch.setattr(generator, "get_questions_from_llm", fake_llm)
    node_ids = ["payments.charge", "db.connect", "auth.login", "auth.logout", "payments.notify"]
    generator.generate_questions(node_ids, graph_in_repo, "medium", count=5)
    assert asked == [2, 2, 2, 2, 2]  # not [5, 5, 5, 5, 5]


# --- parse normalization + retry ------------------------------------------------


def test_parse_questions_normalizes_list_options_and_answer_key() -> None:
    raw = {
        "questions": [
            {
                "question": "Q?",
                "options": ["first", "second", "third", "fourth"],
                "answer": "b) second",
            }
        ]
    }
    (q,) = router.parse_questions(raw, node_id="n1", difficulty="medium", tier=1)
    assert q.options == {"A": "first", "B": "second", "C": "third", "D": "fourth"}
    assert q.correct == "B"


def test_parse_questions_normalizes_lowercase_keys_and_text_answer() -> None:
    raw = {
        "questions": [
            {
                "question": "Q?",
                "options": {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta"},
                "correct": "gamma",
            }
        ]
    }
    (q,) = router.parse_questions(raw, node_id="n1", difficulty="medium", tier=1)
    assert q.options["C"] == "gamma"
    assert q.correct == "C"


def test_router_retries_transient_bad_shapes(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def flaky_call_local(prompt: str, **kwargs) -> dict:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return {"questions": [{"question": "bad", "options": "not a dict"}]}
        return VALID_RAW

    monkeypatch.setattr(local, "is_ollama_running", lambda *a, **k: True)
    monkeypatch.setattr(local, "call_local", flaky_call_local)

    node = get_node(graph, "payments.charge")
    questions = router.get_questions(node, graph, difficulty="medium", count=2)
    assert attempts["n"] == 3
    assert len(questions) == 1


def test_router_gives_up_after_three_attempts(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"n": 0}

    def always_bad(prompt: str, **kwargs) -> dict:
        attempts["n"] += 1
        return {"no_questions": True}

    monkeypatch.setattr(local, "is_ollama_running", lambda *a, **k: True)
    monkeypatch.setattr(local, "call_local", always_bad)

    node = get_node(graph, "payments.charge")
    with pytest.raises(ValueError):
        router.get_questions(node, graph, difficulty="medium", count=2)
    assert attempts["n"] == 3


# --- embedded Modelfile ----------------------------------------------------------


def test_embedded_modelfile_matches_checkout_copy() -> None:
    from pathlib import Path

    repo_modelfile = Path(__file__).resolve().parent.parent / "local" / "Modelfile"
    assert local.MODELFILE_CONTENT == repo_modelfile.read_text(encoding="utf-8")


def test_ensure_modelfile_writes_embedded_copy(in_tmp_repo) -> None:
    from roger import cli

    path = cli._ensure_modelfile()
    assert path == cli.ROGER_DIR / "Modelfile"
    assert path.read_text(encoding="utf-8") == local.MODELFILE_CONTENT


def test_ensure_modelfile_prefers_checkout_copy(in_tmp_repo) -> None:
    from pathlib import Path

    from roger import cli

    checkout = Path("local/Modelfile")
    checkout.parent.mkdir()
    checkout.write_text("FROM custom-model\n", encoding="utf-8")
    assert cli._ensure_modelfile() == checkout
    assert not (cli.ROGER_DIR / "Modelfile").exists()


# --- custom model support ---------------------------------------------------------


def test_call_local_404_custom_model_suggests_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        local.requests, "post", lambda *a, **k: FakeResponse({}, status_code=404)
    )
    with pytest.raises(ModelNotRegisteredError) as excinfo:
        local.call_local("prompt", model="llama3.2:3b")
    assert "ollama pull llama3.2:3b" in str(excinfo.value)
    # The default model keeps the roger init hint instead.
    with pytest.raises(ModelNotRegisteredError) as excinfo:
        local.call_local("prompt")
    assert "roger init" in str(excinfo.value)


class FakeProc:
    def __init__(self, returncode: int):
        self.returncode = returncode


def test_ensure_model_verifies_custom_model_without_create(
    in_tmp_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from roger import cli
    from roger.config import Config, ModelConfig

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return FakeProc(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._ensure_model(Config(model=ModelConfig(local="qwen2.5:7b")))

    assert commands == [["ollama", "show", "qwen2.5:7b"]]  # verify only — never create


def test_ensure_model_fails_when_custom_model_missing(
    in_tmp_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    import typer

    from roger import cli
    from roger.config import Config, ModelConfig

    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: FakeProc(returncode=1))
    with pytest.raises(typer.Exit):
        cli._ensure_model(Config(model=ModelConfig(local="not-pulled:latest")))


def test_ensure_model_registers_default_from_modelfile(
    in_tmp_repo, monkeypatch: pytest.MonkeyPatch
) -> None:
    from roger import cli
    from roger.config import Config

    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return FakeProc(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._ensure_model(Config())

    assert commands[0][:3] == ["ollama", "create", "roger-local"]
    assert (cli.ROGER_DIR / "Modelfile").exists()


def test_call_local_sends_num_ctx_option(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        payloads.append(json)
        return FakeResponse({"message": {"content": '{"questions": []}'}})

    monkeypatch.setattr(local.requests, "post", fake_post)
    local.call_local("prompt", num_ctx=16_384)
    assert payloads[0]["options"] == {"num_ctx": 16_384}
    local.call_local("prompt")  # without num_ctx the Modelfile governs
    assert "options" not in payloads[1]


# --- question quality: labels in prompts, giveaway rejection ---------------------


def test_build_prompt_uses_labels_not_slugs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_graph import REAL_SCHEMA_DATA

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(REAL_SCHEMA_DATA), encoding="utf-8")
    graph = load_graph(str(path))

    node = get_node(graph, "app_helper")  # label "helper", called by "main.py"
    prompt = router.build_prompt(node, graph, "medium", 2)
    assert "Name: helper" in prompt
    assert "main.py" in prompt          # caller shown by its label
    assert "app_main" not in prompt     # slug never leaks into the prompt
    assert "RULES" in prompt
    assert "self-answering" in prompt


def test_is_giveaway_rejects_name_echo_and_verbatim_answers() -> None:
    options = {"A": "usage", "B": "parses args", "C": "sends mail", "D": "retries"}
    assert router._is_giveaway("What is usage()?", options, "A", subject="usage()")

    options2 = {"A": "returns the parsed config object", "B": "x", "C": "y", "D": "z"}
    assert router._is_giveaway(
        "Does load_config returns the parsed config object?", options2, "A", subject="load_config"
    )

    options3 = {
        "A": "opens a pooled connection",
        "B": "closes idle sessions",
        "C": "reloads configuration",
        "D": "starts the scheduler",
    }
    assert not router._is_giveaway(
        "What does connect() do on startup?", options3, "A", subject="db.connect"
    )


def test_parse_questions_drops_self_answering_items() -> None:
    raw = {
        "questions": [
            {
                "question": "What is make_broker_deps?",
                "options": {"A": "make broker deps", "B": "b", "C": "c", "D": "d"},
                "correct": "A",
            },
            {
                "question": "What breaks if the search backend is down?",
                "options": {"A": "startup fails", "B": "silent retry", "C": "cache serves", "D": "nothing"},
                "correct": "B",
            },
        ]
    }
    questions = router.parse_questions(
        raw, node_id="n1", difficulty="medium", tier=1, subject="make_broker_deps"
    )
    assert len(questions) == 1
    assert "search backend" in questions[0].question


def test_generate_questions_skips_nodes_the_model_cannot_handle(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flaky_llm(node, graph, difficulty, count, config=None):
        if node["id"] == "payments.charge":
            raise ValueError("model kept writing giveaway questions")
        return [make_question(node_id=node["id"], text=f"Q {node['id']}?")]

    monkeypatch.setattr(generator, "get_questions_from_llm", flaky_llm)
    questions = generator.generate_questions(
        ["payments.charge", "db.connect"], graph_in_repo, "medium", count=2
    )
    assert len(questions) == 1  # quiz proceeds without the stubborn node
    assert questions[0].node_id == "db.connect"


def test_generate_questions_raises_when_every_node_fails(
    graph_in_repo: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fails(node, graph, difficulty, count, config=None):
        raise ValueError("no valid questions")

    monkeypatch.setattr(generator, "get_questions_from_llm", always_fails)
    with pytest.raises(ValueError):
        generator.generate_questions(["payments.charge"], graph_in_repo, "medium", count=2)


def test_is_giveaway_rejects_length_tell() -> None:
    # Correct option far longer/more detailed than every distractor — the
    # classic amateur-quiz giveaway.
    options = {
        "A": "it validates the card and then charges it via the payment gateway",
        "B": "logs output",
        "C": "sends email",
        "D": "retries once",
    }
    assert router._is_giveaway("What does the processor do?", options, "A", subject="processor")
    # Short name-style options must never trip the length check.
    names = {"A": "process_payment", "B": "notify", "C": "charge", "D": "refund"}
    assert not router._is_giveaway("Which code calls charge?", names, "A", subject="charge")


def test_hash_changes_when_question_style_version_bumps(
    graph: nx.DiGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = generator.hash_node(*_node_and_subgraph(graph, "payments.charge"))
    monkeypatch.setattr(generator, "QUESTION_STYLE_VERSION", 999)
    after = generator.hash_node(*_node_and_subgraph(graph, "payments.charge"))
    assert before != after  # style bumps must invalidate the team-wide cache


def test_prompt_is_certification_style(tmp_path) -> None:
    from tests.test_graph import REAL_SCHEMA_DATA

    path = tmp_path / "graph.json"
    path.write_text(json.dumps(REAL_SCHEMA_DATA), encoding="utf-8")
    graph = load_graph(str(path))
    node = get_node(graph, "app_main")
    prompt = router.build_prompt(node, graph, "hard", 2)
    assert "QUESTION TYPES" in prompt
    assert "Grounded" in prompt         # anti-hallucination rule present
    assert "Cover test" in prompt
    assert "THE MEMORY RULE" in prompt  # no structure-recall questions
    assert "hard" in prompt
    # The worked example is built from the node's own relationships, so a
    # model that copies it still writes about real code, not foreign names.
    assert "WORKED EXAMPLE" in prompt
    assert "Why does main.py hand part of its work to helper" in prompt


def test_is_giveaway_rejects_meta_format_leakage() -> None:
    # The model copied the worked example's format — answer included — into
    # the question text itself.
    options = {"A": "kb_agent.py", "B": "run_task", "C": "_raw_model_step", "D": "helpers"}
    leaked = (
        "Which of these must be updated to pass _is_provider_400()? "
        "Correct: _model_step(), because it calls it. Options: A: kb_agent.py"
    )
    assert router._is_giveaway(leaked, options, "A", subject="_model_step()")
    assert not router._is_giveaway(
        "Which of these must be updated when _is_provider_400() changes?",
        options, "A", subject="_model_step()",
    )


def test_normalize_options_trims_trailing_punctuation() -> None:
    raw = {"A": " knowledge_artifact.py; ", "B": "b", "C": "c", "D": "d,"}
    options = router._normalize_options(raw)
    assert options is not None
    assert options["A"] == "knowledge_artifact.py"
    assert options["D"] == "d"


def test_build_prompt_includes_real_source_when_available(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_graph import REAL_SCHEMA_DATA

    graph_json = tmp_path / "graph.json"
    graph_json.write_text(json.dumps(REAL_SCHEMA_DATA), encoding="utf-8")
    src = tmp_path / "app" / "main.py"
    src.parent.mkdir()
    src.write_text("def main():\n    return helper() or fallback()\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    graph = load_graph(str(graph_json))
    node = get_node(graph, "app_main")  # source_location absent → top of file
    prompt = router.build_prompt(node, graph, "medium", 2)
    assert "SOURCE (excerpt from app/main.py" in prompt
    assert "return helper() or fallback()" in prompt

    # Nodes whose file is gone simply get no SOURCE section.
    node_no_src = get_node(graph, "app_config")
    prompt2 = router.build_prompt(node_no_src, graph, "medium", 2)
    assert "SOURCE (excerpt" not in prompt2


# --- cloze questions (construction-grounded) --------------------------------------


SNIPPET = """\
def prune_stale(session, run_id):
    # remove candidates the generator no longer produces
    stale = session.query(Candidate).filter(Candidate.run_id != run_id)
    count = stale.count()
    stale.delete(synchronize_session=False)
    return count"""


def test_pick_cloze_line_skips_signature_and_comments() -> None:
    import random

    lines = SNIPPET.splitlines()
    for seed in range(10):
        index = router._pick_cloze_line(lines, rng=random.Random(seed))
        assert index is not None
        assert index != 0                      # never the def line
        assert not lines[index].strip().startswith("#")


def test_build_cloze_question_truth_by_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    import random

    monkeypatch.setattr(
        local,
        "call_local",
        lambda *a, **k: {
            "alternatives": [
                "stale.update({Candidate.active: False})",
                "session.expunge_all()",
                "stale.delete(synchronize_session='fetch')",
            ]
        },
    )
    from roger.config import Config

    node = {"id": "n1", "display": "prune_stale", "file": "app/prune.py"}
    rng = random.Random(3)
    question = router.build_cloze_question(node, SNIPPET, "medium", Config(), rng=rng)
    assert question is not None
    assert question.tier == 1
    assert "________" in question.snippet
    # The correct option is the real (removed) line — ground truth we control.
    real_lines = {" ".join(line.split()) for line in SNIPPET.splitlines()}
    assert question.options[question.correct] in real_lines
    assert question.options[question.correct] not in question.snippet
    assert len(set(question.options.values())) == 4


def test_build_cloze_question_rejects_duplicate_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from roger.config import Config

    monkeypatch.setattr(
        local, "call_local", lambda *a, **k: {"alternatives": ["same", "same", "same"]}
    )
    node = {"id": "n1", "display": "prune_stale", "file": "app/prune.py"}
    assert router.build_cloze_question(node, SNIPPET, "medium", Config()) is None


def test_cached_questions_without_snippet_field_still_load(in_tmp_repo) -> None:
    # Backward compat: cache entries written before Question.snippet existed.
    import sqlite3

    from roger import storage

    storage.cache_questions("bc1", "n1", "medium", [make_question()], "roger-local")
    with sqlite3.connect(storage.get_db_path("cache.db")) as conn:
        row = conn.execute("SELECT questions_json FROM question_cache WHERE hash='bc1'").fetchone()
        old_style = row[0].replace(', "snippet": ""', "")
        conn.execute(
            "UPDATE question_cache SET questions_json = ? WHERE hash = 'bc1'", (old_style,)
        )
        conn.commit()
    cached = storage.get_cached_questions("bc1")
    assert cached is not None
    assert cached[0].snippet == ""


def test_build_cloze_question_accepts_bare_list_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from roger.config import Config

    monkeypatch.setattr(
        local,
        "call_local",
        lambda *a, **k: ["stale.update({'active': False})", "session.rollback()", "return 0"],
    )
    node = {"id": "n1", "display": "prune_stale", "file": "app/prune.py"}
    question = router.build_cloze_question(node, SNIPPET, "medium", Config())
    assert question is not None
    assert len(set(question.options.values())) == 4


# --- mutant questions (spot the alteration, zero LLM) -----------------------------


def test_mutate_line_flips_operators() -> None:
    import random

    rng = random.Random(1)
    assert router._mutate_line("    if a == b and c:", rng) in {
        "    if a != b and c:", "    if a == b or c:",
    }
    assert router._mutate_line("    total = max(x, y)", rng) == "    total = min(x, y)"
    assert router._mutate_line("    pass", rng) is None


def test_build_mutant_question_no_llm_and_truth_by_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    def no_llm(*args, **kwargs):
        raise AssertionError("mutant questions must never call the model")

    monkeypatch.setattr(local, "call_local", no_llm)

    node = {"id": "n1", "display": "prune_stale", "file": "app/prune.py"}
    question = router.build_mutant_question(node, SNIPPET, "hard", rng=random.Random(5))
    assert question is not None
    assert question.tier == 0
    # The correct option is the altered line, visible in the shown snippet...
    shown_norms = {" ".join(line.split()) for line in question.snippet.splitlines()}
    assert question.options[question.correct] in shown_norms
    # ...and it is NOT a line of the real source.
    real_norms = {" ".join(line.split()) for line in SNIPPET.splitlines()}
    assert question.options[question.correct] not in real_norms
    # Distractors are genuine real lines.
    for key, value in question.options.items():
        if key != question.correct:
            assert value in real_norms
    # The explanation reveals the real line.
    assert question.explanation.startswith("The real code reads:")


def test_build_mutant_question_unmutable_snippet_returns_none() -> None:
    node = {"id": "n1", "display": "x", "file": "a.py"}
    snippet = "def x():\n    pass\n    return None"
    assert router.build_mutant_question(node, snippet, "hard") is None
