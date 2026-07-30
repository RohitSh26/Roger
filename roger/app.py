"""The Roger app — quiz and ask in the browser, entirely on this machine.

Run via `roger app` (never directly): the CLI anchors the repo root, frees
a port, suppresses Streamlit's telemetry and first-run email prompt via
child-process env, and opens the browser itself.

Streamlit's execution model reruns this script top-to-bottom on every
interaction, so two rules hold throughout:
- Every LLM call sits behind a widget event (button/chat submit) with the
  result parked in st.session_state — tab bodies render on every rerun and
  must never generate at render time.
- Questions are precomputed behind a progress bar. Streamlit is synchronous
  per session: lazy generation can't overlap with thinking time the way the
  terminal's QuestionStream does; it would only turn one visible progress
  bar into a frozen UI between every question.

No CDN assets, ever: st.code/st.markdown are served by the local Streamlit
server; mermaid stays a fenced code block (the terminal shows it the same
way). Nothing on this page talks to anything but localhost.
"""

from __future__ import annotations

import streamlit as st

from roger.ask import answer_question
from roger.config import load_config
from roger.graph import load_graph
from roger.llm.router import ensure_backend
from roger.models import Question
from roger.quiz import language_for_file

OPTION_KEYS = ("A", "B", "C", "D")


@st.cache_resource(show_spinner=False)
def _load_world():
    config = load_config()
    graph = load_graph(config.graph.path)
    return config, graph


def _node_picker(graph, code_count: int) -> list[str]:
    from roger.cli import _pick_quiz_nodes

    config, _ = _load_world()
    return _pick_quiz_nodes(graph, code_count, config.graph.god_node_weight)


def _generate_session(count: int) -> None:
    from roger.session import quiz_blueprint

    config, graph = _load_world()
    ensure_backend(config)
    stream, names, total = quiz_blueprint(graph, config, count, _node_picker)
    questions: list[Question] = []
    bar = st.progress(0.0, text="Preparing your quiz…")
    for question in stream:
        questions.append(question)
        done = min(len(questions), total)
        bar.progress(done / total, text=f"Question {done} of {total} ready…")
    bar.empty()
    st.session_state.quiz = {
        "questions": questions,
        "names": names,
        "index": 0,
        "picks": {},  # index -> chosen key
    }


def _render_snippet(question: Question) -> None:
    if not question.snippet:
        return
    if question.language == "markdown":
        st.markdown(question.snippet)
    else:
        st.code(question.snippet, language=question.language or None, line_numbers=True)


def _quiz_tab() -> None:
    quiz = st.session_state.get("quiz")

    if quiz is None:
        st.markdown("#### Quiz yourself on this repository")
        count = st.select_slider(
            "Questions this session", options=[5, 8, 10, 15], value=5
        )
        if st.button("Start quiz", type="primary"):
            try:
                _generate_session(count)
            except Exception as exc:  # backend down/model missing → say it plainly
                st.error(str(exc))
                return
            st.rerun()
        return

    questions: list[Question] = quiz["questions"]
    index = quiz["index"]

    if index >= len(questions):
        score = sum(
            1 for i, pick in quiz["picks"].items() if pick == questions[i].correct
        )
        st.markdown(f"### Done — {score} of {len(questions)}")
        st.progress(score / max(1, len(questions)))
        if st.button("New quiz"):
            del st.session_state.quiz
            st.rerun()
        return

    question = questions[index]
    who = quiz["names"].get(question.node_id, question.node_id)
    st.caption(f"Question {index + 1} of {len(questions)} · {who}")
    st.markdown(question.question)
    _render_snippet(question)

    picked = quiz["picks"].get(index)
    if picked is None:
        for key in OPTION_KEYS:
            label = question.options.get(key, "")
            if not label:
                continue
            if st.button(f"{key})  {label}", key=f"q{index}-{key}", use_container_width=True):
                quiz["picks"][index] = key
                st.rerun()
        return

    correct = question.correct
    if picked == correct:
        st.success(f"Correct — {correct}) {question.options.get(correct, '')}")
    else:
        st.error(f"You picked {picked}. Correct: {correct}) {question.options.get(correct, '')}")
    if question.explanation:
        st.markdown(f"*{question.explanation}*")
    if st.button("Next question", type="primary"):
        quiz["index"] = index + 1
        st.rerun()


def _ask_tab() -> None:
    chat = st.session_state.setdefault("chat", [])
    for turn in chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    st.markdown("\n".join(f"- {s}" for s in turn["sources"]))

    prompt = st.chat_input("Ask about this codebase…")
    if not prompt:
        return
    chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    config, graph = _load_world()
    with st.chat_message("assistant"):
        with st.spinner("Reading the codebase…"):
            try:
                answer, sources = answer_question(prompt, graph, config)
            except Exception as exc:
                answer, sources = f"✗ {exc}", []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                st.markdown("\n".join(f"- {s}" for s in sources))
    chat.append({"role": "assistant", "text": answer, "sources": sources})


def main() -> None:
    st.set_page_config(page_title="Roger", page_icon="🎯", layout="centered")
    st.title("Roger")
    st.caption("Everything on this page runs on your machine. Nothing leaves it.")
    quiz_tab, ask_tab = st.tabs(["Quiz", "Ask"])
    with quiz_tab:
        _quiz_tab()
    with ask_tab:
        _ask_tab()


if __name__ == "__main__":
    main()
