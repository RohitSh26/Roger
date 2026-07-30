"""The Roger app — quiz and ask in the browser, entirely on this machine.

Run via `roger app` (never directly): the CLI anchors the repo root, frees
a port, suppresses Streamlit's telemetry and first-run email prompt via
child-process env, pins the light theme, and opens the browser itself.

Layout and states implement the Claude Design project "Roger: Codebase
quiz app" (Roger.dc.html); the visual system lives in roger/style.py.

Streamlit reruns this script top-to-bottom on every interaction, so two
rules hold throughout:
- Every LLM call sits behind a widget event with its result parked in
  st.session_state — tab bodies render on every rerun and must never
  generate at render time.
- Questions are precomputed behind a progress bar (Streamlit is
  synchronous per session; lazy generation would freeze the UI between
  questions instead of showing one honest bar).

No CDN assets, ever: system fonts, local Streamlit bundles, nothing else.
"""

from __future__ import annotations

import html
import re
import time
from pathlib import Path

import streamlit as st

from roger import freshness
from roger.ask import answer_question, explain_data, path_data
from roger.config import load_config
from roger.graph import candidate_code_nodes, get_god_nodes, load_graph
from roger.llm.router import ensure_backend
from roger.models import Question
from roger.quiz import language_for_file
from roger.style import STYLE

OPTION_KEYS = ("A", "B", "C", "D")


@st.cache_resource(show_spinner=False)
def _load_graph_cached(path: str, mtime: float):
    """Cached per (path, mtime): `roger update` invalidates it naturally."""
    graph = load_graph(path)
    symbols = candidate_code_nodes(graph)
    files = {str(graph.nodes[n].get("file") or "") for n in symbols}
    return graph, len(symbols), len(files)


def _load_world():
    # Config is NOT cached: it's one cheap TOML read, and caching it made
    # the app keep answering on a stale backend after `roger use …` while
    # the CLI picked up the switch instantly (field-hit).
    config = load_config()
    graph_path = Path(config.graph.path)
    mtime = graph_path.stat().st_mtime if graph_path.exists() else 0.0
    graph, symbols, files = _load_graph_cached(config.graph.path, mtime)
    return config, graph, symbols, files


def _node_picker(graph, code_count: int) -> list[str]:
    from roger.cli import _pick_quiz_nodes

    config, _, _, _ = _load_world()
    return _pick_quiz_nodes(graph, code_count, config.graph.god_node_weight)


def _generate_session(count: int) -> None:
    from roger.session import quiz_blueprint

    config, graph, _, _ = _load_world()
    ensure_backend(config)
    stream, names, total = quiz_blueprint(graph, config, count, _node_picker)
    questions: list[Question] = []
    bar = st.progress(0.0, text="Preparing your quiz…")
    reading = st.empty()
    for question in stream:
        questions.append(question)
        done = min(len(questions), total)
        bar.progress(done / total, text=f"Question {done} of {total} ready…")
        where = names.get(question.node_id, "")
        path = where.split(" (")[-1].rstrip(")") if " (" in where else where
        if path:
            reading.markdown(
                f'<div class="rog-scan" style="text-align:center">reading {path}</div>',
                unsafe_allow_html=True,
            )
    st.session_state.quiz = {
        "questions": questions,
        "names": names,
        "index": 0,
        "picks": {},  # index -> chosen key
    }


def _backend_label() -> str:
    config, _, _, _ = _load_world()
    if config.model.provider.startswith("azure-"):
        return f"Azure Foundry · {config.model.azure_deployment}"
    name = config.model.local.rsplit("/", 1)[-1]
    return f"Ollama · {name}"


def _indexed_note() -> str:
    config, _, _, _ = _load_world()
    commit = (freshness.built_at_commit(config.graph.path) or "")[:7]
    try:
        age_secs = time.time() - Path(config.graph.path).stat().st_mtime
    except OSError:
        return f"at `{commit}`" if commit else ""
    def plural(n: int, unit: str) -> str:
        n = max(1, int(n))
        return f"{n} {unit}{'s' if n != 1 else ''} ago"

    if age_secs < 3600:
        age = plural(age_secs // 60, "minute")
    elif age_secs < 86400:
        age = plural(age_secs // 3600, "hour")
    else:
        age = plural(age_secs // 86400, "day")
    return f"Indexed {age}" + (f" at <code>{commit}</code>" if commit else "")


# --------------------------------------------------------------------------- quiz


def _quiz_start() -> None:
    _, _, symbols, files = _load_world()
    st.markdown("# How well do you know this codebase?")
    st.markdown(
        f'<div class="rog-sub">Roger reads {files:,} files and {symbols:,} '
        "functions here. Questions come straight from your own code — "
        "nothing leaves this machine.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<div class="rog-caption">HOW MANY QUESTIONS</div>',
                unsafe_allow_html=True)
    try:
        count = st.segmented_control(
            "How many questions", options=[5, 10, 20], default=5,
            label_visibility="collapsed",
        )
    except AttributeError:  # older streamlit
        count = st.radio("How many questions", [5, 10, 20], horizontal=True,
                         label_visibility="collapsed")
    st.write("")
    with st.container(key="primary"):
        if st.button("Start the quiz"):
            st.session_state.generating = int(count or 5)
            st.rerun()
    st.markdown(
        '<div class="rog-caption">Takes about four minutes.</div>',
        unsafe_allow_html=True,
    )


def _option_state(index: int, letter: str, question: Question, picked: str | None) -> str:
    if picked is None:
        return "idle"
    if letter == picked and picked == question.correct:
        return "stateok"
    if letter == picked:
        return "stateno"
    if letter == question.correct:
        return "stateans"
    return "statemut"


def _quiz_question(quiz: dict) -> None:
    questions: list[Question] = quiz["questions"]
    index = quiz["index"]
    question = questions[index]
    who = quiz["names"].get(question.node_id, question.node_id)
    picked = quiz["picks"].get(index)

    name = who.split(" (")[0]
    file = " (".join(who.split(" (")[1:]).rstrip(")") if " (" in who else ""
    st.markdown(
        f'<div class="rog-caption"><b>{index + 1:02d} / {len(questions):02d}</b>'
        f" · <code>{name}</code>" + (f" · <code>{file}</code>" if file else "") + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="rog-q">{question.question}</div>', unsafe_allow_html=True)

    if question.snippet:
        if question.language == "markdown":
            with st.container(key=f"docpanel{index}"):
                st.markdown(question.snippet)
        else:
            language = question.language or language_for_file(file)
            with st.container(key=f"codepanel{index}"):
                st.markdown(
                    f'<div class="rog-file"><span>{file or name}</span>'
                    f"<span>{language.capitalize()}</span></div>",
                    unsafe_allow_html=True,
                )
                st.code(question.snippet, language=language or None, line_numbers=True)

    for letter in OPTION_KEYS:
        label = question.options.get(letter, "")
        if not label:
            continue
        state = _option_state(index, letter, question, picked)
        with st.container(key=f"opt{letter}{state}q{index}"):
            if st.button(label, key=f"btn{letter}q{index}") and picked is None:
                quiz["picks"][index] = letter
                st.rerun()

    if picked is None:
        st.markdown(
            '<div class="rog-caption">Pick one. You\'ll see why straight away.</div>',
            unsafe_allow_html=True,
        )
        return
    if question.explanation:
        st.markdown(
            f'<div class="rog-why"><em>{question.explanation}</em></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    with st.container(key="nextbtn"):
        if st.button("Next question", key=f"next{index}"):
            quiz["index"] = index + 1
            st.rerun()


def _quiz_end(quiz: dict) -> None:
    questions: list[Question] = quiz["questions"]
    picks: dict[int, str] = quiz["picks"]
    score = sum(1 for i, pick in picks.items() if pick == questions[i].correct)

    st.markdown(
        '<div class="rog-caption" style="margin-top:8px">THAT\'S THE LOT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"# {score} of {len(questions)}")
    st.progress(score / max(1, len(questions)))

    wrong_dirs: dict[str, int] = {}
    for i, question in enumerate(questions):
        if picks.get(i) != question.correct:
            name = quiz["names"].get(question.node_id, "")
            path = name.split(" (")[1].rstrip(")") if " (" in name else ""
            parent = str(Path(path).parent) if path else ""
            if parent and parent != ".":
                wrong_dirs[parent] = wrong_dirs.get(parent, 0) + 1
    if wrong_dirs:
        worst = max(wrong_dirs, key=lambda d: wrong_dirs[d])
        total_in = sum(
            1 for i, q in enumerate(questions)
            if worst in quiz["names"].get(q.node_id, "")
        )
        st.markdown(
            f'<div class="rog-sub">Shakiest ground: <code>{worst}/</code> — '
            f"{total_in - wrong_dirs[worst]} of {total_in}.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("Review the answers"):
        for i, question in enumerate(questions):
            ok = picks.get(i) == question.correct
            mark = "✓" if ok else "✗"
            st.markdown(
                f"**{mark} {i + 1}.** {question.question}\n\n"
                f"&nbsp;&nbsp;&nbsp;correct: **{question.correct})** "
                f"{question.options.get(question.correct, '')}"
            )
    with st.container(key="linknew"):
        if st.button("Start a new quiz"):
            del st.session_state.quiz
            st.rerun()


def _quiz_generating() -> None:
    count = st.session_state.pop("generating")
    st.write("")
    st.write("")
    try:
        _generate_session(count)
    except Exception as exc:  # backend down/model missing → say it plainly
        st.error(str(exc))
        return
    st.rerun()


def _quiz_tab() -> None:
    quiz = st.session_state.get("quiz")
    if st.session_state.get("generating"):
        _quiz_generating()
    elif quiz is None:
        _quiz_start()
    elif quiz["index"] >= len(quiz["questions"]):
        _quiz_end(quiz)
    else:
        _quiz_question(quiz)


# ---------------------------------------------------------------------------- ask


_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def _render_answer(text: str) -> None:
    """Markdown, but code fences become real st.code panels — line numbers,
    the dark surface, the design's token palette. st.markdown's own fence
    rendering has no line numbers."""
    pos = 0
    for match in _FENCE_RE.finditer(text):
        before = text[pos:match.start()].strip()
        if before:
            st.markdown(before)
        st.code(match.group(2).rstrip(), language=match.group(1) or None,
                line_numbers=True)
        pos = match.end()
    rest = text[pos:].strip()
    if rest:
        st.markdown(rest)


def _suggestions(graph) -> list[str]:
    from roger.graph import looks_like_test_file

    names = [
        str(graph.nodes[n].get("display", n)).removesuffix("()")
        for n in get_god_nodes(graph, top_n=6)
        if not looks_like_test_file(str(graph.nodes[n].get("file") or ""))
    ][:2]
    ideas = []
    if names:
        ideas.append(f"What calls {names[0]}?")
    if len(names) > 1:
        ideas.append(f"How does {names[1]} work?")
    return ideas


def _ask_view(prompt: str | None) -> None:
    config, graph, _, _ = _load_world()
    chat = st.session_state.setdefault("chat", [])
    repo = Path.cwd().name

    if not chat:
        st.markdown(f"# Ask about {repo}")
        st.markdown(
            f'<div class="rog-sub">Roger reads the files it needs and shows '
            f"you which ones. {_indexed_note()}. "
            "Everything stays on this machine.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        ideas = _suggestions(graph)
        columns = st.columns([1, 1, 2])
        for i, idea in enumerate(ideas):
            with columns[i]:
                with st.container(key=f"link{i}"):
                    if st.button(idea, key=f"linkbtn{i}"):
                        st.session_state.pending_question = idea
                        st.rerun()

    for i, turn in enumerate(chat):
        wrapper = "turnuser" if turn["role"] == "user" else "turnroger"
        with st.container(key=f"{wrapper}{i}"):
            with st.chat_message(turn["role"]):
                if turn["role"] == "assistant":
                    _render_answer(turn["text"])
                else:
                    st.markdown(turn["text"])
                if turn.get("sources"):
                    with st.expander(f"Sources · {len(turn['sources'])} files"):
                        st.markdown("\n".join(f"- {s}" for s in turn["sources"]))

    pending = st.session_state.pop("pending_question", None)
    prompt = prompt or pending
    if not prompt:
        return

    chat.append({"role": "user", "text": prompt})
    with st.container(key=f"turnuser{len(chat)}"):
        with st.chat_message("user"):
            st.markdown(prompt)
    with st.container(key=f"turnroger{len(chat) + 1}"):
        with st.chat_message("assistant"):
            thinking = st.empty()
            thinking.markdown(
                '<div class="rog-thinking">Reading the codebase…</div>',
                unsafe_allow_html=True,
            )
            try:
                answer, sources = answer_question(prompt, graph, config)
            except Exception as exc:
                answer, sources = f"✗ {exc}", []
            thinking.empty()
            _render_answer(answer)
            if sources:
                with st.expander(f"Sources · {len(sources)} files"):
                    st.markdown("\n".join(f"- {s}" for s in sources))
    chat.append({"role": "assistant", "text": answer, "sources": sources})


# --------------------------------------------------------------------------- explore


def _explore_options(graph) -> list[str]:
    """Searchable picker entries — the same code-symbol universe the
    index uses, by display name (duplicates resolve to every match)."""
    return sorted({str(graph.nodes[n].get("display") or n) for n in candidate_code_nodes(graph)})


def _explore_goto(label: str) -> None:
    """Neighbor/hop click → explain that symbol (runs before the rerun)."""
    st.session_state.explore_a = label
    st.session_state.explore_b = None


def _edge_row(edge: dict, known: set[str], key: str) -> None:
    """One relationship row — a button when the target is navigable,
    a quiet static row otherwise."""
    inferred = " · inferred" if edge["confidence"] == "INFERRED" else ""
    if edge["display"] in known:
        with st.container(key=f"nb{key}"):
            st.button(
                f"{edge['display']} · {edge['relation']}{inferred}",
                key=f"nbb{key}", help=edge["file"] or None,
                on_click=_explore_goto, args=(edge["display"],),
            )
    else:
        st.markdown(
            f'<div class="rog-edge">{html.escape(edge["display"])}'
            f'<span class="rog-rel"> · {html.escape(edge["relation"])}{inferred}</span></div>',
            unsafe_allow_html=True,
        )


def _explore_ego(graph, name: str, known: set[str]) -> None:
    for node in explain_data(graph, name):
        kind = (
            f'<span class="rog-kind">{html.escape(node["kind"])}</span>'
            if node["kind"] else ""
        )
        meta = html.escape(node["file"] or "?")
        if node["location"]:
            meta += f" · {html.escape(node['location'])}"
        total = len(node["incoming"]) + len(node["outgoing"])
        st.markdown(
            f'<div class="rog-ego"><div class="rog-ego-name">'
            f'{html.escape(node["display"])}{kind}</div>'
            f'<div class="rog-ego-meta">{meta} · {total} connections</div></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        for column, edges, title, tag in (
            (left, node["incoming"], "Used by", "i"),
            (right, node["outgoing"], "Uses", "o"),
        ):
            with column:
                st.markdown(
                    f'<div class="rog-caption">{title.upper()} ({len(edges)})</div>',
                    unsafe_allow_html=True,
                )
                if not edges:
                    st.markdown(
                        '<div class="rog-edge">— nothing recorded</div>',
                        unsafe_allow_html=True,
                    )
                for j, edge in enumerate(edges[:30]):
                    _edge_row(edge, known, f"{tag}{node['id']}{j}")
                if len(edges) > 30:
                    st.markdown(
                        f'<div class="rog-caption">+{len(edges) - 30} more</div>',
                        unsafe_allow_html=True,
                    )


def _explore_path(graph, start: str, end: str, known: set[str]) -> None:
    result = path_data(graph, start, end)
    if "error" in result:
        st.markdown(
            f'<div class="rog-sub">{html.escape(result["error"])}</div>',
            unsafe_allow_html=True,
        )
        return
    hops, links = result["hops"], result["links"]
    st.markdown(
        f'<div class="rog-caption">SHORTEST CONNECTION — '
        f'{len(links)} HOP{"S" if len(links) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    from roger.codemap import path_map_svg

    st.markdown(
        f'<div class="rog-map">{path_map_svg(graph, hops, links)}</div>',
        unsafe_allow_html=True,
    )
    for i, hop in enumerate(hops):
        if hop["display"] in known:
            with st.container(key=f"hop{i}"):
                st.button(
                    hop["display"], key=f"hopb{i}", help=hop["file"] or None,
                    on_click=_explore_goto, args=(hop["display"],),
                )
        else:
            st.markdown(
                f'<div class="rog-edge rog-hopdead">{html.escape(hop["display"])}</div>',
                unsafe_allow_html=True,
            )
        if i < len(links):
            link = links[i]
            # ↓ = the semantic edge points down the chain, ↑ = it points back.
            glyph = "↓" if link["forward"] else "↑"
            conf = (
                f' <span>{html.escape(link["confidence"].lower())}</span>'
                if link["confidence"] else ""
            )
            st.markdown(
                f'<div class="rog-hoplink">{glyph} '
                f'{html.escape(link["relation"])}{conf}</div>',
                unsafe_allow_html=True,
            )


def _explore_view() -> None:
    _, graph, _, _ = _load_world()
    st.markdown("# Walk the code graph")
    st.markdown(
        '<div class="rog-sub">Computed straight from the code graph — no '
        "model runs, nothing leaves this machine, and the answer is the "
        "same every time. Pick a symbol to see everything connected to "
        "it; add a second to trace how the two meet.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    options = _explore_options(graph)
    known = set(options)
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox(
            "Symbol", options, index=None, placeholder="Pick a symbol…",
            key="explore_a", label_visibility="collapsed",
        )
    with c2:
        b = st.selectbox(
            "Connect to", options, index=None,
            placeholder="Second symbol — traces the connection",
            key="explore_b", label_visibility="collapsed",
        )

    if not a and not b:
        st.write("")
        from roger.graph import looks_like_test_file

        starters = [
            d for d, file in (
                (
                    str(graph.nodes[n].get("display") or n),
                    str(graph.nodes[n].get("file") or ""),
                )
                for n in get_god_nodes(graph, top_n=12)
            ) if d in known and not looks_like_test_file(file)
        ][:3]
        if starters:
            st.markdown(
                '<div class="rog-caption">BUSIEST SYMBOLS — START HERE</div>',
                unsafe_allow_html=True,
            )
            columns = st.columns([1, 1, 2])
            for i, display in enumerate(starters):
                with columns[i]:
                    with st.container(key=f"linkg{i}"):
                        st.button(
                            display, key=f"gostart{i}",
                            on_click=_explore_goto, args=(display,),
                        )
        return
    if a and b:
        _explore_path(graph, a, b, known)
    elif a:
        _explore_ego(graph, a, known)
    else:
        _explore_ego(graph, b, known)


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="Roger", page_icon="🎯", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    repo = Path.cwd().name

    provider, model = _backend_label().split(" · ", 1)
    st.markdown(
        f'<div class="rog-head"><span class="rog-repo">roger · {repo}</span>'
        f'<span class="rog-badge"><b>{provider}</b>{model}</span></div>',
        unsafe_allow_html=True,
    )

    # Segmented nav instead of st.tabs — this is what lets st.chat_input
    # live at top level and truly pin to the bottom (design structure call).
    try:
        view = st.segmented_control(
            "view", ["Quiz", "Ask", "Explore"],
            default=st.session_state.get("view", "Quiz"),
            label_visibility="collapsed", key="viewpick",
        )
    except AttributeError:  # older streamlit
        view = st.radio("view", ["Quiz", "Ask", "Explore"], horizontal=True,
                        label_visibility="collapsed")
    view = view or st.session_state.get("view", "Quiz")
    st.session_state.view = view

    if view == "Quiz":
        _quiz_tab()
        return
    if view == "Explore":
        _explore_view()
        return
    prompt = st.chat_input("Ask anything about this codebase…")
    _ask_view(prompt)


if __name__ == "__main__":
    main()
