"""Web quiz: render a static, self-contained HTML page and record its results.

No server of any kind — the page is a local file opened in the browser
(same pattern as the report dashboard). Results come back through
`roger record <CODE>`: the page shows a short answer code at the end, and
this module re-grades it against the pending session saved at render time.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from roger.grader import grade_answer, has_passed, score_answers
from roger.models import Question, QuizAnswer, QuizResult

ROGER_DIR = Path(".roger")
QUIZ_HTML_PATH = ROGER_DIR / "quiz.html"
PENDING_PATH = ROGER_DIR / "quiz-pending.json"

_TEMPLATE_DIRS = [
    Path("templates"),  # running from a Roger checkout
    Path(__file__).resolve().parent.parent / "templates",  # editable install
]


def _load_template():
    # autoescape must stay OFF: the only substitutions are integers and the
    # question JSON blob, and entity-escaping the blob (< → &lt;) corrupts
    # it inside <script type="application/json">. Script-tag breakout is
    # prevented by the "</" → "<\\/" JSON guard instead.
    for directory in _TEMPLATE_DIRS:
        if (directory / "quiz.html.jinja").is_file():
            env = Environment(loader=FileSystemLoader(str(directory)), autoescape=False)
            return env.get_template("quiz.html.jinja")
    from jinja2 import Template

    # Wheel installs don't ship templates/ — fall back to the embedded copy.
    return Template(EMBEDDED_TEMPLATE, autoescape=False)


def render_quiz_html(
    questions: list[Question],
    session_type: str,
    pass_threshold: int,
    node_names: Optional[dict[str, str]] = None,
    out_path: Path = QUIZ_HTML_PATH,
    pending_path: Path = PENDING_PATH,
) -> Path:
    """Write the quiz page + the pending-session file, return the page path."""
    node_names = node_names or {}
    session_id = uuid.uuid4().hex[:8]

    payload = [
        {**asdict(q), "node_label": node_names.get(q.node_id, q.node_id)}
        for q in questions
    ]
    # "<\\/" keeps any literal </script> inside code snippets from
    # terminating the embedding script tag; it is still valid JSON.
    questions_json = json.dumps(payload).replace("</", "<\\/")

    html = _load_template().render(
        questions_json=questions_json,
        session_id=session_id,
        total=len(questions),
        pass_threshold=pass_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    pending_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "session_type": session_type,
                "pass_threshold": pass_threshold,
                "questions": [asdict(q) for q in questions],
            }
        ),
        encoding="utf-8",
    )
    return out_path


def record_answer_code(code: str, pending_path: Path = PENDING_PATH) -> QuizResult:
    """Re-grade a web session from its answer code and return the result.

    The code is the letters the developer picked, in order (e.g. "BCADB"),
    exactly as the finished quiz page displays it.
    """
    if not pending_path.is_file():
        raise ValueError(
            "No pending web quiz found. Run 'roger quiz --web' first — the "
            "answer code only works for the most recent web session."
        )
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    questions = [Question(**q) for q in pending["questions"]]

    letters = code.strip().upper().replace("-", "")
    if len(letters) != len(questions) or any(c not in "ABCD" for c in letters):
        raise ValueError(
            f"Answer code {code!r} does not match the pending quiz "
            f"({len(questions)} questions, letters A-D only)."
        )

    answers = [
        QuizAnswer(
            question=question,
            user_answer=letter,
            is_correct=grade_answer(question, letter),
            time_taken_secs=0.0,
        )
        for question, letter in zip(questions, letters)
    ]
    score = score_answers(answers)
    result = QuizResult(
        session_type=pending.get("session_type", "quiz"),
        answers=answers,
        score=score,
        total=len(questions),
        passed=has_passed(score, len(questions), int(pending.get("pass_threshold", 3))),
        commit_hash=None,
        module_scope=None,
        duration_secs=0.0,
    )
    pending_path.unlink(missing_ok=True)
    return result


# The full page template also lives at templates/quiz.html.jinja for
# checkout installs; a test keeps the two copies identical.
EMBEDDED_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roger — Quiz</title>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 900px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  .meta { color: #8b949e; font-size: .85rem; letter-spacing: .02em; }
  .progress { height: 6px; background: #21262d; border-radius: 3px; margin: .75rem 0 2rem; }
  .progress b { display: block; height: 100%; background: #3fb950; border-radius: 3px;
                transition: width .25s; }
  h2 { font-size: 1.15rem; font-weight: 600; margin: 0 0 1rem; }
  .code { display: flex; background: #161b22; border: 1px solid #30363d;
          border-radius: 8px; margin: 0 0 1.25rem; overflow-x: auto; }
  .gutter { user-select: none; text-align: right; color: #6e7681; padding: 12px 0 12px 12px;
            font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre; }
  .code pre { margin: 0; padding: 12px; flex: 1;
              font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
  .code code { background: transparent; padding: 0; white-space: pre; }
  .opt { display: block; width: 100%; text-align: left; margin: .5rem 0; padding: .7rem .9rem;
         background: #161b22; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; cursor: pointer; }
  .opt:hover:not(:disabled) { border-color: #58a6ff; }
  .opt:disabled { cursor: default; }
  .opt .key { color: #8b949e; font-weight: 700; margin-right: .5rem; }
  .opt.right { border-color: #3fb950; background: #12261e; }
  .opt.wrong { border-color: #f85149; background: #2d1214; }
  .feedback { margin: 1rem 0; padding: .8rem 1rem; border-radius: 8px; display: none; }
  .feedback.ok { display: block; background: #12261e; border: 1px solid #3fb950; }
  .feedback.no { display: block; background: #2d1214; border: 1px solid #f85149; }
  .explain { color: #8b949e; margin-top: .35rem; }
  .next { margin-top: .5rem; padding: .6rem 1.4rem; background: #238636; color: #fff;
          border: 0; border-radius: 8px; font-size: .95rem; cursor: pointer; display: none; }
  .summary { display: none; }
  .score { font-size: 2.4rem; font-weight: 700; margin: .5rem 0; }
  .pass { color: #3fb950; } .fail { color: #f85149; }
  .weak { margin: 1rem 0; padding-left: 1.2rem; color: #8b949e; }
  .record { margin-top: 1.5rem; padding: 1rem; background: #161b22;
            border: 1px solid #30363d; border-radius: 8px; }
  .record code { font: 14px ui-monospace, Menlo, monospace; background: #0d1117;
                 padding: .35rem .6rem; border-radius: 6px; }
  .copy { margin-left: .6rem; padding: .3rem .8rem; background: #21262d; color: #e6edf3;
          border: 1px solid #30363d; border-radius: 6px; cursor: pointer; }
  kbd { color: #8b949e; }
</style>
</head>
<body>
<div class="wrap">
  <div class="meta">Roger · code comprehension quiz · <span id="counter"></span></div>
  <div class="progress"><b id="bar" style="width:0%"></b></div>

  <div id="quiz">
    <div class="meta" id="nodeLabel"></div>
    <h2 id="questionText"></h2>
    <div class="code" id="codeBlock" style="display:none">
      <div class="gutter" id="gutter"></div>
      <pre><code id="snippet"></code></pre>
    </div>
    <div id="options"></div>
    <div class="feedback" id="feedback"></div>
    <button class="next" id="nextBtn">Next &nbsp;<kbd>↵</kbd></button>
    <p class="meta">Keys: <kbd>A</kbd> <kbd>B</kbd> <kbd>C</kbd> <kbd>D</kbd> to answer,
       <kbd>Enter</kbd> for next</p>
  </div>

  <div class="summary" id="summary">
    <div class="meta">Session complete</div>
    <div class="score" id="scoreLine"></div>
    <div id="weakWrap" style="display:none">
      <div class="meta">Review these:</div>
      <ul class="weak" id="weakList"></ul>
    </div>
    <div class="record">
      Record this session in your history:<br><br>
      <code id="recordCmd"></code>
      <button class="copy" id="copyBtn">copy</button>
    </div>
  </div>
</div>

<script id="quiz-data" type="application/json">{{ questions_json }}</script>
<script>
(function () {
  var questions = JSON.parse(document.getElementById("quiz-data").textContent);
  var passThreshold = {{ pass_threshold }};
  var current = 0, picks = [], score = 0, answered = false;

  var el = function (id) { return document.getElementById(id); };

  function show() {
    var q = questions[current];
    answered = false;
    el("counter").textContent = "question " + (current + 1) + " of " + questions.length;
    el("bar").style.width = (100 * current / questions.length) + "%";
    el("nodeLabel").textContent = q.node_label || q.node_id;
    el("questionText").textContent = q.question;

    if (q.snippet) {
      var lines = q.snippet.split("\n");
      el("gutter").textContent = lines.map(function (_, i) { return i + 1; }).join("\n");
      var codeEl = el("snippet");
      codeEl.textContent = q.snippet;
      codeEl.className = q.language ? "language-" + q.language : "";
      // hljs marks elements it has processed and refuses to re-highlight
      // them — clear the marker or only question 1 gets colors.
      delete codeEl.dataset.highlighted;
      if (window.hljs) { try { hljs.highlightElement(codeEl); } catch (e) {} }
      el("codeBlock").style.display = "flex";
    } else {
      el("codeBlock").style.display = "none";
    }

    var wrap = el("options");
    wrap.innerHTML = "";
    ["A", "B", "C", "D"].forEach(function (key) {
      var b = document.createElement("button");
      b.className = "opt";
      b.dataset.key = key;
      b.innerHTML = '<span class="key">' + key + ')</span>';
      b.appendChild(document.createTextNode(q.options[key]));
      b.onclick = function () { answer(key); };
      wrap.appendChild(b);
    });
    el("feedback").className = "feedback";
    el("nextBtn").style.display = "none";
  }

  function answer(key) {
    if (answered) return;
    answered = true;
    var q = questions[current];
    picks.push(key);
    var ok = key === q.correct;
    if (ok) score++;
    var buttons = el("options").children;
    for (var i = 0; i < buttons.length; i++) {
      var b = buttons[i];
      b.disabled = true;
      if (b.dataset.key === q.correct) b.className = "opt right";
      else if (b.dataset.key === key) b.className = "opt wrong";
    }
    var f = el("feedback");
    f.className = "feedback " + (ok ? "ok" : "no");
    f.innerHTML = (ok ? "✓ Correct" : "✗ Incorrect — correct answer: " + q.correct) +
      (q.explanation ? '<div class="explain"></div>' : "");
    if (q.explanation) f.querySelector(".explain").textContent = q.explanation;
    el("nextBtn").style.display = "inline-block";
    el("nextBtn").textContent = current + 1 < questions.length ? "Next  ↵" : "Finish  ↵";
  }

  function next() {
    if (!answered) return;
    current++;
    if (current < questions.length) { show(); return; }
    el("quiz").style.display = "none";
    el("bar").style.width = "100%";
    el("counter").textContent = "done";
    var passed = score >= Math.min(passThreshold, questions.length);
    var s = el("summary");
    s.style.display = "block";
    el("scoreLine").innerHTML = '<span class="' + (passed ? "pass" : "fail") + '">' +
      score + "/" + questions.length + (passed ? " — passed" : " — keep at it") + "</span>";
    var weak = [];
    questions.forEach(function (q, i) {
      if (picks[i] !== q.correct) weak.push(q.node_label || q.node_id);
    });
    if (weak.length) {
      el("weakWrap").style.display = "block";
      el("weakList").innerHTML = "";
      weak.filter(function (v, i, a) { return a.indexOf(v) === i; }).forEach(function (w) {
        var li = document.createElement("li");
        li.textContent = w;
        el("weakList").appendChild(li);
      });
    }
    el("recordCmd").textContent = "roger record " + picks.join("");
    el("copyBtn").onclick = function () {
      navigator.clipboard.writeText(el("recordCmd").textContent);
      el("copyBtn").textContent = "copied";
    };
  }

  document.addEventListener("keydown", function (e) {
    var key = e.key.toUpperCase();
    if (["A", "B", "C", "D"].indexOf(key) !== -1) answer(key);
    if (e.key === "Enter") next();
  });
  el("nextBtn").onclick = next;
  show();
})();
</script>
</body>
</html>
"""
