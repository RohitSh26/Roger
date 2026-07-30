"""Local semantic retrieval — invisible by design.

Reviewed shape: enriched code cards (name + file + signature/docstring/body
head) embedded via the already-required Ollama; vectors in a separate,
never-committed .roger/vectors.db; a full-index scan at query time (numpy —
already present via graphify — with a math.sumprod fallback); consent is
capability-sensed per machine (the model being present IS enablement; only a
refusal is recorded, in ~/.roger/state.json). Every failure degrades silently
to keyword-only retrieval — except an embedding-model digest mismatch, which
hard-disables the semantic channel and schedules a rebuild, because comparing
vectors across embedding spaces returns confidently wrong neighbors.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Callable, Optional

import requests

from roger import graph as g
from roger.config import ROGER_DIR, Config
from dataclasses import dataclass

EMBED_VERSION = 1
EMBED_MODEL = "nomic-embed-text"
VECTORS_PATH = ROGER_DIR / "vectors.db"
MACHINE_STATE_PATH = Path.home() / ".roger" / "state.json"
QUERY_DEADLINE_SECS = 0.5  # cold model loads keep loading server-side; next call is warm
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS cards (
    node_id      TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    card_text    TEXT NOT NULL,
    vec          BLOB
);
"""

# In-process matrix cache: (db mtime, node_ids, matrix)
_matrix_cache: tuple = (None, None, None)


# --- machine state (consent) --------------------------------------------------------


def _read_machine_state() -> dict:
    try:
        return json.loads(MACHINE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_machine_state(state: dict) -> None:
    try:
        MACHINE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MACHINE_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        pass


def embed_prompt_declined() -> bool:
    return bool(_read_machine_state().get("embed_prompt_declined"))


def record_declined() -> None:
    state = _read_machine_state()
    state["embed_prompt_declined"] = True
    _write_machine_state(state)


def clear_declined() -> None:
    state = _read_machine_state()
    state.pop("embed_prompt_declined", None)
    _write_machine_state(state)


# --- model capability sensing ---------------------------------------------------------


def model_digest(config: Config, timeout: float = 2) -> Optional[str]:
    """Digest of the embed model in Ollama, or None if absent/unreachable.

    Presence IS enablement — no flag anywhere. Pinned by digest so an
    upstream re-release under the same tag is detectable. Query-time
    callers keep the tight default; refresh/doctor paths pass a longer
    timeout — a busy Ollama answering /api/tags in 3s must not be
    mistaken for an absent model (field-observed flake).
    """
    try:
        resp = requests.get(f"{config.ollama.url}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    for model in resp.json().get("models", []):
        if str(model.get("name", "")).split(":")[0] == EMBED_MODEL:
            return str(model.get("digest") or "") or None
    return None


def _embed(texts: list[str], config: Config, timeout: float) -> Optional[list[list[float]]]:
    try:
        resp = requests.post(
            f"{config.ollama.url}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=timeout,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        return embeddings if isinstance(embeddings, list) and embeddings else None
    except (requests.RequestException, ValueError):
        return None


# --- cards -----------------------------------------------------------------------------


def card_text(node: dict) -> str:
    """What gets embedded: identity plus actual meaning.

    Name+path alone is semantically empty for code; the docstring and body
    head are where the meaning lives. Vector size is constant regardless of
    text length — richer cards cost nothing at rest.
    """
    head = g.get_source_snippet(node, max_lines=12)[:600]
    return f"{node.get('display', node['id'])}\n{node.get('file', '')}\n{head}".strip()


def card_hash(text: str, digest: str) -> str:
    payload = f"embed-v{EMBED_VERSION}\n{digest}\n{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(VECTORS_PATH, timeout=5)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript(_SCHEMA)
    return conn


def load_card_texts() -> dict[str, str]:
    """Card texts for the keyword channel — docstring-aware search for
    everyone, vectors or not. Empty dict when no index exists."""
    if not VECTORS_PATH.exists():
        return {}
    try:
        with closing(_connect()) as conn:
            return dict(conn.execute("SELECT node_id, card_text FROM cards"))
    except sqlite3.Error:
        return {}


def refresh_index(
    graph,
    config: Config,
    batch_size: int = 64,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Optional[IndexRefresh]:
    """(Re)embed changed cards. Runs inside the freshness flow; never raises.

    Returns coverage stats or None when the model isn't available. A digest
    change invalidates every card (embedding spaces don't mix). progress,
    when given, is called with (embedded, pending total) after each batch.
    """
    digest = model_digest(config, timeout=8)
    if digest is None:
        return None
    try:
        with closing(_connect()) as conn:
            stored = {
                row[0]: (row[1], bool(row[2]))
                for row in conn.execute(
                    "SELECT node_id, content_hash, vec IS NOT NULL FROM cards"
                )
            }
            candidates = g.candidate_code_nodes(graph)
            pending: list[tuple[str, str, str]] = []
            skipped = 0
            for node_id in candidates:
                # One node with an unreadable card (deleted/emptied file,
                # odd encoding) must never kill the other 3,000 — skip it
                # and say so in the stats.
                try:
                    node = g.get_node(graph, node_id)
                    text = card_text(node)
                except Exception:  # noqa: BLE001 - per-node isolation
                    skipped += 1
                    continue
                content = card_hash(text, digest)
                prev = stored.get(node_id)
                # Re-embed on content change AND on a missing vector — a row
                # written upfront by an interrupted build must not look
                # "done" just because its hash matches.
                if prev is None or prev[0] != content or not prev[1]:
                    pending.append((node_id, text, content))

            # Deleted code loses its card; chunked to stay under SQLite's
            # bound-parameter limit on large repos.
            dead = [n for n in stored if n not in set(candidates)]
            for start in range(0, len(dead), 500):
                chunk = dead[start : start + 500]
                conn.execute(
                    "DELETE FROM cards WHERE node_id IN (%s)" % ",".join("?" * len(chunk)),
                    chunk,
                )
            # Every pending card is written up front with an empty vector:
            # done/total progress is real from the first second, and the
            # keyword channel gets fresh card texts immediately — even if
            # embedding is interrupted before it finishes.
            conn.executemany(
                "INSERT OR REPLACE INTO cards (node_id, content_hash, card_text, vec) "
                "VALUES (?, ?, ?, NULL)",
                [(node_id, content, text) for node_id, text, content in pending],
            )
            conn.commit()

            embedded = 0
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                vectors = _embed(
                    [_DOC_PREFIX + text for _, text, _ in batch], config, timeout=120
                )
                if vectors is None or len(vectors) != len(batch):
                    break  # partial coverage is fine; next refresh resumes
                conn.executemany(
                    "UPDATE cards SET vec = ? WHERE node_id = ?",
                    [
                        (_pack(vector), node_id)
                        for (node_id, _, _), vector in zip(batch, vectors)
                    ],
                )
                conn.commit()
                embedded += len(batch)
                if progress is not None:
                    progress(embedded, len(pending))
            for key, value in (
                ("embed_version", str(EMBED_VERSION)),
                ("model", EMBED_MODEL),
                ("digest", digest),
                ("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
                )
            conn.commit()
            total = conn.execute(
                "SELECT COUNT(*), SUM(vec IS NOT NULL) FROM cards"
            ).fetchone()
            return IndexRefresh(
                cards=total[0] or 0, embedded=embedded,
                with_vec=total[1] or 0, skipped=skipped,
            )
    except sqlite3.Error:
        return None


def _pack(vector: list[float]) -> bytes:
    import struct

    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    import struct

    return list(struct.unpack(f"{len(blob) // 4}f", blob))


# --- query time --------------------------------------------------------------------------


def _read_index_state() -> Optional[tuple[dict, int, int]]:
    """(meta, total cards, cards with vectors) or None if unreadable."""
    try:
        with closing(_connect()) as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            total, with_vec = conn.execute(
                "SELECT COUNT(*), SUM(vec IS NOT NULL) FROM cards"
            ).fetchone()
        return meta, total or 0, with_vec or 0
    except sqlite3.Error:
        return None


@dataclass
class IndexRefresh:
    """What one refresh pass did (total cards / newly embedded / covered)."""
    cards: int
    embedded: int
    with_vec: int
    skipped: int = 0  # nodes whose card couldn't be built (file gone/odd)


@dataclass
class IndexStatus:
    """Retrieval mode plus a human-actionable reason when degraded."""
    mode: str  # "semantic+keyword" | "keyword-only"
    model_present: bool
    reason: str = ""
    cards: int = 0
    with_vec: int = 0


def self_heal_index(config: Config, graph_path: str) -> bool:
    """Start a background rebuild of an unhealthy index: missing (fresh
    clone, repo first touched by an agent), interrupted mid-build, stale
    after a model change, or partially covered. Presence of the embed
    model is the consent. Returns True if a build was started."""
    if not needs_refresh(config):
        return False
    from roger import freshness

    return freshness.maybe_refresh_in_background(graph_path, force=True)


def offer_appropriate(config: Config, reoffer: bool = False) -> bool:
    """Should a human flow offer to enable smarter search? (The TTY gate
    and the actual prompt belong to the CLI; the policy lives here.)"""
    if config.model.provider.startswith("azure-"):
        return False
    if embed_prompt_declined() and not reoffer:
        return False
    return model_digest(config) is None  # presence IS enablement


def index_progress() -> Optional[tuple[int, int]]:
    """(embedded, total) card counts — no network calls. None if no index."""
    if not VECTORS_PATH.exists():
        return None
    state = _read_index_state()
    if state is None:
        return None
    _, total, with_vec = state
    return with_vec, total


def needs_refresh(config: Config) -> bool:
    """Should a background refresh (re)build this repo's index?

    True for: no index yet, an interrupted build (meta is only written when
    a build finishes, so a killed build leaves cards without meta), a model
    change, or partial coverage. A 10-minute cool-down since the last
    completed attempt stops a persistently failing embed endpoint from
    turning every context call into a rebuild.
    """
    digest = model_digest(config)
    if digest is None:
        return False
    if not VECTORS_PATH.exists():
        return True
    state = _read_index_state()
    if state is None:
        return False
    meta, total, with_vec = state
    if (
        meta.get("digest") == digest
        and meta.get("embed_version") == str(EMBED_VERSION)
        and with_vec >= total
    ):
        return False
    last = meta.get("updated_at")
    if last:
        try:
            done = time.mktime(time.strptime(last, "%Y-%m-%dT%H:%M:%S"))
            if time.time() - done < 600:
                return False
        except ValueError:
            pass
    return True


def index_status(config: Config) -> IndexStatus:
    """For roger doctor: mode, coverage, and a reason a human can act on."""
    from roger import freshness

    digest = model_digest(config, timeout=8)
    if not VECTORS_PATH.exists():
        return IndexStatus("keyword-only", digest is not None, "index not built yet")
    state = _read_index_state()
    if state is None:
        return IndexStatus("keyword-only", digest is not None, "index unreadable")
    meta, total, with_vec = state
    if digest is None:
        return IndexStatus(
            "keyword-only", False, "embedding model not available (is Ollama running?)"
        )
    if not meta.get("digest"):
        # A finished build always has meta — its absence means the first
        # build is mid-flight (lock held) or died partway.
        reason = (
            f"index building now ({with_vec:,} of {total:,} functions done)"
            if freshness.lock_held()
            else "an index build was interrupted"
        )
        return IndexStatus("keyword-only", True, reason, cards=total, with_vec=with_vec)
    if meta.get("digest") != digest or meta.get("embed_version") != str(EMBED_VERSION):
        return IndexStatus(
            "keyword-only", True, "the embedding model changed",
            cards=total, with_vec=with_vec,
        )
    if with_vec < total:
        return IndexStatus(
            "keyword-only", True, f"index {with_vec:,}/{total:,} complete",
            cards=total, with_vec=with_vec,
        )
    return IndexStatus("semantic+keyword", True, cards=total, with_vec=with_vec)


def semantic_rank(question: str, config: Config, top: int = 30) -> Optional[list[str]]:
    """Node ids ranked by meaning, or None when the channel is unavailable.

    Silent-fallback contract: any failure returns None (keyword-only),
    except a digest mismatch which also returns None but marks the index
    for rebuild — mixed embedding spaces must never be compared.
    """
    import os

    if os.environ.get("ROGER_NO_EMBED"):  # undocumented debugging escape
        return None
    if not VECTORS_PATH.exists():
        return None
    digest = model_digest(config)
    if digest is None:
        return None
    try:
        with closing(_connect()) as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            if meta.get("digest") != digest or meta.get("embed_version") != str(EMBED_VERSION):
                return None  # hard-disable until refresh_index rebuilds
            mtime = VECTORS_PATH.stat().st_mtime
            global _matrix_cache
            if _matrix_cache[0] != mtime:
                rows = conn.execute(
                    "SELECT node_id, vec FROM cards WHERE vec IS NOT NULL"
                ).fetchall()
                if not rows:
                    return None
                ids = [r[0] for r in rows]
                vectors = [_unpack(r[1]) for r in rows]
                try:
                    import numpy as np

                    matrix = np.array(vectors, dtype="float32")
                    norms = np.linalg.norm(matrix, axis=1)
                    norms[norms == 0] = 1.0
                    matrix = matrix / norms[:, None]
                except ImportError:
                    matrix = vectors  # list-of-lists fallback
                _matrix_cache = (mtime, ids, matrix)
    except (sqlite3.Error, OSError):
        return None

    query_vecs = _embed([_QUERY_PREFIX + question], config, timeout=QUERY_DEADLINE_SECS)
    if not query_vecs:
        return None
    query = query_vecs[0]

    _, ids, matrix = _matrix_cache
    try:
        import numpy as np

        q = np.array(query, dtype="float32")
        norm = np.linalg.norm(q) or 1.0
        sims = matrix @ (q / norm)
        order = np.argsort(-sims)[:top]
        return [ids[i] for i in order]
    except ImportError:
        qnorm = math.sqrt(sum(x * x for x in query)) or 1.0
        scored = []
        for node_id, vector in zip(ids, matrix):
            vnorm = math.sqrt(sum(x * x for x in vector)) or 1.0
            if sys.version_info >= (3, 12):
                dot = math.sumprod(query, vector)
            else:
                dot = sum(a * b for a, b in zip(query, vector))
            scored.append((dot / (qnorm * vnorm), node_id))
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:top]]
