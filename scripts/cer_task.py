#!/usr/bin/env python3
"""CER — recursive Task-catcher / calibrator (David's CER logic, buildable form).

Pipeline per David:
  CER(corpus):
    if too big to read at once -> CHUNK, call TCLLM per chunk         (WALL#1, David)
    TPS = TCLLM(text)  = population-based prob the text heads to a goal (logprobs)
    if TPS < threshold -> route to plain TextLLM, return score only
    else (treat as contract):
        extract {actor,input,operation,output,constraint} + per-field score
        subs = TTCLLM(text)   # distinct sub-tasks
        if no sub-task found  -> LEAF, return score+task               (WALL#2, David)
        else recurse per sub-task  -> Multi-CE activates at the leaves

My added guards: chunk overlap (don't split a task at a boundary); max_depth
(recursion can't run away); softmax over the two continuation logprobs for the score.

Backend is pluggable: `LogprobLLM` (real, local Qwen, exposes logprobs) or `MockLLM`
(deterministic, no model) so the control flow runs NOW and termination is provable.
"""
from __future__ import annotations
import math, re, json, sys

WINDOW        = 40     # token budget per TCLLM read (tiny here to exercise chunking; real ~ 8k-100k)
OVERLAP       = 8      # token overlap so a task isn't split at a chunk boundary
TPS_THRESHOLD = 0.5    # < -> TextLLM (no extraction); >= -> treat as contract (TBD: calibrate)
MAX_DEPTH     = 6      # guard: recursion ceiling

# ---------- backend interface ----------
class Backend:
    name = "abstract"
    def tps_logprobs(self, text: str) -> tuple[float, float]:   # (logP_yes, logP_no)
        raise NotImplementedError
    def extract_task(self, text: str) -> dict:                  # 5-tuple + per-field score
        raise NotImplementedError
    def find_subtasks(self, text: str) -> list[str]:            # spans of distinct sub-tasks ([] = atomic)
        raise NotImplementedError

def tps(be: Backend, text: str) -> float:
    """Task-Possibility-Score = softmax of yes/no continuation logprobs (population-based)."""
    ly, ln = be.tps_logprobs(text)
    m = max(ly, ln)
    py, pn = math.exp(ly - m), math.exp(ln - m)
    return py / (py + pn)

def tokenize(text: str) -> list[str]:
    return text.split()

def chunks(toks: list[str], window=WINDOW, overlap=OVERLAP):
    i = 0
    while i < len(toks):
        yield toks[i:i + window]
        if i + window >= len(toks):
            break
        i += max(1, window - overlap)

# ---------- the CER ----------
def CER(be: Backend, corpus: str, depth: int = 0) -> dict:
    toks = tokenize(corpus)

    # WALL#1 (David): too big to read at once -> chunk, TCLLM per chunk
    if len(toks) > WINDOW:
        parts = [CER(be, " ".join(ch), depth) for ch in chunks(toks)]
        kept = [p for p in parts if p.get("task") or p.get("subtasks")]
        return {"depth": depth, "chunked": True, "n_chunks": len(parts),
                "tasks_found": len(kept), "parts": parts}

    t = round(tps(be, corpus), 3)

    # low task-possibility OR depth guard -> score only, route to plain TextLLM
    if t < TPS_THRESHOLD or depth >= MAX_DEPTH:
        return {"depth": depth, "tps": t, "route": "TextLLM", "task": None}

    # treat as contract: extract the 5-tuple
    task = be.extract_task(corpus)
    subs = be.find_subtasks(corpus)

    # WALL#2 (David): no further Trivial Task -> LEAF, return score + task (a Multi-CE target)
    if not subs:
        return {"depth": depth, "tps": t, "task": task, "leaf": True, "route": "CE"}

    # else recurse per sub-task
    children = [CER(be, s, depth + 1) for s in subs]
    return {"depth": depth, "tps": t, "task": task, "subtasks": children}

# ---------- leaves = where Multi-CE activates ----------
def leaves(tree: dict) -> list[dict]:
    out = []
    if tree.get("leaf"):
        out.append(tree)
    for k in ("parts", "subtasks"):
        for c in tree.get(k, []):
            out += leaves(c)
    return out

# ---------- deterministic mock (no model) so the logic runs + termination is checkable ----------
TASK_VERBS = {"summarize", "list", "answer", "translate", "compute", "write",
              "find", "extract", "classify", "compare", "refuse"}

class MockLLM(Backend):
    name = "mock"
    def tps_logprobs(self, text):
        toks = [w.strip(".,:;").lower() for w in text.split()]
        hits = sum(w in TASK_VERBS for w in toks)
        # more task-verbs -> higher logP(yes); deterministic, population-free (mock only)
        return (1.5 * hits, 1.0)            # softmax((1.5*hits,1.0))
    def extract_task(self, text):
        toks = [w.strip(".,:;").lower() for w in text.split()]
        op = next((w for w in toks if w in TASK_VERBS), "")
        return {"actor": "model", "input": text[:30], "operation": op,
                "output": "", "constraint": "only|must" if ("only" in toks or "must" in toks) else "",
                "score": {"actor": .9, "input": 1.0, "operation": .8 if op else .2,
                          "output": .4, "constraint": .6 if ("only" in toks or "must" in toks) else .1}}
    def find_subtasks(self, text):
        # split on sentence/imperative boundaries; a span is a sub-task iff it holds a task-verb;
        # return [] when there is <=1 task-verb span (=> atomic => LEAF => terminates)
        spans = [s.strip() for s in re.split(r"[.;]| then | and then ", text) if s.strip()]
        tasky = [s for s in spans if any(w.strip(".,:;").lower() in TASK_VERBS for w in s.split())]
        return tasky if len(tasky) > 1 else []

def _demo():
    cases = {
        "casual":     "the weather was nice and we walked by the river",
        "one_task":   "summarize the article in one sentence using only the given facts",
        "multi_task": ("first summarize the report; then translate the summary to korean; "
                       "and then list three risks"),
        "huge": " ".join(["please answer the question."] * 30 +
                          ["also translate this to french."] * 30),   # forces chunking
    }
    be = MockLLM()
    for name, text in cases.items():
        tree = CER(be, text)
        lv = leaves(tree)
        print(f"\n=== {name} (tps_root={tree.get('tps','-')}, route={tree.get('route','tree')}) ===")
        print(f"  leaves(=Multi-CE targets): {len(lv)}")
        for l in lv:
            print(f"    op={l['task']['operation']!r:12} tps={l['tps']} constraint={l['task']['constraint']!r}")
        if name == "casual":
            print(f"  -> routed: {tree.get('route')}")
    print("\n[termination] every path: chunk->smaller, or recurse->strictly-smaller spans, "
          "or no-subtask LEAF, or depth>=MAX_DEPTH. No infinite loop.")

if __name__ == "__main__":
    _demo()
