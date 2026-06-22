"""WP-ST-11 U1: LLM backend selector — non-destructive (deepseek default; local opt-in).

Selects which OpenAI-compatible chat backend the harness calls, by env var
`LLM_BACKEND`:

  LLM_BACKEND=deepseek   (default — current behavior, hits DeepSeek public API)
  LLM_BACKEND=local      (Ollama OpenAI-compat endpoint, Qwen2.5-32B-Q8 by default)

The selector is purely additive — it does NOT mutate `oracle_payoff.call_llm`
or `concept_ce._llm_request`. Harness wrapper code (introduced in WP-11 U3+)
calls `get_backend()` and passes the resolved (base_url, model, api_key) into
the LLM client. DeepSeek archived runs remain reproducible by leaving
LLM_BACKEND unset (or setting it to "deepseek").

Manifest discipline (`record_backend_manifest`) writes the resolved backend
+ model + quant + ollama_version + git_commit into a JSON sidecar at run
start, so every Qwen-track artifact carries an unambiguous subject label.
This is the anti-silent-merge guard David named in the WP-11 directive.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_BACKEND = "deepseek"
SUPPORTED       = ("deepseek", "local")

LOCAL_DEFAULTS = {
    # 127.0.0.1 (NOT "localhost") forces IPv4 to avoid colliding with a
    # Docker container or other service listening on IPv6 [::1]:11434.
    # Discovered in WP-11 U2 — Docker proxy on IPv6 :11434 was returning
    # only embedding-model stubs while real Ollama lived on IPv4.
    "base_url":   "http://127.0.0.1:11434/v1",
    "model":      "qwen2.5:32b-instruct-q8_0",
    "api_key":    "ollama",       # Ollama ignores the value; OpenAI clients require non-empty
    "quant":      "q8_0",
    "provider":   "ollama",
}

# Optional env-var overrides for local — useful for sweeping quant / model later.
LOCAL_ENV_KEYS = {
    "base_url": "LOCAL_BASE_URL",
    "model":    "LOCAL_MODEL",
    "api_key":  "LOCAL_API_KEY",
    "quant":    "LOCAL_QUANT",
}


def _maybe_load_env(path: str = ".env") -> None:
    """Defensive convenience — load KEY=VALUE pairs from `.env` if present in the
    repo root. The harness scripts (oracle_payoff, concept_ce) already do this
    at import time; this is for standalone CLI use (python -m cbt.llm_backend).
    Idempotent — does not overwrite existing env vars."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path  = repo_root / path
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        # Best-effort — never crash the selector on a malformed .env line.
        pass


def _resolve_active_backend() -> str:
    """Return the active backend name; default = deepseek (preserves archived behavior)."""
    raw = os.environ.get("LLM_BACKEND", DEFAULT_BACKEND).strip().lower()
    if raw not in SUPPORTED:
        raise RuntimeError(
            f"LLM_BACKEND={raw!r} not in {SUPPORTED}. "
            f"Set LLM_BACKEND=deepseek (default) or local."
        )
    return raw


def get_backend() -> dict:
    """Return the active backend descriptor.

    Schema:
      {
        "backend":   "deepseek" | "local",
        "base_url":  str,
        "model":     str,
        "api_key":   str,              # NEVER logged or returned in manifest as-is
        "quant":     str | None,       # only meaningful for local quantized models
        "provider":  "deepseek" | "ollama",
      }
    """
    name = _resolve_active_backend()
    if name == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        return {
            "backend":  "deepseek",
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model":    os.environ.get("DEEPSEEK_MODEL",    "deepseek-chat"),
            "api_key":  key,
            "quant":    None,
            "provider": "deepseek",
        }
    # local (Ollama)
    return {
        "backend":  "local",
        "base_url": os.environ.get(LOCAL_ENV_KEYS["base_url"], LOCAL_DEFAULTS["base_url"]),
        "model":    os.environ.get(LOCAL_ENV_KEYS["model"],    LOCAL_DEFAULTS["model"]),
        "api_key":  os.environ.get(LOCAL_ENV_KEYS["api_key"],  LOCAL_DEFAULTS["api_key"]),
        "quant":    os.environ.get(LOCAL_ENV_KEYS["quant"],    LOCAL_DEFAULTS["quant"]),
        "provider": "ollama",
    }


def _safe_view(b: dict) -> dict:
    """Backend view safe to print/log/serialize — strips api_key."""
    out = {k: v for k, v in b.items() if k != "api_key"}
    out["api_key_present"] = bool(b.get("api_key"))
    return out


def _git_commit() -> str:
    """Short git HEAD for provenance. 'unknown' if not a repo or git missing."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _ollama_version() -> str | None:
    """Capture installed ollama version if present (provenance only)."""
    try:
        r = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def record_backend_manifest(path: str) -> str:
    """Write a JSON manifest at `path` describing the active backend.

    The manifest is the anti-silent-merge label David named in the WP-11 directive:
    every Qwen-track artifact MUST carry one so future readers cannot accidentally
    mix Qwen rows with DeepSeek rows.

    Never includes the api_key value (only a `api_key_present` boolean).
    Returns the manifest path.
    """
    b = get_backend()
    manifest = {
        **_safe_view(b),
        "ollama_version": _ollama_version() if b["backend"] == "local" else None,
        "git_commit":     _git_commit(),
        "recorded_at":    datetime.now(timezone.utc).isoformat(),
        "env_LLM_BACKEND": os.environ.get("LLM_BACKEND"),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def smoke_ping(b: dict | None = None, timeout: int = 5) -> dict:
    """Hit the backend's /v1/models endpoint as a liveness check.
    Returns {ok, status, models_count, error?}.
    Never sends or logs the api_key value (just a bool 'api_key_present')."""
    import urllib.error
    import urllib.request
    if b is None:
        b = get_backend()
    url = b["base_url"].rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {b['api_key']}"} if b.get("api_key") else {},
    )
    out = {
        "endpoint":        url,
        "api_key_present": bool(b.get("api_key")),
    }
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            doc  = json.loads(body)
            models = doc.get("data") or doc.get("models") or []
            ids = [m.get("id") or m.get("name") for m in models if isinstance(m, dict)]
            out.update({
                "ok":           True,
                "status":       r.status,
                "models_count": len(models),
                "models":       ids[:8],
            })
            return out
    except urllib.error.HTTPError as e:
        out.update({"ok": False, "status": e.code, "error": str(e)[:160]})
        return out
    except Exception as e:  # noqa: BLE001
        out.update({"ok": False, "status": None, "error": str(e)[:160]})
        return out


def _cli() -> int:
    _maybe_load_env()
    p = argparse.ArgumentParser(description="LLM backend selector — WP-ST-11")
    p.add_argument("--print",      action="store_true", help="Print resolved backend (safe view, no key)")
    p.add_argument("--smoke-ping", action="store_true", help="Hit /v1/models on the active backend")
    p.add_argument("--manifest",   metavar="PATH", help="Write a backend manifest to PATH")
    args = p.parse_args()

    if not (args.print or args.smoke_ping or args.manifest):
        p.print_help()
        return 0

    b = get_backend()
    if args.print:
        print(json.dumps(_safe_view(b), indent=2))
    if args.smoke_ping:
        ping = smoke_ping(b)
        print(json.dumps(ping, indent=2))
        if not ping.get("ok"):
            return 1
    if args.manifest:
        path = record_backend_manifest(args.manifest)
        print(f"[manifest] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
