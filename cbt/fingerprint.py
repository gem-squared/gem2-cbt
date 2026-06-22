"""Dataset + config fingerprinting and checkpoint namespacing (U1 + U4).

Provides:
  compute_dataset_hash(data_dir)      — SHA256 over train.jsonl + test.jsonl
  compute_config_hash(args)           — hash of training hyperparameters
  write_manifest(...)                 — write provenance JSON to data_dir
  ckpt_path(...)                      — namespaced checkpoint path
  freeze_dataset_hash(...)            — U4 freeze-gate: lock canonical hash
  assert_frozen_hash(data_dir)        — FAIL FAST: assert hash unchanged since freeze

Checkpoints are stored at:
  {data_dir}/checkpoints/{dataset_hash}/{config}_seed{seed}.pt
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import torch

FROZEN_HASH_FILE = "frozen_dataset_hash.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_hash(data_dir: str) -> str:
    """SHA256 over train.jsonl then test.jsonl — 16-hex prefix."""
    h = hashlib.sha256()
    for name in ("train.jsonl", "test.jsonl"):
        path = os.path.join(data_dir, name)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def compute_config_hash(args) -> str:
    """Stable hash of training hyperparameters; excludes runtime-only fields."""
    cfg = {k: v for k, v in sorted(vars(args).items())
           if k not in ("data", "out", "time_budget", "fresh", "only")}
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def _git_commit(repo_root: str) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def write_manifest(data_dir: str, dataset_hash: str, config_name: str,
                   args, dataset_version: str = "v0") -> str:
    """Write provenance manifest; return manifest path."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = {
        "dataset_hash": dataset_hash,
        "dataset_version": dataset_version,
        "train_sha256": sha256_file(os.path.join(data_dir, "train.jsonl")),
        "test_sha256": sha256_file(os.path.join(data_dir, "test.jsonl")),
        "generator_seed": getattr(args, "seed", None),
        "config_hash": compute_config_hash(args),
        "config_name": config_name,
        "git_commit": _git_commit(repo_root),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    path = os.path.join(data_dir, f"manifest_{dataset_hash}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def ckpt_path(data_dir: str, dataset_hash: str, config_name: str, seed: int) -> str:
    """Return namespaced checkpoint path, creating the directory if needed."""
    ckpt_dir = os.path.join(data_dir, "checkpoints", dataset_hash)
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{config_name}_seed{seed}.pt")


def freeze_dataset_hash(data_dir: str, dataset_hash: str,
                        notes: str = "", probe_results: dict = None) -> str:
    """U4 freeze-gate: lock canonical hash. Write frozen_dataset_hash.json.
    Returns path to the freeze record."""
    path = os.path.join(data_dir, FROZEN_HASH_FILE)
    record = {
        "frozen_hash": dataset_hash,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "probe_results": probe_results or {},
    }
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def assert_frozen_hash(data_dir: str) -> str:
    """FAIL FAST: assert current dataset hash matches frozen hash.
    Raises RuntimeError with 'dataset hash mismatch' if different.
    Returns the verified hash on success."""
    path = os.path.join(data_dir, FROZEN_HASH_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"FROZEN HASH FILE MISSING: {path} — run U4 freeze-gate first")
    frozen = json.load(open(path))["frozen_hash"]
    current = compute_dataset_hash(data_dir)
    if current != frozen:
        raise RuntimeError(
            f"dataset hash mismatch: frozen={frozen} current={current} — "
            "dataset has changed since freeze. Aborting.")
    return current
