"""Accuracy gates.

Each step writes results/gates/<gate_name>.json with its metrics and PASS/FAIL.
The next step calls `require_gate(...)` first and refuses to run unless that gate passed.
Thresholds come from configs and are never lowered by code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from common.paths import REPO_ROOT, load_paths


def _gate_path(gate_name: str) -> Path:
    return load_paths()["results"]["gates"] / f"{gate_name}.json"


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_gate(gate_name: str, checks: dict[str, dict], extra: dict | None = None) -> dict:
    """Write a gate file.

    `checks` maps a check name to {"value": float, "threshold": float, "op": ">=" | "<=" | "=="}.
    The gate passes only if every check passes.
    """
    ops = {">=": lambda v, t: v >= t, "<=": lambda v, t: v <= t, "==": lambda v, t: v == t}
    evaluated = {}
    for name, c in checks.items():
        ok = bool(ops[c.get("op", ">=")](c["value"], c["threshold"]))
        evaluated[name] = {**c, "op": c.get("op", ">="), "passed": ok}
    status = "PASS" if evaluated and all(c["passed"] for c in evaluated.values()) else "FAIL"
    gate = {
        "gate": gate_name,
        "status": status,
        "checks": evaluated,
        "extra": extra or {},
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = _gate_path(gate_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return gate


def read_gate(gate_name: str) -> dict | None:
    path = _gate_path(gate_name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def accept_failed_gate(gate_name: str, decided_by: str, reason: str) -> dict:
    """Record that the user explicitly accepted a FAILED gate. The status stays FAIL; the decision
    is stored next to it so every later report shows the model did not reach its threshold."""
    gate = read_gate(gate_name)
    if gate is None:
        sys.exit(f"[gate] '{gate_name}' does not exist")
    gate["user_decision"] = {"accepted": True, "decided_by": decided_by, "reason": reason,
                             "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _gate_path(gate_name).write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    return gate


def require_gate(gate_name: str, allow_user_accepted: bool = False) -> dict:
    """Exit with a clear message unless `gate_name` exists and has status PASS.

    allow_user_accepted=True also lets a FAILED gate through if the user explicitly accepted it
    (see accept_failed_gate); a warning is printed so this is never silent.
    """
    gate = read_gate(gate_name)
    if gate is None:
        sys.exit(f"[gate] '{gate_name}' has not been run yet ({_gate_path(gate_name)} missing). "
                 f"Finish that step first.")
    if gate["status"] != "PASS" and allow_user_accepted and gate.get("user_decision", {}).get("accepted"):
        d = gate["user_decision"]
        print(f"[gate] WARNING: '{gate_name}' is {gate['status']} but was accepted by {d['decided_by']} "
              f"on {d['timestamp'][:10]}: {d['reason']}")
        return gate
    if gate["status"] != "PASS":
        failed = [k for k, c in gate["checks"].items() if not c["passed"]]
        sys.exit(f"[gate] '{gate_name}' is {gate['status']} (failed checks: {failed}). "
                 f"This step will not start until it passes.")
    return gate
