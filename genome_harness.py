"""
genome_harness.py — SANDBOXED STRUCTURAL VALIDATION of a candidate genome.

Standalone, stdlib-only. Mounted into the Docker sandbox and run there against an untrusted
candidate genome, so no LLM-authored code executes on the host. It checks that the genome
honours its contract and that its prompts enforce the grounded-metric protocol (the genome
no longer judges success — grounding.py does — so there is nothing to regression-test here;
instead we verify the prompts will actually elicit gradeable code).

The REAL fitness (does this genome's prompt make the coder produce working, honestly-graded
code?) is measured separately, host-side, by self_improver via canary tasks — that needs
the LLM and cannot run in the sandbox.
"""

import json

REQUIRED_CALLABLES = ("craft_hypothesis_prompt", "craft_code_prompt", "craft_debug_prompt")
REQUIRED_ATTRS = ("GENOME_VERSION",)

# Tokens that prove a prompt enforces the protocol that makes external grounding possible.
_HYPOTHESIS_MUST = ("OBJ_TOKEN", "FAIL_TOKEN")
_HYPOTHESIS_QUALITY = ("success_metric", "comparison", "json", "falsifiable", "measurable")
_CODE_MUST = ("HYP_TOKEN", "CRIT_TOKEN", "METRIC")
_CODE_QUALITY = ("self-contained", "raw code", "real measured", "do not print success")
_DEBUG_MUST = ("HYP_TOKEN", "CODE_TOKEN", "ERROR_TOKEN", "CRIT_TOKEN", "METRIC")
_DEBUG_QUALITY = ("root cause", "repair", "do not start over", "corrected")


def evaluate_genome(module) -> dict:
    """Validate + structurally score a genome module. Never raises.

    Returns {"ok": bool, "score": float, "version": int|None, "report": str}. `score` is a
    structural-quality gradient only; promotion is decided by self_improver using canary
    fitness on top of this.
    """
    notes = []

    for name in REQUIRED_CALLABLES:
        if not callable(getattr(module, name, None)):
            notes.append(f"MISSING callable: {name}")
    version = getattr(module, "GENOME_VERSION", None)
    if not isinstance(version, int):
        notes.append("GENOME_VERSION must be an int")
        version = None
    if notes:
        return {"ok": False, "score": 0.0, "version": version, "report": "; ".join(notes)}

    hyp_ok, hyp_bonus = _check_prompt(
        module.craft_hypothesis_prompt, ("OBJ_TOKEN", "FAIL_TOKEN"),
        must_contain=_HYPOTHESIS_MUST, quality=_HYPOTHESIS_QUALITY, notes=notes,
        label="craft_hypothesis_prompt")
    code_ok, code_bonus = _check_prompt(
        module.craft_code_prompt, ("HYP_TOKEN", "python", "CRIT_TOKEN"),
        must_contain=_CODE_MUST, quality=_CODE_QUALITY, notes=notes,
        label="craft_code_prompt")
    debug_ok, debug_bonus = _check_prompt(
        module.craft_debug_prompt, ("HYP_TOKEN", "rust", "CODE_TOKEN", "ERROR_TOKEN", "CRIT_TOKEN"),
        must_contain=_DEBUG_MUST, quality=_DEBUG_QUALITY, notes=notes,
        label="craft_debug_prompt")

    ok = hyp_ok and code_ok and debug_ok
    score = (10.0 if ok else 0.0) + hyp_bonus + code_bonus + debug_bonus
    report = f"structural {'pass' if ok else 'FAIL'}; quality+{hyp_bonus + code_bonus + debug_bonus}"
    if notes:
        report += " | " + "; ".join(notes)
    return {"ok": ok, "score": score, "version": version, "report": report}


def _check_prompt(fn, args, must_contain, quality, notes, label):
    """Returns (ok, quality_bonus). Records problems into `notes`."""
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"{label} raised: {exc!r}")
        return False, 0.0
    if not isinstance(out, str) or not out.strip():
        notes.append(f"{label} must return a non-empty str")
        return False, 0.0
    low = out.lower()
    for token in must_contain:
        if token not in out and token.lower() not in low:
            notes.append(f"{label} output missing required token {token!r}")
            return False, 0.0
    bonus = float(sum(1 for t in quality if t.lower() in low))
    return True, bonus


def load_module_from_path(path, name="genome_under_test"):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load genome from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_genome_file(path) -> dict:
    try:
        module = load_module_from_path(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "score": 0.0, "version": None, "report": f"load failed: {exc!r}"}
    return evaluate_genome(module)


RESULT_SENTINEL = "###GENOME_RESULT###"


def build_runner_script() -> str:
    return (
        "import json, sys\n"
        "sys.path.insert(0, '/workspace')\n"
        "report = {'ok': False, 'score': 0.0, 'version': None, 'report': 'runner did not complete'}\n"
        "try:\n"
        "    import _candidate_genome as cand\n"
        "    import genome_harness as H\n"
        "    report = H.evaluate_genome(cand)\n"
        "except Exception as exc:\n"
        "    report = {'ok': False, 'score': 0.0, 'version': None, 'report': 'runner exception: %r' % (exc,)}\n"
        f"print('{RESULT_SENTINEL}' + json.dumps(report))\n"
        "print('SUCCESS' if report.get('ok') else 'FAILURE')\n"
    )


def parse_runner_output(logs: str) -> dict:
    for line in (logs or "").splitlines():
        line = line.strip()
        if line.startswith(RESULT_SENTINEL):
            try:
                return json.loads(line[len(RESULT_SENTINEL):])
            except json.JSONDecodeError:
                break
    return {"ok": False, "score": 0.0, "version": None, "report": "no parseable result from sandbox"}
