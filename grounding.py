"""
grounding.py — HARNESS-OWNED, NON-EVOLVABLE success grounding.

This is the fix for the "self-certified SUCCESS" gaming. A program no longer proves
success by printing a keyword it chose itself. Instead the Principal Investigator commits
to a measurable acceptance criterion BEFORE any code is written, the implementation emits a
single metric line, and THIS module — not the generated code, not the evolvable genome —
decides pass/fail by checking the metric against the criterion.

Protocol the code must follow (instructed via the prompts):
    print exactly one line:   METRIC <name>=<number>

The harness checks  <number> <comparison> <target>  where (name, target, comparison) is the
criterion. A trivial stub that prints "SUCCESS" now scores zero, because there is no metric
to check — it must actually produce the named number, and clear the threshold, in a
container that exited cleanly.
"""

import math
import re

_CONTAINER_FAIL = "[CONTAINER_FAIL]"
_METRIC_RE = re.compile(r"METRIC\s+([A-Za-z0-9_]+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# Used when the PI fails to commit to a real criterion — deliberately strict so a missing
# criterion cannot become an easy win.
DEFAULT_CRITERION = {"metric": "result", "target": 1.0, "comparison": ">="}

_COMPARATORS = {
    ">=": lambda a, t: a >= t,
    ">": lambda a, t: a > t,
    "<=": lambda a, t: a <= t,
    "<": lambda a, t: a < t,
    "==": lambda a, t: math.isclose(a, t, rel_tol=1e-6, abs_tol=1e-9),
}


def normalize_criterion(raw):
    """Coerce a PI-proposed criterion into a valid {metric, target, comparison} dict."""
    crit = dict(DEFAULT_CRITERION)
    if isinstance(raw, dict):
        metric = raw.get("metric") or raw.get("success_metric") or raw.get("name")
        if isinstance(metric, str) and metric.strip():
            crit["metric"] = re.sub(r"[^A-Za-z0-9_]", "_", metric.strip())[:40] or "result"
        comp = str(raw.get("comparison", "")).strip()
        if comp in _COMPARATORS:
            crit["comparison"] = comp
        try:
            crit["target"] = float(raw.get("target"))
        except (TypeError, ValueError):
            pass
    return crit


def criterion_instruction(criterion):
    """Human-readable instruction injected into the code/debug prompts."""
    c = normalize_criterion(criterion)
    return (
        f"SUCCESS CRITERION (checked externally by the harness — you cannot self-certify):\n"
        f"Your program MUST print exactly one line of the form `METRIC {c['metric']}=<number>` "
        f"where <number> is the REAL measured value produced by actually running your approach. "
        f"The run counts as solved only if `{c['metric']} {c['comparison']} {c['target']}` and the "
        f"program exits without error. Do NOT print the word SUCCESS — it is ignored. Compute the "
        f"metric honestly; a hardcoded or fabricated value defeats the purpose."
    )


def parse_metrics(output):
    """Extract all METRIC name=value pairs (last value for a name wins)."""
    metrics = {}
    for name, value in _METRIC_RE.findall(output or ""):
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


def evaluate(test_output, criterion):
    """Authoritative verdict. Returns {is_solved, score, detail, metrics}.

    score is graded so the debug loop gets a gradient:
        1.0  -> criterion met (solved)
        0.4  -> ran cleanly and emitted the metric, but threshold missed
        0.1  -> ran cleanly but never emitted the required metric
        0.0  -> container error / no usable output
    """
    text = (test_output or "")
    crit = normalize_criterion(criterion)
    metrics = parse_metrics(text)

    if _CONTAINER_FAIL in text or not text.strip():
        return {"is_solved": False, "score": 0.0,
                "detail": "container failed or produced no output", "metrics": metrics}

    if crit["metric"] not in metrics:
        return {"is_solved": False, "score": 0.1,
                "detail": f"ran, but never emitted the required `METRIC {crit['metric']}=<number>` line",
                "metrics": metrics}

    value = metrics[crit["metric"]]
    passed = _COMPARATORS[crit["comparison"]](value, crit["target"])
    if passed:
        return {"is_solved": True, "score": 1.0,
                "detail": f"{crit['metric']}={value} satisfies {crit['comparison']} {crit['target']}",
                "metrics": metrics}
    return {"is_solved": False, "score": 0.4,
            "detail": f"{crit['metric']}={value} FAILS {crit['comparison']} {crit['target']}",
            "metrics": metrics}
