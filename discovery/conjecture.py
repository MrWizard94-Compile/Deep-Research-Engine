"""A conjecture = an LLM-authored program that infers a task's hidden rule and predicts for the
query inputs. We run it in the honest Docker sandbox and score it TWO ways:

  * fit_score  — accuracy on the in-regime queries (what induction optimizes)
  * ood_score  — accuracy on the held-out regime it was never shown (the jump signature)

The behavioral descriptor (bd) is the prediction vector itself, so novelty (diversity of
predictions) and the jump metric (OOD accuracy) are measured in the same space.
"""

import os
import re
import sys

# Make the parent engine dir importable regardless of cwd (llm_bridge lives there).
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_bridge import query_llm, OLLAMA_CODE_MODEL  # noqa: E402

_PRED_RE = re.compile(r"PRED\s+(\d+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def craft_conjecture_prompt(task, queries, extra=""):
    ex = "\n".join(f"  x={x} -> y={round(float(y), 6)}" for x, y in task.fit_examples())
    qs = "\n".join(f"  [{i}] x={x}" for i, x in enumerate(queries))
    guidance = ("\nADDITIONAL STEER:\n" + extra.strip() + "\n") if extra.strip() else "\n"
    return f"""You are reverse-engineering a fixed hidden rule y = f(x).
{task.blurb}

Worked examples (input -> output):
{ex}

Method you MUST follow in your program:
1. HYPOTHESIZE ONE functional FORM for f with a few free parameters. Weigh the full range of simple
   laws — polynomial (a*x**2+b*x+c), power law (a*x**k), linear (a*x+b), exponential (a*exp(b*x)),
   logarithmic (a*log(x)+b), trend+periodic (a*x+b*sin(c*x+d)) — and commit to the ONE most likely to
   be the true generating law for THIS data. Choosing the form well is the hard part.
2. FIT the parameters to the worked examples. numpy and scipy ARE installed — scipy.optimize.curve_fit
   is ideal: give it your hypothesized form and let it find the least-squares best-fit parameters.
   Do NOT hardcode guessed constants — FIT them to the data.
3. Pick a form that could be the TRUE generating law, one that EXTRAPOLATES. A high-degree polynomial
   or a lookup table will match these points and then diverge far outside — that is failure.
{guidance}Then predict y for EACH query using your FITTED rule, and print exactly one line per query, in order:
  PRED <index>=<value>

Query inputs:
{qs}

Output ONLY the PRED lines. Return ONLY the raw Python program — no markdown, no commentary."""


def _strip_fences(text):
    if not isinstance(text, str):
        return "print('NO_OUTPUT')"
    if "```" in text:
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    return text.strip() + "\n"


def _parse_preds(output, n):
    preds = {}
    for idx, val in _PRED_RE.findall(output or ""):
        try:
            i = int(idx)
            if 0 <= i < n:
                preds[i] = float(val)
        except ValueError:
            continue
    return preds


def generate_and_grade(task, sandbox, query_fn=query_llm, model=OLLAMA_CODE_MODEL, extra="", timeout_note=""):
    """Generate one conjecture for `task`, run it in `sandbox`, and grade fit + OOD."""
    queries = task.all_inputs
    n = len(queries)
    n_fit = len(task.fit_inputs)

    prompt = craft_conjecture_prompt(task, queries, extra)
    raw = query_fn(prompt, model=model)
    code = _strip_fences(raw)

    sandbox.write_file("_conjecture.py", code)
    logs = sandbox.run_container_test("python3 _conjecture.py")
    preds = _parse_preds(logs, n)

    def grade(indexed_inputs):
        """Return (hit_rate, graded_mean). Graded gives partial credit by relative error, so a
        right-frame/slightly-off conjecture reads as a gradient, not a flat miss (linear decay to
        0 at 100% error). Binary hit_rate stays as the strict 'exactly right' measure."""
        hits, graded = 0, 0.0
        for idx, x in indexed_inputs:
            yh = preds.get(idx)
            if yh is None:
                continue
            yt = task.true_fn(x)
            if task.is_close(yh, yt):
                hits += 1
                graded += 1.0
            else:
                relerr = abs(yh - yt) / (abs(yt) + task.tol_abs + 1e-9)
                graded += max(0.0, 1.0 - relerr)
        m = len(indexed_inputs)
        return hits / max(1, m), graded / max(1, m)

    fit_indexed = list(enumerate(task.fit_inputs))
    ood_indexed = [(n_fit + j, x) for j, x in enumerate(task.ood_inputs)]
    fit_score, fit_graded = grade(fit_indexed)
    ood_score, ood_graded = grade(ood_indexed)

    # Behavioral descriptor: the prediction vector (None where the program emitted nothing).
    bd = [preds.get(i) for i in range(n)]

    return {
        "code": code,
        "preds": preds,
        "bd": bd,
        "fit_score": fit_score,      # strict hit-rate in the fit regime
        "ood_score": ood_score,      # strict hit-rate out of distribution (the jump, binary)
        "fit_graded": fit_graded,    # partial-credit fit
        "ood_graded": ood_graded,    # partial-credit generalization (sensitive gradient)
        "n_fit": n_fit,
        "n_ood": len(task.ood_inputs),
        "ran": len(preds) > 0,
    }
