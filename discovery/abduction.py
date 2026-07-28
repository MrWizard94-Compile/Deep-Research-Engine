"""The abduction attempt: infer a hidden latent program from examples and predict held-out inputs.

Condition 2 (no gradient): each call is an INDEPENDENT one-shot attempt. The default gives the model
NO feedback about how close a prior attempt was — it must LEAP to the mechanism, not hill-climb to it.
The prompt states the SPACE of possible rules (digits/parity/divisibility/accumulation/cases) but
never the specific frame — analogous to telling a scientist "the mechanism is some computation,"
which levels the field without handing over the answer.

Condition 3 (held-out grading): fit and OOD are graded separately by EXACT integer match; a program
that reproduces the examples but diverges on the larger held-out inputs (an over-fit / lookup) fails.
Only inferring the true mechanism generalizes -> `jump`.
"""
import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_bridge import query_llm, OLLAMA_CODE_MODEL  # noqa: E402
import conjecture  # reuse _strip_fences + _parse_preds  # noqa: E402


def craft_abduction_prompt(task, queries, prior=""):
    ex = "\n".join(f"  x={x} -> y={task.true_fn(x)}" for x in task.fit_inputs)
    qs = "\n".join(f"  [{i}] x={x}" for i, x in enumerate(queries))
    # `prior` is used ONLY in search mode (gradient ON); empty for the pure one-shot abduction test.
    hint = ("\nOne earlier program fit the examples but you should reconsider the RULE itself:\n"
            + prior + "\n") if prior else "\n"
    return f"""You are reverse-engineering a hidden deterministic rule y = f(x) for positive integers.
{task.blurb}

Examples (input -> output):
{ex}

Infer the underlying RULE, then write a Python program that computes y for each query x using that
rule. The rule may involve the DIGITS of x, parity, divisibility, running totals over 1..x, or
different cases for different ranges of x — it need NOT be a smooth arithmetic formula. Do NOT
hardcode a lookup of the examples; find the GENERAL rule so it also holds for larger x you were not
shown.
{hint}Print exactly one line per query, in order:
  PRED <index>=<value>

Query inputs:
{qs}

Return ONLY the raw Python program — no markdown, no commentary."""


def attempt(task, sandbox, query_fn=query_llm, model=OLLAMA_CODE_MODEL, prior=""):
    """One independent attempt. Returns fit/OOD exact-match scores and whether it generalized."""
    queries = task.all_inputs
    n = len(queries)
    n_fit = len(task.fit_inputs)

    code = conjecture._strip_fences(query_fn(craft_abduction_prompt(task, queries, prior), model=model))
    sandbox.write_file("_abduct.py", code)
    logs = sandbox.run_container_test("python3 _abduct.py")
    preds = conjecture._parse_preds(logs, n)

    def score(indexed):
        hits = sum(1 for i, x in indexed if task.is_close(preds.get(i), task.true_fn(x)))
        return hits / max(1, len(indexed))

    fit = score(list(enumerate(task.fit_inputs)))
    ood = score([(n_fit + j, x) for j, x in enumerate(task.ood_inputs)])
    return {
        "code": code,
        "fit": fit,               # exact-match rate on shown examples
        "ood": ood,               # exact-match rate on held-out inputs (the abduction signal)
        "ran": len(preds) > 0,
        "jump": ood >= 0.999,     # inferred the true mechanism -> generalizes perfectly OOD
        "preds": preds,
    }
