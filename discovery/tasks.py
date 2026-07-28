"""Contamination-safe discovery tasks.

Each task hides a generative law. The candidate sees a FIT regime (worked examples) and must
predict for query inputs — some in-regime, some from a held-out OOD regime it is graded on but
never told about. Instances are RANDOMIZED (random constants/exponents), so a correct OOD
prediction requires actually inferring the law, not recalling it. That is the whole point: it
lets a null result mean something (we beat the GR-thought-experiment contamination problem).

The "jump" signature is generalization: an epicycle nails the fit regime and diverges OOD; the
real frame holds far outside the sample.
"""

import math
import random


class Task:
    def __init__(self, name, blurb, true_fn, fit_inputs, ood_inputs, tol_rel=0.02, tol_abs=1e-6):
        self.name = name
        self.blurb = blurb
        self.true_fn = true_fn          # host-side only, for grading — NEVER shown to the model
        self.fit_inputs = list(fit_inputs)
        self.ood_inputs = list(ood_inputs)
        self.tol_rel = tol_rel
        self.tol_abs = tol_abs

    @property
    def all_inputs(self):
        return self.fit_inputs + self.ood_inputs

    def fit_examples(self):
        return [(x, self.true_fn(x)) for x in self.fit_inputs]

    def is_close(self, yhat, ytrue):
        try:
            return math.isclose(float(yhat), float(ytrue), rel_tol=self.tol_rel, abs_tol=self.tol_abs)
        except (TypeError, ValueError):
            return False


def power_law_task(seed=None):
    """y = A * x^k, non-integer k. A polynomial fit nails x in [1,10] and diverges by x=50;
    the reframe ('it's a power law — fit in log-log') generalizes. Kepler-flavored, synthetic."""
    r = random.Random(seed)
    A = round(r.uniform(0.6, 3.0), 3)
    k = round(r.uniform(1.35, 2.65), 3)
    return Task(
        name=f"power-law(A={A},k={k})",
        blurb="A fixed hidden rule maps each input x (a positive number) to an output y. The rule "
              "is smooth and holds for x far beyond the examples.",
        true_fn=lambda x: A * (x ** k),
        fit_inputs=list(range(1, 11)),
        ood_inputs=list(range(50, 61)),
        tol_rel=0.03,
    )


def trend_plus_period_task(seed=None):
    """y = a*x + b*sin(2*pi*x/p). A local linear/poly fit tracks the sample and loses the period
    OOD; recovering the trend+period structure generalizes."""
    r = random.Random(seed)
    a = round(r.uniform(0.5, 2.0), 3)
    b = round(r.uniform(1.0, 4.0), 3)
    p = r.choice([5, 6, 7, 8])
    return Task(
        name=f"trend+period(a={a},b={b},p={p})",
        blurb="A fixed hidden rule maps each integer input x to an output y by combining a steady "
              "trend with a repeating pattern. It holds far beyond the shown range.",
        true_fn=lambda x: a * x + b * math.sin(2 * math.pi * x / p),
        fit_inputs=list(range(0, 16)),
        ood_inputs=list(range(40, 52)),
        tol_rel=0.06,
        tol_abs=0.6,
    )


def quadratic_task(seed=None):
    """y = a*x^2 + b*x + c. The sweet spot: with scipy available a conjecture that hypothesizes the
    quadratic FORM fits and generalizes, but wrong forms (exp, linear) or an over-fit high-degree
    polynomial still diverge OOD. Frame/degree choice is the abduction — and it's within a small
    model's reach, so the generator SOMETIMES succeeds, which is exactly what makes the selection
    question testable."""
    r = random.Random(seed)
    a = round(r.uniform(0.5, 3.0), 3)
    b = round(r.uniform(-3.0, 3.0), 3)
    c = round(r.uniform(-5.0, 5.0), 3)
    return Task(
        name=f"quadratic(a={a},b={b},c={c})",
        blurb="A fixed hidden rule maps each input x to an output y smoothly, and it holds for x far "
              "beyond the shown range.",
        true_fn=lambda x: a * x * x + b * x + c,
        fit_inputs=list(range(1, 13)),
        ood_inputs=list(range(40, 51)),
        tol_rel=0.05,
        tol_abs=1.0,
    )


def deceptive_power_task(seed=None):
    """DECEPTIVE by construction — numerically verified (scratchpad/verify_deception.py).

    True law y = A*x^k with NON-integer k in [2.15, 2.55], shown on a SHORT window x=3..7. With only
    five points, an easy-to-hypothesize WRONG frame — a plain quadratic (a*x^2+b*x+c) or cubic —
    fits the window to tolerance (fit-hit-rate 1.0) yet diverges by x=40..50 (ood 0.0). The true
    power law is the ONLY form that BOTH fits AND generalizes, and it too reaches fit 1.0. So the
    in-regime fit signal CANNOT distinguish the generalizing frame from the trap: several forms tie
    at the fit ceiling.

    Window placement matters: x=3..7 (not x=1..5) makes the quadratic trap ROBUST across the whole
    A/k range by pure relative error — near x=1 a quadratic deviates most from a power law, so the
    trap there depended on abs-tol slack and only held for small A. Verified in scratchpad/
    verify_window.py: quad AND cube tie truth at fit 1.0 in-window, ood 0.0, for every k in-range.

    That is the whole point. This is the abduction trap:
      * Greedy strict-improvement can PRUNE the jump — it keeps the first fit=1.0 form it meets and
        discards every later tie, so a generalizing power law arriving after a quadratic trap (same
        fit) is thrown away.
      * A protected novelty niche can RETAIN the jump — trap and truth have wildly different OOD
        predictions, i.e. they are behaviorally novel, so the archive keeps both alive.
    Unlike power_law_task (fit=1.0 reachable only by the truth => non-deceptive, greedy wins), here
    fit is deliberately uninformative near its optimum — the condition novelty search targets."""
    r = random.Random(seed)
    A = round(r.uniform(0.6, 3.0), 3)
    k = round(r.uniform(2.15, 2.55), 3)  # k-range where quadratic AND cubic both tie the truth in-window
    return Task(
        name=f"deceptive-power(A={A},k={k})",
        blurb="A fixed hidden rule maps each input x (a positive number) to an output y. The rule "
              "is smooth and holds for x far beyond the examples.",
        true_fn=lambda x: A * (x ** k),
        fit_inputs=list(range(3, 8)),     # SHORT window (x=3..7) — underdetermined on purpose
        ood_inputs=list(range(40, 51)),
        tol_rel=0.05,
        tol_abs=0.1,                      # small: keeps grading essentially relative (A-independent)
    )


TASKS = {
    "power-law": power_law_task,
    "periodic": trend_plus_period_task,
    "quadratic": quadratic_task,
    "deceptive": deceptive_power_task,
}


def make_task(kind="power-law", seed=None):
    if kind not in TASKS:
        raise ValueError(f"unknown task '{kind}'; have {sorted(TASKS)}")
    return TASKS[kind](seed=seed)
