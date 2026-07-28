"""Randomized latent-program tasks — the frame-invention (abduction) test.

tasks.py offered a FIXED menu of smooth forms (power law, quadratic) that a model RECOGNIZES.
Here each instance's hidden rule is a randomly COMPOSED small integer program: the STRUCTURE, not
just the constants, is randomized. "Recognize a known form" cannot work — the model must POSIT the
right computational mechanism (the frame) and have it GENERALIZE to held-out inputs.

The three jump-test conditions:
  1. randomize the frame            -> this file (novel composition per instance)
  2. no gradient / one-shot         -> run_abduction.py (independent attempts, no fit feedback)
  3. grade on held-out OOD          -> exact integer match on inputs never shown

Honest limit (stated, not hidden): the MOTIFS (digit-sum, parity, accumulation, divisor count, ...)
are known primitives; only their COMPOSITION is novel per instance — ARC's principle. We resist
memorization of the specific mapping, not of the primitives. `true_fn` and `description` are
host-side ONLY, for grading/logging, and are NEVER shown to the model.
"""
import math
import random

MAXV = 10 ** 7  # reject explosive programs (keep outputs inferable/printable)


# ── base motifs: factory(rng) -> (fn: int->int, label) ─────────────────────────
def _affine(r):
    a = r.choice([1, 2, 3, -1, -2]); b = r.randint(-5, 5)
    return (lambda x: a * x + b), f"{a}*x+{b}"

def _digitsum(r):
    return (lambda x: sum(int(d) for d in str(abs(x)))), "digitsum(x)"

def _numdigits(r):
    return (lambda x: len(str(abs(x)))), "numdigits(x)"

def _reverse(r):
    return (lambda x: int(str(abs(x))[::-1])), "reverse(x)"

def _altsum(r):
    return (lambda x: sum((-1) ** i * int(d) for i, d in enumerate(str(abs(x))))), "altdigitsum(x)"

def _mod(r):
    m = r.randint(2, 9); c = r.randint(0, 4)
    return (lambda x: x % m + c), f"x%{m}+{c}"

def _popcount(r):
    return (lambda x: bin(abs(x)).count("1")), "popcount(x)"

def _gcdc(r):
    c = r.randint(2, 12)
    return (lambda x: math.gcd(abs(x), c)), f"gcd(x,{c})"

def _ndiv(r):
    return (lambda x: sum(1 for k in range(1, abs(x) + 1) if x % k == 0) if x else 0), "numdivisors(x)"

BASE = [_affine, _digitsum, _numdigits, _reverse, _altsum, _mod, _popcount, _gcdc, _ndiv]


# ── combinators over sub-expressions -> (fn, desc) ─────────────────────────────
def _build(r, depth):
    """Recursively assemble a random program of the given max depth."""
    if depth <= 1 or r.random() < 0.35:
        return r.choice(BASE)(r)
    kind = r.choice(["compose", "add", "parity", "threshold", "accum"])
    if kind == "compose":
        f, df = _build(r, depth - 1); g, dg = _build(r, depth - 1)
        return (lambda x: f(g(x))), f"({df})∘({dg})"
    if kind == "add":
        f, df = _build(r, depth - 1); g, dg = _build(r, depth - 1)
        return (lambda x: f(x) + g(x)), f"({df})+({dg})"
    if kind == "parity":
        f, df = _build(r, depth - 1); g, dg = _build(r, depth - 1)
        return (lambda x: f(x) if x % 2 == 0 else g(x)), f"even(x)?({df}):({dg})"
    if kind == "threshold":
        t = r.randint(8, 22); f, df = _build(r, depth - 1); g, dg = _build(r, depth - 1)
        return (lambda x: f(x) if x < t else g(x)), f"x<{t}?({df}):({dg})"
    f, df = _build(r, depth - 1)  # accum: running total over 1..x
    return (lambda x: sum(f(k) for k in range(1, abs(x) + 1))), f"sum_k=1..x[{df}]"


class LatentTask:
    def __init__(self, name, true_fn, description, fit_inputs, ood_inputs):
        self.name = name
        self.true_fn = true_fn          # HOST-ONLY grading; never shown to the model
        self.description = description   # HOST-ONLY mechanism label; never shown
        self.blurb = (
            "A fixed deterministic rule maps each positive integer x to an integer y. The rule is "
            "some computation — it may use the digits of x, parity, divisibility, running totals "
            "over 1..x, or different cases for different x — and it is NOT necessarily a smooth "
            "arithmetic formula. It holds for every x."
        )
        self.fit_inputs = list(fit_inputs)
        self.ood_inputs = list(ood_inputs)

    @property
    def all_inputs(self):
        return self.fit_inputs + self.ood_inputs

    def fit_examples(self):
        return [(x, self.true_fn(x)) for x in self.fit_inputs]

    def is_close(self, yhat, ytrue):
        """Exact integer match — these are discrete programs, no tolerance."""
        try:
            return int(round(float(yhat))) == int(ytrue)
        except (TypeError, ValueError):
            return False


def _degenerate(fn, xs):
    """Reject trivial/uninteresting/unsafe instances so the test stays informative."""
    try:
        ys = [fn(x) for x in xs]
    except Exception:  # noqa: BLE001 - any runtime issue => reject and resample
        return True
    if any(abs(int(y)) > MAXV for y in ys):
        return True
    if len(set(ys)) < 3:          # near-constant: nothing to infer
        return True
    if ys == list(xs):            # identity: trivial
        return True
    if all(y == ys[0] + i * (ys[1] - ys[0]) for i, y in enumerate(ys)):
        return True               # pure arithmetic progression: a plain line, not a "frame"
    return False


def make_latent_task(seed=0, complexity=2, fit_inputs=None, ood_inputs=None):
    """A random latent-program task. complexity 1=single motif, 2=one combinator, 3=deeper."""
    fit_inputs = list(fit_inputs or range(1, 25))     # shown to the model
    ood_inputs = list(ood_inputs or range(30, 45))    # held-out, larger => real extrapolation
    xs = fit_inputs + ood_inputs
    depth = {1: 1, 2: 2, 3: 3}.get(complexity, 2)
    for t in range(300):
        r = random.Random((seed * 1000003) ^ (t * 97) ^ (complexity * 7))
        fn, desc = _build(r, depth)
        if not _degenerate(fn, xs):
            return LatentTask(f"latent(c={complexity},seed={seed})", fn, desc, fit_inputs, ood_inputs)
    # near-impossible fallback: a guaranteed-valid non-degenerate instance
    return LatentTask(f"latent-fallback(seed={seed})",
                      lambda x: sum(int(d) for d in str(2 * x + 3)) + (x % 4),
                      "digitsum(2*x+3)+x%4", fit_inputs, ood_inputs)
