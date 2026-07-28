"""The A/B selectors. Both see ONLY in-regime signal (fit + behavior); neither is ever shown
ood_score — OOD is the held-out outcome, not a selection input. Each selector also steers the
next generation (feedback), so selection actually shapes the search trajectory:

  GreedySelector   — keep the single best-by-FIT conjecture; steer toward exploiting it.
                     This is the engine's real gate: strict-improvement, hostile to reframes.
  NoveltySelector  — a quality-diversity archive; keep behaviorally-novel, minimally-viable
                     conjectures ALIVE even when their fit is low (the protected niche), and
                     steer toward unexplored behavior. This is what lets an epicycle-losing but
                     frame-opening conjecture survive its ugly adolescence.
"""

_EPS = 1e-9


def bd_distance(a, b):
    """Normalized distance between two prediction vectors (behavioral descriptors), robust to
    scale and to missing predictions. Returns [0,1]: a missing prediction on either side counts
    as maximally different at that index."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        va = a[i] if i < len(a) else None
        vb = b[i] if i < len(b) else None
        if va is None or vb is None:
            total += 1.0
            continue
        total += abs(va - vb) / (abs(va) + abs(vb) + _EPS)  # bounded in [0,1)
    return total / n


class GreedySelector:
    name = "greedy"

    def __init__(self):
        self.best = None
        self.retained = []  # the exploit path — every conjecture that became the best-so-far

    def consider(self, c):
        if self.best is None or c["fit_score"] > self.best["fit_score"]:
            self.best = c
            self.retained.append(c)
            return True
        return False

    def guidance(self):
        if not self.best:
            return ""
        return (
            f"A previous attempt reached fit {self.best['fit_score']:.2f} on the examples with this code:\n"
            f"{self.best['code'][:500]}\n"
            "Improve it into a rule that fits the examples MORE accurately."
        )

    def answer(self):
        return self.best

    def retained_pop(self):
        return self.retained


class NoveltySelector:
    name = "novelty"

    def __init__(self, novelty_threshold=0.15, viability_fit=0.1, k=3):
        self.archive = []
        self.nt = novelty_threshold      # how behaviorally-different a conjecture must be to keep
        self.viability_fit = viability_fit  # a LOW floor: must run + be non-garbage, NOT beat the best
        self.k = k

    def _novelty(self, bd):
        if not self.archive:
            return 1.0
        dists = sorted(bd_distance(bd, a["bd"]) for a in self.archive)
        kk = dists[: self.k]
        return sum(kk) / len(kk) if kk else 1.0

    def consider(self, c):
        # Protected niche: keep if behaviorally novel AND minimally viable — crucially NOT
        # required to beat the incumbent's fit. That is the whole difference from greedy.
        if not c["ran"] or c["fit_score"] < self.viability_fit:
            return False
        if self._novelty(c["bd"]) >= self.nt:
            self.archive.append(c)
            return True
        return False

    def guidance(self):
        if not self.archive:
            return ""
        forms = []
        for a in self.archive[-4:]:
            head = (a["code"] or "").strip().splitlines()
            forms.append("- " + (head[0][:120] if head else "(empty)"))
        return (
            "Forms already explored (do NOT repeat their structure):\n" + "\n".join(forms) +
            "\nPropose a rule with a STRUCTURALLY DIFFERENT functional form that still matches the examples."
        )

    def answer(self):
        # Committed pick: the best-fitting archive member (you'd still ship a good one).
        return max(self.archive, key=lambda a: a["fit_score"]) if self.archive else None

    def retained_pop(self):
        return self.archive
