"""The decisive A/B. Same generator + same budget under two selection regimes (greedy fit-climb
vs protected-novelty niche), on a contamination-safe task. OOD is graded but NEVER fed to a
selector. We log the ENTIRE candidate stream so 'couldn't propose it' can be told apart from
'proposed it and killed it'.

Run from the engine dir:
    python discovery/run_ab.py --task power-law --seed 1 --budget 8 --model qwen2.5:7b
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox import DockerSandbox  # noqa: E402
import tasks  # noqa: E402
import conjecture  # noqa: E402
from selectors import GreedySelector, NoveltySelector  # noqa: E402


GEN_THRESH = 0.5  # a conjecture "made the jump" if it gets >= half the held-out OOD points right


def run_regime(task, sandbox, selector, budget, model):
    stream = []
    all_ood_graded = []
    n_gen = 0        # generalizing conjectures this regime GENERATED
    n_gen_kept = 0   # ...of those, how many the selector KEPT (retained vs pruned)
    for step in range(budget):
        c = conjecture.generate_and_grade(task, sandbox, model=model, extra=selector.guidance())
        kept = selector.consider(c)
        all_ood_graded.append(c["ood_graded"])
        is_gen = c["ood_score"] >= GEN_THRESH
        if is_gen:
            n_gen += 1
            n_gen_kept += 1 if kept else 0
        stream.append({
            "step": step, "fit": round(c["fit_score"], 3), "ood": round(c["ood_score"], 3),
            "ood_graded": round(c["ood_graded"], 3), "ran": c["ran"], "kept": kept,
            "generalized": is_gen,
        })
        print(f"    [{selector.name}] step {step}: fit={c['fit_score']:.2f} "
              f"ood={c['ood_score']:.2f} ood_graded={c['ood_graded']:.2f} "
              f"ran={c['ran']} {'JUMP ' if is_gen else ''}{'KEPT' if kept else 'discarded'}")
    ans = selector.answer()
    pop = selector.retained_pop()
    return {
        "selector": selector.name,
        "answer_fit": round(ans["fit_score"], 3) if ans else 0.0,
        "answer_ood": round(ans["ood_score"], 3) if ans else 0.0,
        "answer_ood_graded": round(ans["ood_graded"], 3) if ans else 0.0,
        "best_ood_retained": round(max((c["ood_score"] for c in pop), default=0.0), 3),
        "best_ood_graded_retained": round(max((c["ood_graded"] for c in pop), default=0.0), 3),
        "best_ood_graded_generated": round(max(all_ood_graded, default=0.0), 3),
        # Count-unconfounded deception metric: of the JUMPS this regime generated, how many survived
        # selection. Per-conjecture, so novelty's larger archive gives it no free advantage here.
        "generalizers_generated": n_gen,
        "generalizers_kept": n_gen_kept,
        "generalizers_pruned": n_gen - n_gen_kept,
        "retained": len(pop),
        "budget": budget,
        "stream": stream,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="power-law")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("DISCOVERY_MODEL", "qwen2.5:7b"))
    args = ap.parse_args()

    task = tasks.make_task(args.task, seed=args.seed)
    print(f"TASK {task.name}  fit={task.fit_inputs[0]}..{task.fit_inputs[-1]} "
          f"OOD={task.ood_inputs[0]}..{task.ood_inputs[-1]}  model={args.model} budget={args.budget}")
    sandbox = DockerSandbox()

    print("\n== GREEDY (fit-climb) ==")
    greedy = run_regime(task, sandbox, GreedySelector(), args.budget, args.model)
    print("\n== NOVELTY (protected niche) ==")
    novelty = run_regime(task, sandbox, NoveltySelector(), args.budget, args.model)

    print("\n────────── RESULT ──────────")
    for r in (greedy, novelty):
        print(f"  {r['selector']:8s}  answer_fit={r['answer_fit']:.2f} answer_ood_graded={r['answer_ood_graded']:.2f}  "
              f"best_ood_generated={r['best_ood_graded_generated']:.2f}  "
              f"best_ood(binary)={r['best_ood_retained']:.2f}  retained={r['retained']}")

    # ── Deception measure: did the selector KEEP the jumps it GENERATED? Per-conjecture, so
    #    novelty's larger archive earns it nothing here — this is a pure selection-decision metric.
    print("\n────────── GENERALIZER RETENTION (the jump: ood>=0.5) ──────────")
    for r in (greedy, novelty):
        g, kp, pr = r["generalizers_generated"], r["generalizers_kept"], r["generalizers_pruned"]
        rate = (kp / g) if g else 0.0
        print(f"  {r['selector']:8s}  generated={g}  kept={kp}  pruned={pr}  retention={rate:.0%}")

    g_gen, g_kept, g_pruned = greedy["generalizers_generated"], greedy["generalizers_kept"], greedy["generalizers_pruned"]
    n_gen, n_kept, n_pruned = novelty["generalizers_generated"], novelty["generalizers_kept"], novelty["generalizers_pruned"]
    g_rate = (g_kept / g_gen) if g_gen else 0.0
    n_rate = (n_kept / n_gen) if n_gen else 0.0
    print("  [single seed — a hint, not proof]")
    if g_gen == 0 and n_gen == 0:
        best_fit = max(greedy["answer_fit"], novelty["answer_fit"])
        print(f"  → BELOW THRESHOLD: neither regime generated a jump (best fit={best_fit:.2f}). "
              "Generator-limited — NOT yet a selection result.")
    elif g_pruned > 0 and n_rate > g_rate:
        print(f"  → NOVELTY RETAINED JUMPS GREEDY PRUNED: greedy discarded {g_pruned} generalizer(s) "
              f"it generated (retention {g_rate:.0%}); novelty retention {n_rate:.0%}. "
              "Hypothesis SUPPORTED on this seed — multi-seed to confirm.")
    elif g_pruned > 0:
        print(f"  → greedy pruned {g_pruned} generalizer(s), but novelty didn't retain them better "
              f"(greedy {g_rate:.0%} vs novelty {n_rate:.0%}). No clear advantage this seed.")
    else:
        print("  → greedy kept every jump it generated (no pruning to rescue) — fit wasn't deceptive this run.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(PARENT, "workspace", f"discovery_ab_{args.task}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": task.name, "model": args.model, "seed": args.seed,
                   "greedy": greedy, "novelty": novelty}, f, indent=2)
    print(f"  full streams -> {out}")


if __name__ == "__main__":
    main()
