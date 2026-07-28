"""The frame-invention (abduction) experiment.

For each structural complexity level, generate several randomized latent-program tasks and give the
model N INDEPENDENT one-shot attempts per task (default: no feedback between attempts = no gradient).
Grade on held-out OOD by exact match. Report the abduction-ceiling curve: the fraction of tasks where
the model ever inferred the true mechanism (generalized perfectly OOD = "the jump").

    python discovery/run_abduction.py --complexity sweep --seeds 4 --shots 4 --model qwen2.5-coder:14b
    python discovery/run_abduction.py --complexity 2 --seeds 6 --shots 6 --mode search   # gradient ON

Modes:
  oneshot (default) — each attempt is independent; measures pure one-shot abduction.
  search            — carry the best-fitting prior program forward as a hint; measures whether
                      gradient-guided SEARCH rescues what one-shot abduction misses (the contrast).
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
import latent_programs  # noqa: E402
import abduction  # noqa: E402


def run_task(task, sandbox, shots, model, mode):
    attempts = []
    prior = ""
    for s in range(shots):
        a = abduction.attempt(task, sandbox, model=model, prior=(prior if mode == "search" else ""))
        attempts.append(a)
        print(f"      shot {s}: fit={a['fit']:.2f} ood={a['ood']:.2f} ran={a['ran']} "
              f"{'JUMP' if a['jump'] else ''}")
        # search mode only: keep the best-fitting program so far as the next hint (gradient ON)
        if mode == "search" and a["ran"] and a["fit"] > 0 and (not prior or a["fit"] >= 0.5):
            prior = a["code"][:600]
    best_ood = max((a["ood"] for a in attempts), default=0.0)
    best_fit = max((a["fit"] for a in attempts), default=0.0)
    jumped = any(a["jump"] for a in attempts)
    return {
        "task": task.name, "mechanism": task.description,   # mechanism logged host-side for our eyes
        "best_fit": round(best_fit, 3), "best_ood": round(best_ood, 3),
        "jumped": jumped, "shots": shots,
        "stream": [{"fit": round(a["fit"], 3), "ood": round(a["ood"], 3),
                    "ran": a["ran"], "jump": a["jump"]} for a in attempts],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexity", default="sweep", help="1 | 2 | 3 | sweep")
    ap.add_argument("--seeds", type=int, default=4, help="latent tasks per complexity level")
    ap.add_argument("--shots", type=int, default=4, help="independent attempts per task")
    ap.add_argument("--mode", default="oneshot", choices=["oneshot", "search"])
    ap.add_argument("--model", default=os.environ.get("DISCOVERY_MODEL", "qwen2.5:7b"))
    args = ap.parse_args()

    levels = [1, 2, 3] if args.complexity == "sweep" else [int(args.complexity)]
    sandbox = DockerSandbox()
    print(f"ABDUCTION TEST  mode={args.mode} model={args.model} shots={args.shots} "
          f"seeds/level={args.seeds}  complexity={levels}")
    print("(each task: hidden rule is a RANDOM latent program; OOD is held-out larger x; "
          "jump = perfect OOD generalization)\n")

    results = {"mode": args.mode, "model": args.model, "levels": {}}
    for c in levels:
        print(f"===== complexity {c} =====")
        tasks_out = []
        for seed in range(args.seeds):
            task = latent_programs.make_latent_task(seed=seed, complexity=c)
            print(f"  task c{c}.seed{seed}  [mechanism hidden from model]")
            tasks_out.append(run_task(task, sandbox, args.shots, args.model, args.mode))
        jump_rate = sum(1 for t in tasks_out if t["jumped"]) / len(tasks_out)
        mean_best_ood = sum(t["best_ood"] for t in tasks_out) / len(tasks_out)
        mean_best_fit = sum(t["best_fit"] for t in tasks_out) / len(tasks_out)
        results["levels"][c] = {"jump_rate": round(jump_rate, 3),
                                "mean_best_ood": round(mean_best_ood, 3),
                                "mean_best_fit": round(mean_best_fit, 3),
                                "tasks": tasks_out}
        print(f"  -> complexity {c}: JUMP RATE {jump_rate:.0%}  "
              f"(mean best OOD {mean_best_ood:.2f}, mean best fit {mean_best_fit:.2f})\n")

    print("────────── ABDUCTION CEILING ──────────")
    for c in levels:
        L = results["levels"][c]
        print(f"  complexity {c}:  jump_rate={L['jump_rate']:.0%}  "
              f"fit={L['mean_best_fit']:.2f} -> ood={L['mean_best_ood']:.2f}")
    any_jump = any(results["levels"][c]["jump_rate"] > 0 for c in levels)
    top = max(levels)
    if not any_jump:
        print("  → NO JUMP at any complexity: the model never inferred a randomized mechanism that "
              "generalized. Frame-invention not demonstrated (could be generator-limited — note the model).")
    elif results["levels"][top]["jump_rate"] > 0:
        print(f"  → jumps occurred even at complexity {top}. The gap between fit and ood is the tell: "
              "where fit is high but ood low, the model FIT the examples without finding the frame.")
    else:
        print("  → jumps only at low complexity (single/near-single motifs = recognition). Frame-invention "
              "degrades as structure grows — the honest signature of search-not-abduction.")
    print("  [randomized frames per instance; contamination-resistant but motifs are known primitives]")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(PARENT, "workspace", f"abduction_{args.mode}_{stamp}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  full results -> {out}")


if __name__ == "__main__":
    main()
