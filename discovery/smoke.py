"""Smoke test the discovery primitive: generate one conjecture, run it in the real sandbox,
grade fit + OOD. Run from the engine dir:  python discovery/smoke.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox import DockerSandbox  # noqa: E402
import tasks  # noqa: E402
import conjecture  # noqa: E402

task = tasks.make_task("power-law", seed=42)
print(f"task: {task.name}")
print(f"  fit inputs {task.fit_inputs[0]}..{task.fit_inputs[-1]} | OOD {task.ood_inputs[0]}..{task.ood_inputs[-1]}")

sandbox = DockerSandbox()
r = conjecture.generate_and_grade(task, sandbox)

print(f"ran: {r['ran']}")
print(f"fit_score:  {r['fit_score']:.2f}  ({r['fit_hits']}/{r['n_fit']})")
print(f"ood_score:  {r['ood_score']:.2f}  ({r['ood_hits']}/{r['n_ood']})  <- the jump signature")
print(f"bd (first 6 preds): {r['bd'][:6]}")
print(f"--- code head ---\n{r['code'][:400]}")
