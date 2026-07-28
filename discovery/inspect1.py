"""Show ONE conjecture's actual code + sandbox output, to diagnose why fit is stuck at ~0."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (PARENT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox import DockerSandbox  # noqa: E402
from llm_bridge import query_llm  # noqa: E402
import tasks  # noqa: E402
import conjecture  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-coder:7b"
task = tasks.make_task("power-law", seed=1)
print(f"task {task.name}  true: f(1)={task.true_fn(1):.3f} f(10)={task.true_fn(10):.3f} f(50)={task.true_fn(50):.3f}")

queries = task.all_inputs
prompt = conjecture.craft_conjecture_prompt(task, queries)
raw = query_llm(prompt, model=model)
code = conjecture._strip_fences(raw)
print("\n=== GENERATED CODE ===")
print(code[:1600])

sb = DockerSandbox()
sb.write_file("_conjecture.py", code)
logs = sb.run_container_test("python3 _conjecture.py")
print("\n=== SANDBOX OUTPUT (head) ===")
print(logs[:700])

preds = conjecture._parse_preds(logs, len(queries))
nf = len(task.fit_inputs)
print("\n=== preds vs true ===")
for i in [0, 1, 2, nf, len(queries) - 1]:
    x = queries[i]
    print(f"  x={x}: pred={preds.get(i)}  true={round(task.true_fn(x), 3)}")
