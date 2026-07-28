import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import console_utf8  # noqa: F401  -- forces UTF-8 stdout/stderr before any emoji print

# Import the pre-defined ResearchState class alongside the app
import importlib
import shutil
import time
import json
from engine import app, ResearchState, RECURSION_LIMIT
from failure_registry import FailureRegistry
from llm_bridge import query_llm
from sandbox import DockerSandbox
from self_improver import SelfImprover
from objectives import ObjectiveProvider
import strategies

def execute_pre_mortem_pass(goal_prompt):
    print("\n🔮 [BOOT] Initiating Local Pre-Mortem Pessimistic Analysis...")
    analysis_prompt = f"OBJECTIVE: {goal_prompt}\nList 3 naive approaches that will fail. Format strictly as JSON: {{ 'anticipated_failures': [ {{ 'naive_approach': '...', 'why_it_will_fail': '...' }} ] }}"

    try:
        response = query_llm(analysis_prompt, structured=True)

        if not response or not isinstance(response, dict) or "anticipated_failures" not in response:
            print("⚠️ Model output messy JSON. Injecting default architectural safety constraints...")
            response = {
                "anticipated_failures": [
                    {"naive_approach": "Standard Von Neumann Shared Bus Memory Allocation", "why_it_will_fail": "Creates classic Von Neumann memory bottlenecks under massive context pooling loads."},
                    {"naive_approach": "Naively Expanding Context via Dense Floating-Point Attention Matrices", "why_it_will_fail": "Triggers quadratic O(N^2) memory consumption spikes, killing hardware execution performance."},
                    {"naive_approach": "Applying Simple Top-K Probability Softmax Layer Filters", "why_it_will_fail": "Retains the probabilistic, unreliable nature of token predictions instead of enforcing deterministic code execution logic."}
                ]
            }

        failures = response.get("anticipated_failures", [])
        FailureRegistry().seed_pre_mortem(failures)
        print(f"📉 Experience Ingested. Seeded {len(failures)} initial constraints directly to local memory.")
    except Exception as e:
        print(f"⚠️ Pre-mortem layer bypassed smoothly: {e}")

def run_research_cycle(user_objective):
    """One full research cycle: PI -> Lab -> Docker -> Review until solved or capped.

    Returns (solved, winning_code_filename). The code filename is tracked so a winning
    solution can be archived before the next cycle overwrites the workspace.
    """
    initial_context = ResearchState(
        objective=user_objective,
        current_hypothesis="",
        code_filename="",
        test_output="",
        target_language="python",
        iteration=1,
        is_solved=False,
        debug_attempt=0,
        pi_degraded=False,
        criterion={},
        review_detail="",
    )
    solved = False
    last_code_file = ""
    for state_update in app.stream(initial_context, {"recursion_limit": RECURSION_LIMIT}):
        for node_state in state_update.values():
            if isinstance(node_state, dict):
                if node_state.get("code_filename"):
                    last_code_file = node_state["code_filename"]
                if node_state.get("is_solved"):
                    solved = True
    return solved, last_code_file


def archive_solution(objective, code_file, generation, solutions_dir="solutions"):
    """Copy a winning code file into a durable solutions archive with a manifest entry."""
    if not code_file:
        return None
    src = os.path.join("workspace", code_file)
    if not os.path.exists(src):
        return None
    os.makedirs(solutions_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(solutions_dir, f"gen{generation}_{stamp}_{code_file}")
    shutil.copyfile(src, dst)

    manifest_path = os.path.join(solutions_dir, "solutions_manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = []
    manifest.append({
        "generation": generation,
        "objective": objective,
        "solution_file": dst,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return dst


def run_self_improvement(improver):
    """One self-improvement generation: mutate the genome, validate, promote or roll back."""
    print("\n🧬 [META] Self-improvement generation: evolving the strategy genome...")
    perf = improver.build_performance_context()
    result = improver.run_generation(perf)
    if result["promoted"]:
        importlib.reload(strategies)  # hot-swap the genome for the next research cycle
        print(f"✅ [META] Genome promoted v{result['from_version']} -> v{result['to_version']} "
              f"(fitness {result['incumbent_fitness']} -> {result['candidate_fitness']}, "
              f"canaries {result['incumbent_canaries']} -> {result['candidate_canaries']}). "
              f"Archived: {result['archived']}")
    else:
        print(f"🛡️  [META] Genome held (v{result['from_version']}). Reason: {result['reason']}")
    return result


if __name__ == "__main__":
    # FIRST: refuse to run if another engine instance is alive — concurrent runs corrupt
    # the shared workspace and contend for the single GPU. (Before any other work.)
    import single_instance
    single_instance.acquire_or_exit()

    print("="*60)
    print("     LOCAL OLLAMA OPEN-ENDED RESEARCH ENGINE RUNTIME    ")
    print("="*60)

    # META_GENERATIONS caps the run; 0 means run forever (open-ended).
    generations = int(os.environ.get("META_GENERATIONS", "5"))

    objectives = ObjectiveProvider()
    improver = SelfImprover(sandbox=DockerSandbox())
    execute_pre_mortem_pass(objectives.current)

    horizon = "∞" if generations == 0 else str(generations)
    print(f"\n🚀 Commencing OPEN-ENDED Research Loop ({horizon} generations). "
          f"Never halts on a win — solves, banks the solution, advances the frontier, and keeps evolving.")

    gen = 0
    while generations == 0 or gen < generations:
        gen += 1
        objective = objectives.current
        print("\n" + "#"*60)
        print(f"# GENERATION {gen}/{horizon}  (genome v{strategies.GENOME_VERSION}, solved so far: {objectives.solved_count})")
        print(f"# OBJECTIVE: {objective}")
        print("#"*60)

        solved, code_file = run_research_cycle(objective)

        if solved:
            saved = archive_solution(objective, code_file, gen)
            print(f"\n🏆 Objective SOLVED. Winning code banked -> {saved}")
            next_objective = objectives.advance(objective, saved)
            print(f"➡️  Advancing frontier. New objective:\n    {next_objective}")
            execute_pre_mortem_pass(next_objective)  # seed pessimistic constraints for the new goal
        else:
            print("\n📉 Objective not solved within the iteration cap. Keeping it; evolving strategy.")

        # Open-ended core: evolve the genome EVERY generation, win or lose.
        run_self_improvement(improver)

    print(f"\n🧾 Open-ended run complete. Objectives solved: {objectives.solved_count}. "
          f"Solutions -> solutions/ | Genome history -> workspace/genome_log.json | Objectives -> workspace/objectives_log.json")
