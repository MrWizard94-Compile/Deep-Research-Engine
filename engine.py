import console_utf8  # noqa: F401  -- forces UTF-8 stdout/stderr before any emoji print
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from failure_registry import FailureRegistry
from llm_bridge import query_llm, OLLAMA_CODE_MODEL
from sandbox import DockerSandbox
import strategies  # the evolvable genome (prompt-craft); referenced as strategies.<fn> so reloads apply live
import grounding   # NON-evolvable, harness-owned success verdict

registry = FailureRegistry()
sandbox = DockerSandbox()

# How many hypotheses per research cycle, and how many self-repair attempts per hypothesis
# before it is retired. Both env-configurable.
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "12"))
MAX_DEBUG_ATTEMPTS = int(os.environ.get("MAX_DEBUG_ATTEMPTS", "3"))
# Safety net for LangGraph's super-step counter: worst case per hypothesis is
# PI+Lab+Docker+Review plus (Debug+Docker+Review) per repair, across every iteration.
RECURSION_LIMIT = MAX_ITERATIONS * (4 + 3 * MAX_DEBUG_ATTEMPTS) + 25

EXT_MAP = {"python": "exp.py", "rust": "main.rs", "cpp": "main.cpp", "java": "Main.java"}


class ResearchState(TypedDict):
    objective: str
    current_hypothesis: str
    code_filename: str
    test_output: str
    target_language: str
    iteration: int
    is_solved: bool
    debug_attempt: int
    pi_degraded: bool  # True when hypothesis generation failed (timeout/unparseable) -> cannot "solve"
    criterion: dict    # PI's pre-committed measurable acceptance criterion {metric, target, comparison}
    review_detail: str  # grounding verdict detail, fed back into the debug prompt


def _filename_for(lang):
    return EXT_MAP.get(str(lang).lower(), "exp.py")


def _clean_code(code_content):
    """Normalise a code-model response into raw source (handle dict fallback + md fences)."""
    if isinstance(code_content, dict):
        return "# Fallback code stub due to parsing mutation\nprint('FAILURE')"
    if "```" in code_content:
        lines = code_content.splitlines()
        code_content = "\n".join(l for l in lines if not l.strip().startswith("```"))
    return code_content


def node_principal_investigator(state: ResearchState):
    print(f"\n🧠 [STATE: Principal Investigator] (Iteration {state['iteration']})")
    failures = registry.get_summary(limit=3)
    if failures:
        print(f"📋 Recent Failure History:\n{failures}")

    prompt = strategies.craft_hypothesis_prompt(state['objective'], failures)

    # Primary: the reasoning model. If it degrades (timeout / unparseable), fall back to the
    # faster code model for the hypothesis rather than accepting a canned stub that would
    # then "solve" trivially. Only if BOTH fail is the iteration marked degraded.
    data = query_llm(prompt, structured=True)
    if isinstance(data, dict) and data.get("__fallback__"):
        print("⚠️ Reasoning model degraded on hypothesis — retrying via the code model...")
        data = query_llm(prompt, structured=True, model=OLLAMA_CODE_MODEL)

    degraded = (not isinstance(data, dict)) or bool(data.get("__fallback__"))
    if not isinstance(data, dict):
        data = {"hypothesis": "", "target_language": "python"}

    hypo = data.get("hypothesis") or f"Degraded hypothesis (iteration {state['iteration']})"
    if not degraded:
        is_vetoed, _ = registry.is_redundant(hypo)
        if is_vetoed:
            hypo = f"{hypo} (Mutation Shift {state['iteration']})"
    else:
        print("⛔ Hypothesis generation failed on both models — this iteration cannot count as a solve.")

    # The PI commits to a measurable criterion up front; grounding.py judges against it.
    criterion = grounding.normalize_criterion(data)
    print(f"🎯 Acceptance criterion: {criterion['metric']} {criterion['comparison']} {criterion['target']}")

    return {
        "current_hypothesis": hypo,
        "target_language": data.get("target_language", "python"),
        "debug_attempt": 0,       # fresh hypothesis -> reset the repair budget
        "pi_degraded": degraded,
        "criterion": criterion,
        "review_detail": "",
    }


def route_after_pi(state: ResearchState):
    # A degraded hypothesis skips code generation entirely and goes straight to retirement —
    # no point coding (and accidentally "solving") a fallback stub.
    return "Review" if state.get("pi_degraded") else "Lab"


def node_lab_engineer(state: ResearchState):
    print("⚡ [STATE: Lab Engineer] Coding solution module...")

    criterion_text = grounding.criterion_instruction(state.get('criterion'))
    prompt = strategies.craft_code_prompt(state['current_hypothesis'], state['target_language'], criterion_text)
    code_content = _clean_code(query_llm(prompt, model=OLLAMA_CODE_MODEL))

    filename = _filename_for(state['target_language'])
    sandbox.write_file(filename, code_content)
    print(f"💾 Code file generated: workspace/{filename}")
    return {"code_filename": filename}


def node_debug(state: ResearchState):
    attempt = state.get('debug_attempt', 0) + 1
    filename = state['code_filename'] or _filename_for(state['target_language'])
    print(f"🔧 [STATE: Debug Engineer] Repair attempt {attempt}/{MAX_DEBUG_ATTEMPTS} on workspace/{filename}")

    current_code = sandbox.read_file(filename)
    # Feed the coder both the raw sandbox output AND the harness's grounded verdict, so it
    # knows whether it crashed, forgot the METRIC line, or simply missed the target.
    error = str(state['test_output'])
    detail = state.get('review_detail')
    if detail:
        error = f"{error}\n\nHARNESS VERDICT: {detail}"
    criterion_text = grounding.criterion_instruction(state.get('criterion'))
    prompt = strategies.craft_debug_prompt(state['current_hypothesis'], state['target_language'],
                                           current_code, error, criterion_text)
    fixed = _clean_code(query_llm(prompt, model=OLLAMA_CODE_MODEL))

    sandbox.write_file(filename, fixed)
    print(f"💾 Patched code rewritten: workspace/{filename}")
    return {"debug_attempt": attempt}


def node_sandbox_runtime(state: ResearchState):
    print("🐳 [STATE: Sandbox Verification Container]")

    lang = str(state['target_language']).lower()
    if lang == "rust": cmd = "rustc main.rs && ./main"
    elif lang == "cpp": cmd = "g++ main.cpp -o main && ./main"
    elif lang == "java": cmd = "javac Main.java && java Main"
    else: cmd = "python3 exp.py"

    runtime_logs = sandbox.run_container_test(cmd)

    print("--- 🖥️  Raw Container Execution Output ---")
    print(runtime_logs)
    print("-----------------------------------------")

    return {"test_output": runtime_logs}


def node_peer_review(state: ResearchState):
    print("🧐 [STATE: Peer Review Analysis]")

    # Degraded hypothesis: retire immediately. It NEVER counts as solved, no matter what any
    # stale output says — this is what stops a timeout from forging a win.
    if state.get('pi_degraded'):
        print("📉 Hypothesis generation was degraded — retiring iteration (cannot be a solve).")
        registry.log_failure("[PI-DEGRADED]",
                             "hypothesis generation failed on both models (likely timeout); no real hypothesis",
                             state['target_language'])
        return {"is_solved": False, "iteration": state['iteration'] + 1}

    raw_logs = state['test_output']
    if isinstance(raw_logs, dict):
        raw_logs = str(raw_logs)

    # Authoritative, NON-gameable verdict: the harness checks the committed metric.
    verdict = grounding.evaluate(raw_logs, state.get('criterion'))
    detail = verdict.get("detail", "")

    if verdict.get("is_solved"):
        print(f"🎉 Breakthrough Verified by the harness: {detail}")
        return {"is_solved": True, "review_detail": detail}

    print(f"📉 Not solved: {detail}")
    attempt = state.get('debug_attempt', 0)
    if attempt < MAX_DEBUG_ATTEMPTS:
        # Don't log/abandon yet — route to the Debug node to repair this very code.
        print(f"🔁 Dispatching self-repair ({attempt + 1}/{MAX_DEBUG_ATTEMPTS}).")
        return {"is_solved": False, "review_detail": detail}

    # Repair budget exhausted -> retire this hypothesis and advance the iteration counter.
    print(f"📉 Hypothesis retired after {MAX_DEBUG_ATTEMPTS} repair attempts. Logging failure context.")
    registry.log_failure(state['current_hypothesis'], detail[:200], state['target_language'])
    return {"is_solved": False, "iteration": state['iteration'] + 1, "review_detail": detail}


def route_after_review(state: ResearchState):
    if state['is_solved']:
        return END
    if state['iteration'] > MAX_ITERATIONS:
        return END
    # A degraded iteration is never debugged (there is no real code to repair) — go straight
    # back to the PI for a fresh hypothesis.
    if state.get('pi_degraded'):
        return "PI"
    if state.get('debug_attempt', 0) < MAX_DEBUG_ATTEMPTS:
        return "Debug"
    return "PI"


workflow = StateGraph(ResearchState)
workflow.add_node("PI", node_principal_investigator)
workflow.add_node("Lab", node_lab_engineer)
workflow.add_node("Debug", node_debug)
workflow.add_node("Docker", node_sandbox_runtime)
workflow.add_node("Review", node_peer_review)
workflow.set_entry_point("PI")
workflow.add_conditional_edges("PI", route_after_pi, {"Lab": "Lab", "Review": "Review"})
workflow.add_edge("Lab", "Docker")
workflow.add_edge("Debug", "Docker")
workflow.add_edge("Docker", "Review")
workflow.add_conditional_edges("Review", route_after_review, {"Debug": "Debug", "PI": "PI", END: END})
app = workflow.compile()
