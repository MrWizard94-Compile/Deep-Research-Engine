# Self-Improvement Architecture

The engine improves itself by evolving a **genome** under a sandboxed safety gate. The
orchestrator never rewrites itself, so a bad mutation can degrade quality but can never
brick the runtime — and the gate rejects degradations before they land.

## Components

| File | Role | May the LLM rewrite it? |
|------|------|--------------------------|
| `strategies.py` | **Genome** — pure PROMPT-craft functions only (no verdict). | **Yes** (only this) |
| `grounding.py` | **Non-evolvable success authority.** Checks a measurable METRIC against the PI's pre-committed criterion. The code cannot self-certify. | No |
| `genome_harness.py` | Stdlib-only STRUCTURAL gate (does the genome honour the prompt contract?). Runs in Docker against candidates. | No |
| `self_improver.py` | One generation: ask → static scan → sandbox structural-validate → **canary fitness** → promote only on STRICT improvement. | No |
| `engine.py` | LangGraph orchestrator. References the genome as `strategies.<fn>` so hot-reload applies live. | No |
| `objectives.py` | Supplies research goals; advances to a fresh frontier after each solve. | No |
| `main.py` | Open-ended meta-loop: research cycle → (archive solution + advance objective on win) → self-improvement every generation → reload → repeat. | No |

## Open-ended loop (main.py)

The loop **never halts on a win**. Each generation:
1. Runs one research cycle on the current objective (PI → Lab → Docker → Review, capped at 12 iterations).
2. **On solve:** the winning code is copied to `solutions/` (with a manifest), then
   `ObjectiveProvider.advance()` rolls onto the next frontier objective (seed backlog
   first, then LLM-generated) and a fresh pre-mortem seeds constraints for it.
3. **Win or lose:** the genome self-improvement pass runs, so the strategy keeps evolving.

`META_GENERATIONS` caps the run (default 5); **set it to `0` to run forever.**

Per-role models ([llm_bridge.py](../llm_bridge.py)): `OLLAMA_MODEL` (reasoning, default
`deepseek-r1-highctx`) drives hypotheses/pre-mortem; `OLLAMA_CODE_MODEL` (default
`qwen2.5-coder:14b`) drives all code generation — a reasoning model cannot finish a
code-gen call on modest hardware before timing out.

Artifacts: `solutions/` (banked wins + manifest), `workspace/objectives_log.json`
(solved-objective history), `workspace/genome_log.json` (every promote/reject verdict),
`genome_archive/` (superseded genomes).

## Objective grounding — non-gameable success (grounding.py)

Success is decided by the HARNESS, not the generated code. The PI commits, in its
hypothesis JSON, to a measurable criterion `{success_metric, target, comparison}`. The code
must print `METRIC <name>=<value>` with the REAL measured value; `grounding.evaluate` checks
`value <comparison> target` in a cleanly-exited container. Printing `SUCCESS` does nothing.
This kills the trivial-stub gaming (a 2×2 matmul printing SUCCESS now scores 0).

## Honest fitness — no random walk (self_improver.py)

Genome promotion requires **strict** fitness improvement, where fitness = `100 ×
canaries_solved + structural_score`. Canaries are fixed known-answer tasks (e.g. sum 1..1000
== 500500); the candidate genome's `craft_code_prompt` is run through the real code model +
sandbox + grounding to see if it actually elicits working, correctly-measured code. Equal
fitness holds the incumbent — so a good genome correctly goes dormant instead of churning
version numbers (the old loop promoted 10× at a flat 95.0). Set `CANARY_FITNESS=0` for
structural-only scoring (faster, weaker).

## Genome contract (enforced by `genome_harness.evaluate_genome`)

```
GENOME_VERSION: int
craft_hypothesis_prompt(objective, failure_summary) -> str          # references both args; asks for success_metric/target/comparison JSON
craft_code_prompt(hypothesis, target_language, criterion) -> str    # references hypothesis + criterion; instructs the METRIC line
craft_debug_prompt(hypothesis, target_language, code, error, criterion) -> str  # references all + METRIC; REPAIR not restart
```
The genome is PROMPT-CRAFT ONLY — the success verdict lives in `grounding.py`, not here, so
evolution can never produce a gameable success check. The genome must import nothing but
stdlib `re` and perform no I/O. It also carries a
`SANDBOX_CAPABILITIES` string describing exactly what the container provides (Python +
numpy/pandas/scipy/…, Rust std-only via `rustc`, g++, JDK 21) so the coder stops reaching
for libraries/crates that cannot exist in the sandbox.

## Self-debug loop (engine.py)

A failing program is no longer thrown away. The graph is:

```
PI -> Lab -> Docker -> Review --(solved)--------> END
                          |--(fail, budget left)--> Debug -> Docker -> Review ...
                          |--(fail, budget spent)-> PI (new hypothesis)
                          |--(iteration cap)------> END
```

`Debug` feeds the coder its *previous code + the exact sandbox error* and asks it to
REPAIR (not restart). Up to `MAX_DEBUG_ATTEMPTS` (default 3) repairs per hypothesis before
it is retired and logged. Only on retirement is the failure recorded and the iteration
counter advanced. `FailureRegistry.get_summary` now surfaces the *reason* each hypothesis
failed (not just its name), so the PI proposes around known failure modes instead of
repeating them. Caps: `MAX_ITERATIONS` (default 12), `MAX_DEBUG_ATTEMPTS` (default 3),
both env-configurable; `RECURSION_LIMIT` is derived from them.

## Promotion pipeline (one generation)

1. **Performance context** built from `workspace/failure_memory.json`.
2. **LLM** is asked to rewrite the genome under the hard contract.
3. **Static safety scan** (host, no execution) — rejects any I/O / OS / dynamic-exec
   construct (`import os`, `subprocess`, `open(`, `eval`, `exec`, `__import__`, …).
4. **Sandbox validation** (Docker) — the candidate + harness are mounted into the
   container and run there. Must pass every golden regression case and both prompt
   contracts. No LLM-authored code ever executes on the host.
5. **Promote iff** `ok` and `candidate_score >= incumbent_score`. The incumbent is
   archived to `genome_archive/genome_v<n>_<stamp>.py`; rollback is a single file copy.
6. On promotion, `main.py` calls `importlib.reload(strategies)` so the next research
   cycle uses the improved genome immediately.

## Scoring gradient

`score = (golden_cases_passed * 10) + prompt_quality_token_bonuses`. Correctness
dominates (the gate requires 8/8 golden cases); the quality bonus gives evolution a
gradient to climb toward richer, more effective prompts.

## Knobs

- `META_GENERATIONS` (env, default `3`) — number of research→improve generations.
- `OLLAMA_MODEL`, `OLLAMA_URL` (env) — model + endpoint for the bridge.

## Safety properties

- Orchestrator is immutable at runtime; only the genome surface changes.
- Untrusted candidate code runs **only** inside the Docker sandbox.
- A candidate must clear a static scan **and** a sandboxed regression suite **and** beat
  no-regression on score before it can touch disk.
- Every promotion is reversible (archived incumbent + `workspace/genome_log.json` audit trail).
