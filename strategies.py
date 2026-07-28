"""
strategies.py - THE EVOLVABLE GENOME (prompt-craft only).

The ONLY file the self-improver may rewrite. Pure, side-effect-free PROMPT functions.
The success VERDICT lives in non-evolvable grounding.py so the genome can never evolve a
gameable success check. This genome only phrases prompts:

    * craft_hypothesis_prompt  - one falsifiable hypothesis + measurable criterion (JSON)
    * craft_code_prompt        - self-contained program that emits METRIC
    * craft_debug_prompt       - repair the failing program (not restart)

engine.py references these as strategies.<fn> so a promoted genome hot-reloads live.

CONTRACT (enforced by genome_harness.evaluate_genome - do not break it):
    GENOME_VERSION: int
    craft_hypothesis_prompt(objective: str, failure_summary: str) -> str
    craft_code_prompt(hypothesis: str, target_language: str, criterion: str) -> str
    craft_debug_prompt(hypothesis: str, target_language: str, code: str, error: str, criterion: str) -> str

Import-free except stdlib re. No I/O. Auditable pure prompt genome.
"""

import re

GENOME_VERSION = 3

# Ground truth about the Docker sandbox (see Dockerfile). Stops the coder reaching for
# libraries/crates that cannot exist in the container.
SANDBOX_CAPABILITIES = (
    "EXECUTION SANDBOX (hard constraints - code that violates these will not run):\n"
    "- python: run as `python3 exp.py`. Standard library PLUS preinstalled: numpy, pandas, "
    "scipy, requests, redis, pyzmq. No internet, no pip install at runtime.\n"
    "- rust: compiled with `rustc main.rs` directly (NO Cargo, NO Cargo.toml). "
    "STANDARD LIBRARY ONLY - external crates (rand, ndarray, serde, etc.) are UNAVAILABLE "
    "and will fail with 'unresolved import'. Implement RNG/math/etc. yourself with std.\n"
    "- cpp: compiled with `g++ main.cpp -o main`. C++ standard library only, no third-party libs.\n"
    "- java: single file `Main.java`, `javac`/`java`, JDK 21 standard library only.\n"
    "Pick the language best suited to a self-contained, dependency-free implementation."
)

# Shared coding discipline: optimized for correct, efficient solutions under sandbox limits.
# Explicit playbook targets common hard tasks (DP, sieves, search) without timing out.
_CODING_DISCIPLINE = (
    "IMPLEMENTATION RULES (follow strictly):\n"
    "1. Implement a CORRECT algorithm end-to-end. Prefer standard efficient methods:\n"
    "   - primes / primality counts: Sieve of Eratosthenes (O(n log log n)); never trial-divide "
    "     every integer when n >= 1e5\n"
    "   - edit distance / LCS / coin-change / knapsack: classic DP tables; initialize bases carefully\n"
    "   - N-queens / combinatorial search: backtracking with pruning; count DISTINCT solutions\n"
    "   - factorial trailing zeros: count factors of 5 (floor(n/5)+floor(n/25)+...), not full n!\n"
    "   - closed forms (sum 1..n = n*(n+1)/2) when exact and cheaper than loops\n"
    "2. Complexity: for large bounds (n >= 1e5) you MUST use O(n log n) or better. Naive nested "
    "loops time out. Preallocate arrays; avoid per-iteration allocations in hot loops.\n"
    "3. Edge cases: empty inputs, n=0/1, inclusive vs exclusive bounds, off-by-one, integer "
    "overflow (prefer 64-bit ints / bigints where sums or products grow).\n"
    "4. HONEST METRIC: compute the metric by actually running the algorithm. Do NOT invent, "
    "guess, or hardcode a placeholder constant. The REAL MEASURED value must appear on the "
    "METRIC line. A fabricated number is a failed experiment.\n"
    "5. Print EXACTLY one gradeable line to stdout in this form (name matches the criterion):\n"
    "      METRIC <name>=<number>\n"
    "   integer or float; no units; no extra text on that line. Example: METRIC total=500500\n"
    "   Do not print SUCCESS - the harness ignores it. Extra diagnostics only on OTHER lines.\n"
    "6. Exit 0 with no uncaught exceptions / panics / compile errors.\n"
    "7. SELF-CONTAINED: single file, only sandbox-available libraries, no network, no files.\n"
    "8. Return RAW CODE only - no markdown fences, no prose before or after the code."
)

_LANG_HINTS = {
    "python": (
        "Python specifics: file is exp.py, top-level script entry. Use stdlib freely; "
        "numpy/pandas/scipy are available but pure Python is fine for algorithms. "
        "Use range carefully for exclusive upper bounds. Print with "
        "print(f\"METRIC name={value}\") or print(\"METRIC name=%s\" % value)."
    ),
    "rust": (
        "Rust specifics: single main.rs via rustc (no Cargo). std only. Prefer i64/u64 for "
        "counts. Put logic in fn main(); print with println!(\"METRIC name={}\", value). "
        "Avoid unwrap on fallible ops; pre-size Vec when possible."
    ),
    "cpp": (
        "C++ specifics: single main.cpp, g++. Standard headers only "
        "(iostream, vector, algorithm, cstdint, string, cmath). Use long long for large counts. "
        "Print with: cout << \"METRIC name=\" << value << \"\\n\";"
    ),
    "java": (
        "Java specifics: public class Main in Main.java, JDK 21 stdlib only. Use long for large "
        "counts. Print with: System.out.println(\"METRIC name=\" + value);"
    ),
}

# Lightweight algorithm playbook injected into code prompts so the model picks the right
# method without rediscovering it under time pressure.
_ALGORITHM_PLAYBOOK = (
    "ALGORITHM PLAYBOOK (pick the matching recipe when the hypothesis fits):\n"
    "- Counting primes below N: boolean sieve; mark multiples from p*p; count True entries; "
    "  remember 0 and 1 are not prime; exclusive upper bound means indices 0..N-1.\n"
    "- Levenshtein edit distance: (m+1)x(n+1) DP; dp[i][0]=i, dp[0][j]=j; min of insert/"
    "  delete/substitute; final cell is the answer.\n"
    "- LCS length: (m+1)x(n+1) DP; if equal take diag+1 else max(left, up).\n"
    "- Coin change (number of combinations): 1D DP ways[0]=1; outer loop coins, inner amounts; "
    "  order matters for combinations vs permutations - coins outer gives combinations.\n"
    "- Trailing zeros of n!: sum floor(n/5^k) for k=1,2,... while 5^k <= n.\n"
    "- N-queens solutions: place row by row; track used cols and both diagonals; count leaves.\n"
    "When the hypothesis is performance-oriented (throughput, speedup, power), still implement "
    "a real workload, measure the named metric, and print it - never stub the number."
)


def _lang_hint(target_language: str) -> str:
    key = (target_language or "python").strip().lower()
    if key in ("py", "python3"):
        key = "python"
    elif key in ("rs",):
        key = "rust"
    elif key in ("c++", "cxx"):
        key = "cpp"
    return _LANG_HINTS.get(key, _LANG_HINTS["python"])


def _normalize_failure_block(failure_summary: str) -> str:
    """Light cleanup so empty/None summaries still produce a usable block."""
    text = failure_summary if isinstance(failure_summary, str) else str(failure_summary)
    text = text.strip()
    if not text or text.lower() in ("none", "n/a", "no failures", "no runs recorded yet."):
        return (
            "(none yet - explore a novel, concrete approach; still make the criterion "
            "strict and measurable so a wrong program cannot pass by accident)"
        )
    # Collapse runaway whitespace without changing content tokens the harness checks.
    return re.sub(r"[ \t]+\n", "\n", text)


def craft_hypothesis_prompt(objective: str, failure_summary: str) -> str:
    """Prompt the Principal Investigator for ONE hypothesis AND a measurable criterion."""
    failures = _normalize_failure_block(failure_summary)
    return f"""
You are the Principal Investigator of an autonomous computational research lab.

OBJECTIVE:
{objective}

RECENT ATTEMPTS THAT FAILED (do not repeat these - extract the WHY of each failure and
avoid that entire class of mistake: wrong algorithm family, unavailable dependency,
unmeasurable claim, metric mismatch, timeout, off-by-one, hardcoded fake metric, etc.):
{failures}

Task: Propose ONE specific, unique, FALSIFIABLE computational hypothesis that directly
advances the OBJECTIVE above. Prefer an approach demonstrable with a SELF-CONTAINED,
dependency-free program that finishes in a few seconds inside a constrained sandbox.

Requirements for a good hypothesis:
- Concrete mechanism (data structure, algorithm, scheduling policy, numeric method), not a slogan.
- Empirically testable in one short run with a single numeric outcome.
- Distinct from every failed attempt listed above (mutate the mechanism, not just the wording).
- Prefer deterministic methods over probabilistic ones when both could work.
- Scope the experiment so it is implementable with sandbox-legal libraries only.

You must also commit, IN ADVANCE, to a MEASURABLE acceptance criterion. Success is judged
externally by the harness against this criterion - the program cannot self-certify.
Choose:
  * success_metric: a single snake_case metric name that genuinely reflects the claim
    (e.g. max_error, speedup_ratio, throughput_ops, memory_bytes, prime_count, total,
    edit_distance, lcs_length, coin_ways)
  * target: a numeric threshold that is meaningful but achievable in a short run
  * comparison: one of ">=", ">", "<=", "<", "=="
Prefer "==" when the hypothesis predicts an exact integer; use ">="/" <=" for continuous
performance/error metrics. Never choose a target a trivial stub can hit by printing a constant.
The metric name you choose will be the name the program must print on its METRIC line.

Format the response strictly as a single JSON object. No conversational filler text,
no markdown fences, no commentary outside the JSON.

Schema:
{{
  "hypothesis": "one-sentence unique architecture / algorithm theory",
  "target_language": "python|rust|cpp|java",
  "success_metric": "metric_name",
  "target": <number>,
  "comparison": ">="
}}
""".strip()


def craft_code_prompt(hypothesis: str, target_language: str, criterion: str) -> str:
    """Prompt the Lab Engineer to emit a self-contained program that REPORTS its metric."""
    lang = (target_language or "python").strip()
    return f"""
ROLE: Lab Engineer. Emit a complete, correct program - nothing else.

HYPOTHESIS TO IMPLEMENT:
{hypothesis}

Language Target: {lang}

{SANDBOX_CAPABILITIES}

{_lang_hint(lang)}

{criterion}

{_ALGORITHM_PLAYBOOK}

{_CODING_DISCIPLINE}

Task checklist:
- Translate the hypothesis into a working algorithm that actually performs the work.
- Choose an efficient method from the playbook when it applies (sieve, DP, closed form).
- Measure the required metric honestly (REAL MEASURED value from execution, not a guess).
- Print the required METRIC line exactly once in the form METRIC <name>=<value>.
- Keep the program self-contained and sandbox-legal.
- Do not print SUCCESS.
- Return RAW CODE only.

Begin the program now.
""".strip()


def craft_debug_prompt(hypothesis: str, target_language: str, code: str, error: str, criterion: str) -> str:
    """Prompt the coder to FIX its previous program given the exact sandbox error/verdict."""
    lang = (target_language or "python").strip()
    return f"""
ROLE: Debug Engineer. REPAIR the existing program - do not start over with a different approach.

HYPOTHESIS (keep the same approach):
{hypothesis}

Language: {lang}

{SANDBOX_CAPABILITIES}

{_lang_hint(lang)}

{criterion}

--- YOUR PREVIOUS CODE ---
{code}
--- SANDBOX RESULT / ERROR ---
{error}
--- END ---

Diagnose the ROOT CAUSE from the sandbox result (common classes, check in order):
  A) unavailable import / missing crate or library  -> remove it; reimplement with allowed APIs
  B) compile/type/borrow error                     -> fix types, ownership, signatures
  C) runtime crash (index, null, overflow, panic)  -> guard bounds; fix off-by-ones
  D) missing or misnamed METRIC line               -> print exactly `METRIC <name>=<number>`
  E) metric present but misses target              -> fix the algorithm / counting logic
  F) timeout / too slow                            -> switch to an efficient algorithm (sieve, DP)
  G) wrong inclusive/exclusive bound or off-by-one -> re-read the problem limits carefully
  H) wrong DP recurrence / init                    -> re-check base cases and loop order

Then return a CORRECTED, complete program that preserves the original approach, actually
computes the metric, and prints the required METRIC line. Do not print SUCCESS.
This is a REPAIR pass: keep the same high-level method unless the root cause proves that
method is impossible under sandbox constraints (e.g. forbidden crate) - only then swap to
an equivalent sandbox-legal implementation of the SAME measurement.

{_ALGORITHM_PLAYBOOK}

{_CODING_DISCIPLINE}

Return RAW CODE only - no markdown fences, no explanation before or after the code.
""".strip()
