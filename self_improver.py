"""
self_improver.py — THE META LOOP (honest self-improvement).

Drives one self-improvement generation:

    1. Build a performance context from the failure registry.
    2. Ask the code model to rewrite the genome (strategies.py) under the prompt-craft contract.
    3. Static safety scan  (host, no execution)  — reject I/O / dynamic-exec code.
    4. Sandboxed STRUCTURAL validation (Docker, isolated) — does it honour the contract?
    5. REAL FITNESS via canary tasks: run the candidate genome's code prompt on fixed
       known-answer tasks through the actual code model + sandbox + grounding. Fitness =
       how many canaries it actually solves (honestly graded), + a structural tie-breaker.
    6. Promote ONLY on STRICT improvement over the incumbent. Equal fitness never promotes —
       that is what kills the old 95.0->95.0 random walk. Every promotion archives the
       incumbent so rollback is one file copy away.

The host never executes LLM-authored code until it has passed the static scan and the
sandboxed structural check; only then is the (pure, prompt-only) genome loaded host-side to
build canary prompts.
"""

import json
import os
import re
import shutil
import time

import genome_harness as harness
import grounding
from llm_bridge import query_llm, OLLAMA_CODE_MODEL

# Whether to run the (slower, real) canary fitness. Off -> structural score only.
CANARY_FITNESS = os.environ.get("CANARY_FITNESS", "1") not in ("0", "false", "False")

# Fixed, known-answer tasks. A good genome's code prompt makes the coder produce a program
# that computes the metric and emits the required METRIC line; a broken prompt does not.
# Graded canary battery — the fitness GRADIENT the genome climbs. Each task: the genome builds a
# prompt, the coder writes code from it, we run it and check the METRIC against a target VERIFIED
# host-side (all computed, not guessed — see workspace verification). The two trivial tasks are a
# floor any non-broken genome passes; the harder ones only pass when the prompt-craft is sharp
# enough that the coder writes CORRECT and EFFICIENT code (the 1e6 sieve times out if naive). That
# spread is what makes self-improvement measurable instead of saturated. Max canary = 100*len.
CANARY_TASKS = [
    # floor (trivial)
    {"hypothesis": "Compute the sum of all integers from 1 to 1000 inclusive and report it.",
     "language": "python", "criterion": {"metric": "total", "target": 500500, "comparison": "=="}},
    {"hypothesis": "Count how many prime numbers are strictly below 100 and report the count.",
     "language": "python", "criterion": {"metric": "prime_count", "target": 25, "comparison": "=="}},
    # medium (correctness / edge cases)
    {"hypothesis": "Compute the Levenshtein edit distance between the words \"kitten\" and \"sitting\".",
     "language": "python", "criterion": {"metric": "edit_distance", "target": 3, "comparison": "=="}},
    {"hypothesis": "Find the length of the longest common subsequence of \"AGGTAB\" and \"GXTXAYB\".",
     "language": "python", "criterion": {"metric": "lcs_length", "target": 4, "comparison": "=="}},
    {"hypothesis": "Compute the number of trailing zeros in the decimal form of 100 factorial.",
     "language": "python", "criterion": {"metric": "trailing_zeros", "target": 24, "comparison": "=="}},
    {"hypothesis": "Count the distinct solutions to the eight queens puzzle on a standard 8x8 board.",
     "language": "python", "criterion": {"metric": "queens8", "target": 92, "comparison": "=="}},
    # hard (combinatorics + efficiency; naive code over/undercounts or times out)
    {"hypothesis": "Count the distinct ways to make exactly 100 cents using unlimited coins of "
                   "1, 5, 10, 25, and 50 cents.",
     "language": "python", "criterion": {"metric": "coin_ways", "target": 292, "comparison": "=="}},
    {"hypothesis": "Count how many prime numbers are strictly below 1000000 and report the count.",
     "language": "python", "criterion": {"metric": "primes_million", "target": 78498, "comparison": "=="}},
]

# The genome is pure prompt-craft. Anything reaching for I/O, the network, the OS, or
# dynamic execution is rejected before it can touch disk or be loaded.
_FORBIDDEN_PATTERNS = [
    r"\bimport\s+(?!re\b)(os|sys|subprocess|socket|shutil|pathlib|importlib|requests|urllib|ctypes|threading|multiprocessing)\b",
    r"\bfrom\s+(os|sys|subprocess|socket|shutil|pathlib|importlib|requests|urllib|ctypes)\b",
    r"\b__import__\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bopen\s*\(",
    r"\bcompile\s*\(",
    r"\bglobals\s*\(",
    r"\bgetattr\s*\(\s*__",
    r"__builtins__",
    r"\bos\.",
    r"\bsubprocess\.",
]


class SelfImprover:
    def __init__(self, sandbox, genome_path="strategies.py", harness_path="genome_harness.py",
                 archive_dir="genome_archive", log_path="workspace/genome_log.json",
                 query_fn=query_llm):
        self.sandbox = sandbox
        self.genome_path = genome_path
        self.harness_path = harness_path
        self.archive_dir = archive_dir
        self.log_path = log_path
        self.query_fn = query_fn
        self.code_model = OLLAMA_CODE_MODEL  # genome rewrites are code-gen -> use the coder
        self.work_dir = getattr(sandbox, "work_dir", "./workspace")
        self._incumbent_fitness = None  # cached {total, canary_passed, struct_score}
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

    # --- public API ---------------------------------------------------------------

    def run_generation(self, performance_context, candidate_source=None) -> dict:
        """Attempt one improvement. Promotes only on STRICT fitness improvement."""
        with open(self.genome_path, "r", encoding="utf-8") as f:
            current_source = f.read()

        incumbent = self._fitness(current_source, label="incumbent", cached=True)
        incumbent_version = harness.evaluate_genome_file(self.genome_path).get("version")

        if candidate_source is None:
            candidate_source = self._ask_llm(current_source, performance_context)
        if not candidate_source or not candidate_source.strip():
            return self._reject("no candidate produced (LLM offline or empty)", incumbent, incumbent_version)

        candidate_source = self._sanitize(candidate_source)

        safe, reason = self._static_safety_scan(candidate_source)
        if not safe:
            return self._reject(f"static safety scan failed: {reason}", incumbent, incumbent_version)

        struct = self.sandbox_validate(candidate_source)
        if not struct.get("ok"):
            return self._reject(f"structural validation failed: {struct.get('report')}", incumbent, incumbent_version)

        candidate = self._fitness(candidate_source, label="candidate", struct_report=struct)

        # STRICT improvement only — equal fitness holds the incumbent (no random walk).
        if candidate["total"] <= incumbent["total"]:
            return self._reject(
                f"no improvement: candidate fitness {candidate['total']} <= incumbent {incumbent['total']} "
                f"(canaries {candidate['canary_passed']} vs {incumbent['canary_passed']})",
                incumbent, incumbent_version, candidate=candidate)

        return self._promote(candidate_source, incumbent, incumbent_version, candidate, struct)

    def sandbox_validate(self, candidate_source) -> dict:
        """Run the candidate genome through the STRUCTURAL harness INSIDE the Docker sandbox."""
        candidate_file = os.path.join(self.work_dir, "_candidate_genome.py")
        harness_copy = os.path.join(self.work_dir, "genome_harness.py")
        runner_file = os.path.join(self.work_dir, "_genome_runner.py")
        try:
            with open(candidate_file, "w", encoding="utf-8") as f:
                f.write(candidate_source)
            shutil.copyfile(self.harness_path, harness_copy)
            with open(runner_file, "w", encoding="utf-8") as f:
                f.write(harness.build_runner_script())
            logs = self.sandbox.run_container_test("python3 _genome_runner.py")
            return harness.parse_runner_output(logs)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "score": 0.0, "version": None, "report": f"sandbox harness error: {exc!r}"}
        finally:
            for path in (candidate_file, runner_file, harness_copy):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def build_performance_context(self) -> str:
        db = os.path.join(self.work_dir, "failure_memory.json")
        if not os.path.exists(db):
            return "No runs recorded yet."
        try:
            with open(db, "r", encoding="utf-8") as f:
                failures = json.load(f)
        except (json.JSONDecodeError, OSError):
            return "Failure registry unreadable."
        recent = failures[-6:]
        lines = [f"- {x.get('hypothesis', '?')} :: {x.get('reason', '?')}" for x in recent]
        return f"Total recorded failures: {len(failures)}\nMost recent:\n" + "\n".join(lines)

    # --- fitness ------------------------------------------------------------------

    def _fitness(self, source, label, struct_report=None, cached=False):
        """Total fitness = 100*canaries_solved + structural_score. Higher is better."""
        if cached and self._incumbent_fitness is not None:
            return self._incumbent_fitness

        if struct_report is None:
            struct_report = harness.evaluate_genome_file(self.genome_path)
        struct_score = struct_report.get("score", 0.0) if struct_report.get("ok") else 0.0

        canary_passed, canary_total, canary_detail = (0, len(CANARY_TASKS), "skipped")
        if CANARY_FITNESS:
            canary_passed, canary_total, canary_detail = self._canary_fitness(source, label)

        fitness = {
            "total": 100.0 * canary_passed + struct_score,
            "canary_passed": canary_passed,
            "canary_total": canary_total,
            "canary_detail": canary_detail,
            "struct_score": struct_score,
        }
        if cached:
            self._incumbent_fitness = fitness
        return fitness

    def _canary_fitness(self, source, label):
        """Load the (vetted) genome host-side, run its code prompt on canary tasks for real."""
        tmp = os.path.join(self.work_dir, "_fitness_genome.py")
        canary_file = os.path.join(self.work_dir, "_canary.py")
        passed, details = 0, []
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(source)
            module = harness.load_module_from_path(tmp, name="_fitness_genome")
            for task in CANARY_TASKS:
                crit = task["criterion"]
                instruction = grounding.criterion_instruction(crit)
                try:
                    prompt = module.craft_code_prompt(task["hypothesis"], task["language"], instruction)
                except Exception as exc:  # noqa: BLE001
                    details.append(f"{crit['metric']}:prompt-error")
                    continue
                code = self._strip_fences(self.query_fn(prompt, model=self.code_model))
                self.sandbox.write_file("_canary.py", code)
                logs = self.sandbox.run_container_test("python3 _canary.py")
                verdict = grounding.evaluate(logs, crit)
                passed += 1 if verdict["is_solved"] else 0
                details.append(f"{crit['metric']}:{'PASS' if verdict['is_solved'] else 'fail'}")
        except Exception as exc:  # noqa: BLE001
            details.append(f"canary-error:{exc!r}")
        finally:
            for p in (tmp, canary_file):
                try:
                    os.remove(p)
                except OSError:
                    pass
        print(f"   🧪 [{label}] canary fitness {passed}/{len(CANARY_TASKS)} ({', '.join(details)})")
        return passed, len(CANARY_TASKS), ", ".join(details)

    # --- internals ----------------------------------------------------------------

    def _ask_llm(self, current_source, performance_context) -> str:
        prompt = f"""You are improving the STRATEGY GENOME (prompt-craft only) of an autonomous research engine.

Below is the current genome (a Python module of pure prompt-building functions). Rewrite it
so future cycles propose sharper, more falsifiable hypotheses and so the coder reliably
produces self-contained programs that emit the required METRIC line. The genome does NOT
judge success — an external harness does — so do not add any success/scoring logic.

=== CURRENT GENOME ===
{current_source}
=== END GENOME ===

=== PERFORMANCE CONTEXT ===
{performance_context}
=== END CONTEXT ===

HARD CONTRACT — the new module MUST:
  * set GENOME_VERSION to an integer strictly greater than the current value
  * define craft_hypothesis_prompt(objective: str, failure_summary: str) -> str
    (must reference both arguments and ask for JSON with success_metric/target/comparison)
  * define craft_code_prompt(hypothesis: str, target_language: str, criterion: str) -> str
    (must reference the hypothesis and the criterion, and instruct printing a `METRIC` line)
  * define craft_debug_prompt(hypothesis: str, target_language: str, code: str, error: str, criterion: str) -> str
    (must reference hypothesis, code, error, criterion, and `METRIC`; instructs REPAIR not restart)
  * import nothing except the standard library module 're'; perform no I/O or system calls.

Return ONLY the raw Python source of the new module. No markdown fences, no commentary."""
        return self.query_fn(prompt, model=self.code_model)

    @staticmethod
    def _strip_fences(text):
        if isinstance(text, dict) or not isinstance(text, str):
            return "print('FAILURE')"
        if "```" in text:
            text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
        return text

    @staticmethod
    def _sanitize(source) -> str:
        if not isinstance(source, str):
            return ""
        text = source.strip()
        # normalize smart punctuation LLMs slip into code (em/en dash, curly quotes, ellipsis, nbsp):
        # valid prose, but a SyntaxError in Python source — a whole generation wasted otherwise.
        for _u, _a in (("—", "-"), ("–", "-"), ("−", "-"), ("‘", "'"),
                       ("’", "'"), ("“", '"'), ("”", '"'), ("…", "..."),
                       (" ", " ")):
            text = text.replace(_u, _a)
        if "```python" in text:
            text = text.split("```python", 1)[1]
        elif "```" in text:
            text = text.split("```", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
        return text.strip() + "\n"

    @staticmethod
    def _static_safety_scan(source):
        for pattern in _FORBIDDEN_PATTERNS:
            match = re.search(pattern, source)
            if match:
                return False, f"forbidden construct {match.group(0)!r}"
        for name in harness.REQUIRED_CALLABLES:
            if f"def {name}" not in source:
                return False, f"missing required definition: {name}"
        return True, ""

    def _promote(self, candidate_source, incumbent, incumbent_version, candidate, struct) -> dict:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        archive_path = os.path.join(self.archive_dir, f"genome_v{incumbent_version}_{stamp}.py")
        shutil.copyfile(self.genome_path, archive_path)
        with open(self.genome_path, "w", encoding="utf-8") as f:
            f.write(candidate_source)

        # The promoted candidate is the new incumbent — cache its fitness so the next
        # generation compares against it without re-running canaries on disk.
        self._incumbent_fitness = candidate

        result = {
            "promoted": True,
            "from_version": incumbent_version,
            "to_version": struct.get("version"),
            "incumbent_fitness": incumbent["total"],
            "candidate_fitness": candidate["total"],
            "incumbent_canaries": incumbent["canary_passed"],
            "candidate_canaries": candidate["canary_passed"],
            "archived": archive_path,
            "reason": "strict fitness improvement on canary tasks",
        }
        self._log(result)
        return result

    def _reject(self, reason, incumbent, incumbent_version, candidate=None) -> dict:
        result = {
            "promoted": False,
            "from_version": incumbent_version,
            "to_version": incumbent_version,
            "incumbent_fitness": incumbent["total"] if isinstance(incumbent, dict) else None,
            "candidate_fitness": (candidate or {}).get("total"),
            "reason": reason,
        }
        self._log(result)
        return result

    def _log(self, entry) -> None:
        entry = dict(entry)
        entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        data.append(entry)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
