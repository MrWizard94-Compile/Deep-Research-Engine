import console_utf8  # noqa: F401  -- forces UTF-8 stdout/stderr before any emoji print
import os
import shutil
import subprocess
import tempfile
import urllib.request
import json
import re

# Per-role model routing. DeepSeek-R1 reasons well but, asked to write a full program,
# emits an unbounded wall of chain-of-thought that never finishes on modest hardware — so
# code generation is delegated to a non-reasoning coder that emits just the program.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1-highctx")          # reasoning / structured
OLLAMA_CODE_MODEL = os.environ.get("OLLAMA_CODE_MODEL", "qwen2.5-coder:14b")  # code generation
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
# Reasoning models (DeepSeek-R1) emit long chains of thought, and a 14B model with a 32K
# context that spills onto CPU runs at only a few tokens/sec — a single call can take many
# minutes. Default generously; override with OLLAMA_TIMEOUT (seconds).
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "1800"))

# ── Backend selection ─────────────────────────────────────────────────────────
# LLM_BACKEND=grok routes every generation through the Grok Build CLI (grok-build-0.1) using the
# local SuperGrok subscription — a far stronger generator than the local Ollama models, with NO API
# key (the CLI carries the auth) and no per-call cost beyond the subscription. Default stays "ollama"
# so the stack is local-first unless deliberately armed.
#
# Grok is an AGENT: for substantial prompts it wants to WRITE files, not return text (plan/read-only
# mode just narrates and yields nothing). So we let it BUILD FREELY in an isolated per-call workdir
# under GROK_WORKSPACE (its own venv, outside the Docker sandbox), tell it to write the finished
# artifact to grok_output.txt, and harvest that file. Containment is by cwd — accepted deliberately.
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
_GROK_DEFAULT = os.path.join(os.path.expanduser("~"), ".grok", "bin", "grok.exe")
GROK_BIN = os.environ.get("GROK_BIN") or (_GROK_DEFAULT if os.path.exists(_GROK_DEFAULT) else "grok")
GROK_MODEL = os.environ.get("GROK_MODEL", "")   # empty -> CLI default model (verified working)
GROK_TIMEOUT = int(os.environ.get("GROK_TIMEOUT", "420"))
GROK_WORKSPACE = os.environ.get("GROK_WORKSPACE", r"C:\WPAI\grok-workspace")

# Fallback payloads kept in one place so structured callers get a consistent shape.
# The "__fallback__" marker lets callers detect a DEGRADED result (the model timed out or
# emitted unparseable output) so a fake hypothesis can never be mistaken for a real one.
_STRUCTURED_FALLBACK = {
    "__fallback__": True,
    "hypothesis": "Isolated Native Computational Architecture Fallback",
    "target_language": "python",
    "test_command": "python3 exp.py",
}


def _query_grok(prompt, structured=False):
    """Generate via the Grok Build CLI (SuperGrok subscription, no API key). Grok is an agent, so we
    let it build in an isolated per-call workdir (its own venv on PATH, outside the Docker sandbox),
    instruct it to write the finished artifact to grok_output.txt, and harvest that file."""
    print("--- 🤖 Querying Grok Build CLI (SuperGrok · isolated build) ---")
    os.makedirs(GROK_WORKSPACE, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="call_", dir=GROK_WORKSPACE)
    directive = ("\n\nIMPORTANT: You are being run headless. Write your COMPLETE final answer — ONLY the "
                 "raw requested artifact, no commentary, no markdown fences — to a file named exactly "
                 "grok_output.txt in the current directory. That file is the sole deliverable.")
    if structured:
        directive += " The artifact must be a single raw valid JSON object."
    prompt_path = os.path.join(workdir, "prompt.txt")
    try:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt + directive)
        env = dict(os.environ)  # give grok the isolated venv's python for any building/testing it does
        venv_scripts = os.path.join(GROK_WORKSPACE, ".venv", "Scripts")
        if os.path.isdir(venv_scripts):
            env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
        cmd = [GROK_BIN, "--prompt-file", prompt_path, "--permission-mode", "bypassPermissions",
               "--output-format", "json", "--disable-web-search"]
        if GROK_MODEL:
            cmd += ["-m", GROK_MODEL]
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=GROK_TIMEOUT, env=env)
        text = _harvest_grok(workdir, proc)
        if not text:
            raise RuntimeError(f"grok produced no artifact (rc={proc.returncode}): {(proc.stderr or '')[:200]}")
        return _extract_json(text) if structured else text
    except Exception as e:  # noqa: BLE001 - degrade to the same fallback shape as the Ollama path
        print(f"❌ Grok Build CLI error: {e}")
        return dict(_STRUCTURED_FALLBACK) if structured else "ERROR: Endpoint Offline"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _harvest_grok(workdir, proc):
    """Pull the artifact grok wrote: the named output file first, then any grok_output.* it made,
    then the JSON-envelope `text` field, then raw stdout."""
    named = os.path.join(workdir, "grok_output.txt")
    if os.path.isfile(named):
        with open(named, encoding="utf-8", errors="replace") as f:
            t = f.read().strip()
        if t:
            return t
    for name in sorted(os.listdir(workdir)):
        if name.startswith("grok_output") and os.path.isfile(os.path.join(workdir, name)):
            with open(os.path.join(workdir, name), encoding="utf-8", errors="replace") as f:
                t = f.read().strip()
            if t:
                return t
    try:
        return (json.loads(proc.stdout or "{}").get("text") or "").strip()
    except Exception:  # noqa: BLE001
        return (proc.stdout or "").strip()


def query_llm(prompt, structured=False, model=None):
    model = model or OLLAMA_MODEL
    if LLM_BACKEND == "grok":
        return _query_grok(prompt, structured)
    print(f"--- 🖥️  Querying Local Ollama Engine ({model}) ---")

    if structured:
        prompt += "\nYour response must be a single, raw, valid JSON object matching the schema. Do not enclose it in any markdown text wrapper fences."

    # Direct system option mapping parameters passing context tokens directly to native endpoints
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # Keep in sync with the Modelfile. 8192 fits this 6 GB GPU far better than 32768
            # (the KV cache is what spills a model onto the CPU). Override via OLLAMA_NUM_CTX.
            "num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
            "temperature": 0.2,
        },
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            raw_text = res_body.get("response", "").strip()

        return _extract_json(raw_text) if structured else raw_text

    except Exception as e:
        print(f"❌ Native Ollama Channel Connection Error: {str(e)}")
        if structured:
            return dict(_STRUCTURED_FALLBACK)
        return "ERROR: Endpoint Offline"


def _extract_json(text):
    if not text:
        return dict(_STRUCTURED_FALLBACK)
    try:
        # 1. Strip the internal reasoning blocks emitted by DeepSeek-R1 (it uses <think>,
        #    older variants used <thought>) so they cannot corrupt the JSON parser.
        clean_text = re.sub(r'<think(?:ing)?>.*?</think(?:ing)?>', '', text, flags=re.DOTALL)
        clean_text = re.sub(r'<thought>.*?</thought>', '', clean_text, flags=re.DOTALL).strip()

        # 2. Strip out accidental markdown fences if applied by the model layers
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]

        clean_text = clean_text.strip()

        # 3. As a last resort, isolate the outermost JSON object if the model wrapped it
        #    in stray prose despite instructions.
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', clean_text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
    except Exception:
        print("⚠️ Core JSON extraction failed. Falling back to structured heuristic tracking model.")
        return {
            "__fallback__": True,
            "hypothesis": "Deterministic Distributed State Tracking Topology",
            "target_language": "python",
            "test_command": "python3 exp.py",
        }
