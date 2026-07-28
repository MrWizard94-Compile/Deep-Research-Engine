"""
objectives.py — research-goal supply for the open-ended loop.

Starts from a seed backlog of concrete objectives. When the backlog is exhausted it asks
the reasoning model to propose the next frontier objective, building on what was just
solved. A deterministic fallback guarantees the loop never stalls for lack of a goal.
"""

import json
import os
import time

from llm_bridge import query_llm

SEED_OBJECTIVES = [
    "Design a stable, scalable computational architecture that addresses power draw, scaling memory limits, and expanding context windows.",
    "Design a memory-efficient sparse data structure supporting sub-linear nearest-neighbour queries over high-dimensional vectors.",
    "Design a deterministic scheduler that maximizes throughput on heterogeneous CPU/GPU hardware under a fixed power budget.",
    "Design a cache-replacement policy that provably beats LRU on adversarial access patterns without unbounded metadata.",
    "Design a lock-free concurrent queue that sustains linear throughput scaling to 64 producers without starvation.",
]


class ObjectiveProvider:
    """Hands out one objective at a time; advances to a fresh frontier after each solve."""

    def __init__(self, seeds=None, log_path="workspace/objectives_log.json", query_fn=query_llm):
        queue = list(seeds if seeds is not None else SEED_OBJECTIVES)
        self._queue = queue
        self.query_fn = query_fn
        self.log_path = log_path
        self.solved_count = 0
        self._current = self._queue.pop(0) if self._queue else self._fallback_objective(0)

    @property
    def current(self):
        return self._current

    def advance(self, solved_objective, solution_path=None):
        """Record a solved objective and move on to the next frontier. Returns the new objective."""
        self.solved_count += 1
        self._log({
            "solved_objective": solved_objective,
            "solution": solution_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        if self._queue:
            self._current = self._queue.pop(0)
        else:
            self._current = self._generate_next(solved_objective)
        return self._current

    def _generate_next(self, solved_objective):
        prompt = (
            "You are directing an autonomous research program. You have just SOLVED this objective:\n"
            f"  {solved_objective}\n\n"
            "Propose the NEXT, more ambitious research objective that builds on it and pushes a new "
            "frontier in efficient, deterministic computation. Make it concrete and falsifiable, one "
            'sentence. Respond as strict JSON: {"objective": "..."}'
        )
        try:
            data = self.query_fn(prompt, structured=True)
            if isinstance(data, dict):
                obj = data.get("objective")
                if isinstance(obj, str) and len(obj.strip()) > 15:
                    return obj.strip()
        except Exception:
            pass
        return self._fallback_objective(self.solved_count)

    @staticmethod
    def _fallback_objective(n):
        return (
            f"Design a novel deterministic computational architecture (frontier {n + 1}) that improves "
            "energy efficiency and memory scaling beyond the previously solved design."
        )

    def _log(self, entry):
        data = []
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = []
        data.append(entry)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
