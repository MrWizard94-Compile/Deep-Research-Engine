import json, os
from difflib import SequenceMatcher

class FailureRegistry:
    def __init__(self, db_path="workspace/failure_memory.json"):
        self.db_path = db_path

    def log_failure(self, hypothesis, reason, metrics):
        self._write({"hypothesis": hypothesis, "reason": reason, "metrics": metrics})

    def seed_pre_mortem(self, failures_list):
        for fail in failures_list:
            self._write({"hypothesis": fail['naive_approach'], "reason": f"[PRE-MORTEM] {fail['why_it_will_fail']}", "metrics": "Theoretical Block"})

    def is_redundant(self, new_hypothesis):
        if not os.path.exists(self.db_path): return False, None
        with open(self.db_path, 'r') as f: failures = json.load(f)
        for fail in failures:
            if SequenceMatcher(None, new_hypothesis.lower(), fail['hypothesis'].lower()).ratio() > 0.82:
                return True, fail['reason']
        return False, None

    def _write(self, entry):
        data = []
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f: data = json.load(f)
        data.append(entry)
        with open(self.db_path, 'w') as f: json.dump(data, f, indent=2)

    def get_summary(self, limit=5):
        if not os.path.exists(self.db_path): return ""
        with open(self.db_path, 'r') as f: failures = json.load(f)
        lines = []
        for x in failures[-limit:]:
            reason = str(x.get('reason', '')).strip()
            # Keep the failure CLASS (the start of the error) but cap length so the PI
            # prompt stays lean — an over-long prompt makes the reasoning model time out.
            if len(reason) > 140:
                reason = reason[:140] + "…"
            why = f" | WHY IT FAILED: {reason}" if reason else ""
            lines.append(f"- BLOCK: {x['hypothesis']}{why}")
        return "\n".join(lines)
