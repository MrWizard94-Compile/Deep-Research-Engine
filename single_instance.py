"""
single_instance.py — refuse to run two engines at once.

Concurrent runs share the workspace (genome_log, failure_memory, strategies.py, exp.py)
and the single local GPU, so two of them corrupt each other's data. This guard makes the
engine single-instance via a PID lock file with stale-lock detection, so a crashed run does
not block a restart.
"""

import atexit
import ctypes
import os


def _pid_alive(pid):
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_or_exit(lock_path="workspace/.engine.lock"):
    """Acquire the single-instance lock or exit(1) if another engine holds it."""
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)

    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                other = int((f.read().strip() or "0"))
        except (ValueError, OSError):
            other = 0
        if other and other != os.getpid() and _pid_alive(other):
            print(f"ERROR: Another engine instance is already running (PID {other}). "
                  f"Refusing to start a second - it would corrupt shared state. "
                  f"Stop it first, or remove {lock_path} if it is stale.")
            raise SystemExit(1)
        # Stale lock from a dead process — reclaim it.

    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def _release():
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                if int((f.read().strip() or "0")) == os.getpid():
                    os.remove(lock_path)
        except (ValueError, OSError):
            pass

    atexit.register(_release)
    return lock_path
