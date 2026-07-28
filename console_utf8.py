"""Force UTF-8 on stdout/stderr so the engine's emoji status output never crashes on
Windows consoles (which default to cp1252 and raise UnicodeEncodeError on emoji).

Importing this module early applies the fix process-wide and idempotently, regardless of
which module is the entry point. Every module that prints imports it at the top.
"""

import sys


def force_utf8():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


force_utf8()
