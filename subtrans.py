#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subtrans - Anime Subtitle Translator CLI
Entry point wrapper.
"""

import sys
from pathlib import Path

# Configure Windows console to UTF-8 to prevent charmap encoding errors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from subtrans.cli import app

if __name__ == "__main__":
    app()
