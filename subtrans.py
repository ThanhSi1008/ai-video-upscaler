#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subtrans - Anime Subtitle Translator CLI
Entry point wrapper.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from subtrans.cli import app

if __name__ == "__main__":
    app()
