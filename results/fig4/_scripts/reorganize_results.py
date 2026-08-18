#!/usr/bin/env python3
"""Wrapper: run reproduce_repo/reorganize_results.py from any cwd."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "reproduce_repo"))

from reorganize_results import main  # noqa: E402

if __name__ == "__main__":
    main()
