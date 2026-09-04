#!/usr/bin/env python3
"""First-wave pipeline sanity only. Do not interpret ΔP / C(S) here."""
from analyze_common_state_forks import main

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--sanity-only", *sys.argv[1:]]
    main()
