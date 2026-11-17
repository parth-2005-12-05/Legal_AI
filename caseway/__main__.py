"""Allows `python -m caseway ...` (delegates to caseway/cli.py)."""

from .cli import main

raise SystemExit(main())
