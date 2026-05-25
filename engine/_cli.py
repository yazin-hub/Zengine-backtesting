"""
engine/_cli.py — Console-script entry point
=============================================

Registered as ``zengine-ui`` in pyproject.toml so users can launch the
Streamlit UI without knowing the exact file path:

    zengine-ui          # after: pip install -e .
    streamlit run app.py   # alternative direct invocation

Notes
-----
Streamlit apps cannot be invoked with a plain ``python -m`` because Streamlit
needs to own the process lifecycle.  This shim locates app.py relative to the
installed package and delegates to ``streamlit run``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_ui() -> None:
    """Launch the ZEngine Strategy Lab Streamlit application."""
    # app.py lives at the repo/package root — one level above engine/
    app_path = Path(__file__).parent.parent / "app.py"
    if not app_path.exists():
        sys.exit(
            f"[zengine-ui] Cannot find app.py at {app_path!r}.\n"
            "Make sure you installed the package from the repository root "
            "(pip install -e .)."
        )
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--", *sys.argv[1:]]
    sys.exit(subprocess.call(cmd))
