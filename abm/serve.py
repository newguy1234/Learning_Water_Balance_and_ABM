"""Start a Solara page (the ABM UI or the sensitivity results) and block until Ctrl+C."""

from __future__ import annotations

import os
import socket
import subprocess
import sys

from stem.params import PROJECT

SERVER_FLAG = "ABM_SOLARA_SERVER"


def is_server():
    """True inside the process Solara serves from (so a served script does not serve again)."""
    return os.environ.get(SERVER_FLAG) == "1" or "solara" in sys.argv[0].replace("\\", "/").lower()


def free_port(port, tries=20):
    """The first port from `port` up that nothing is listening on."""
    for candidate in range(port, port + tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", candidate))
            return candidate
        except OSError:
            continue
    raise OSError(f"no free port between {port} and {port + tries - 1}")


def serve(script, port=8770, open_browser=True, reload=False):
    """`solara run script`. reload=False runs --production (Solara's reloader re-imports
    numpy from the project's .venv and crashes)."""
    if is_server():
        return None
    port = free_port(port)
    command = [sys.executable, "-m", "solara", "run", os.path.abspath(script), "--port", str(port)]
    if not open_browser:
        command.append("--no-open")
    if not reload:
        command.append("--production")
    print(f"Opening http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        return subprocess.call(command, cwd=PROJECT, env={**os.environ, SERVER_FLAG: "1"})
    except KeyboardInterrupt:
        return 0
