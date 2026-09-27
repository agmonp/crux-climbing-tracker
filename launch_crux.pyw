"""Start the local CRUX server in the background and open its browser UI."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import Tk, messagebox

APP_ROOT = Path(__file__).resolve().parent
APP_URL = "http://127.0.0.1:8765"
HEALTH_URL = f"{APP_URL}/api/health"
STARTUP_TIMEOUT_SECONDS = 40.0
POLL_SECONDS = 0.25


def health_is_ready() -> bool:
    """Return true only for the CRUX health endpoint, not any process on 8765."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1) as response:
            payload = json.load(response)
            return response.status == 200 and payload.get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def port_is_in_use() -> bool:
    """Detect an unrelated local service before trying to start CRUX."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(0.5)
        return client.connect_ex(("127.0.0.1", 8765)) == 0


def show_error(message: str) -> None:
    """Show startup failures because this launcher intentionally has no console."""
    root = Tk()
    root.withdraw()
    try:
        messagebox.showerror("CRUX — שגיאת הפעלה", message, parent=root)
    finally:
        root.destroy()


def start_server() -> subprocess.Popen[bytes]:
    """Create a detached server process and retain logs for troubleshooting."""
    python = APP_ROOT / ".venv" / "Scripts" / "pythonw.exe"
    entrypoint = APP_ROOT / "run_app.py"
    if not python.is_file():
        raise FileNotFoundError(
            "סביבת Python של האפליקציה חסרה. יש להריץ את שלבי ההתקנה ב־README.md."
        )
    log_dir = APP_ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / "server.log").open("ab", buffering=0)
    creation_flags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    try:
        return subprocess.Popen(
            [str(python), str(entrypoint)],
            cwd=APP_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=creation_flags,
            close_fds=True,
            env=environment,
        )
    finally:
        log.close()


def main() -> int:
    if health_is_ready():
        webbrowser.open_new_tab(APP_URL)
        return 0
    if port_is_in_use():
        show_error(
            "פורט 8765 כבר נמצא בשימוש על ידי תוכנה אחרת. "
            "יש לסגור אותה ולפתוח שוב את CRUX."
        )
        return 1
    try:
        process = start_server()
    except OSError as error:
        show_error(str(error))
        return 1
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if health_is_ready():
            webbrowser.open_new_tab(APP_URL)
            return 0
        if process.poll() is not None:
            show_error(
                "האפליקציה לא הצליחה לעלות. פרטי התקלה נשמרו בקובץ "
                "data\\logs\\server.log."
            )
            return 1
        time.sleep(POLL_SECONDS)
    process.terminate()
    show_error(
        "הפעלת האפליקציה ארכה יותר מ־40 שניות. "
        "פרטי התקלה נשמרו בקובץ data\\logs\\server.log."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
