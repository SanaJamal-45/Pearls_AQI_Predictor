"""
run.py

Starts both the FastAPI backend and the Streamlit dashboard.
Run from the project root:

    python run.py

The backend starts on http://localhost:8000 and the dashboard on http://localhost:8501.
"""

import subprocess
import sys
import time
import signal
import os


def main():
    procs = []

    # --- FastAPI backend ---
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    procs.append(backend)
    print("[run] FastAPI backend starting on http://localhost:8000 ...")

    # Give the backend a moment to boot before launching the dashboard
    time.sleep(3)

    # --- Streamlit dashboard ---
    dashboard = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/dashboard.py", "--server.port", "8501"],
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )
    procs.append(dashboard)
    print("[run] Streamlit dashboard starting on http://localhost:8501 ...")

    # --- Wait until one of them dies, then clean up ---
    def shutdown(signum, frame):
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        shutdown(None, None)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=5)


if __name__ == "__main__":
    main()
