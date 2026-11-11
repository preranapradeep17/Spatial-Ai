#!/usr/bin/env python3
"""Spatial Quest — Universal Portable Project Launcher.

Always anchors working directory and Python module resolution to the directory
containing this file so running from ANY location or extracted ZIP serves the
exact same authoritative frontend and backend assets.
"""
import argparse
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_cur_pp = os.environ.get("PYTHONPATH", "")
if REPO_ROOT not in _cur_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{_cur_pp}" if _cur_pp else REPO_ROOT


def ensure_environment():
    """Ensures Python environment has required dependencies without hardcoding external paths."""
    try:
        import fastapi  # noqa: F401
        import torch  # noqa: F401
        import uvicorn  # noqa: F401
        return
    except ImportError:
        local_venv = os.path.join(REPO_ROOT, ".venv", "bin", "python3")
        candidates = [
            local_venv,
            os.path.expanduser("~/anaconda3/bin/python3"),
            os.path.expanduser("~/miniconda3/bin/python3"),
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for py in candidates:
            if os.path.exists(py) and os.path.realpath(py) != os.path.realpath(sys.executable):
                try:
                    res = subprocess.run(
                        [py, "-c", "import fastapi, torch, uvicorn; print('ok')"],
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=8,
                    )
                    if res.returncode == 0:
                        print(f"\033[1;33m[!] Switching to Python environment with dependencies: {py}\033[0m")
                        os.execv(py, [py] + sys.argv)
                except Exception:
                    continue

        print(
            "\033[1;31m[✗] Missing dependencies (fastapi, torch, uvicorn).\033[0m\n"
            f"Run inside '{REPO_ROOT}':\n"
            "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
        )
        sys.exit(1)


ensure_environment()


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Checks whether a port is already bound before opening the browser."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return False
        except OSError:
            return True


def print_banner():
    print(f"""
\033[1;36m======================================================================
  SPATIAL QUEST — 3D EMBODIED VISION-LANGUAGE INVESTIGATION SYSTEM
  Root: {REPO_ROOT}
======================================================================\033[0m
\033[1;32m✓ Core CV Model:\033[0m SpatialDetector (From-Scratch CNN, SUN RGB-D + Sector 4)
\033[1;32m✓ AI Partner:\033[0m Google Gemini API + From-Scratch Transformer Fallback
\033[1;32m✓ 3D Frontend:\033[0m Modern Dark Investigation UI + Human Rigged Investigator
""")


def launch_server_and_browser(host: str = "0.0.0.0", port: int = 8000, open_browser: bool = True):
    if is_port_in_use(port, host):
        print("=" * 65)
        print(f"⚠️  PORT {port} IS ALREADY IN USE!")
        print("=" * 65)
        print("An older server process is still listening on this port.")
        print("Stop it before launching so you do not open a stale frontend:")
        print(f"  kill -9 $(lsof -t -i :{port})")
        print("Or run on another port:")
        print(f"  python3 run.py --server --port {port + 1}")
        print("=" * 65)
        sys.exit(1)

    print(f"\n\033[1;33m[+] Launching Spatial Quest Server on http://localhost:{port} ...\033[0m")
    if open_browser:
        def open_web():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_web, daemon=True).start()

    import uvicorn
    from backend.app.main import app
    uvicorn.run(app, host=host, port=port, reload=False)


def run_tests():
    print("\n\033[1;33m[+] Running Automated Test Suite ...\033[0m\n")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=REPO_ROOT)


def run_vision_benchmark():
    print("\n\033[1;33m[+] Evaluating Custom Detector on Real-World SUN RGB-D Benchmark ...\033[0m\n")
    subprocess.run([sys.executable, "scripts/evaluate_detector.py"], cwd=REPO_ROOT)


def run_domain_transfer():
    print("\n\033[1;33m[+] Running Dual Domain-Transfer Evaluation ...\033[0m\n")
    subprocess.run([sys.executable, "scripts/evaluate_unity_transfer.py"], cwd=REPO_ROOT)


def run_language_model():
    print("\n\033[1;33m[+] Training & Verifying Custom Transformer Language Model from Scratch ...\033[0m\n")
    subprocess.run([sys.executable, "scripts/train_language_model.py"], cwd=REPO_ROOT)


def run_interactive_chat():
    print("\n\033[1;33m[+] Initializing AI Investigation Partner ...\033[0m")
    from backend.game.facility import seed_facility_scene
    seed_facility_scene()
    from backend.agent.investigation_agent import global_investigation_agent
    print("\033[1;32m[✓] AI Investigation Partner Online. Type your query or 'exit'.\033[0m\n")

    while True:
        try:
            query = input("\033[1;36mInvestigator > \033[0m").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                break
            res = global_investigation_agent.run(query)
            if res.get("tool_calls"):
                for tc in res["tool_calls"]:
                    print(f"  \033[1;34m[⚡ Tool Execution]: {tc['tool']}({tc['arguments']})\033[0m")
            print(f"\033[1;32mAI Partner >\033[0m {res['final_response']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break


def main():
    parser = argparse.ArgumentParser(description="Spatial Quest Universal Launcher")
    parser.add_argument("--server", "-s", action="store_true", help="Launch web server directly")
    parser.add_argument("--chat", "-c", "--drone", action="store_true", help="Interactive CLI investigation chat")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    args, unknown = parser.parse_known_args()
    if unknown and unknown[0] == "server":
        args.server = True

    print_banner()
    if args.server:
        launch_server_and_browser(host=args.host, port=args.port, open_browser=not args.no_browser)
        return
    if args.chat:
        run_interactive_chat()
        return

    print("Select an action:")
    print("  \033[1;37m[1]\033[0m \033[1mLaunch 3D Investigation Command & Backend Server\033[0m (Recommended)")
    print("  \033[1;37m[2]\033[0m Run Full Automated Test Suite")
    print("  \033[1;37m[3]\033[0m Evaluate Vision Detector on Real SUN RGB-D Benchmark")
    print("  \033[1;37m[4]\033[0m Run Dual Domain-Transfer Evaluation")
    print("  \033[1;37m[5]\033[0m Train & Verify Custom Transformer Language Model")
    print("  \033[1;37m[6]\033[0m Interactive Terminal Chat with AI Investigation Partner")
    print("  \033[1;37m[Q]\033[0m Exit")

    choice = input("\nEnter choice [1-6, default=1]: ").strip().lower()
    if choice in ["1", "", "s"]:
        launch_server_and_browser(host=args.host, port=args.port, open_browser=not args.no_browser)
    elif choice == "2":
        run_tests()
    elif choice == "3":
        run_vision_benchmark()
    elif choice == "4":
        run_domain_transfer()
    elif choice == "5":
        run_language_model()
    elif choice == "6":
        run_interactive_chat()
    elif choice in ["q", "quit", "exit"]:
        print("Goodbye!")
    else:
        print("Invalid option. Launching default Web Terminal & Server...")
        launch_server_and_browser(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
