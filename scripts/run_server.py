"""Run script for Spatial Quest FastAPI backend with automatic local network IP detection for mobile devices."""
import argparse
import os
import socket
import sys

# Ensure repository root is CWD and on sys.path regardless of where this script is called from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
current_pp = os.environ.get("PYTHONPATH", "")
if REPO_ROOT not in current_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{current_pp}" if current_pp else REPO_ROOT

import uvicorn
from backend.app.main import app


def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Checks whether a port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return False
        except OSError:
            return True


def get_local_ip() -> str:
    """Detects primary local network IP address."""
    # Method 1: Outbound socket check
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 2: System network interface check (macOS / Linux)
    import subprocess
    for iface in ("en0", "en1", "wlan0", "eth0"):
        try:
            res = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True, timeout=1)
            candidate = res.stdout.strip()
            if candidate and not candidate.startswith("127."):
                return candidate
        except Exception:
            pass

    # Method 3: Hostname lookup
    try:
        _, _, ips = socket.gethostbyname_ex(socket.gethostname())
        for candidate in ips:
            if candidate and not candidate.startswith("127."):
                return candidate
    except Exception:
        pass

    return "127.0.0.1"


def main():
    parser = argparse.ArgumentParser(description="Spatial Quest Server Launcher")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    args = parser.parse_args()

    port = args.port
    host = args.host
    local_ip = get_local_ip()

    if is_port_in_use(port, host):
        print("=" * 65)
        print(f"⚠️  PORT {port} IS CURRENTLY IN USE!")
        print("=" * 65)
        print("A previous server process is still holding this port.")
        print()
        print("To free port 8000:")
        print("  1. Press Ctrl + C in the terminal window where it was started, OR")
        print(f"  2. Run: kill -9 $(lsof -t -i :{port})")
        print()
        print("Or run on a different port immediately:")
        print(f"  .venv/bin/python3 scripts/run_server.py --port {port + 1}")
        print("=" * 65)
        sys.exit(1)

    print("=" * 65)
    print("🚀 SPATIAL QUEST — 3D EMBODIED AI INVESTIGATION SERVER")
    print("=" * 65)
    print("📍 Local Machine Access:")
    print(f"   👉 http://localhost:{port}")
    print(f"   👉 http://127.0.0.1:{port}")
    print()
    print("📱 Phone & Tablet Access (Same Wi-Fi Network):")
    print(f"   👉 http://{local_ip}:{port}")
    print()
    print("💡 Mobile Features Enabled:")
    print("   • Touch-drag on 3D canvas to look around")
    print("   • Left-thumb Virtual D-Pad for walking (▲ ◄ ▼ ►)")
    print("   • Right-thumb Action Buttons (✋ Interact, 👁️ Scan, 🔄 180°)")
    print("   • Mobile View Switcher (3D Scene | AI Assistant | Objectives)")
    print("   • Real-time Investigation Partner & Human Investigator")
    print("=" * 65)
    print(f"Starting server on {host}:{port}...")
    print("=" * 65)

    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
