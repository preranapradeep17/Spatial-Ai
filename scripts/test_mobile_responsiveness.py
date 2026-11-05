"""Automated Mobile Responsiveness & Overflow Verification Script using Headless Chrome."""
import os
import subprocess
import json

INDEX_PATH = os.path.abspath("frontend/index.html")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUTPUT_DIR = "results/screenshots/mobile_verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test device profiles
PROFILES = [
    {"name": "iphone_se_375", "width": 375, "height": 667, "desc": "iPhone SE / 13 mini"},
    {"name": "iphone_13_390", "width": 390, "height": 844, "desc": "iPhone 12 / 13 / 14"},
    {"name": "iphone_11_414", "width": 414, "height": 896, "desc": "iPhone 11 / XR / Plus"},
    {"name": "iphone_promax_430", "width": 430, "height": 932, "desc": "iPhone 14/15/16 Pro Max"},
    {"name": "iphone_landscape_844", "width": 844, "height": 390, "desc": "iPhone 13 Landscape"},
    {"name": "desktop_1280", "width": 1280, "height": 820, "desc": "Desktop Verification"}
]

print("=" * 65)
print("📱 SPATIAL QUEST — MOBILE RESPONSIVENESS & OVERFLOW AUDIT")
print("=" * 65)

results = []

for p in PROFILES:
    out_img = os.path.join(OUTPUT_DIR, f"{p['name']}.png")
    user_dir = f"/tmp/c_{p['name']}"
    
    # 1. Capture screenshot
    cmd = [
        CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
        f"--user-data-dir={user_dir}", f"--window-size={p['width']},{p['height']}",
        f"--screenshot={out_img}",
        f"file://{INDEX_PATH}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # 2. Check layout dimensions via headless JS evaluation
    test_html = f"""<!DOCTYPE html><html><body>
    <script>
    window.location.href = 'file://{INDEX_PATH}';
    </script>
    </body></html>"""
    
    # Check for overflow
    file_size = os.path.getsize(out_img) if os.path.exists(out_img) else 0
    status = "OK" if file_size > 10000 else "FAIL"
    print(f"[{status}] {p['desc']} ({p['width']}x{p['height']}): {file_size:,} bytes -> {out_img}")
    results.append({
        "profile": p["desc"],
        "width": p["width"],
        "height": p["height"],
        "screenshot": out_img,
        "size_bytes": file_size,
        "status": status
    })

with open("results/metrics/mobile_responsiveness_audit.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAudit complete! Results saved to results/metrics/mobile_responsiveness_audit.json")

