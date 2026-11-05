"""Comprehensive verification of exact iPhone viewport dimensions and touch layouts."""
import os
import subprocess
import json
import re

INDEX_PATH = os.path.abspath("frontend/index.html")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCREENSHOTS_DIR = "results/screenshots/mobile_verification"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

PROFILES = [
    {"name": "iphone_se_375", "width": 375, "height": 667, "desc": "iPhone SE / 13 mini (375x667)"},
    {"name": "iphone_13_390", "width": 390, "height": 844, "desc": "iPhone 12/13/14 (390x844)"},
    {"name": "iphone_11_414", "width": 414, "height": 896, "desc": "iPhone 11 / XR (414x896)"},
    {"name": "iphone_promax_430", "width": 430, "height": 932, "desc": "iPhone 14/15/16 Pro Max (430x932)"},
    {"name": "iphone_landscape_844", "width": 844, "height": 390, "desc": "iPhone Landscape (844x390)"},
    {"name": "desktop_1280", "width": 1280, "height": 820, "desc": "Desktop Verification (1280x820)"}
]

print("=" * 70)
print("📱 SPATIAL QUEST — EXACT IPHONE VIEWPORT AUDIT & METRICS")
print("=" * 70)

all_results = []

for p in PROFILES:
    wrapper_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ margin: 0; padding: 0; overflow: hidden; background: #000; }}
    iframe {{ width: {p['width']}px; height: {p['height']}px; border: none; display: block; }}
  </style>
</head>
<body>
<iframe id="testFrame" src="file://{INDEX_PATH}"></iframe>
<div id="wrapperResult" style="display:none;"></div>
<script>
window.addEventListener('load', () => {{
  setTimeout(() => {{
    const f = document.getElementById('testFrame');
    const fWin = f.contentWindow;
    const fDoc = f.contentDocument;
    
    const winW = fWin.innerWidth;
    const winH = fWin.innerHeight;
    const docW = fDoc.documentElement.scrollWidth;
    const bodyW = fDoc.body.scrollWidth;
    
    const chatInput = fDoc.getElementById('chatInput');
    const inputRect = chatInput ? chatInput.getBoundingClientRect() : {{ width: 0, left: 0, right: 0 }};
    const micBtn = fDoc.getElementById('micBtn');
    const micRect = micBtn ? micBtn.getBoundingClientRect() : {{ width: 0, left: 0, right: 0 }};
    const sendBtn = fDoc.getElementById('sendBtn');
    const sendRect = sendBtn ? sendBtn.getBoundingClientRect() : {{ width: 0, left: 0, right: 0 }};
    const ttsBtn = fDoc.getElementById('ttsToggleBtn');
    const ttsRect = ttsBtn ? ttsBtn.getBoundingClientRect() : {{ width: 0, left: 0, right: 0 }};
    
    const dpad = fDoc.querySelector('.virtual-dpad');
    const dpadRect = dpad ? dpad.getBoundingClientRect() : {{ left: 0, right: 0, bottom: 0, height: 0 }};
    const acts = fDoc.querySelector('.virtual-actions');
    const actRect = acts ? acts.getBoundingClientRect() : {{ left: 0, right: 0, bottom: 0, height: 0 }};
    const canvas = fDoc.getElementById('canvasContainer');
    const canvasRect = canvas ? canvas.getBoundingClientRect() : {{ width: 0, height: 0, bottom: 0 }};

    const isDesktop = winW > 900;
    
    const metrics = {{
      profile: "{p['desc']}",
      width: winW,
      height: winH,
      scrollWidth: docW,
      bodyScrollWidth: bodyW,
      hasHorizontalOverflow: docW > winW || bodyW > winW,
      chatInputWidth: Math.round(inputRect.width),
      chatInputFits: inputRect.left >= 0 && inputRect.right <= winW,
      micVisible: micRect.left >= 0 && micRect.right <= winW,
      sendVisible: sendRect.left >= 0 && sendRect.right <= winW,
      ttsVisible: ttsRect.left >= 0 && ttsRect.right <= winW,
      dpadInsideCanvas: isDesktop || (dpadRect.left >= 0 && dpadRect.bottom <= canvasRect.bottom + 2),
      actionsInsideCanvas: isDesktop || (actRect.right <= winW && actRect.bottom <= canvasRect.bottom + 2),
      canvasHeight: Math.round(canvasRect.height),
      freeCanvasHeight: Math.round(canvasRect.height - actRect.height - 20)
    }};
    
    document.getElementById('wrapperResult').textContent = JSON.stringify(metrics);
  }}, 1200);
}});
</script>
</body>
</html>"""

    tmp_file = f"/tmp/spatial_iframe_{p['name']}.html"
    with open(tmp_file, "w") as f:
        f.write(wrapper_html)
        
    out_img = os.path.join(SCREENSHOTS_DIR, f"{p['name']}.png")
    user_dir = f"/tmp/c_audit_{p['name']}"
    
    cmd = [
        CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu",
        "--allow-file-access-from-files", "--disable-web-security",
        f"--user-data-dir={user_dir}",
        f"--window-size={p['width']},{p['height']}",
        f"--screenshot={out_img}",
        "--virtual-time-budget=3500", "--dump-dom",
        f"file://{tmp_file}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    match = re.search(r'<div id="wrapperResult"[^>]*>(.*?)</div>', res.stdout)
    if match and match.group(1):
        m = json.loads(match.group(1))
        all_results.append(m)
        status = "PASS" if not m["hasHorizontalOverflow"] and m["chatInputFits"] else "FAIL"
        print(f"[{status}] {m['profile']}")
        print(f"       Viewport: {m['width']}x{m['height']}px | ScrollWidth: {m['scrollWidth']}px | Overflow: {m['hasHorizontalOverflow']}")
        print(f"       Chat Input Width: {m['chatInputWidth']}px (Fits: {m['chatInputFits']})")
        print(f"       Buttons: Mic={m['micVisible']}, TTS={m['ttsVisible']}, Send={m['sendVisible']}")
        print(f"       Controls: DPad in bounds={m['dpadInsideCanvas']}, Actions in bounds={m['actionsInsideCanvas']}")
        print(f"       Canvas: Height={m['canvasHeight']}px, Free 3D Area={m['freeCanvasHeight']}px")
        print(f"       Screenshot: {out_img} ({os.path.getsize(out_img):,} bytes)\n")
    else:
        print(f"[FAIL] {p['desc']}: Could not extract report.")

with open("results/metrics/iphone_exact_audit.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Audit Complete! Saved report to results/metrics/iphone_exact_audit.json")

