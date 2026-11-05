"""Verify DOM element bounding boxes and test for horizontal overflow across iPhone profiles."""
import os
import subprocess
import json
import re

INDEX_PATH = os.path.abspath("frontend/index.html")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PROFILES = [
    {"name": "iphone_se_375", "width": 375, "height": 667, "desc": "iPhone SE / 13 mini"},
    {"name": "iphone_13_390", "width": 390, "height": 844, "desc": "iPhone 12 / 13 / 14"},
    {"name": "iphone_11_414", "width": 414, "height": 896, "desc": "iPhone 11 / XR"},
    {"name": "iphone_promax_430", "width": 430, "height": 932, "desc": "iPhone 14/15/16 Pro Max"},
]

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    base_html = f.read()

# Audit script that writes layout report to a DOM element
audit_script = """
<div id="layoutAuditResult" style="display:none;"></div>
<script>
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    const winW = window.innerWidth;
    const docW = document.documentElement.scrollWidth;
    const bodyW = document.body.scrollWidth;
    
    const overflowing = [];
    document.querySelectorAll('*').forEach(el => {
      const rect = el.getBoundingClientRect();
      if (rect.right > winW + 1 && !el.closest('.inspector-drawer') && !el.closest('.modal-backdrop')) {
        overflowing.push({
          tag: el.tagName,
          id: el.id,
          cls: el.className,
          right: Math.round(rect.right),
          overflow: Math.round(rect.right - winW)
        });
      }
    });

    const chatInput = document.getElementById('chatInput');
    const inputRect = chatInput ? chatInput.getBoundingClientRect() : null;
    const micRect = document.getElementById('micBtn') ? document.getElementById('micBtn').getBoundingClientRect() : null;
    const sendRect = document.getElementById('sendBtn') ? document.getElementById('sendBtn').getBoundingClientRect() : null;
    const ttsRect = document.getElementById('ttsToggleBtn') ? document.getElementById('ttsToggleBtn').getBoundingClientRect() : null;
    
    const dpadRect = document.querySelector('.virtual-dpad') ? document.querySelector('.virtual-dpad').getBoundingClientRect() : null;
    const actRect = document.querySelector('.virtual-actions') ? document.querySelector('.virtual-actions').getBoundingClientRect() : null;
    const canvasRect = document.getElementById('canvasContainer') ? document.getElementById('canvasContainer').getBoundingClientRect() : null;

    const report = {
      windowWidth: winW,
      docScrollWidth: docW,
      bodyScrollWidth: bodyW,
      hasHorizontalOverflow: docW > winW || bodyW > winW,
      overflowingElementsCount: overflowing.length,
      overflowingSample: overflowing.slice(0, 5),
      chatInputWidth: inputRect ? Math.round(inputRect.width) : 0,
      chatInputFullyVisible: inputRect ? (inputRect.left >= 0 && inputRect.right <= winW) : false,
      micVisible: micRect ? (micRect.left >= 0 && micRect.right <= winW) : false,
      sendVisible: sendRect ? (sendRect.left >= 0 && sendRect.right <= winW) : false,
      ttsVisible: ttsRect ? (ttsRect.left >= 0 && ttsRect.right <= winW) : false,
      dpadVisible: dpadRect ? (dpadRect.left >= 0 && dpadRect.bottom <= canvasRect.bottom) : false,
      actionsVisible: actRect ? (actRect.right <= winW && actRect.bottom <= canvasRect.bottom) : false,
      canvasHeight: canvasRect ? Math.round(canvasRect.height) : 0,
      freeCanvasHeight: canvasRect && actRect ? Math.round(canvasRect.height - actRect.height - 30) : 0
    };
    
    document.getElementById('layoutAuditResult').textContent = JSON.stringify(report);
  }, 200);
});
</script>
"""

test_html_path = "/tmp/spatial_layout_audit.html"
with open(test_html_path, "w", encoding="utf-8") as f:
    f.write(base_html.replace("</body>", audit_script + "</body>"))

print("Running DOM Layout Verification across iPhone screen widths...\n")
all_reports = []

for p in PROFILES:
    user_dir = f"/tmp/c_audit_{p['name']}"
    cmd = [
        CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
        f"--user-data-dir={user_dir}", f"--window-size={p['width']},{p['height']}",
        "--virtual-time-budget=2000", "--dump-dom",
        f"file://{test_html_path}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
    dom = res.stdout
    
    match = re.search(r'<div id="layoutAuditResult"[^>]*>(.*?)</div>', dom)
    if match and match.group(1):
        data = json.loads(match.group(1))
        all_reports.append({"profile": p["desc"], **data})
        print(f"=== {p['desc']} ({p['width']}x{p['height']}) ===")
        print(f"  Horizontal Overflow: {'NO (PASS)' if not data['hasHorizontalOverflow'] else 'YES (FAIL)'}")
        print(f"  Doc Scroll Width: {data['docScrollWidth']}px / Body: {data['bodyScrollWidth']}px (Window: {data['windowWidth']}px)")
        print(f"  Overflowing Elements Count: {data['overflowingElementsCount']}")
        print(f"  Chat Input Width: {data['chatInputWidth']}px (Fully Visible: {data['chatInputFullyVisible']})")
        print(f"  Action Buttons Visible: Mic={data['micVisible']}, TTS={data['ttsVisible']}, Send={data['sendVisible']}")
        print(f"  D-Pad & Actions: DPad={data['dpadVisible']}, Actions={data['actionsVisible']}")
        print(f"  Canvas Height: {data['canvasHeight']}px, Free 3D Height: {data['freeCanvasHeight']}px\n")
    else:
        print(f"Profile {p['name']}: Could not extract report from DOM.")

with open("results/metrics/mobile_dom_overflow_audit.json", "w") as f:
    json.dump(all_reports, f, indent=2)

print("Saved report to results/metrics/mobile_dom_overflow_audit.json")
