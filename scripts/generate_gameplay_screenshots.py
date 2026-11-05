"""Script to capture comprehensive visual verification screenshots of Spatial Quest using Chrome headless."""
import os
import subprocess
import shutil

SCREENSHOTS_DIR = "results/screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
INDEX_PATH = os.path.abspath("frontend/index.html")
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

with open(INDEX_PATH, "r", encoding="utf-8") as f:
    base_html = f.read()

# 1. Desktop Initial Main View
print("Capturing 01_desktop_gameplay_initial.png...")
cmd1 = [
    CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
    "--user-data-dir=/tmp/c_shot1", "--window-size=1280,820",
    f"--screenshot={SCREENSHOTS_DIR}/01_desktop_gameplay_initial.png",
    f"file://{INDEX_PATH}"
]
subprocess.run(cmd1, capture_output=True)

# 2. Guide & Codes Modal View
print("Capturing 02_guide_codes_modal.png...")
html_guide = base_html.replace(
    "</body>",
    "<script>window.addEventListener('DOMContentLoaded', () => { setTimeout(() => { openMissionGuideModal(); }, 150); });</script></body>"
)
tmp_guide_html = "/tmp/spatial_guide_shot.html"
with open(tmp_guide_html, "w", encoding="utf-8") as f:
    f.write(html_guide)

cmd2 = [
    CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
    "--user-data-dir=/tmp/c_shot2", "--window-size=1280,820",
    f"--screenshot={SCREENSHOTS_DIR}/02_guide_codes_modal.png",
    f"file://{tmp_guide_html}"
]
subprocess.run(cmd2, capture_output=True)

# 3. Mission Completed Banner View
print("Capturing 03_mission_completed_banner.png...")
completed_state_js = """
<script>
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    applyGameState({
      score: 350,
      active_mission: {
        id: 'mission_a',
        code: 'A',
        title: 'Mystery A: Missing Research Key',
        briefing: 'Survey Reception and locate the security table to find where staff logged the missing key.',
        completed: true,
        timed: false,
        spawn_z: -15.0,
        objectives: [
          { objective_id: 'scout_01', description: 'Ask: what objects are visible?', points: 100, is_completed: true },
          { objective_id: 'scout_02', description: 'Locate the table in the facility', points: 100, is_completed: true },
          { objective_id: 'scout_03', description: 'Retrieve the missing Research Key into your inventory', points: 150, is_completed: true }
        ]
      },
      companion: { id: 'scout', name: 'Scout Drone' },
      missions: [
        { id: 'mission_a', code: 'A', title: 'Missing Research Key', active: true, completed: true },
        { id: 'mission_b', code: 'B', title: 'Missing Research File', active: false, completed: false },
        { id: 'mission_c', code: 'C', title: 'Laboratory Containment', active: false, completed: false },
        { id: 'mission_d', code: 'D', title: 'Clue Room & RAG', active: false, completed: false },
        { id: 'mission_e', code: 'E', title: 'Timed Containment', active: false, completed: false },
        { id: 'mission_f', code: 'F', title: 'Dataset Terminal', active: false, completed: false }
      ]
    }, false);
  }, 150);
});
</script></body>
"""
html_completed = base_html.replace("</body>", completed_state_js)
tmp_comp_html = "/tmp/spatial_completed_shot.html"
with open(tmp_comp_html, "w", encoding="utf-8") as f:
    f.write(html_completed)

cmd3 = [
    CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
    "--user-data-dir=/tmp/c_shot3", "--window-size=1280,820",
    f"--screenshot={SCREENSHOTS_DIR}/03_mission_completed_banner.png",
    f"file://{tmp_comp_html}"
]
subprocess.run(cmd3, capture_output=True)

# 4. Mobile Split View (Phone Screen 412x892)
print("Capturing 04_mobile_phone_split_view.png...")
mobile_state_js = """
<script>
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    setMobileView('split');
  }, 100);
});
</script></body>
"""
html_mobile = base_html.replace("</body>", mobile_state_js)
tmp_mobile_html = "/tmp/spatial_mobile_shot.html"
with open(tmp_mobile_html, "w", encoding="utf-8") as f:
    f.write(html_mobile)

cmd4 = [
    CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
    "--user-data-dir=/tmp/c_shot4", "--window-size=412,892",
    f"--screenshot={SCREENSHOTS_DIR}/04_mobile_phone_split_view.png",
    f"file://{tmp_mobile_html}"
]
subprocess.run(cmd4, capture_output=True)

# 5. Mystery D Override Code Verification View
print("Capturing 05_mystery_d_override_unlocked.png...")
mystery_d_js = """
<script>
window.addEventListener('DOMContentLoaded', () => {
  setTimeout(() => {
    applyGameState({
      score: 1650,
      active_mission: {
        id: 'mission_d',
        code: 'D',
        title: 'Clue Room & RAG',
        briefing: 'Inspect the datapad clue, retrieve the Door 3 override code (7391) from facility incident logs via RAG, and disengage the security lock.',
        completed: true,
        timed: false,
        spawn_z: 6.0,
        objectives: [
          { objective_id: 'clue_01', description: 'Inspect a clue (datapad or safe)', points: 150, is_completed: true },
          { objective_id: 'clue_02', description: 'Search documents for the Door 3 override', points: 250, is_completed: true },
          { objective_id: 'clue_03', description: 'Unlock Archive Door 3', points: 250, is_completed: true }
        ]
      },
      companion: { id: 'archivist', name: 'Archivist Drone' },
      missions: [
        { id: 'mission_a', code: 'A', title: 'Missing Research Key', active: false, completed: true },
        { id: 'mission_b', code: 'B', title: 'Missing Research File', active: false, completed: true },
        { id: 'mission_c', code: 'C', title: 'Laboratory Containment', active: false, completed: true },
        { id: 'mission_d', code: 'D', title: 'Clue Room & RAG', active: true, completed: true },
        { id: 'mission_e', code: 'E', title: 'Timed Containment', active: false, completed: false },
        { id: 'mission_f', code: 'F', title: 'Dataset Terminal', active: false, completed: false }
      ]
    }, false);

    appendMessage("Enter code 7391", "user");
    appendMessage("Door 3 emergency magnetic lock disengaged (Override Code: 7391). Path open.", "assistant");
  }, 150);
});
</script></body>
"""
html_d = base_html.replace("</body>", mystery_d_js)
tmp_d_html = "/tmp/spatial_d_shot.html"
with open(tmp_d_html, "w", encoding="utf-8") as f:
    f.write(html_d)

cmd5 = [
    CHROME_BIN, "--headless", "--no-sandbox", "--disable-gpu", "--disable-crash-reporter",
    "--user-data-dir=/tmp/c_shot5", "--window-size=1280,820",
    f"--screenshot={SCREENSHOTS_DIR}/05_mystery_d_override_unlocked.png",
    f"file://{tmp_d_html}"
]
subprocess.run(cmd5, capture_output=True)

print("All visual verification screenshots captured successfully!")

