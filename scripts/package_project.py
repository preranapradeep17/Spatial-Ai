"""Clean Portable ZIP Packager for Spatial Quest.

Creates a self-contained project archive excluding machine-specific virtualenvs (.venv),
Git history (.git), bytecode caches (__pycache__), OS metadata (.DS_Store, __MACOSX),
local secrets (.env), temporary scratch outputs, and the 66 MB raw ZIP central directory cache.
Verifies all required frontend assets and expected UI markers inside the generated archive.
"""
import argparse
import os
import sys
import zipfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_ROOT = "Spatial-Quest"

EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "__MACOSX",
    "scratch",
}
EXCLUDED_EXTS = {".pyc", ".pyo", ".zip"}
EXCLUDED_RELPATHS = {
    ".env",
    ".DS_Store",
    "data/sunrgbd/metadata/sunrgbd_cd.bin",
}

REQUIRED_ARCHIVE_FILES = [
    f"{ARCHIVE_ROOT}/run.py",
    f"{ARCHIVE_ROOT}/requirements.txt",
    f"{ARCHIVE_ROOT}/.env.example",
    f"{ARCHIVE_ROOT}/scripts/run_server.py",
    f"{ARCHIVE_ROOT}/backend/app/main.py",
    f"{ARCHIVE_ROOT}/frontend/index.html",
    f"{ARCHIVE_ROOT}/frontend/three.min.js",
    f"{ARCHIVE_ROOT}/frontend/assets/styles.css",
    f"{ARCHIVE_ROOT}/frontend/assets/GLTFLoader.js",
    f"{ARCHIVE_ROOT}/frontend/assets/depth_capture.js",
    f"{ARCHIVE_ROOT}/frontend/assets/investigator3d.js",
    f"{ARCHIVE_ROOT}/frontend/assets/investigator.glb",
    f"{ARCHIVE_ROOT}/frontend/assets/facility3d.js",
    f"{ARCHIVE_ROOT}/frontend/assets/map_ui.js",
    f"{ARCHIVE_ROOT}/models/vision/spatial_detector_best.pt",
    f"{ARCHIVE_ROOT}/models/language_model/transformer_best.pt",
    f"{ARCHIVE_ROOT}/models/language_model/tokenizer.json",
]


def package_project(output_zip: str) -> str:
    out_abs = os.path.abspath(output_zip)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    tmp_abs = out_abs + ".tmp"

    file_count = 0
    with zipfile.ZipFile(tmp_abs, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith("._"))
            for fn in sorted(filenames):
                if fn == ".DS_Store" or fn.startswith("._"):
                    continue
                if os.path.splitext(fn)[1].lower() in EXCLUDED_EXTS:
                    continue
                full_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(full_path, REPO_ROOT).replace(os.sep, "/")
                if rel_path in EXCLUDED_RELPATHS or rel_path.endswith(".staging"):
                    continue
                if "/.staging/" in rel_path:
                    continue
                arcname = f"{ARCHIVE_ROOT}/{rel_path}"
                zf.write(full_path, arcname)
                file_count += 1

    os.replace(tmp_abs, out_abs)

    # Verify archive integrity & expected UI presence
    with zipfile.ZipFile(out_abs, "r") as zf:
        names = set(zf.namelist())
        missing = [req for req in REQUIRED_ARCHIVE_FILES if req not in names]
        if missing:
            raise RuntimeError(f"ZIP verification failed, missing required files: {missing}")
        html = zf.read(f"{ARCHIVE_ROOT}/frontend/index.html").decode("utf-8", errors="ignore")
        if "Investigation Objectives" not in html or "AI Investigation Partner" not in html:
            raise RuntimeError("ZIP verification failed: frontend/index.html missing expected UI markers!")
        if "Companion Drone Dialogue Terminal" in html:
            raise RuntimeError("ZIP verification failed: stale Companion Drone UI detected in archive!")

    size_mb = os.path.getsize(out_abs) / (1024.0 * 1024.0)
    print(f"Created verified clean package: {out_abs} ({file_count} files, {size_mb:.1f} MB)")
    return out_abs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Spatial Quest into a clean, relocation-safe ZIP.")
    parser.add_argument(
        "--output",
        default="/tmp/Spatial-Quest-Clean.zip",
        help="Destination ZIP path (default /tmp/Spatial-Quest-Clean.zip)",
    )
    args = parser.parse_args()
    package_project(args.output)

