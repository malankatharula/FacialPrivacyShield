"""
export_eval500.py

Copies the exact 500 images used in the 500-image evaluation into one flat
folder, writes a manifest, and zips it for upload to Google Drive
(for running the official LowKey baseline on Colab).

Additive only: it READS the results file and the LFW images and never
modifies them. It refuses to run if the output folder already has files.

Usage (from the project root, with your venv active):
    python export_eval500.py
"""

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THIS LINE: the results file from your 500-image evaluation
# (the file whose lines start with {"index": 0, "identity": ..., "img_path": ...})
RESULTS_FILE = Path(r"C:\projects\FacialPrivacyShield\experiments\eval_full\results_full_500images_vggreal_rgb01.jsonl")
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\projects\FacialPrivacyShield")
OUT_DIR = PROJECT_ROOT / "experiments" / "eval500_inputs"
ZIP_PATH = PROJECT_ROOT / "experiments" / "eval500_inputs.zip"
EXPECTED_COUNT = 500


def load_records(path: Path):
    """Accepts a JSON list, a JSON object containing the list, or JSON Lines."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict) and "img_path" in value[0]:
                return value
        sys.exit("ERROR: found a JSON object, but no list of records with 'img_path' inside it.")

    # JSON Lines: one record per line
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    if not RESULTS_FILE.exists():
        sys.exit(f"ERROR: results file not found:\n  {RESULTS_FILE}\nEdit RESULTS_FILE at the top of this script.")

    records = load_records(RESULTS_FILE)
    records = sorted(records, key=lambda r: r["index"])
    print(f"Loaded {len(records)} records from {RESULTS_FILE.name}")

    if len(records) != EXPECTED_COUNT:
        print(f"WARNING: expected {EXPECTED_COUNT} records, found {len(records)}. Continuing anyway.")

    # Safety checks before copying anything
    names = [Path(r["img_path"]).name for r in records]
    if len(set(names)) != len(names):
        sys.exit("ERROR: two images share the same filename, so they can't go in one flat folder.")

    missing = [r["img_path"] for r in records if not Path(r["img_path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} image(s) not found, e.g.:")
        for p in missing[:5]:
            print("  ", p)
        sys.exit(1)

    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"ERROR: {OUT_DIR} already has files. Delete or rename it first.")
    if ZIP_PATH.exists():
        sys.exit(f"ERROR: {ZIP_PATH} already exists. Delete or rename it first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy images (copy2 keeps the original file untouched)
    for r in records:
        shutil.copy2(r["img_path"], OUT_DIR / Path(r["img_path"]).name)

    # Manifest: which image is which, in evaluation order
    manifest_path = OUT_DIR / "eval500_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "identity", "filename", "original_path"])
        for r in records:
            writer.writerow([r["index"], r["identity"], Path(r["img_path"]).name, r["img_path"]])

    # Zip with the folder name at the top, ready to unzip in Colab
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT_DIR.iterdir()):
            z.write(p, arcname=f"eval500_inputs/{p.name}")

    n_imgs = sum(1 for p in OUT_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_ids = len({r["identity"] for r in records})
    print(f"Copied {n_imgs} images from {n_ids} identities into:\n  {OUT_DIR}")
    print(f"Manifest:\n  {manifest_path}")
    print(f"Zip ready to upload:\n  {ZIP_PATH}  ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
