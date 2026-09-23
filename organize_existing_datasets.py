"""
Organize Existing Datasets into Letter Folders
─────────────────────────────────────────────
Scans the ./datasets directory for legacy flat files:
  e.g. dataset_<letter>_<timestamp>.csv
and organizes them into:
  datasets/<letter>/dataset_<letter>_<id>.csv

Usage:
  python organize_existing_datasets.py          # Preview only (dry-run)
  python organize_existing_datasets.py --apply  # Move and rename files
"""

import re
import sys
import shutil
from pathlib import Path

DATASET_DIR = Path("./datasets")

def get_next_id_in_folder(folder: Path, letter: str) -> int:
    pattern = re.compile(rf"^dataset_{re.escape(letter)}_(\d+)\.csv$", re.IGNORECASE)
    existing = []
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file():
                m = pattern.match(f.name)
                if m:
                    try:
                        existing.append(int(m.group(1)))
                    except ValueError:
                        pass
    return max(existing, default=0) + 1

def main():
    apply_changes = "--apply" in sys.argv
    if not DATASET_DIR.exists():
        print("datasets/ directory not found.")
        return

    # Match files like dataset_<letter>_<timestamp>.csv directly inside DATASET_DIR
    legacy_pattern = re.compile(r"^dataset_([A-Za-z0-9_\-]+)_(\d{8}_\d{6})\.csv$", re.IGNORECASE)
    
    files_to_organize = []
    for f in sorted(DATASET_DIR.iterdir()):
        if f.is_file():
            m = legacy_pattern.match(f.name)
            if m:
                raw_label = m.group(1)
                clean_label = raw_label.upper() if (len(raw_label) == 1 and raw_label.isalpha()) else raw_label
                files_to_organize.append((f, clean_label))

    if not files_to_organize:
        print("No legacy flat files found in datasets/ directory. All files are already organized!")
        return

    print(f"Found {len(files_to_organize)} legacy dataset files in datasets/:\n")
    
    # Track assigned IDs per letter
    current_ids = {}

    for file_path, letter in files_to_organize:
        target_dir = DATASET_DIR / letter
        if letter not in current_ids:
            current_ids[letter] = get_next_id_in_folder(target_dir, letter)
        else:
            current_ids[letter] += 1
        
        assigned_id = current_ids[letter]
        new_filename = f"dataset_{letter}_{assigned_id}.csv"
        target_path = target_dir / new_filename

        action_str = "[MOVE]" if apply_changes else "[PLAN]"
        print(f"  {action_str} {file_path.name}  -->  datasets/{letter}/{new_filename}")

        if apply_changes:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(target_path))

    if not apply_changes:
        print("\n[Preview mode] No files were moved.")
        print("To apply these changes, run:")
        print("  python organize_existing_datasets.py --apply")
    else:
        print(f"\nSuccessfully organized {len(files_to_organize)} files into letter folders!")

if __name__ == "__main__":
    main()
