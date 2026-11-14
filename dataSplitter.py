"""
Donor-Preserving Train/Val/Test Dataset Splitter.
Strictly isolates donors into dedicated split directories to prevent spatial data leakage.
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==============================================================================
# Configuration & Directory Setup
# ==============================================================================
BASE_DIR = Path("/workspace")
SOURCE_DIR = BASE_DIR / "patch"
DATASET_DIR = BASE_DIR / "dataset"

SPLIT_DIRS = {
    "train": DATASET_DIR / "train",
    "val": DATASET_DIR / "val",
    "test": DATASET_DIR / "test",
}

for split_path in SPLIT_DIRS.values():
    split_path.mkdir(parents=True, exist_ok=True)

# Strict Donor Assignments
TRAIN_DONORS = ["H20.33.001", "H20.33.002", "H20.33.005"]
VAL_DONORS = ["H19.33.004"]
TEST_DONORS = ["H20.33.004"]


def get_split_type(folder_name: str) -> str | None:
    """Identify destination split based on donor prefix in folder name."""
    for donor in TRAIN_DONORS:
        if donor in folder_name:
            return "train"
    for donor in VAL_DONORS:
        if donor in folder_name:
            return "val"
    for donor in TEST_DONORS:
        if donor in folder_name:
            return "test"
    return None


def main():
    if not SOURCE_DIR.exists():
        print(f"[!] Error: Source patch directory '{SOURCE_DIR}' does not exist.")
        return

    subdirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    tasks = []
    skipped_folders = []

    for folder in subdirs:
        split_type = get_split_type(folder.name)
        if not split_type:
            skipped_folders.append(folder.name)
            continue

        target_dir = SPLIT_DIRS[split_type]

        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                unique_filename = f"{folder.name}_{file_path.name}"
                dst_path = target_dir / unique_filename
                tasks.append((file_path, dst_path, split_type))

    print(f"Subdirectories found: {len(subdirs)}")
    print(f"Total image patches scheduled for transfer: {len(tasks):,}")
    if skipped_folders:
        print(f"Skipped subdirectories (no matching donor): {skipped_folders}")

    stats = {"train": 0, "val": 0, "test": 0}

    for src_path, dst_path, split_type in tqdm(tasks, desc="Splitting and moving patches"):
        shutil.move(str(src_path), str(dst_path))
        stats[split_type] += 1

    for folder in subdirs:
        try:
            folder.rmdir()
        except OSError:
            pass

    try:
        SOURCE_DIR.rmdir()
        print(f"Cleaned up empty root source directory: {SOURCE_DIR}")
    except OSError:
        pass

    print("\n" + "=" * 50)
    print("Donor-Level Split Summary")
    print("=" * 50)
    print(f"Train Set: {stats['train']:,} patches")
    print(f"Val Set:   {stats['val']:,} patches")
    print(f"Test Set:  {stats['test']:,} patches")
    print(f"Target location: {DATASET_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()