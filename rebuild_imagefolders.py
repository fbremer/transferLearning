#!/usr/bin/env python3
"""Rebuild the ImageFolder trees from the photograph release.

Usage: python rebuild_imagefolders.py /path/to/polygonia-image-sets [data]
"""
import csv
import shutil
import sys
from pathlib import Path


JOBS = (
    ("splits/polygonia_224_3cat_oversample-20.tsv", "polygonia_224_3cat_oversample-20"),
    ("splits/dorsal_1024_4class.tsv", "dorsal_1024_4class"),
)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python rebuild_imagefolders.py /path/to/polygonia-image-sets [data]")
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    for tsv_name, dataset in JOBS:
        count = 0
        with (root / tsv_name).open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["split"] == "alias":
                    continue
                dest = out / dataset / row["split"] / row["class"] / row["filename"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / row["source"], dest)
                count += 1
        print(f"{dataset}: {count} files")


if __name__ == "__main__":
    main()
