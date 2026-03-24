#!/usr/bin/env python3
"""
Download FlyWire connectome data for the fly brain model.

Checks that the required data files are present in data/.
"""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

REQUIRED_FILES = {
    '2025_Completeness_783.csv': 3_300_000,
    '2025_Connectivity_783.parquet': 96_000_000,
}


def main():
    print("🪰 Fly Brain Data Checker\n")
    print(f"Data directory: {DATA_DIR}\n")

    missing = []
    for fname, expected_size in REQUIRED_FILES.items():
        fpath = DATA_DIR / fname
        if fpath.exists():
            actual = fpath.stat().st_size
            if actual > expected_size * 0.8:
                print(f"  ✓ {fname} ({actual / 1_000_000:.1f} MB)")
                continue
        missing.append(fname)

    if not missing:
        print("\n✅ All required data files are present!")
        print("   You're ready to run: python fly_chat.py")
        return

    print(f"\n⚠️  Missing {len(missing)} file(s):")
    for f in missing:
        print(f"   • {f}")
    print("\nThese files should have been included via Git LFS.")
    print("Try: git lfs pull")


if __name__ == '__main__':
    main()
