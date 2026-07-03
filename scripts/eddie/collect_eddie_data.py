"""
collect_eddie_data.py
─────────────────────
Transfers completed XGBoost assay results from the datastore to the local
data/eddie/ directory. Run this once all Eddie jobs have finished.

Usage:
    python scripts/collect_eddie_data.py

Options:
    --dry-run    Print what would be copied without transferring anything
    --subfolder  Transfer only a specific subfolder (e.g. xgboost_pairwise)
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASTORE_BASE = (
    "/Volumes/INCR-NolanLab/ActiveProjects/Harry/"
    "SpatialLocationManifolds2025/data/xgboost"
)
LOCAL_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "eddie"
)

SUBFOLDERS = [
    "xgboost_cell_number_assay",
    "xgboost_cell_number_assay_of",
    "xgboost_medlat_ngs_vr",
    "xgboost_medlat_ngs_of",
    "xgboost_pairwise",
]


def check_datastore_mounted():
    if not os.path.isdir(DATASTORE_BASE):
        print(f"ERROR: Datastore not accessible at:\n  {DATASTORE_BASE}")
        print("Make sure the INCR-NolanLab volume is mounted (Finder → Network).")
        sys.exit(1)
    print(f"Datastore mounted ✓  {DATASTORE_BASE}")


def make_local_dirs(subfolders):
    os.makedirs(LOCAL_BASE, exist_ok=True)
    for sf in subfolders:
        path = os.path.join(LOCAL_BASE, sf)
        os.makedirs(path, exist_ok=True)
    print(f"Local directories ready: {LOCAL_BASE}/")


def count_files(path):
    if not os.path.isdir(path):
        return 0
    return sum(len(files) for _, _, files in os.walk(path))


def rsync_subfolder(subfolder, dry_run=False):
    src = os.path.join(DATASTORE_BASE, subfolder) + "/"   # trailing / = contents
    dst = os.path.join(LOCAL_BASE, subfolder) + "/"

    n_src_before = count_files(src)
    n_dst_before = count_files(dst)

    if n_src_before == 0:
        print(f"  [{subfolder}]  No files on datastore yet — skipping")
        return 0, 0

    cmd = [
        "rsync",
        "--archive",          # preserve timestamps, permissions
        "--verbose",
        "--progress",
        "--human-readable",
        "--stats",
        "--exclude=.DS_Store",
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, dst]

    print(f"\n{'='*60}")
    print(f"  {subfolder}")
    print(f"  src: {src}  ({n_src_before} files)")
    print(f"  dst: {dst}  ({n_dst_before} files already)")
    if dry_run:
        print("  *** DRY RUN — no files will be transferred ***")
    print(f"{'='*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  WARNING: rsync exited with code {result.returncode}")

    n_dst_after = count_files(dst)
    new_files = n_dst_after - n_dst_before
    return n_src_before, new_files


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be transferred without copying')
    parser.add_argument('--subfolder', default=None,
                        choices=SUBFOLDERS,
                        help='Transfer only a specific subfolder')
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"  Eddie data collection  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'─'*60}")

    check_datastore_mounted()

    subfolders = [args.subfolder] if args.subfolder else SUBFOLDERS
    make_local_dirs(subfolders)

    total_src   = 0
    total_new   = 0
    t_start     = datetime.now()

    for sf in subfolders:
        n_src, n_new = rsync_subfolder(sf, dry_run=args.dry_run)
        total_src += n_src
        total_new += n_new

    elapsed = (datetime.now() - t_start).seconds
    print(f"\n{'─'*60}")
    print(f"  Done in {elapsed}s")
    print(f"  Datastore files available : {total_src}")
    print(f"  New files transferred      : {total_new}")
    print(f"  Local base                 : {LOCAL_BASE}/")
    if args.dry_run:
        print("  *** Dry run — nothing was actually transferred ***")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
