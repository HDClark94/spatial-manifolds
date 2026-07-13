"""
collect_border_scores.py
─────────────────────────
Rsync shifted border score parquets from the datastore into a local COHORT12
data folder. Run this locally once all Eddie jobs have finished.

Usage:
    python scripts/eddie/collect_border_scores.py
    python scripts/eddie/collect_border_scores.py --dry-run
    python scripts/eddie/collect_border_scores.py --cohort_path /path/to/COHORT12/border_scores/
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

DATASTORE_SRC = (
    "/Volumes/INCR-NolanLab/ActiveProjects/Harry/"
    "SpatialLocationManifolds2025/data/border_scores_of/"
)

# Default local destination: data/eddie/border_scores_of/ inside the repo
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOCAL = os.path.join(_REPO_ROOT, "data", "eddie", "border_scores_of")


def check_datastore():
    src = DATASTORE_SRC.rstrip('/')
    if not os.path.isdir(src):
        print(f"ERROR: Datastore not accessible at:\n  {src}")
        print("Make sure the INCR-NolanLab volume is mounted (Finder → Network).")
        sys.exit(1)
    print(f"Datastore mounted ✓  {src}")


def count_files(path):
    if not os.path.isdir(path):
        return 0
    return sum(len(fs) for _, _, fs in os.walk(path))


def rsync(src, dst, dry_run=False):
    os.makedirs(dst, exist_ok=True)
    n_before = count_files(dst)

    cmd = [
        "rsync", "--archive", "--verbose", "--progress",
        "--human-readable", "--stats", "--exclude=.DS_Store",
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src.rstrip('/') + '/', dst.rstrip('/') + '/']

    print(f"\n{'='*60}")
    print(f"  src : {src}")
    print(f"  dst : {dst}  ({n_before} files already)")
    if dry_run:
        print("  *** DRY RUN — no files will be transferred ***")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: rsync exited with code {result.returncode}")

    n_after = count_files(dst)
    return n_after - n_before


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be transferred without copying')
    parser.add_argument('--cohort_path', default=DEFAULT_LOCAL,
                        help=f'Local destination directory (default: {DEFAULT_LOCAL})')
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"  Border score collection  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'─'*60}")

    check_datastore()

    t_start  = datetime.now()
    new_files = rsync(DATASTORE_SRC, args.cohort_path, dry_run=args.dry_run)
    elapsed  = (datetime.now() - t_start).seconds

    print(f"\n{'─'*60}")
    print(f"  Done in {elapsed}s")
    print(f"  New files transferred : {new_files}")
    print(f"  Local path            : {args.cohort_path}")
    if args.dry_run:
        print("  *** Dry run — nothing was actually transferred ***")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
