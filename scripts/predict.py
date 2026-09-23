"""Batch QC of DXA studies: a folder or .zip of DICOM files -> results table (CSV + XLSX).

One row per image (ТЗ 2.5). Unreadable files and processing errors become Failure rows.
"""
import argparse
from pathlib import Path

from dxaqc.batch import run_batch, zip_folder
from dxaqc.service import QCService


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, help="folder with studies or a .zip archive")
    ap.add_argument("--out", type=Path, default=Path("results.csv"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--vis", type=Path, default=None, help="folder for overlays (PNG + DICOM Secondary Capture); zipped next to --out")
    ap.add_argument("--policy", choices=["balanced", "screening"], default=None,
                    help="operating point: balanced (default, F1-optimal per type) or screening (catches ~80 % of each type)")
    args = ap.parse_args()

    df = run_batch(args.input, QCService(args.models, policy=args.policy), args.vis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    df.to_excel(args.out.with_suffix(".xlsx"), index=False)
    if args.vis and args.vis.exists():
        archive = args.out.with_name(args.out.stem + "_vis.zip")
        zip_folder(args.vis, archive)
        print(f"overlays -> {archive}")
    ok = (df.processing_status == "Success").sum()
    print(f"{len(df)} images, {ok} processed, {len(df) - ok} failed -> {args.out} (+ .xlsx); "
          f"mean time {df.time_of_processing.mean():.2f}s")


if __name__ == "__main__":
    main()
