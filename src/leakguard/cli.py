"""Command line: `leakguard data.csv --target y`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import __version__
from .checks import SEVERITIES
from .scan import scan


def _read(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = ";" if _sniff_semicolon(path) else ","
    return pd.read_csv(path, sep=sep)


def _sniff_semicolon(path: str) -> bool:
    # the UCI Bank Marketing CSVs use ';'
    with open(path, encoding="utf-8", errors="ignore") as fh:
        first = fh.readline()
    return first.count(";") > first.count(",")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="leakguard", description="Find data leakage in a CSV or Parquet file.")
    p.add_argument("path", help="training data (.csv or .parquet)")
    p.add_argument("--target", required=True, help="column to predict")
    p.add_argument("--time-col", help="date/time column, enables the past-vs-future check")
    p.add_argument("--id-cols", default="", help="comma-separated identifier columns")
    p.add_argument("--test", help="separate test file, enables the train/test overlap check")
    p.add_argument("--top-k", type=int, default=5, help="columns that get the drop-column test (default 5)")
    p.add_argument("--max-rows", type=int, default=20_000, help="sample larger files down to this (default 20000)")
    p.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    p.add_argument("--fail-on", choices=[*SEVERITIES[:3], "none"], default="high",
                   help="exit with code 1 if a finding is this severe or worse (default: high)")
    p.add_argument("--version", action="version", version=f"leakguard {__version__}")
    args = p.parse_args(argv)

    try:
        df = _read(args.path)
        test_df = _read(args.test) if args.test else None
        report = scan(
            df, args.target,
            time_col=args.time_col,
            id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()],
            test_df=test_df,
            top_k=args.top_k,
            max_rows=args.max_rows,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"leakguard: error: {e}", file=sys.stderr)
        return 2

    print({"text": str, "markdown": lambda r: r.to_markdown(), "json": lambda r: r.to_json()}[args.format](report))
    if args.fail_on != "none" and report.has_issues(args.fail_on):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
