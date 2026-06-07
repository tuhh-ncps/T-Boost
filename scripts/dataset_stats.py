#!/usr/bin/env python3
"""Compute dataset statistics for event log CSVs and write to a text file.

Scans provided input directories for CSV files, detects common column names
for case id, activity and timestamp, and computes per-file metrics:
- cases (unique case ids)
- events (rows)
- activities (unique activity values)
- median case length (events per case)
- median case timespan (days)
- max case timespan (days)
- min case timespan (seconds)

Usage: python scripts/dataset_stats.py --input-dirs data_csv data --output results/dataset_stats.txt
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CASE_CANDIDATES = [
    "case_id",
    "case:concept:name",
    "case",
    "Case ID",
    "caseid",
    "case-id",
]

ACTIVITY_CANDIDATES = [
    "activity",
    "concept:name",
    "Activity",
    "activityname",
    "task",
]

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "time:timestamp",
    "time",
    "date",
    "datetime",
    "start_time",
    "event_time",
]


def detect_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # try fuzzy: candidate appearing as substring
    for cand in candidates:
        for c in columns:
            if cand in c.lower():
                return c
    return None


def stats_for_file(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"file": str(path), "error": f"read error: {e}"}

    if df.shape[0] == 0:
        return {"file": str(path), "error": "empty file"}

    cols = list(df.columns)
    case_col = detect_column(cols, CASE_CANDIDATES)
    act_col = detect_column(cols, ACTIVITY_CANDIDATES)
    ts_col = detect_column(cols, TIMESTAMP_CANDIDATES)

    if case_col is None or act_col is None or ts_col is None:
        return {
            "file": str(path), "error": "missing expected columns", "found_columns": cols,
        }

    # parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    n_nat = df[ts_col].isna().sum()
    if n_nat > 0:
        # drop rows without timestamps for timespan computations
        df = df.dropna(subset=[ts_col])
        if df.shape[0] == 0:
            return {"file": str(path), "error": "all timestamps invalid"}

    total_events = int(df.shape[0])
    total_activities = int(df[act_col].nunique(dropna=True))

    grp = df.groupby(case_col)[ts_col].agg(["count", "min", "max"]).rename(
        columns={"count": "events", "min": "min_ts", "max": "max_ts"}
    )

    case_counts = grp["events"].astype(int)
    time_spans = (grp["max_ts"] - grp["min_ts"]).dt.total_seconds()

    # If a case has single event, timespan is 0
    time_spans = time_spans.fillna(0.0)

    median_case_length = float(np.median(case_counts))
    median_timespan_days = float(np.median(time_spans) / 86400.0)
    max_timespan_days = float(time_spans.max() / 86400.0)
    min_timespan_seconds = float(time_spans.min())

    return {
        "file": str(path),
        "cases": int(grp.shape[0]),
        "events": total_events,
        "activities": total_activities,
        "median_case_length": median_case_length,
        "median_timespan_days": median_timespan_days,
        "max_timespan_days": max_timespan_days,
        "min_timespan_seconds": min_timespan_seconds,
        "bad_timestamps": int(n_nat),
    }


def find_csvs(dirs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            for f in sorted(p.glob("*.csv")):
                paths.append(f)
        elif p.is_file() and p.suffix.lower() == ".csv":
            paths.append(p)
        else:
            # allow glob patterns
            for f in sorted(glob.glob(d)):
                fp = Path(f)
                if fp.exists() and fp.suffix.lower() == ".csv":
                    paths.append(fp)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dataset statistics for event logs")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=["data_csv"],
        help="Directories or CSV files to scan (default: data_csv)",
    )
    parser.add_argument(
        "--output",
        default="results/dataset_stats.txt",
        help="Output text file to write metrics",
    )
    args = parser.parse_args(argv)

    csvs = find_csvs(args.input_dirs)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if not csvs:
        lines.append("No CSV files found in: %s" % (", ".join(args.input_dirs)))
    for p in csvs:
        res = stats_for_file(p)
        if res is None:
            lines.append(f"{p}: unknown error")
            continue
        if "error" in res:
            lines.append(f"{p}: ERROR: {res.get('error')}\n  columns: {res.get('found_columns','-')}")
            continue

        lines.append(f"File: {res['file']}")
        lines.append(f"- Cases: {res['cases']}")
        lines.append(f"- Events: {res['events']}")
        lines.append(f"- Activities: {res['activities']}")
        lines.append(f"- Median case length (events): {res['median_case_length']:.2f}")
        lines.append(f"- Median timespan (days): {res['median_timespan_days']:.3f}")
        lines.append(f"- Max timespan (days): {res['max_timespan_days']:.3f}")
        lines.append(f"- Min timespan (seconds): {res['min_timespan_seconds']:.1f}")
        lines.append(f"- Rows with invalid timestamps: {res.get('bad_timestamps',0)}")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote dataset stats to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
