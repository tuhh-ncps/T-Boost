#!/usr/bin/env python3
"""Preprocess event logs from data/ into per-file CSV outputs split by next_time range.

For each input log file, this script creates up to four CSV files in data_csv/:
- <dataset>_second.csv: rows with next_time_seconds <= 60
- <dataset>_minute.csv: rows with 60 < next_time_seconds <= 3600
- <dataset>_hour.csv: rows with 3600 < next_time_seconds <= 86400
- <dataset>_day.csv: rows with next_time_seconds > 86400

Each row is enriched with process-oriented features:
- prefix_sequence: all preceding activities in the case (comma-separated)
- prefix_delta_t_sequence: time gaps between consecutive prefix events (comma-separated)
- time_span: duration within the prefix history (max prefix timestamp - min prefix timestamp)
- next_time: time to next event in the same case, represented in --time-unit
- next_time_seconds: time to next event in seconds (used for per-unit file splitting)
- next_activity: activity label of next event in the same case
- time_unit: unit used for time_span and next_time

Input selection:
    Use --data-dir for the input folder. Optionally use --dataset to process
    only one file from that folder.

Example:
    python scripts/data_preprocess_per_unit.py --data-dir data --output-dir data_csv

Usage:
    --data-dir PATH   Input directory with .xes/.zip/.csv logs (default: data)
    --output-dir PATH Output directory for generated CSV files (default: data_csv)
    --time-unit UNIT  seconds, minutes, hours, or days (default: seconds)
    --dataset NAME    Process only one dataset file
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog, Trace


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data_csv")
SUPPORTED_SUFFIXES = {".xes", ".zip", ".csv"}


CASE_COL_CANDIDATES = [
    "case:concept:name",
    "trace_concept:name",
    "@@case_index",
    "case_index",
    "case id",
    "case_id",
    "caseid",
    "caseid",
]
ACTIVITY_COL_CANDIDATES = [
    "concept:name",
    "activity",
    "event",
    "task",
    "activityid",
    "activity id",
]
TIMESTAMP_COL_CANDIDATES = [
    "time:timestamp",
    "complete timestamp",
    "timestamp",
    "date",
    "datetime",
    "completetimestamp",
    "complete_timestamp",
]

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert logs in data/ to enriched CSV files split by next_time ranges."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Input directory containing .xes, .zip, or .csv log files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated CSV files (default: data_csv)",
    )
    parser.add_argument(
        "--time-unit",
        choices=["seconds", "minutes", "hours", "days"],
        default="seconds",
        help="Unit used for time_span and next_time (default: seconds)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset file to process (e.g., helpdesk.csv). If omitted, processes all files in --data-dir",
    )
    return parser.parse_args()


def list_input_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {data_dir}")

    files = [
        p
        for p in sorted(data_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No supported files found in {data_dir} (expected .xes/.zip/.csv)")
    return files


def parse_timestamp(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def to_event_log(data) -> EventLog:
    if isinstance(data, EventLog):
        return data

    converted = pm4py.convert_to_event_log(data)
    if not isinstance(converted, EventLog):
        raise TypeError(f"Could not convert data type {type(data).__name__} to EventLog")
    return converted


def _normalize_col_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def _find_matching_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    by_normalized = {_normalize_col_name(col): col for col in df.columns}
    for candidate in candidates:
        match = by_normalized.get(_normalize_col_name(candidate))
        if match is not None:
            return match
    return None


def read_csv_as_event_log(file_path: Path) -> EventLog:
    df = pd.read_csv(file_path, low_memory=False)

    case_col = _find_matching_column(df, CASE_COL_CANDIDATES)
    activity_col = _find_matching_column(df, ACTIVITY_COL_CANDIDATES)
    timestamp_col = _find_matching_column(df, TIMESTAMP_COL_CANDIDATES)

    missing_labels = []
    if case_col is None:
        missing_labels.append("case id")
    if activity_col is None:
        missing_labels.append("activity")
    if timestamp_col is None:
        missing_labels.append("timestamp")
    if missing_labels:
        raise ValueError(
            f"CSV {file_path.name} is missing required columns for: {', '.join(missing_labels)}"
        )

    renamed = df.rename(
        columns={
            case_col: "case:concept:name",
            activity_col: "concept:name",
            timestamp_col: "time:timestamp",
        }
    ).copy()

    renamed["case:concept:name"] = renamed["case:concept:name"].astype(str).str.strip()
    renamed["concept:name"] = renamed["concept:name"].astype(str).str.strip()
    renamed["time:timestamp"] = pd.to_datetime(renamed["time:timestamp"], errors="coerce", utc=True)

    renamed = renamed[
        (renamed["case:concept:name"] != "")
        & (renamed["concept:name"] != "")
        & renamed["time:timestamp"].notna()
    ].copy()
    if renamed.empty:
        raise ValueError(f"CSV {file_path.name} has no valid rows after parsing case/activity/timestamp")

    formatted = pm4py.format_dataframe(
        renamed,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
    )
    return to_event_log(pm4py.convert_to_event_log(formatted))


def read_log(file_path: Path) -> EventLog:
    """Read .xes/.zip/.csv into an EventLog using PM4Py."""
    if file_path.suffix.lower() == ".xes":
        return to_event_log(pm4py.read_xes(str(file_path)))

    if file_path.suffix.lower() == ".csv":
        return read_csv_as_event_log(file_path)

    with zipfile.ZipFile(file_path, "r") as archive:
        xes_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".xes")
            and "/__macosx/" not in f"/{name.lower()}"
            and not Path(name).name.startswith("._")
        ]
        if not xes_members:
            raise ValueError(f"No .xes member found in archive: {file_path.name}")

        with archive.open(xes_members[0], "r") as zipped_stream:
            with tempfile.NamedTemporaryFile(suffix=".xes", delete=False) as temp_file:
                shutil.copyfileobj(zipped_stream, temp_file)
                temp_path = Path(temp_file.name)

    try:
        return to_event_log(pm4py.read_xes(str(temp_path)))
    finally:
        temp_path.unlink(missing_ok=True)


def duration_in_unit(start: datetime, end: datetime, unit: str) -> float:
    seconds = (end - start).total_seconds()
    if unit == "seconds":
        return seconds
    if unit == "minutes":
        return seconds / SECONDS_PER_MINUTE
    if unit == "hours":
        return seconds / SECONDS_PER_HOUR
    return seconds / SECONDS_PER_DAY


def collect_columns(log: EventLog) -> List[str]:
    """Collect stable CSV columns from trace/event attributes plus derived fields."""
    base_cols = {
        "case_index",
        "event_index",
        "activity",
        "timestamp",
        "prefix_sequence",
        "prefix_delta_t_sequence",
        "time_span",
        "next_time",
        "next_time_seconds",
        "next_activity",
        "time_unit",
    }

    for trace in log:
        for key in trace.attributes.keys():
            base_cols.add(f"trace_{key}")
        for event in trace:
            for key in event.keys():
                base_cols.add(str(key))

    preferred_order = [
        "case_index",
        "event_index",
        "activity",
        "timestamp",
        "prefix_sequence",
        "prefix_delta_t_sequence",
        "time_span",
        "next_time",
        "next_time_seconds",
        "next_activity",
        "time_unit",
    ]

    others = sorted(c for c in base_cols if c not in preferred_order)
    return preferred_order + others


def row_for_event(
    trace: Trace,
    event_index: int,
    case_index: int,
    activities: List[str],
    timestamps: List[datetime | None],
    unit: str,
) -> Dict[str, str]:
    event = trace[event_index]
    current_ts = timestamps[event_index]
    has_next = event_index + 1 < len(timestamps)
    next_ts = timestamps[event_index + 1] if has_next else None

    if current_ts is not None and next_ts is not None:
        next_time_seconds = max(0.0, (next_ts - current_ts).total_seconds())
    else:
        next_time_seconds = 0.0
    next_time = duration_in_unit(current_ts, next_ts, unit) if (current_ts is not None and next_ts is not None) else 0.0

    prefix = ",".join(a for a in activities[:event_index] if a)

    prefix_ts = [t for t in timestamps[:event_index] if t is not None]
    prefix_delta_t: List[float] = []
    previous_ts = None
    for prefix_ts_value in timestamps[:event_index]:
        if previous_ts is None or prefix_ts_value is None:
            prefix_delta_t.append(0.0)
        else:
            delta_value = duration_in_unit(previous_ts, prefix_ts_value, unit)
            prefix_delta_t.append(max(0.0, float(delta_value)))
        previous_ts = prefix_ts_value

    if len(prefix_ts) >= 2:
        prefix_span = duration_in_unit(min(prefix_ts), max(prefix_ts), unit)
    else:
        prefix_span = 0.0
    prefix_span = max(0.0, float(prefix_span))

    if has_next:
        candidate_next_activity = activities[event_index + 1].strip()
        next_activity = candidate_next_activity if candidate_next_activity else "END"
    else:
        next_activity = "END"

    row: Dict[str, str] = {
        "case_index": str(case_index),
        "event_index": str(event_index + 1),
        "activity": activities[event_index],
        "timestamp": current_ts.isoformat() if current_ts is not None else "",
        "prefix_sequence": prefix,
        "prefix_delta_t_sequence": ",".join(f"{float(delta):.6f}" for delta in prefix_delta_t),
        "time_span": f"{float(prefix_span):.6f}",
        "next_time": f"{float(next_time):.6f}",
        "next_time_seconds": f"{float(next_time_seconds):.6f}",
        "next_activity": next_activity,
        "time_unit": unit,
    }

    for key, value in trace.attributes.items():
        row[f"trace_{key}"] = "" if value is None else str(value)

    for key, value in event.items():
        row[str(key)] = "" if value is None else str(value)

    return row


def iter_enriched_rows(log: EventLog, unit: str) -> Iterable[Dict[str, str]]:
    for case_index, trace in enumerate(log, start=1):
        activities = [str(event.get("concept:name", "")).strip() for event in trace]
        timestamps = [parse_timestamp(event.get("time:timestamp")) for event in trace]

        for event_index in range(len(trace)):
            yield row_for_event(trace, event_index, case_index, activities, timestamps, unit)


def output_name(base_stem: str, bucket_name: str) -> str:
    return f"{base_stem}_{bucket_name}.csv"


def classify_next_time_bucket(next_time_seconds: float) -> str:
    # Inclusive boundaries avoid dropping transition values (60, 3600, 86400).
    if next_time_seconds <= SECONDS_PER_MINUTE:
        return "second"
    if next_time_seconds <= SECONDS_PER_HOUR:
        return "minute"
    if next_time_seconds <= SECONDS_PER_DAY:
        return "hour"
    return "day"


def write_split_csvs(log: EventLog, input_file: Path, output_dir: Path, unit: str) -> List[Path]:
    columns = collect_columns(log)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_bucket: Dict[str, List[Dict[str, str]]] = {
        "second": [],
        "minute": [],
        "hour": [],
        "day": [],
    }

    for row in iter_enriched_rows(log, unit):
        next_time_seconds = float(row.get("next_time_seconds", 0.0))
        bucket_name = classify_next_time_bucket(next_time_seconds)
        rows_by_bucket[bucket_name].append(row)

    written_files: List[Path] = []
    for bucket_name, rows in rows_by_bucket.items():
        output_path = output_dir / output_name(input_file.stem, bucket_name)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in columns})
        written_files.append(output_path)

    return written_files


def process_file(input_file: Path, output_dir: Path, unit: str) -> List[Path]:
    log = read_log(input_file)
    return write_split_csvs(log, input_file, output_dir, unit)


def main() -> None:
    args = parse_args()

    if args.dataset:
        input_file = args.data_dir / args.dataset
        if not input_file.exists():
            raise FileNotFoundError(f"Dataset not found: {input_file}")
        files = [input_file]
    else:
        files = list_input_files(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_file in files:
        output_files = process_file(input_file, args.output_dir, args.time_unit)
        for output_file in output_files:
            print(f"Processed {input_file.name} -> {output_file}")


if __name__ == "__main__":
    main()
