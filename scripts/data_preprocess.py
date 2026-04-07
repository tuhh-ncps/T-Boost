#!/usr/bin/env python3
"""Preprocess event logs from data/ into per-file CSV outputs.

For each input log file, this script creates one CSV in data_csv/ and enriches
rows with process-oriented features:
- prefix_sequence: all preceding activities in the case (comma-separated)
- time_span: case-level duration (newest timestamp - oldest timestamp)
- next_time: time to next event in the same case
- next_activity: activity label of next event in the same case
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

import pm4py
from pm4py.objects.log.obj import EventLog, Trace


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data_csv")
SUPPORTED_SUFFIXES = {".xes", ".zip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert logs in data/ to enriched CSV files in data_csv/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Input directory containing .xes or .zip log files (default: data)",
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
        raise FileNotFoundError(f"No supported files found in {data_dir} (expected .xes/.zip)")
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


def read_log(file_path: Path) -> EventLog:
    """Read .xes directly, or first .xes member from .zip, using PM4Py."""
    if file_path.suffix.lower() == ".xes":
        return to_event_log(pm4py.read_xes(str(file_path)))

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
        return seconds / 60.0
    if unit == "hours":
        return seconds / 3600.0
    return seconds / 86400.0


def collect_columns(log: EventLog) -> List[str]:
    """Collect stable CSV columns from trace/event attributes plus derived fields."""
    base_cols = {
        "case_index",
        "event_index",
        "activity",
        "timestamp",
        "prefix_sequence",
        "time_span",
        "next_time",
        "next_activity",
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
        "time_span",
        "next_time",
        "next_activity",
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

    # Required behavior: when next timestamp is missing, fill next_time with 0.
    if current_ts is not None and next_ts is not None:
        next_time = duration_in_unit(current_ts, next_ts, unit)
    else:
        next_time = 0.0

    prefix = ",".join(a for a in activities[:event_index] if a)

    # time_span is computed from timestamps in the prefix sequence only.
    prefix_ts = [t for t in timestamps[:event_index] if t is not None]
    if len(prefix_ts) >= 2:
        prefix_span = duration_in_unit(min(prefix_ts), max(prefix_ts), unit)
    else:
        prefix_span = 0.0

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
        "time_span": f"{float(prefix_span):.6f}",
        "next_time": f"{float(next_time):.6f}",
        "next_activity": next_activity,
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


def output_name(input_file: Path) -> str:
    # Keep name aligned with source file while normalizing extension to .csv.
    return f"{input_file.stem}.csv"


def write_csv(log: EventLog, output_path: Path, unit: str) -> None:
    columns = collect_columns(log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in iter_enriched_rows(log, unit):
            writer.writerow({col: row.get(col, "") for col in columns})


def process_file(input_file: Path, output_dir: Path, unit: str) -> Path:
    log = read_log(input_file)
    output_path = output_dir / output_name(input_file)
    write_csv(log, output_path, unit)
    return output_path


def main() -> None:
    args = parse_args()
    files = list_input_files(args.data_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_file in files:
        output_file = process_file(input_file, args.output_dir, args.time_unit)
        print(f"Processed {input_file.name} -> {output_file}")


if __name__ == "__main__":
    main()
