#!/usr/bin/env python3
"""Analyze XES event logs with PM4Py.

For each .xes file in the input data directory, this script loads the event log,
builds a directly-follows graph from the most frequent activities, computes
summary statistics for traces grouped by their last activity, and writes CSV
reports plus PDF plots to the output directory.

Input selection:
    Folder-based only via --data-dir. This script processes all .xes files in
    the folder and does not provide a single-file filter flag.

Example:
    python scripts/data_analysis.py --data-dir data --output-dir results/data_analysis

Usage:
    --data-dir PATH        Directory containing .xes files (default: data)
    --output-dir PATH      Directory for CSV/PDF outputs (default: results/data_analysis)
    --top-activities N     Keep the top N activities in the DFG plot (default: 20)
    --time-unit UNIT       seconds, minutes, hours, or days (default: hours)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace


TIME_FACTORS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


@dataclass
class LastActivityStats:
    case_count: int = 0
    variants: set[Tuple[str, ...]] = field(default_factory=set)
    variant_counts: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    variant_exec_times: Dict[Tuple[str, ...], List[float]] = field(default_factory=dict)
    seq_lengths: List[int] = field(default_factory=list)
    exec_times: List[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze XES files using PM4Py and generate DFG + last-activity summaries/plots.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing .xes files (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "data_analysis",
        help="Directory to write analysis outputs (default: results/data_analysis)",
    )
    parser.add_argument(
        "--top-activities",
        type=int,
        default=20,
        help="Top-N most frequent activities to keep for DFG plot (default: 20)",
    )
    parser.add_argument(
        "--time-unit",
        choices=["seconds", "minutes", "hours", "days"],
        default="hours",
        help="Time unit for execution-time metrics (default: hours)",
    )
    return parser.parse_args()


def find_xes_files(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".xes")
    if not files:
        raise FileNotFoundError(f"No .xes files found in {data_dir}")
    return files


def load_event_log(path: Path) -> EventLog:
    data = pm4py.read_xes(str(path))
    if isinstance(data, EventLog):
        return data

    converted = pm4py.convert_to_event_log(data)
    if not isinstance(converted, EventLog):
        raise TypeError(f"Could not convert {path.name} to EventLog")
    return converted


def clone_log_structure(log: EventLog) -> EventLog:
    return EventLog(
        attributes=dict(getattr(log, "attributes", {})),
        extensions=dict(getattr(log, "extensions", {})),
        classifiers=dict(getattr(log, "classifiers", {})),
        omni_present=dict(getattr(log, "omni_present", {})),
        properties=dict(getattr(log, "properties", {})),
    )


def filter_log_top_activities(log: EventLog, top_n: int) -> EventLog:
    if top_n <= 0:
        return log

    counts: Dict[str, int] = {}
    for trace in log:
        for event in trace:
            activity = str(event.get("concept:name", "")).strip()
            if not activity:
                continue
            counts[activity] = counts.get(activity, 0) + 1

    if not counts:
        return log

    top_activities = {
        name
        for name, _count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    }

    filtered = clone_log_structure(log)
    for trace in log:
        new_trace = Trace(
            attributes=dict(getattr(trace, "attributes", {})),
            properties=dict(getattr(trace, "properties", {})),
        )
        for event in trace:
            activity = str(event.get("concept:name", "")).strip()
            if activity in top_activities:
                new_trace.append(event)
        if len(new_trace) > 0:
            filtered.append(new_trace)

    return filtered


def save_dfg_plot(log: EventLog, output_path: Path) -> None:
    dfg, start_activities, end_activities = pm4py.discover_dfg(log)
    if not dfg:
        raise ValueError("DFG is empty after activity filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm4py.save_vis_dfg(
        dfg,
        start_activities,
        end_activities,
        str(output_path),
        aggregation_measure="frequency",
        rankdir="TB",
    )


def to_datetime(value) -> datetime | None:
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


def build_last_activity_stats(log: EventLog, time_unit: str) -> List[Dict[str, float | int | str]]:
    stats: Dict[str, LastActivityStats] = {}
    case_points: List[Tuple[str, str, float]] = []
    factor = TIME_FACTORS[time_unit]

    for trace in log:
        activities: List[str] = []
        timestamps: List[datetime] = []

        for event in trace:
            activity = str(event.get("concept:name", "")).strip()
            if activity:
                activities.append(activity)

            ts = to_datetime(event.get("time:timestamp"))
            if ts is not None:
                timestamps.append(ts)

        if not activities:
            continue

        last_activity = activities[-1]
        variant = tuple(activities)
        seq_length = len(activities)

        if len(timestamps) >= 2:
            exec_time = (max(timestamps) - min(timestamps)).total_seconds() / factor
            exec_time = max(0.0, exec_time)
        else:
            exec_time = 0.0

        rec = stats.get(last_activity)
        if rec is None:
            rec = LastActivityStats()
            stats[last_activity] = rec

        rec.case_count += 1
        rec.variants.add(variant)
        rec.variant_counts[variant] = rec.variant_counts.get(variant, 0) + 1
        rec.variant_exec_times.setdefault(variant, []).append(exec_time)
        rec.seq_lengths.append(seq_length)
        rec.exec_times.append(exec_time)
        case_points.append((last_activity, " > ".join(variant), exec_time))

    rows: List[Dict[str, float | int | str]] = []
    for last_activity, rec in sorted(stats.items(), key=lambda x: (-x[1].case_count, x[0])):
        row = {
            "last_activity": last_activity,
            "case_count": rec.case_count,
            "variant_count": len(rec.variants),
            "shortest_sequence_length": min(rec.seq_lengths),
            "longest_sequence_length": max(rec.seq_lengths),
            "avg_sequence_length": mean(rec.seq_lengths),
            "shortest_execution_time": min(rec.exec_times),
            "longest_execution_time": max(rec.exec_times),
            "avg_execution_time": mean(rec.exec_times),
        }
        rows.append(row)

    return rows, stats, case_points


def plot_task5(stats: Dict[str, LastActivityStats], output_pdf: Path) -> None:
    if not stats:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    for last_activity, rec in sorted(stats.items(), key=lambda x: (-x[1].case_count, x[0])):
        counts = sorted(rec.variant_counts.values(), reverse=True)
        if not counts:
            continue
        cum = np.cumsum(np.array(counts, dtype=float))
        cum_pct = 100.0 * cum / cum[-1]
        ranks = np.arange(1, len(counts) + 1)
        ax.plot(ranks, cum_pct, linewidth=1.6, alpha=0.85, label=last_activity)

    ax.set_xlabel("Variant rank")
    ax.set_ylabel("Cumulative % of cases")
    ax.set_title("Variant contribution curves (Pareto)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_task6_trace_timespan_scatter(
    stats: Dict[str, LastActivityStats],
    case_points: Sequence[Tuple[str, str, float]],
    output_pdf: Path,
    mapping_csv: Path,
    time_unit: str,
) -> None:
    """Plot encoded unique traces (Y) vs timespan (X), ordered by trace code."""
    if not stats or not case_points:
        return

    # Build stable variant encoding (Y-axis).
    variant_meta: List[Tuple[str, str, int]] = []
    for last_activity, rec in sorted(stats.items(), key=lambda x: x[0]):
        for variant, count in rec.variant_counts.items():
            variant_meta.append((last_activity, " > ".join(variant), count))
    variant_meta.sort(key=lambda x: (x[0], x[1]))

    variant_to_code: Dict[Tuple[str, str], Tuple[int, str, int]] = {}
    for idx, (last_activity, variant_label, count) in enumerate(variant_meta, start=1):
        variant_to_code[(last_activity, variant_label)] = (idx, f"T{idx:04d}", count)

    rows: List[Dict[str, str | int | float]] = []

    for last_activity, variant_label, timespan in case_points:
        key = (last_activity, variant_label)
        if key not in variant_to_code:
            continue
        encoded_y, trace_code, variant_case_count = variant_to_code[key]
        rows.append(
            {
                "trace_code": trace_code,
                "encoded_y": encoded_y,
                "last_activity": last_activity,
                "timespan": timespan,
                "case_count": variant_case_count,
                "trace_variant": variant_label,
            }
        )

    rows.sort(key=lambda r: (str(r["trace_code"]), float(r["timespan"])))

    x_vals = [float(r["timespan"]) for r in rows]
    y_vals = [int(r["encoded_y"]) for r in rows]
    sizes = [36.0 for _ in rows]

    fig, ax = plt.subplots(figsize=(12, max(6, min(14, 0.03 * len(case_points)))))
    ax.scatter(x_vals, y_vals, s=sizes, alpha=0.7, color="#1f77b4", edgecolors="black", linewidths=0.3)

    ax.set_xlabel(f"Time span ({time_unit})")
    ax.set_ylabel("Unique trace (encoded label)")
    ax.set_title("Task 6: Unique traces vs timespan (ordered by trace code)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)

    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with mapping_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trace_code", "encoded_y", "last_activity", "timespan", "case_count", "trace_variant"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "last_activity",
                "case_count",
                "variant_count",
                "shortest_sequence_length",
                "longest_sequence_length",
                "avg_sequence_length",
                "shortest_execution_time",
                "longest_execution_time",
                "avg_execution_time",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_task3(rows: Sequence[Dict[str, float | int | str]], output_pdf: Path, time_unit: str) -> None:
    if not rows:
        return

    labels = [str(r["last_activity"]) for r in rows]
    avg_len = [float(r["avg_sequence_length"]) for r in rows]
    min_len = [float(r["shortest_sequence_length"]) for r in rows]
    max_len = [float(r["longest_sequence_length"]) for r in rows]
    avg_time = [float(r["avg_execution_time"]) for r in rows]
    min_time = [float(r["shortest_execution_time"]) for r in rows]
    max_time = [float(r["longest_execution_time"]) for r in rows]

    x = list(range(len(labels)))
    bar_width = 0.38

    fig, ax_counter = plt.subplots(figsize=(max(14, 1.2 * len(labels)), 8))

    len_err_low = [a - b for a, b in zip(avg_len, min_len)]
    len_err_high = [b - a for a, b in zip(avg_len, max_len)]
    ax_counter.bar(
        [i - bar_width / 2 for i in x],
        avg_len,
        width=bar_width,
        yerr=[len_err_low, len_err_high],
        capsize=4,
        color="#1f77b4",
        label="Avg sequence length",
    )
    ax_counter.set_ylabel("Counter axis: sequence length")

    ax_clock = ax_counter.twinx()
    time_err_low = [a - b for a, b in zip(avg_time, min_time)]
    time_err_high = [b - a for a, b in zip(avg_time, max_time)]
    ax_clock.bar(
        [i + bar_width / 2 for i in x],
        avg_time,
        width=bar_width,
        yerr=[time_err_low, time_err_high],
        capsize=4,
        color="#ff7f0e",
        alpha=0.85,
        label=f"Avg execution time ({time_unit})",
    )
    ax_clock.set_ylabel(f"Clock axis: execution time ({time_unit})")

    ax_counter.set_xticks(x)
    ax_counter.set_xticklabels(labels, rotation=45, ha="right")
    ax_counter.set_title("Task 3: Sequence length and execution time by last activity")

    h1, l1 = ax_counter.get_legend_handles_labels()
    h2, l2 = ax_clock.get_legend_handles_labels()
    ax_counter.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_task4(rows: Sequence[Dict[str, float | int | str]], output_pdf: Path) -> None:
    if not rows:
        return

    labels = [str(r["last_activity"]) for r in rows]
    case_count = [float(r["case_count"]) for r in rows]
    variant_count = [float(r["variant_count"]) for r in rows]
    avg_len = [float(r["avg_sequence_length"]) for r in rows]
    min_len = [float(r["shortest_sequence_length"]) for r in rows]
    max_len = [float(r["longest_sequence_length"]) for r in rows]

    total_cases = sum(case_count)
    # Case-count weighted percentage: (variant/case) weighted by case share.
    # This simplifies to variant_count / total_cases * 100.
    percentage = [((v / total_cases) * 100.0) if total_cases > 0 else 0.0 for v in variant_count]

    x = list(range(len(labels)))
    bar_width = 0.38

    fig, ax_pct = plt.subplots(figsize=(max(14, 1.2 * len(labels)), 8))
    ax_pct.bar(
        [i - bar_width / 2 for i in x],
        percentage,
        width=bar_width,
        color="#2ca02c",
        label="Weighted variant percentage (%)",
    )
    ax_pct.set_ylabel("Percentage axis (%)")

    ax_counter = ax_pct.twinx()
    len_err_low = [a - b for a, b in zip(avg_len, min_len)]
    len_err_high = [b - a for a, b in zip(avg_len, max_len)]
    ax_counter.bar(
        [i + bar_width / 2 for i in x],
        avg_len,
        width=bar_width,
        yerr=[len_err_low, len_err_high],
        capsize=4,
        color="#9467bd",
        alpha=0.85,
        label="Avg sequence length",
    )
    ax_counter.set_ylabel("Counter axis: sequence length")

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(labels, rotation=45, ha="right")
    ax_pct.set_title("Task 4: Sequence diversity (%) and sequence length by last activity")

    h1, l1 = ax_pct.get_legend_handles_labels()
    h2, l2 = ax_counter.get_legend_handles_labels()
    ax_pct.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    plt.close(fig)


def analyze_file(xes_path: Path, output_dir: Path, top_activities: int, time_unit: str) -> None:
    stem = xes_path.stem
    dataset_dir = output_dir / stem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{stem}] Loading log")
    log = load_event_log(xes_path)

    print(f"[{stem}] Task 1: DFG with top-{top_activities} activities")
    filtered_log = filter_log_top_activities(log, top_activities)
    dfg_pdf = dataset_dir / f"{stem}_dfg_top_{top_activities}.pdf"
    save_dfg_plot(filtered_log, dfg_pdf)

    print(f"[{stem}] Task 2: Last-activity summary table")
    summary_rows, stats, case_points = build_last_activity_stats(log, time_unit)
    summary_csv = dataset_dir / f"{stem}_last_activity_summary.csv"
    write_summary_csv(summary_csv, summary_rows)

    print(f"[{stem}] Task 3: Bar plot (counter + clock axis)")
    task3_pdf = dataset_dir / f"{stem}_task3_counter_clock.pdf"
    plot_task3(summary_rows, task3_pdf, time_unit)

    print(f"[{stem}] Task 4: Bar plot (percentage + counter axis)")
    task4_pdf = dataset_dir / f"{stem}_task4_percentage_counter.pdf"
    plot_task4(summary_rows, task4_pdf)

    print(f"[{stem}] Task 5: Variant contribution curve (Pareto)")
    task5_pdf = dataset_dir / f"{stem}_task5_variant_pareto.pdf"
    plot_task5(stats, task5_pdf)

    print(f"[{stem}] Task 6: Scatter unique traces vs timespan")
    task6_pdf = dataset_dir / f"{stem}_task6_trace_timespan_scatter.pdf"
    task6_csv = dataset_dir / f"{stem}_task6_trace_label_mapping.csv"
    plot_task6_trace_timespan_scatter(stats, case_points, task6_pdf, task6_csv, time_unit)

    print(f"[{stem}] Done")
    print(f"  - DFG: {dfg_pdf}")
    print(f"  - Table: {summary_csv}")
    print(f"  - Plot3: {task3_pdf}")
    print(f"  - Plot4: {task4_pdf}")
    print(f"  - Plot5: {task5_pdf}")
    print(f"  - Plot6: {task6_pdf}")
    print(f"  - Plot6 labels: {task6_csv}")


def main() -> None:
    args = parse_args()
    xes_files = find_xes_files(args.data_dir)

    for xes_path in xes_files:
        analyze_file(xes_path, args.output_dir, args.top_activities, args.time_unit)


if __name__ == "__main__":
    main()
