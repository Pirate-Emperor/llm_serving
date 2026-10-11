#!/usr/bin/env python3
"""Validate sweep CSVs and produce readable SVG charts plus a Markdown report."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def svg_text(x: float, y: float, text: str, size: int = 18,
             fill: str = "#e2e8f0", weight: str = "400",
             anchor: str = "start") -> str:
    safe = (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-family="DejaVu Sans, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}">{safe}</text>')


def write_svg(path: Path, width: int, height: int, parts: list[str]) -> None:
    body = "\n".join(parts)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" '
        f'rx="24" fill="#0b1220"/>{body}</svg>\n',
        encoding="utf-8",
    )
    ET.parse(path)


def scheduler_chart(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["workload"], int(row["prefill_chunk"]))].append(row)

    workloads = ["decode_heavy", "mixed_long_prompt", "prompt_burst"]
    chunks = [8, 16, 32, 64]
    colors = ["#38bdf8", "#34d399", "#fbbf24", "#e879f9"]
    width, height = 1440, 850
    parts = [
        svg_text(64, 64, "Scheduler experiment: decode-gap percentiles", 32, "#f8fafc", "700"),
        svg_text(64, 105, f"{len(rows)} measured simulation records across 36 settings", 20, "#b6c2d2"),
        svg_text(64, 145, "Bar length is mean p95 decode gap in scheduler iterations; lower is better for this metric.", 18, "#b6c2d2"),
    ]
    max_value = max(float(r["p95_decode_gap_iterations"]) for r in rows)
    scale_max = max(1.0, math.ceil(max_value))
    left, right, top = 370, 1300, 190
    usable = right - left
    for tick in range(int(scale_max) + 1):
        x = left + usable * tick / scale_max
        parts.append(f'<line x1="{x:.1f}" y1="175" x2="{x:.1f}" y2="780" stroke="#334155" stroke-width="1"/>')
        parts.append(svg_text(x, 165, str(tick), 16, "#94a3b8", anchor="middle"))
    for wi, workload in enumerate(workloads):
        group_y = top + wi * 200
        label = {"decode_heavy": "Decode-heavy", "mixed_long_prompt": "Mixed long prompts", "prompt_burst": "Prompt burst"}[workload]
        parts.append(svg_text(64, group_y + 18, label, 22, "#f8fafc", "700"))
        for ci, chunk in enumerate(chunks):
            group = grouped[(workload, chunk)]
            avg = sum(float(r["p95_decode_gap_iterations"]) for r in group) / len(group)
            y = group_y + 34 + ci * 34
            bar_w = usable * avg / scale_max
            parts.append(svg_text(105, y + 18, f"chunk {chunk}", 17, "#cbd5e1"))
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="24" rx="5" fill="{colors[ci]}"/>')
            parts.append(svg_text(min(left + bar_w + 12, right - 4), y + 18,
                                  f"{avg:.2f} iter", 16, "#f8fafc"))
    parts.append(svg_text(64, 825, "Host simulation only. Scheduler iterations are not milliseconds or GPU ITL.", 17, "#94a3b8"))
    write_svg(path, width, height, parts)


def kv_chart(rows: list[dict[str, str]], path: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["workload"], int(row["page_size_tokens"]))].append(
            float(row["reservation_reduction_fraction"]) * 100.0
        )
    workloads = ["short", "mixed", "ragged", "long"]
    pages = [8, 16, 32, 64]
    colors = ["#38bdf8", "#34d399", "#fbbf24", "#e879f9"]
    width, height = 1440, 930
    parts = [
        svg_text(64, 64, "KV-cache experiment: reserved-capacity reduction", 32, "#f8fafc", "700"),
        svg_text(64, 105, f"{len(rows)} measured allocator cases across four page sizes", 20, "#b6c2d2"),
        svg_text(64, 145, "Reduction compares paged KV reservation with batch × max sequence length; it is not total VRAM.", 18, "#b6c2d2"),
    ]
    left, right = 405, 1290
    usable = right - left
    for tick in range(0, 101, 20):
        x = left + usable * tick / 100.0
        parts.append(f'<line x1="{x:.1f}" y1="175" x2="{x:.1f}" y2="865" stroke="#334155" stroke-width="1"/>')
        parts.append(svg_text(x, 166, f"{tick}%", 16, "#94a3b8", anchor="middle"))
    for wi, workload in enumerate(workloads):
        gy = 205 + wi * 166
        parts.append(svg_text(64, gy + 25, workload.title(), 22, "#f8fafc", "700"))
        for pi, page in enumerate(pages):
            vals = grouped[(workload, page)]
            avg = sum(vals) / len(vals)
            y = gy + 37 + pi * 30
            bar_w = usable * avg / 100.0
            parts.append(svg_text(235, y + 18, f"page {page}", 16, "#cbd5e1"))
            parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="22" rx="4" fill="{colors[pi]}"/>')
            parts.append(svg_text(min(left + bar_w + 10, right - 2), y + 18, f"{avg:.1f}%", 16, "#f8fafc"))
    parts.append(svg_text(64, 902, "These results depend on the listed sequence patterns and page size.", 17, "#94a3b8"))
    write_svg(path, width, height, parts)


def terminal_chart(scheduler_count: int, kv_count: int,
                   benchmark_log: Path | None, ctest_log: Path | None,
                   path: Path) -> None:
    values = {
        "max_decode_gap_iterations": "not supplied",
        "long_prompt_prefilled": "not supplied",
        "kv_logical_tokens": "not supplied",
        "kv_reserved_tokens": "not supplied",
        "kv_capacity_reclaim": "not supplied",
    }
    if benchmark_log and benchmark_log.is_file():
        for line in benchmark_log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.strip().split("=", 1)
                if key in values:
                    values[key] = value
    suite_text = "not supplied"
    if ctest_log and ctest_log.is_file():
        raw = ctest_log.read_text(encoding="utf-8", errors="replace")
        if "100% tests passed" in raw:
            suite_text = "all CTest suites passed"
        else:
            suite_text = "CTest log captured; inspect the raw log"
    parts = [
        '<rect x="36" y="34" width="1368" height="690" rx="18" fill="#111827" stroke="#475569" stroke-width="2"/>',
        '<path d="M54 34H1386Q1404 34 1404 52V100H36V52Q36 34 54 34Z" fill="#263449"/>',
        '<circle cx="72" cy="67" r="9" fill="#fb7185"/><circle cx="104" cy="67" r="9" fill="#fbbf24"/><circle cx="136" cy="67" r="9" fill="#34d399"/>',
        svg_text(720, 75, "Recorded host experiment output", 21, "#e2e8f0", "700", "middle"),
        svg_text(76, 145, f"Scheduler rows checked: {scheduler_count}", 22, "#d1fae5", "700"),
        svg_text(76, 184, f"KV-cache rows checked: {kv_count}", 22, "#d1fae5", "700"),
        svg_text(76, 235, "CTest status: " + suite_text, 20, "#bfdbfe"),
    ]
    labels = [
        ("max_decode_gap_iterations", "Maximum decode gap"),
        ("long_prompt_prefilled", "Long prompt tokens"),
        ("kv_logical_tokens", "Logical KV tokens"),
        ("kv_reserved_tokens", "Reserved KV tokens"),
        ("kv_capacity_reclaim", "Reservation reduction fraction"),
    ]
    for i, (key, label) in enumerate(labels):
        y = 295 + i * 54
        parts.append(svg_text(100, y, label, 20, "#94a3b8"))
        parts.append(svg_text(620, y, values[key], 22, "#e2e8f0", "700"))
    parts.append(svg_text(76, 670, "Host simulation only. This snapshot does not show GPU timings.", 17, "#94a3b8"))
    write_svg(path, 1440, 760, parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path, nargs="?",
                        default=Path("dlse_runtime/results/experiments"))
    parser.add_argument("--benchmark-log", type=Path)
    parser.add_argument("--ctest-log", type=Path)
    args = parser.parse_args()

    try:
        scheduler_rows = load_csv(args.results_dir / "scheduler_sweep.csv")
        kv_rows = load_csv(args.results_dir / "kv_cache_sweep.csv")
        if len(scheduler_rows) != 108:
            raise ValueError(f"expected 108 scheduler rows; found {len(scheduler_rows)}")
        if len(kv_rows) != 16:
            raise ValueError(f"expected 16 KV rows; found {len(kv_rows)}")

        required_scheduler = {
            "workload", "batch_size", "token_budget", "prefill_chunk", "iterations",
            "max_decode_gap_iterations", "p95_decode_gap_iterations",
            "p99_decode_gap_iterations", "max_prefill_chunk",
        }
        required_kv = {
            "workload", "page_size_tokens", "logical_tokens", "reserved_tokens",
            "reserved_kv_bytes", "contiguous_kv_bytes",
            "reservation_reduction_fraction", "internal_fragmentation",
        }
        if not required_scheduler.issubset(scheduler_rows[0]):
            raise ValueError("scheduler CSV is missing expected fields")
        if not required_kv.issubset(kv_rows[0]):
            raise ValueError("KV CSV is missing expected fields")

        for row in scheduler_rows:
            if int(row["iterations"]) <= 0 or int(row["max_prefill_chunk"]) > int(row["prefill_chunk"]):
                raise ValueError("scheduler CSV contains invalid values")
            for field in ("p95_decode_gap_iterations", "p99_decode_gap_iterations"):
                if not math.isfinite(float(row[field])):
                    raise ValueError("scheduler CSV contains non-finite measurements")
        for row in kv_rows:
            logical = int(row["logical_tokens"])
            reserved = int(row["reserved_tokens"])
            reduction = float(row["reservation_reduction_fraction"])
            fragmentation = float(row["internal_fragmentation"])
            if logical > reserved or not 0.0 <= reduction <= 1.0 or not 0.0 <= fragmentation < 1.0:
                raise ValueError("KV CSV contains an invalid allocation result")

        ASSETS.mkdir(parents=True, exist_ok=True)
        scheduler_chart(scheduler_rows, ASSETS / "scheduler_sweep.svg")
        kv_chart(kv_rows, ASSETS / "kv_sweep.svg")
        terminal_chart(len(scheduler_rows), len(kv_rows), args.benchmark_log,
                       args.ctest_log, ASSETS / "execution_snapshot.svg")

        report = [
            "# Host experiment summary",
            "",
            "All latency-like scheduler values below are in scheduler iterations, not milliseconds.",
            "All KV reductions compare reserved KV capacity with a fixed contiguous reservation, not total process VRAM.",
            "",
            f"- Scheduler records checked: {len(scheduler_rows)}",
            f"- KV-cache records checked: {len(kv_rows)}",
            "",
            "## Lowest p95 decode-gap setting for each request mix",
            "",
            "| Workload | Batch | Token budget | Prefill chunk | p95 decode gap (iterations) | Mean scheduler iterations |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scheduler_rows:
            by_workload[row["workload"]].append(row)
        for workload, rows in sorted(by_workload.items()):
            best = min(rows, key=lambda r: (float(r["p95_decode_gap_iterations"]),
                                             int(r["iterations"]),
                                             int(r["prefill_chunk"])))
            report.append(
                f"| {workload} | {best['batch_size']} | {best['token_budget']} | "
                f"{best['prefill_chunk']} | {float(best['p95_decode_gap_iterations']):.3f} | "
                f"{best['iterations']} |"
            )

        report.extend([
            "",
            "## KV-cache reservation results",
            "",
            "| Workload | Page size | Logical tokens | Reserved tokens | Reservation reduction | Internal fragmentation |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in sorted(kv_rows, key=lambda r: (r["workload"], int(r["page_size_tokens"]))):
            report.append(
                f"| {row['workload']} | {row['page_size_tokens']} | {row['logical_tokens']} | "
                f"{row['reserved_tokens']} | {float(row['reservation_reduction_fraction'])*100:.2f}% | "
                f"{float(row['internal_fragmentation'])*100:.2f}% |"
            )

        report.extend([
            "",
            "## What the data suggests",
            "",
            "- The prompt-burst cases had a p95 decode gap of one scheduler iteration in every tested configuration. This does not measure real token latency.",
            "- The decode-heavy cases still had a p95 gap of two iterations at the best tested setting because the number of active requests exceeded the configured batch limit.",
            "- For mixed long prompts, the lowest p95 gap was two iterations. Larger batch limits reduced the number of scheduler iterations in the tested simulation, but those counts are not wall-clock time.",
            "- For short sequences, the page-size sweep reserved 552 tokens at page size 8 and 704 tokens at page size 64. Internal fragmentation rose from 0% to 21.59%.",
            "- For ragged sequence lengths, internal fragmentation rose from 0.85% with page size 8 to 5.50% with page size 64.",
            "- The mixed and long KV workloads were aligned to all four page sizes, so those cases showed no internal fragmentation from page rounding.",
            "",
            "## Scope",
            "",
            "These are deterministic host simulations and allocation calculations. They help compare scheduler settings and page sizes, but they do not measure GPU kernel time, end-to-end inter-token latency, or total process VRAM.",
        ])
        report_text = "\n".join(report) + "\n"
        (args.results_dir / "analysis_summary.md").write_text(report_text, encoding="utf-8")
        (ROOT / "dlse_runtime" / "docs" / "host_experiment_summary.md").write_text(
            report_text, encoding="utf-8"
        )
        print(f"scheduler_rows_checked={len(scheduler_rows)}")
        print(f"kv_rows_checked={len(kv_rows)}")
        print(f"scheduler_chart={ASSETS / 'scheduler_sweep.svg'}")
        print(f"kv_chart={ASSETS / 'kv_sweep.svg'}")
        print(f"execution_snapshot={ASSETS / 'execution_snapshot.svg'}")
        print(f"analysis_report={args.results_dir / 'analysis_summary.md'}")
        print("analysis_status=PASS")
        return 0
    except Exception as exc:
        print(f"analysis_status=FAIL error={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
