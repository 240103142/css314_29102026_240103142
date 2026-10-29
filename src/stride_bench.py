"""Measure a NumPy reduction over a large, contiguous float64 buffer."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from src.salt import get_student_params


ROOT = Path(__file__).resolve().parents[1]


def run(elements: int, warmups: int = 5, repeats: int = 5) -> list[dict[str, object]]:
    data = np.ones(elements, dtype=np.float64)
    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 64, 128):
        view = data[::stride]
        for _ in range(warmups):
            float(np.sum(view, dtype=np.float64))
        samples: list[int] = []
        checksum = 0.0
        for _ in range(repeats):
            start = time.perf_counter_ns()
            checksum = float(np.sum(view, dtype=np.float64))
            samples.append(time.perf_counter_ns() - start)
        median_ns = int(statistics.median(samples))
        bytes_touched = int(view.size * data.itemsize)
        rows.append(
            {
                "stride_elements": stride,
                "elements_touched": int(view.size),
                "median_time_ns": median_ns,
                "latency_ns_per_element": median_ns / view.size,
                "effective_bytes_per_second": bytes_touched / (median_ns / 1e9),
                "checksum": checksum,
                "warmups": warmups,
                "recorded_repeats": repeats,
                "kernel": "numpy float64 sum over strided view",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--elements", type=int)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    params = get_student_params(args.student_id)
    rows = run(args.elements or int(params["buffer_elements"]), args.warmups, args.repeats)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    output = out_dir / "stride_results.tsv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "stride_run.json").write_text(
        json.dumps({"student_id": args.student_id, "rows": len(rows)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
