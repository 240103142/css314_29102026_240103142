"""Measure unpadded, 64-byte padded, hardware-line padded and local counters."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from src.profile_env import collect_host_profile


ROOT = Path(__file__).resolve().parents[1]


def _worker(shared, worker_id: int, iterations: int, index: int, privatized: bool, barrier, lock) -> None:
    barrier.wait()
    if privatized:
        local = 0
        for _ in range(iterations):
            local += 1
        with lock:
            shared[worker_id] = local
    else:
        for _ in range(iterations):
            shared[index] += 1


def _one_run(processes: int, iterations: int, mode: str, line_elements: int) -> tuple[int, int]:
    ctx = mp.get_context("spawn")
    if mode == "unpadded":
        stride = 1
    elif mode == "padded_64":
        stride = 8
    elif mode == "padded_hardware_line":
        stride = max(1, line_elements)
    elif mode == "privatized":
        stride = 1
    else:
        raise ValueError(mode)
    slots = (processes - 1) * stride + 1 if mode != "privatized" else processes
    shared = ctx.RawArray("q", slots)
    barrier = ctx.Barrier(processes + 1)
    lock = ctx.Lock()
    procs = [
        ctx.Process(
            target=_worker,
            args=(shared, worker, iterations, worker * stride, mode == "privatized", barrier, lock),
        )
        for worker in range(processes)
    ]
    for proc in procs:
        proc.start()
    start = time.perf_counter_ns()
    barrier.wait()
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter_ns() - start
    if any(proc.exitcode != 0 for proc in procs):
        raise RuntimeError(f"worker failed: {[proc.exitcode for proc in procs]}")
    total = sum(int(shared[i]) for i in range(slots))
    return elapsed, total


def run(iterations: int, runs: int) -> list[dict[str, object]]:
    profile = collect_host_profile()
    line_bytes = profile["cpu"].get("cache_line_bytes") or 64  # type: ignore[index,union-attr]
    line_elements = max(1, int(line_bytes) // 8)
    rows: list[dict[str, object]] = []
    modes = ("unpadded", "padded_64", "padded_hardware_line", "privatized")
    samples: dict[tuple[str, int], list[int]] = {}
    for mode in modes:
        for p in (1, 2, 4, 8):
            timings = []
            totals = []
            for _ in range(runs):
                elapsed, total = _one_run(p, iterations, mode, line_elements)
                timings.append(elapsed)
                totals.append(total)
            if len(set(totals)) != 1:
                raise RuntimeError(f"inconsistent counter totals for {mode}, P={p}: {totals}")
            samples[(mode, p)] = timings
            rows.append(
                {
                    "mode": mode,
                    "processes": p,
                    "iterations_per_process": iterations,
                    "median_time_ns": int(statistics.median(timings)),
                    "samples_ns": ",".join(map(str, timings)),
                    "speedup": "",
                    "karp_flatt_e": "",
                    "expected_total": p * iterations,
                    "observed_total": totals[0],
                    "count_correct": totals[0] == p * iterations,
                    "padding_bytes": {
                        "unpadded": 8,
                        "padded_64": 64,
                        "padded_hardware_line": int(line_bytes),
                        "privatized": 0,
                    }[mode],
                    "hardware_cache_line_bytes": int(line_bytes),
                }
            )
    for row in rows:
        mode, p = str(row["mode"]), int(row["processes"])
        t1 = int(statistics.median(samples[(mode, 1)]))
        tp = int(row["median_time_ns"])
        speedup = t1 / tp if tp else 0.0
        row["speedup"] = speedup
        row["karp_flatt_e"] = ((p * tp / t1) - 1) / (p - 1) if p > 1 and t1 else ""
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5_000_000)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    rows = run(args.iterations, args.runs)
    out = ROOT / "results" / "false_sharing_karp_flatt.tsv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
