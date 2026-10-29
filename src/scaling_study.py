"""Strong/weak scaling study of a five-point NumPy stencil in shared memory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import statistics
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from src.profile_env import collect_host_profile
from src.salt import get_student_params


ROOT = Path(__file__).resolve().parents[1]
_IN = None
_OUT = None
_SHM_IN = None
_SHM_OUT = None


def _attach_shared(in_name: str, out_name: str, n: int) -> None:
    global _IN, _OUT, _SHM_IN, _SHM_OUT
    _SHM_IN = shared_memory.SharedMemory(name=in_name)
    _SHM_OUT = shared_memory.SharedMemory(name=out_name)
    _IN = np.ndarray((n, n), dtype=np.float64, buffer=_SHM_IN.buf)
    _OUT = np.ndarray((n, n), dtype=np.float64, buffer=_SHM_OUT.buf)


def _stencil_chunk(bounds: tuple[int, int]) -> None:
    start, end = bounds
    if end <= start:
        return
    target = _OUT[start:end, 1:-1]
    np.copyto(target, _IN[start:end, 1:-1])
    target += _IN[start - 1 : end - 1, 1:-1]
    target += _IN[start + 1 : end + 1, 1:-1]
    target += _IN[start:end, :-2]
    target += _IN[start:end, 2:]


def stencil_reference(source: np.ndarray) -> np.ndarray:
    out = np.zeros_like(source)
    target = out[1:-1, 1:-1]
    np.copyto(target, source[1:-1, 1:-1])
    target += source[:-2, 1:-1]
    target += source[2:, 1:-1]
    target += source[1:-1, :-2]
    target += source[1:-1, 2:]
    return out


def _chunks(n: int, p: int) -> list[tuple[int, int]]:
    rows = np.array_split(np.arange(1, n - 1), p)
    return [(int(chunk[0]), int(chunk[-1]) + 1) for chunk in rows if chunk.size]


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.round(array, 6).astype("<f8", copy=False).tobytes()).hexdigest()


def _measure_case(n: int, p: int, seed: int, iterations: int, runs: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    source = rng.random((n, n), dtype=np.float64)
    expected = stencil_reference(source)
    shm_in = shared_memory.SharedMemory(create=True, size=source.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=source.nbytes)
    shared_in = np.ndarray(source.shape, dtype=source.dtype, buffer=shm_in.buf)
    shared_out = np.ndarray(source.shape, dtype=np.float64, buffer=shm_out.buf)
    np.copyto(shared_in, source)
    shared_out.fill(0.0)
    bounds = _chunks(n, p)
    ctx = mp.get_context("spawn")
    samples: list[int] = []
    try:
        with ctx.Pool(
            processes=p,
            initializer=_attach_shared,
            initargs=(shm_in.name, shm_out.name, n),
        ) as pool:
            pool.map(_stencil_chunk, bounds)
            # Exclude process startup and first-touch page faults from steady-state samples.
            for _ in range(runs):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    pool.map(_stencil_chunk, bounds)
                samples.append((time.perf_counter_ns() - start) // iterations)
        actual = np.array(shared_out, copy=True)
    finally:
        del shared_in, shared_out
        shm_in.close()
        shm_in.unlink()
        shm_out.close()
        shm_out.unlink()
    median = int(statistics.median(samples))
    max_error = float(np.max(np.abs(expected - actual)))
    return {
        "dimension": n,
        "processes": p,
        "iterations_per_sample": iterations,
        "sample_times_ns_per_iteration": samples,
        "median_time_ns": median,
        "interior_cells": (n - 2) ** 2,
        "flops_per_iteration": 4 * (n - 2) ** 2,
        "validation_max_abs_error": max_error,
        "validation_pass": bool(np.allclose(expected, actual, rtol=0.0, atol=1e-12)),
        "reference_sha256_rounded_6": _digest(expected),
        "parallel_sha256_rounded_6": _digest(actual),
    }


def _serial_fraction(speedup: float, p: int) -> float | None:
    if p <= 1 or speedup <= 0:
        return None
    return (1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p)


def run(student_id: str, iterations: int, runs: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    params = get_student_params(student_id)
    profile = collect_host_profile()
    cores = int(profile["cpu"].get("physical_cores") or mp.cpu_count())  # type: ignore[index,union-attr]
    ps = [p for p in (1, 2, 4, 8) if p <= cores]
    n = int(params["N"])
    strong = [_measure_case(n, p, int(params["seed"]), iterations, runs) for p in ps]
    t1 = int(next(row["median_time_ns"] for row in strong if row["processes"] == 1))
    serial_fractions = []
    for row in strong:
        p, tp = int(row["processes"]), int(row["median_time_ns"])
        speedup = t1 / tp
        row["speedup"] = speedup
        row["parallel_efficiency_percent"] = speedup / p * 100.0
        row["karp_flatt_serial_fraction"] = _serial_fraction(speedup, p)
        if p > 1 and row["karp_flatt_serial_fraction"] is not None:
            serial_fractions.append(float(row["karp_flatt_serial_fraction"]))
    avg_s = max(0.0, statistics.mean(serial_fractions)) if serial_fractions else 0.0

    weak: list[dict[str, object]] = []
    for p in ps:
        n_p = math.ceil(n * math.sqrt(p))
        row = _measure_case(n_p, p, int(params["seed"]), iterations, runs)
        row["work_multiplier"] = (n_p - 2) ** 2 / (n - 2) ** 2
        row["pdf_formula_scaled_speedup"] = (p * (n - 2) ** 2 * t1) / (
            (n_p - 2) ** 2 * int(row["median_time_ns"])
        )
        # Throughput ratio is the Gustafson-style speedup for work scaled with P.
        row["scaled_throughput_speedup"] = (
            ((n_p - 2) ** 2 / int(row["median_time_ns"]))
            / (((n - 2) ** 2) / t1)
        )
        row["gustafson_estimate_from_strong_s"] = p - avg_s * (p - 1)
        weak.append(row)

    area = (n - 2) ** 2
    flops = 4 * area
    ideal_bytes = 16 * area  # one input stream and one output stream
    no_reuse_bytes = 48 * area  # five input reads plus one output write
    ai_ideal = flops / ideal_bytes
    ai_no_reuse = flops / no_reuse_bytes
    stream_rows_path = ROOT / "results" / "stride_results.tsv"
    measured_bandwidth = None
    if stream_rows_path.exists():
        with stream_rows_path.open(encoding="utf-8") as f:
            stride_rows = list(csv.DictReader(f, delimiter="\t"))
        measured_bandwidth = max(float(row["effective_bytes_per_second"]) for row in stride_rows)
    peak_at_measured_bandwidth = ai_ideal * measured_bandwidth if measured_bandwidth else None
    one_strong = next(row for row in strong if int(row["processes"]) == 1)
    observed_flops_per_second = flops / (int(one_strong["median_time_ns"]) / 1e9)
    observed_effective_bandwidth = observed_flops_per_second / ai_ideal
    measured_bound_fraction = (
        observed_effective_bandwidth / measured_bandwidth if measured_bandwidth else None
    )
    roofline = {
        "student_id": student_id,
        "kernel": "five-point 2D stencil: center plus four neighbors, four additions per interior cell",
        "dimension": n,
        "flops_per_interior_cell": 4,
        "ideal_streaming_bytes_per_cell": 16,
        "no_cache_reuse_bytes_per_cell": 48,
        "arithmetic_intensity_ideal_flop_per_byte": ai_ideal,
        "arithmetic_intensity_no_reuse_flop_per_byte": ai_no_reuse,
        "peak_theoretical_memory_bandwidth_bytes_per_second": None,
        "peak_bandwidth_note": "Not exposed by this host OS profile; measured stride throughput is reported as an empirical bound.",
        "measured_peak_stride_bandwidth_bytes_per_second": measured_bandwidth,
        "roofline_flops_per_second_at_measured_stride_bandwidth": peak_at_measured_bandwidth,
        "observed_single_process_stencil_flops_per_second": observed_flops_per_second,
        "observed_effective_bandwidth_bytes_per_second": observed_effective_bandwidth,
        "fraction_of_measured_stride_bandwidth": measured_bound_fraction,
        "cache_sizes_bytes": profile["cpu"].get("cache"),  # type: ignore[index,union-attr]
        "classification": "bandwidth-sensitive by low arithmetic intensity; the stencil did not saturate the measured stride-bandwidth bound",
    }
    return strong + weak, {"student_id": student_id, "strong_scaling": strong, "weak_scaling": weak, "roofline": roofline}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    rows, report = run(args.student_id, args.iterations, args.runs)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tsv = out_dir / "scaling_metrics.tsv"
    with tsv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "scaling_run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "roofline_summary.json").write_text(json.dumps(report["roofline"], indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {tsv}")
    print(f"Wrote {out_dir / 'roofline_summary.json'}")


if __name__ == "__main__":
    main()
