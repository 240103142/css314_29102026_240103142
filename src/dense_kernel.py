"""Compare i-j-k and i-k-j dense matrix multiplication loop orders."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from src.salt import get_student_params


ROOT = Path(__file__).resolve().parents[1]


def naive_ijk(a: list[float], b: list[float], n: int) -> list[float]:
    c = [0.0] * (n * n)
    for i in range(n):
        ai, ci = i * n, i * n
        for j in range(n):
            value = 0.0
            for k in range(n):
                value += a[ai + k] * b[k * n + j]
            c[ci + j] = value
    return c


def interchanged_ikj(a: list[float], b: list[float], n: int) -> list[float]:
    c = [0.0] * (n * n)
    for i in range(n):
        ai, ci = i * n, i * n
        for k in range(n):
            aik, bk = a[ai + k], k * n
            for j in range(n):
                c[ci + j] += aik * b[bk + j]
    return c


def _digest(values: list[float]) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float64), 6).astype("<f8", copy=False)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def benchmark(student_id: str, n: int, runs: int) -> dict[str, object]:
    params = get_student_params(student_id)
    rng = np.random.default_rng(int(params["seed"]))
    a = rng.random((n, n), dtype=np.float64).ravel().tolist()
    b = rng.random((n, n), dtype=np.float64).ravel().tolist()
    samples: dict[str, list[int]] = {"naive_ijk": [], "interchanged_ikj": []}
    outputs: dict[str, list[float]] = {}
    for name, kernel in (("naive_ijk", naive_ijk), ("interchanged_ikj", interchanged_ikj)):
        for _ in range(runs):
            start = time.perf_counter_ns()
            result = kernel(a, b, n)
            samples[name].append(time.perf_counter_ns() - start)
            outputs[name] = result
    naive = np.asarray(outputs["naive_ijk"], dtype=np.float64)
    optimized = np.asarray(outputs["interchanged_ikj"], dtype=np.float64)
    max_error = float(np.max(np.abs(naive - optimized)))
    tolerance = 1e-12
    medians = {name: int(statistics.median(vals)) for name, vals in samples.items()}
    return {
        "student_id": student_id,
        "dimension": n,
        "runs": runs,
        "timings_ns": {
            "naive_ijk": samples["naive_ijk"],
            "interchanged_ikj": samples["interchanged_ikj"],
            "median_naive_ijk": medians["naive_ijk"],
            "median_interchanged_ikj": medians["interchanged_ikj"],
        },
        "speedup_naive_over_interchanged": medians["naive_ijk"] / medians["interchanged_ikj"],
        "validation": {
            "allclose_atol_1e-12": bool(np.allclose(naive, optimized, rtol=0.0, atol=tolerance)),
            "max_abs_error": max_error,
            "tolerance": tolerance,
            "naive_sha256_rounded_6": _digest(outputs["naive_ijk"]),
            "interchanged_sha256_rounded_6": _digest(outputs["interchanged_ikj"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--dimension", type=int)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    params = get_student_params(args.student_id)
    n = args.dimension or int(params["N"]) // 2
    result = benchmark(args.student_id, n, args.runs)
    out = ROOT / "results" / "loop_interchange.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
