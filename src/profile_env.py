"""Collect host topology and write the salted practicum parameters."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.salt import get_student_params


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _sysctl_all() -> dict[str, str]:
    if platform.system() != "Darwin":
        return {}
    try:
        result = subprocess.run(
            ["sysctl", "-a"], capture_output=True, text=True, check=False
        )
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            values[key] = value
    return values


def _sysctl(name: str, all_values: dict[str, str]) -> str | None:
    if name in all_values:
        return all_values[name]
    try:
        result = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _as_int(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", value.replace(",", ""))
    return int(match.group()) if match else None


def _linux_cache() -> dict[str, int | None]:
    root = Path("/sys/devices/system/cpu/cpu0/cache")
    result: dict[str, int | None] = {"L1_data_bytes": None, "L2_bytes": None, "L3_bytes": None}
    if not root.exists():
        return result
    for index in root.glob("index*"):
        try:
            level = (index / "level").read_text().strip()
            kind = (index / "type").read_text().strip()
            size_text = (index / "size").read_text().strip().upper()
        except OSError:
            continue
        match = re.fullmatch(r"(\d+)([KMG])?", size_text)
        if not match:
            continue
        size = int(match.group(1)) * {None: 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
        if level == "1" and kind == "Data":
            result["L1_data_bytes"] = size
        elif level == "2":
            result["L2_bytes"] = size
        elif level == "3":
            result["L3_bytes"] = size
    return result


def collect_host_profile() -> dict[str, object]:
    system = platform.system()
    all_values = _sysctl_all()
    cpu_model = (
        _sysctl("machdep.cpu.brand_string", all_values)
        if system == "Darwin"
        else platform.processor() or None
    )
    cpu_model = cpu_model or platform.processor() or platform.machine()

    physical = None
    logical = os.cpu_count() or multiprocessing.cpu_count()
    cache_line = None
    cache: dict[str, object]
    perf_levels: list[dict[str, object]] = []
    base_frequency_hz = None
    frequency_source = "not exposed by the operating system"

    if system == "Darwin":
        physical = _as_int(_sysctl("hw.physicalcpu", all_values))
        logical = _as_int(_sysctl("hw.logicalcpu", all_values)) or logical
        cache_line = _as_int(_sysctl("hw.cachelinesize", all_values))
        cache = {
            "L1_data_bytes_sysctl": _as_int(_sysctl("hw.l1dcachesize", all_values)),
            "L2_bytes_sysctl": _as_int(_sysctl("hw.l2cachesize", all_values)),
            "L3_bytes_sysctl": _as_int(_sysctl("hw.l3cachesize", all_values)),
            "sharing_note": "Apple Silicon has per-core/per-cluster caches; see performance_levels.",
        }
        for level in range(4):
            count = _as_int(_sysctl(f"hw.perflevel{level}.physicalcpu", all_values))
            if not count:
                continue
            perf_levels.append(
                {
                    "name": _sysctl(f"hw.perflevel{level}.name", all_values),
                    "physical_cores": count,
                    "logical_cores": _as_int(_sysctl(f"hw.perflevel{level}.logicalcpu", all_values)),
                    "L1_data_bytes": _as_int(_sysctl(f"hw.perflevel{level}.l1dcachesize", all_values)),
                    "L2_bytes": _as_int(_sysctl(f"hw.perflevel{level}.l2cachesize", all_values)),
                }
            )
        base_frequency_hz = _as_int(_sysctl("hw.cpufrequency", all_values))
        if base_frequency_hz:
            frequency_source = "sysctl hw.cpufrequency"
    elif system == "Linux":
        try:
            import psutil  # type: ignore[import-not-found]

            physical = psutil.cpu_count(logical=False)
            freq = psutil.cpu_freq()
            if freq and freq.min:
                base_frequency_hz = int(freq.min * 1_000_000)
                frequency_source = "psutil reported minimum frequency"
        except ImportError:
            pass
        cache = _linux_cache()
        try:
            line_file = Path("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
            cache_line = int(line_file.read_text().strip())
        except (OSError, ValueError):
            pass
    else:
        cache = {"L1_data_bytes": None, "L2_bytes": None, "L3_bytes": None}

    if physical is None:
        try:
            import psutil  # type: ignore[import-not-found]

            physical = psutil.cpu_count(logical=False)
        except ImportError:
            physical = None

    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "kernel_version": platform.release(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "executable_name": Path(sys.executable).name,
        },
        "cpu": {
            "model": cpu_model,
            "architecture": platform.machine(),
            "microarchitecture": "Apple M1" if "Apple M1" in cpu_model else None,
            "base_frequency_hz": base_frequency_hz,
            "base_frequency_source": frequency_source,
            "physical_cores": physical,
            "logical_threads": logical,
            "smt_detected": bool(physical and logical and logical > physical),
            "cache_line_bytes": cache_line,
            "cache": cache,
            "performance_levels": perf_levels,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "host_profile.json").write_text(
        json.dumps(collect_host_profile(), indent=2) + "\n", encoding="utf-8"
    )
    (RESULTS / "salted_params.json").write_text(
        json.dumps(get_student_params(args.student_id), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {RESULTS / 'host_profile.json'}")
    print(f"Wrote {RESULTS / 'salted_params.json'}")


if __name__ == "__main__":
    main()
