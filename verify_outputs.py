"""Validate generated benchmark outputs and emit a salted SHA-256 receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.salt import get_student_params


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REQUIRED_FILES = (
    "host_profile.json",
    "salted_params.json",
    "stride_results.tsv",
    "loop_interchange.json",
    "false_sharing_karp_flatt.tsv",
    "scaling_metrics.tsv",
    "roofline_summary.json",
)


def _read_json(name: str) -> dict[str, object]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def verify(student_id: str) -> tuple[list[str], str]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (RESULTS / name).is_file()]
    if missing:
        return [f"Missing result files: {', '.join(missing)}"], ""

    expected_params = get_student_params(student_id)
    actual_params = _read_json("salted_params.json")
    if actual_params != expected_params:
        errors.append("salted_params.json does not match the requested student ID")
    host = _read_json("host_profile.json")
    if not host.get("cpu") or not host.get("kernel_version"):
        errors.append("host_profile.json is missing CPU or kernel information")

    with (RESULTS / "stride_results.tsv").open(encoding="utf-8") as f:
        stride = list(csv.DictReader(f, delimiter="\t"))
    if [int(row["stride_elements"]) for row in stride] != [1, 2, 4, 8, 16, 32, 64, 128]:
        errors.append("stride_results.tsv does not contain the required stride set")
    if any(not float(row["checksum"]) > 0 for row in stride):
        errors.append("stride benchmark contains an invalid reduction checksum")

    loop = _read_json("loop_interchange.json")
    if not loop.get("validation", {}).get("allclose_atol_1e-12", False):  # type: ignore[union-attr]
        errors.append("loop interchange results do not match within 1e-12")

    with (RESULTS / "false_sharing_karp_flatt.tsv").open(encoding="utf-8") as f:
        sharing = list(csv.DictReader(f, delimiter="\t"))
    required_modes = {"unpadded", "padded_64", "padded_hardware_line", "privatized"}
    required_ps = {1, 2, 4, 8}
    seen = {(row["mode"], int(row["processes"])) for row in sharing}
    if seen != {(mode, p) for mode in required_modes for p in required_ps}:
        errors.append("false-sharing table is missing one or more mode/process combinations")
    for row in sharing:
        if row["count_correct"].lower() != "true":
            errors.append(f"counter validation failed for {row['mode']} at P={row['processes']}")
            break

    with (RESULTS / "scaling_metrics.tsv").open(encoding="utf-8") as f:
        scaling = list(csv.DictReader(f, delimiter="\t"))
    if not scaling or any(row.get("validation_pass", "").lower() != "true" for row in scaling):
        errors.append("one or more scaling stencil checksums failed")

    if not _read_json("roofline_summary.json").get("arithmetic_intensity_ideal_flop_per_byte"):
        errors.append("roofline summary is missing arithmetic intensity")

    if errors:
        return errors, ""

    digest = hashlib.sha256()
    digest.update(student_id.encode("utf-8"))
    for name in REQUIRED_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((RESULTS / name).read_bytes())
    receipt = {
        "student_id": student_id,
        "algorithm": "SHA-256",
        "integrity_digest": digest.hexdigest(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checked_files": list(REQUIRED_FILES),
        "note": "Digest for integrity checking; not a public-key digital signature.",
    }
    receipt_path = RESULTS / "verification_receipt.sig"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return [], receipt["integrity_digest"]  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", default=None)
    args = parser.parse_args()
    if args.student_id is None:
        try:
            args.student_id = str(_read_json("salted_params.json")["student_id"])
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot determine student ID: {exc}") from exc
    errors, digest = verify(args.student_id)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("[PASS] Milestone 0: Host topology and salted parameters validated.")
    print("[PASS] Milestone 1: Stride and loop-interchange results validated.")
    print("[PASS] Milestone 2: Counter totals and Karp-Flatt rows validated.")
    print("[PASS] Milestone 3: Strong/weak scaling stencil results validated.")
    print(f"[PASS] Milestone 4: SHA-256 receipt emitted ({digest[:16]}...).")


if __name__ == "__main__":
    main()
