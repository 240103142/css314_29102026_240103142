"""Student-ID-derived, reproducible practicum parameters."""

from __future__ import annotations

import hashlib
import re


def get_student_params(student_id: str) -> dict[str, int | str]:
    """Return the parameters specified by the practicum handout.

    The handout specifies an eight-digit ID. Keeping it as text preserves any
    leading zeroes if the function is reused for another student.
    """
    student_id = str(student_id).strip()
    if not re.fullmatch(r"\d{8}", student_id):
        raise ValueError("student_id must contain exactly 8 digits")

    h = int(hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8], 16)
    return {
        "student_id": student_id,
        "N": 1024 + ((h % 9) * 64),
        "buffer_elements": 16_777_216,
        "seed": h % 100_007,
    }
