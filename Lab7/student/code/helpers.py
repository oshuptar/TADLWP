"""Shared helpers for Lab7 teacher code."""

from pathlib import Path
from typing import Tuple


def load_text_and_context() -> Tuple[str, str]:
    """Read full training text from ``../input.txt`` (Lab7/teacher/input.txt). Context is a short prefix so generation continues the same text."""
    input_path = Path(__file__).resolve().parent.parent / "input.txt"
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()
    text = " ".join(raw.split()).strip()
    if not text:
        raise ValueError(f"{input_path} is empty or whitespace only.")
    return text
