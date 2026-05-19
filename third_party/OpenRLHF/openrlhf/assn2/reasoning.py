from __future__ import annotations

import re

_FINAL_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


def build_prompt(question: str) -> str:
    """A simple, consistent reasoning prompt used by this assignment."""
    return (
        "You are a careful reasoner. Solve step-by-step. "
        "At the end, output the final answer on its own line as: '#### <answer>'\n\n"
        f"Question: {question}\n\nAnswer:\n"
    )


def extract_final(text: str) -> str | None:
    """Extract the '#### <answer>' final answer token from a model output."""
    m = None
    for m in _FINAL_RE.finditer(text):
        pass
    return m.group(1) if m else None
