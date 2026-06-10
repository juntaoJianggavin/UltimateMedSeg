#!/usr/bin/env python3
"""Check local config path references across docs and scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK_RE = re.compile(
    r"\[[^\]]*\]\(([^)\s]+\.ya?ml(?:#[^)\s]+)?)(?:\s+\"[^\"]*\")?\)",
    re.I,
)
BACKTICK_RE = re.compile(r"`((?:configs/)[^`\s]+\.ya?ml)`", re.I)
PLAIN_RE = re.compile(r"(?<![`\w])((?:configs/)[^\s\)`\]]+\.ya?ml)", re.I)

SCAN_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".sh"}
SKIP_FRAGMENTS = ("xxx", "{", "*", "YourDataset")


def is_skipped(target: str) -> bool:
    if target.startswith(("http://", "https://", "mailto:")):
        return True
    return any(part in target for part in SKIP_FRAGMENTS)


def resolve_target(source: Path, target: str) -> Path:
    target = target.split("#")[0]
    if target.startswith("configs/"):
        return ROOT / target
    return (source.parent / target).resolve()


def iter_references():
    seen: set[tuple[str, str]] = set()
    for path in sorted(ROOT.rglob("*")):
        if path.suffix not in SCAN_SUFFIXES or ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in (LINK_RE, BACKTICK_RE, PLAIN_RE):
            for match in pattern.finditer(text):
                target = match.group(1)
                if is_skipped(target):
                    continue
                key = (str(path.relative_to(ROOT)), target)
                if key in seen:
                    continue
                seen.add(key)
                line = text[: match.start()].count("\n") + 1
                yield path, line, target


def main() -> int:
    broken = []
    checked = 0
    for source, line, target in iter_references():
        checked += 1
        resolved = resolve_target(source, target)
        if not resolved.exists():
            broken.append(
                (
                    str(source.relative_to(ROOT)),
                    line,
                    target,
                    str(resolved.relative_to(ROOT))
                    if resolved.is_relative_to(ROOT)
                    else str(resolved),
                )
            )

    if broken:
        print(f"Broken config path references: {len(broken)}/{checked}")
        for item in broken:
            print(f"{item[0]}:{item[1]} -> {item[2]}")
        return 1

    print(f"Config path references OK: {checked} checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
