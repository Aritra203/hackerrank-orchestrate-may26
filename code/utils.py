from __future__ import annotations

import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)
HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_front_matter(text: str) -> str:
    return FRONT_MATTER_RE.sub("", text, count=1)


def strip_markup(text: str) -> str:
    cleaned = (text or "").replace("\u00a0", " ")
    cleaned = MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\\", " ")
    return cleaned


def split_sentences(text: str) -> List[str]:
    clean = normalize_whitespace(text.replace("\u00a0", " "))
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+[-*]\s+|\n+", clean)
    return [part.strip() for part in parts if part and part.strip()]


def chunk_text(text: str, max_words: int = 170, overlap: int = 35) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current).strip())
            if overlap and len(current) > 1:
                tail: List[str] = []
                tail_words = 0
                for sentence in reversed(current):
                    words = len(tokenize(sentence))
                    if tail_words + words > overlap and tail:
                        break
                    tail.append(sentence)
                    tail_words += words
                current = list(reversed(tail))
                current_words = sum(len(tokenize(sentence)) for sentence in current)
            else:
                current = []
                current_words = 0

    for sentence in sentences:
        sentence_words = len(tokenize(sentence))
        if current and current_words + sentence_words > max_words:
            flush()
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[dict[str, str]] = []
        for row in reader:
            rows.append({(key or "").strip().lower(): (value or "") for key, value in row.items()})
        return rows


def write_csv_rows(path: Path, rows: Sequence[dict[str, str]], fieldnames: Sequence[str]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def word_counter(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def path_to_company(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "visa" in parts:
        return "Visa"
    if "claude" in parts:
        return "Claude"
    if "hackerrank" in parts:
        return "HackerRank"
    return "Unknown"


def safe_get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def compact_snippet(text: str, max_length: int = 350) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3].rstrip() + "..."


def heading_depth_score(text: str) -> int:
    return len(HEADING_RE.findall(text or ""))
