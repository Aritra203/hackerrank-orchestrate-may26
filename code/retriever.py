from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from utils import chunk_text, compact_snippet, normalize_whitespace, path_to_company, read_text, strip_front_matter, strip_markup, tokenize, word_counter


@dataclass(frozen=True)
class CorpusChunk:
    path: Path
    company: str
    product_area: str
    title: str
    text: str
    tokens: tuple[str, ...]

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}"


class CorpusRetriever:
    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root
        self.chunks: list[CorpusChunk] = []
        self.document_frequency: Counter[str] = Counter()
        self._load_corpus()

    def _load_corpus(self) -> None:
        for path in sorted(self.corpus_root.rglob("*.md")):
            if path.name.lower() == "index.md":
                continue
            company = path_to_company(path)
            raw_text = read_text(path)
            body = strip_markup(strip_front_matter(raw_text))
            title = self._extract_title(body, path)
            product_area = self._infer_product_area(path)
            for chunk in self._make_chunks(body, title):
                tokens = tuple(tokenize(f"{title} {chunk}"))
                if not tokens:
                    continue
                corpus_chunk = CorpusChunk(
                    path=path,
                    company=company,
                    product_area=product_area,
                    title=title,
                    text=compact_snippet(chunk, max_length=2000),
                    tokens=tokens,
                )
                self.chunks.append(corpus_chunk)
                for token in set(tokens):
                    self.document_frequency[token] += 1

    def _extract_title(self, body: str, path: Path) -> str:
        match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        if match:
            return normalize_whitespace(match.group(1))
        return path.stem.replace("-", " ").replace("_", " ").title()

    def _infer_product_area(self, path: Path) -> str:
        lower_parts = [part.lower() for part in path.parts]
        if "visa" in lower_parts:
            if "small-business" in lower_parts:
                return "small_business"
            if "consumer" in lower_parts:
                if "travel-support" in lower_parts:
                    return "travel_support"
                return "consumer_support"
            return "visa_support"

        if "claude" in lower_parts:
            if "claude-code" in lower_parts:
                return "claude_code"
            if "claude-desktop" in lower_parts:
                return "desktop"
            if "claude-api-and-console" in lower_parts:
                return "api_and_console"
            if "connectors" in lower_parts:
                return "connectors"
            if "privacy-and-legal" in lower_parts:
                return "privacy_and_legal"
            if "team-and-enterprise-plans" in lower_parts:
                return "team_and_enterprise_plans"
            if "pro-and-max-plans" in lower_parts:
                return "pro_and_max_plans"
            if "mobile-apps" in lower_parts:
                return "mobile_apps"

            if "account-management" in lower_parts:
                return "account_management"
            if "conversation-management" in lower_parts:
                return "conversation_management"
            if "features-and-capabilities" in lower_parts:
                return "features_and_capabilities"
            if "get-started-with-claude" in lower_parts:
                return "get_started_with_claude"
            if "personalization-and-settings" in lower_parts:
                return "personalization_and_settings"
            if "troubleshooting" in lower_parts:
                return "troubleshooting"
            if "usage-and-limits" in lower_parts:
                return "usage_and_limits"
            if "claude" in lower_parts and len(lower_parts) >= 2:
                return "claude"
            return "claude_support"

        if "hackerrank" in lower_parts:
            if "hackerrank_community" in lower_parts:
                if "account-settings" in lower_parts:
                    return "community"
                if "subscriptions-payments-and-billing" in lower_parts:
                    return "billing"
                if "mock-interviews" in lower_parts:
                    return "mock_interviews"
                if "certifications" in lower_parts:
                    return "certifications"
                if "contests" in lower_parts:
                    return "contests"
                if "practice-coding-challenges" in lower_parts:
                    return "practice_coding_challenges"
                if "getting-started" in lower_parts:
                    return "getting_started"
                if "additional-resources" in lower_parts:
                    return "additional_resources"
                return "community"

            if "chakra" in lower_parts:
                return "chakra"
            if "screen" in lower_parts:
                return "screen"
            if "engage" in lower_parts:
                return "engage"
            if "integrations" in lower_parts:
                return "integrations"
            if "skillup" in lower_parts:
                return "skillup"
            if "general-help" in lower_parts:
                if "contact-us" in lower_parts:
                    return "general_support"
                if "important-notifications" in lower_parts:
                    return "notifications"
                if "release-notes" in lower_parts:
                    return "release_notes"
                return "general_help"
            return "hackerrank_support"

        return "general"

    def _make_chunks(self, body: str, title: str) -> List[str]:
        text = normalize_whitespace(body)
        if not text:
            return []

        sections = [segment.strip() for segment in re.split(r"\n\s*\n", body) if segment.strip()]
        chunks: list[str] = []
        for section in sections:
            cleaned = normalize_whitespace(section)
            if not cleaned:
                continue
            if len(tokenize(cleaned)) <= 200:
                chunks.append(cleaned)
                continue
            chunks.extend(chunk_text(cleaned, max_words=160, overlap=30))

        if not chunks:
            chunks = chunk_text(text, max_words=160, overlap=30)

        return chunks

    def _score_chunk(self, query_tokens: Sequence[str], chunk: CorpusChunk, company: Optional[str], area_hint: Optional[str]) -> float:
        if company and chunk.company != company:
            return -1.0

        chunk_counter = Counter(chunk.tokens)
        query_set = list(dict.fromkeys(query_tokens))
        score = 0.0
        coverage = 0

        for token in query_set:
            if token in chunk_counter:
                df = self.document_frequency.get(token, 1)
                idf = math.log((1 + len(self.chunks)) / (1 + df)) + 1.0
                score += (1.0 + math.log(1 + chunk_counter[token])) * idf
                coverage += 1

        if query_set:
            score += 3.0 * (coverage / len(query_set))

        title_tokens = set(tokenize(chunk.title))
        title_overlap = len(title_tokens.intersection(query_set))
        if title_overlap:
            score += 2.0 * title_overlap

        if area_hint and area_hint != "unknown" and area_hint in chunk.product_area:
            score += 4.0

        if company and chunk.company == company:
            score += 1.0

        if any(term in chunk.searchable_text.lower() for term in ["how do i", "how can i", "faq", "frequently asked questions"]):
            score += 0.5

        return score

    def search(self, query: str, company: Optional[str] = None, area_hint: Optional[str] = None, top_k: int = 5) -> List[CorpusChunk]:
        query_tokens = tokenize(query)
        scored_chunks = [
            (self._score_chunk(query_tokens, chunk, company, area_hint), chunk)
            for chunk in self.chunks
        ]
        article_best: dict[Path, tuple[float, CorpusChunk]] = {}
        for score, chunk in scored_chunks:
            if score <= 0:
                continue
            current = article_best.get(chunk.path)
            if current is None or score > current[0]:
                article_best[chunk.path] = (score, chunk)

        ranked_articles = sorted(article_best.values(), key=lambda item: (item[0], len(item[1].text)), reverse=True)
        return [chunk for score, chunk in ranked_articles[:top_k] if score > 0]

    def best_match(self, query: str, company: Optional[str] = None, area_hint: Optional[str] = None) -> Optional[CorpusChunk]:
        results = self.search(query, company=company, area_hint=area_hint, top_k=1)
        return results[0] if results else None

    def group_by_article(self, chunks: Iterable[CorpusChunk]) -> list[CorpusChunk]:
        return list(chunks)
