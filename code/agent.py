from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from classifier import Classification, classify_request, infer_company
from decision import Decision, decide
from retriever import CorpusRetriever
from utils import normalize_whitespace, read_csv_rows, write_csv_rows


OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


@dataclass
class TriageAgent:
    repo_root: Path
    corpus_root: Path

    def __post_init__(self) -> None:
        self.retriever = CorpusRetriever(self.corpus_root)

    def triage_row(self, row: dict[str, str]) -> dict[str, str]:
        issue = normalize_whitespace(row.get("issue", ""))
        subject = normalize_whitespace(row.get("subject", ""))
        explicit_company = normalize_whitespace(row.get("company", ""))
        company = infer_company(issue, subject, explicit_company)
        classification = classify_request(issue=issue, subject=subject, company=company)

        evidence = self.retriever.best_match(
            query=f"{subject} {issue}",
            company=company if company != "Unknown" else None,
            area_hint=classification.product_area_hint,
        )

        decision = decide(
            ticket_text=f"{subject}\n{issue}",
            classification=classification,
            evidence=evidence,
        )
        return {
            "status": decision.status,
            "product_area": decision.product_area,
            "response": decision.response,
            "justification": decision.justification,
            "request_type": decision.request_type,
        }

    def run(self, input_csv: Path, output_csv: Path) -> list[dict[str, str]]:
        rows = read_csv_rows(input_csv)
        outputs = [self.triage_row(row) for row in rows]
        write_csv_rows(output_csv, outputs, OUTPUT_COLUMNS)
        return outputs
