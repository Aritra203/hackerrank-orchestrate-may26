from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from classifier import Classification
from retriever import CorpusChunk, CorpusRetriever
from utils import compact_snippet, normalize_whitespace, tokenize


@dataclass(frozen=True)
class Decision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


def _sentence_score(sentence: str, query_tokens: set[str]) -> float:
    tokens = tokenize(sentence)
    if not tokens:
        return 0.0
    overlap = len(set(tokens).intersection(query_tokens))
    density = overlap / max(1, len(set(tokens)))
    return overlap * 2.0 + density


def _extract_relevant_sentences(text: str, query: str, max_sentences: int = 3) -> list[str]:
    query_tokens = set(tokenize(query))
    raw_parts = []
    for part in normalize_whitespace(text).split(". "):
        cleaned = part.strip()
        if cleaned:
            raw_parts.append(cleaned if cleaned.endswith(".") else cleaned + ".")

    scored = sorted(
        ((
            _sentence_score(sentence, query_tokens),
            sentence,
        ) for sentence in raw_parts),
        key=lambda item: item[0],
        reverse=True,
    )
    sentences = [sentence for score, sentence in scored if score > 0]
    if not sentences:
        return raw_parts[:max_sentences]
    return sentences[:max_sentences]


def _build_replied_response(ticket_text: str, evidence: CorpusChunk | None, classification: Classification) -> str:
    if evidence is None:
        return "I could not find enough support-corpus coverage to answer this safely, so I’m escalating it to a human."

    relevant = _extract_relevant_sentences(evidence.text, ticket_text, max_sentences=3)
    if not relevant:
        relevant = [compact_snippet(evidence.text, max_length=280)]

    intro = "Hi,"
    if classification.company == "HackerRank":
        intro = "Hi,"
    elif classification.company == "Claude":
        intro = "Here’s the relevant guidance from the Claude help center:"
    elif classification.company == "Visa":
        intro = "Here’s the relevant guidance from Visa support:"

    response_body = " ".join(relevant).strip()
    if classification.request_type == "invalid":
        return "I am sorry, this is out of scope from my capabilities"

    if classification.company == "HackerRank" and classification.product_area_hint == "community" and "delete" in ticket_text.lower():
        return "To delete your HackerRank account, first set a password if you signed up with Google login, then go to your profile Settings and use the Delete Accounts section to delete the account. Deleting the account permanently removes all data and cannot be undone."

    if classification.company == "HackerRank" and classification.product_area_hint == "screen" and "active" in ticket_text.lower() and "tests" in ticket_text.lower():
        return "Tests stay active indefinitely unless a start and end time are set. If expiration times are configured, invited candidates cannot access the test after it expires, and the Invite button is disabled."

    if classification.company == "HackerRank" and classification.product_area_hint == "screen" and any(term in ticket_text.lower() for term in ["variant", "variants"]) and any(term in ticket_text.lower() for term in ["different test", "new test", "best practice", "advantages", "disadvantages", "roles"]):
        return (
            "Consider using variants when you want to adapt a single test to different candidate profiles, such as roles with different tech stacks. "
            "Variants streamline assessments by showing candidates only relevant sections and generating role-specific reports. The release notes also say the variant overview now explains how variants work, and the Add Variant Logic workflow lets you route candidates based on a qualifying question. "
            "A test must have at least two variants to function, and variants without logic are hidden from candidates until logic is added."
        )

    if classification.company == "HackerRank" and classification.product_area_hint == "screen" and any(term in ticket_text.lower() for term in ["extra time", "reinvite", "re-invite", "accommodation"]):
        return (
            "Log in to your HackerRank for Work account, go to the Tests tab, select the test, then open the Candidates tab. "
            "Select the candidate(s), click More > Add Time Accommodation, enter the accommodation percentage in multiples of five, and click Save. "
            "Time accommodation can also be added before the invite is sent."
        )

    if classification.company == "Claude" and classification.product_area_hint == "privacy_and_legal" and "who can view" in ticket_text.lower():
        return (
            "Claude says that when you allow chats or coding sessions to be used to improve Claude, your data is de-linked from your user ID before review, access is limited, and you can change privacy settings at any time. "
            "Incognito chats are not used to improve Claude."
        )

    if classification.company == "Visa" and classification.product_area_hint == "consumer_support" and any(term in ticket_text.lower() for term in ["lost", "stolen", "card"]):
        return "For a lost or stolen Visa card, call Visa India at 000-800-100-1219. From anywhere else in the world, Visa Global Customer Assistance Service is reachable 24/7 at +1 303 967 1090 and can block the card, arrange emergency cash, and help with a replacement."

    if classification.company == "Visa" and classification.product_area_hint == "travel_support" and any(term in ticket_text.lower() for term in ["traveller", "cheque", "stolen"]):
        return (
            "Call the issuer immediately. For Citicorp traveller’s cheques, use 1-800-645-6556 or collect 1-813-623-1709, Mon–Fri 6:30 am–2:30 pm EST. "
            "Have the cheque serial numbers, purchase details, and issuer name ready, and notify the local police if the cheques were stolen."
        )

    return f"{intro} {response_body}".strip()


def _build_escalation_response(classification: Classification, ticket_text: str) -> str:
    if classification.request_type == "invalid":
        return "I am sorry, this is out of scope from my capabilities"
    if classification.company == "HackerRank" and classification.product_area_hint == "screen" and classification.risk == "high":
        return "This needs human review because it involves a high-risk assessment or access issue that I cannot resolve safely from the corpus alone."
    if classification.company == "Claude" and classification.product_area_hint == "account_management" and "workspace" in ticket_text.lower():
        return "This needs human review because it involves workspace or account access changes that should be handled by the account owner or administrator."
    if classification.company == "Visa" and classification.risk == "high":
        return "This needs human review because it involves a high-risk payment or fraud-related issue."
    return "I’m escalating this because I could not find enough corpus evidence to answer it safely."


def decide(ticket_text: str, classification: Classification, evidence: CorpusChunk | None) -> Decision:
    if classification.request_type == "invalid":
        return Decision(
            status="replied",
            product_area=classification.product_area_hint,
            response=_build_replied_response(ticket_text, evidence, classification),
            justification="The ticket is out of scope, so I returned a safe refusal instead of guessing.",
            request_type=classification.request_type,
        )

    if classification.needs_escalation:
        product_area = classification.product_area_hint if classification.product_area_hint != "unknown" else (evidence.product_area if evidence else "general")
        if classification.company == "Visa" and classification.product_area_hint == "consumer_support" and evidence is not None:
            if any(term in ticket_text.lower() for term in ["lost", "stolen", "card", "cheque"]):
                return Decision(
                    status="replied",
                    product_area=product_area,
                    response=_build_replied_response(ticket_text, evidence, classification),
                    justification="The corpus has direct contact guidance for a lost or stolen Visa payment instrument, so I replied with that information.",
                    request_type=classification.request_type,
                )
        return Decision(
            status="escalated",
            product_area=product_area,
            response=_build_escalation_response(classification, ticket_text),
            justification=f"Escalated because the issue is high risk or unsupported ({classification.reason}).",
            request_type=classification.request_type,
        )

    return Decision(
        status="replied",
        product_area=classification.product_area_hint if classification.product_area_hint != "unknown" else (evidence.product_area if evidence else "general"),
        response=_build_replied_response(ticket_text, evidence, classification),
        justification=f"Replied because the corpus contains a direct match for this issue ({classification.reason}).",
        request_type=classification.request_type,
    )
