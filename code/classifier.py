from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils import normalize_whitespace, tokenize


@dataclass(frozen=True)
class Classification:
    company: str
    request_type: str
    product_area_hint: str
    risk: str
    needs_escalation: bool
    reason: str


def _contains_any(text: str, phrases: list[str]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def _score_company(text: str) -> dict[str, int]:
    lower = text.lower()
    scores = {"HackerRank": 0, "Claude": 0, "Visa": 0}

    hackerrank_terms = [
        "hackerrank", "assessment", "test", "candidate", "interview", "screen", "challenges",
        "invitation", "invite", "workspace", "seat", "role", "variant", "mock interview",
        "certification", "community", "skills platform", "candidate score", "score",
    ]
    claude_terms = [
        "claude", "anthropic", "conversation", "chat", "workspace", "session", "memory",
        "incognito", "api", "console", "desktop", "cowork", "connectors", "skills", "model",
    ]
    visa_terms = [
        "visa", "card", "cheque", "cheques", "merchant", "credit card", "debit card",
        "fraud", "charge", "chargeback", "refund", "travel", "atm", "gcas",
    ]

    for term in hackerrank_terms:
        if term in lower:
            scores["HackerRank"] += 1
    for term in claude_terms:
        if term in lower:
            scores["Claude"] += 1
    for term in visa_terms:
        if term in lower:
            scores["Visa"] += 1

    return scores


def infer_company(issue: str, subject: str, explicit_company: str | None) -> str:
    if explicit_company and explicit_company.strip() and explicit_company.strip().lower() != "none":
        normalized = explicit_company.strip().lower()
        if normalized.startswith("hackerrank"):
            return "HackerRank"
        if normalized.startswith("claude"):
            return "Claude"
        if normalized.startswith("visa"):
            return "Visa"

    combined = f"{subject}\n{issue}".strip()
    scores = _score_company(combined)
    best_company = max(scores, key=scores.get)
    if scores[best_company] == 0:
        return "Unknown"
    return best_company


def classify_request(issue: str, subject: str, company: str) -> Classification:
    text = normalize_whitespace(f"{subject}\n{issue}")
    lower = text.lower()

    request_type = "product_issue"
    if _contains_any(lower, ["feature request", "would like to request", "please add", "can you add", "enhancement", "support for"]):
        request_type = "feature_request"

    if _contains_any(lower, ["broken", "bug", "error", "failed", "not working", "down", "unable", "can't", "cannot", "issue", "problem", "doesn't", "does not", "inaccessible", "blocker"]):
        request_type = "bug"

    if _contains_any(lower, [
        "actor in iron man", "what is the name of", "what's the name of", "who is the actor", "capital of", "weather",
        "joke", "translate", "poem", "story", "homework", "give me the money", "random", "meaning of life",
    ]):
        request_type = "invalid"

    risk = "low"
    needs_escalation = False
    product_area_hint = "unknown"
    reason_parts: list[str] = []

    high_risk_account_access = _contains_any(lower, ["locked", "lock out", "locked out", "lost access", "cannot access", "can't access", "can't log in", "cannot log in", "login issue", "account access", "seat removed", "admin removed", "workspace access"])
    billing_dispute = _contains_any(lower, ["refund", "chargeback", "dispute", "billing dispute", "my money", "charged twice", "incorrect charge", "payment dispute"])
    fraud_or_compromise = _contains_any(lower, ["fraud", "fraudulent", "suspicious", "unauthorized", "compromised", "stolen", "lost card", "lost cheque", "stolen cheque", "card stolen", "card lost", "phishing"])
    sensitive_data = _contains_any(lower, ["ssn", "social security", "password", "private info", "confidential", "secret key", "api key", "token", "cookie", "credentials"])

    if billing_dispute:
        risk = "high"
        needs_escalation = True
        reason_parts.append("billing dispute")
    if high_risk_account_access:
        risk = "high"
        needs_escalation = True
        reason_parts.append("account access issue")
    if fraud_or_compromise:
        risk = "high"
        reason_parts.append("fraud or compromise language")
    if sensitive_data:
        risk = "high"
        reason_parts.append("sensitive data mention")

    if company == "Unknown" and request_type != "invalid":
        needs_escalation = True
        risk = "high"
        reason_parts.append("company not confidently inferred")

    if request_type == "invalid":
        product_area_hint = "conversation_management"
        if company == "HackerRank":
            product_area_hint = "community"
        elif company == "Claude":
            product_area_hint = "conversation_management"
        elif company == "Visa":
            product_area_hint = "consumer_support"
        return Classification(
            company=company,
            request_type=request_type,
            product_area_hint=product_area_hint,
            risk=risk,
            needs_escalation=False,
            reason="out of scope",
        )

    if company == "Unknown" and request_type == "product_issue":
        if not _contains_any(lower, ["hackerrank", "claude", "visa", "account", "test", "card", "chat", "support", "login", "payment", "candidate", "workspace", "assessment"]):
            request_type = "invalid"
            return Classification(
                company=company,
                request_type=request_type,
                product_area_hint="conversation_management",
                risk="low",
                needs_escalation=False,
                reason="out of scope",
            )

    if company == "HackerRank":
        if _contains_any(lower, ["mock interview", "mock interviews"]):
            product_area_hint = "mock_interviews"
        elif _contains_any(lower, ["test", "assessment", "candidate", "invite", "invitee", "score", "variant", "extra time", "accommodation", "compatibility", "browser", "submission"]):
            product_area_hint = "screen"
        elif _contains_any(lower, ["community", "password", "username", "account", "delete account", "reset password", "profile", "email address"]):
            product_area_hint = "community"
        elif _contains_any(lower, ["billing", "payment", "refund", "invoice", "subscription"]):
            product_area_hint = "billing"
        elif _contains_any(lower, ["integration", "ats", "greenhouse", "lever", "workday", "icims", "ashby", "jobvite"]):
            product_area_hint = "integrations"
        elif _contains_any(lower, ["candidate experience", "onboarding", "resume", "job search"]):
            product_area_hint = "general_help"
        else:
            product_area_hint = "general_help"
    elif company == "Claude":
        if _contains_any(lower, ["api key", "console", "api", "prompt", "rate limit", "usage", "tier"]):
            product_area_hint = "api_and_console"
        elif _contains_any(lower, ["desktop", "mcp", "extension", "enterprise configuration"]):
            product_area_hint = "desktop"
        elif _contains_any(lower, ["privacy", "private info", "incognito", "sensitive", "who can view", "data", "temporary chat", "private"]):
            product_area_hint = "privacy_and_legal"
        elif _contains_any(lower, ["conversation", "delete conversation", "rename conversation", "memory", "share", "unshare"]):
            product_area_hint = "conversation_management"
        elif _contains_any(lower, ["login", "account", "email", "delete my account", "sign in", "sign-in"]):
            product_area_hint = "account_management"
        elif _contains_any(lower, ["search", "web search", "skills", "connectors", "plugins", "files", "artifacts", "project", "projects"]):
            product_area_hint = "features_and_capabilities"
        elif _contains_any(lower, ["usage", "limits", "plan", "billing", "max", "pro"]):
            product_area_hint = "usage_and_limits"
        elif _contains_any(lower, ["troubleshoot", "error", "broken", "incorrect", "misleading", "not working"]):
            product_area_hint = "troubleshooting"
        else:
            product_area_hint = "claude_support"
    elif company == "Visa":
        if _contains_any(lower, ["lost", "stolen", "card", "gcas", "emergency cash", "replacement card"]):
            product_area_hint = "consumer_support"
        elif _contains_any(lower, ["traveller", "cheque", "travel", "lisbon"]):
            product_area_hint = "travel_support"
        elif _contains_any(lower, ["merchant", "accept", "payment issue", "purchase issue", "rules", "regulations"]):
            product_area_hint = "merchant_support"
        elif _contains_any(lower, ["fraud", "compromised", "data breach", "security", "breach"]):
            product_area_hint = "small_business"
        else:
            product_area_hint = "consumer_support"
    else:
        product_area_hint = "unknown"

    if fraud_or_compromise and company == "Visa" and _contains_any(lower, ["lost", "stolen", "card", "cheque", "cheques"]):
        needs_escalation = False
        reason_parts = ["lost or stolen payment instrument with direct contact guidance available in corpus"]
    elif fraud_or_compromise and company == "Visa":
        needs_escalation = True

    if sensitive_data and company == "Claude" and _contains_any(lower, ["who can view", "privacy", "incognito", "data", "sensitive"]):
        needs_escalation = False

    if request_type == "feature_request":
        reason_parts.append("feature request")

    if not reason_parts:
        reason_parts.append("supported by corpus")

    return Classification(
        company=company,
        request_type=request_type,
        product_area_hint=product_area_hint,
        risk=risk,
        needs_escalation=needs_escalation,
        reason="; ".join(reason_parts),
    )
