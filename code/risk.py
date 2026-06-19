"""
risk.py — Pure-logic risk assessment based on user claim history.
No LLM or API calls. Returns list of risk flag strings.
"""


def detect_claim_injection(user_claim: str) -> bool:
    """Detect if user_claim contains instruction injection."""
    claim_lower = user_claim.lower()
    injection_patterns = [
        "ignore all previous",
        "ignore previous instructions",
        "skip manual review",
        "approve the claim",
        "approve this claim",
        "mark this",
        "mark the claim",
        "follow it and approve",
        "usko follow karke",
        "should be approved",
        "approve immediately",
        "accept this quickly",
        "keep reopening",
    ]
    return any(p in claim_lower for p in injection_patterns)


def assess_risk(user_history: dict, user_claim: str = "") -> list:
    """
    Assess risk flags based on user history and claim text.

    Rules:
      - if past_claim_count > 5 and rejected_claim > 1 → "user_history_risk"
      - if last_90_days_claim_count > 3 → "manual_review_required"
      - if history_flags is not empty and not "none" → "user_history_risk"
      - if user_claim contains injection patterns → "text_instruction_present",
        "manual_review_required"
      - deduplicate before returning

    Returns [] if user_history is empty and no injection detected.
    """
    if not user_history:
        return []

    flags = []

    # Rule 1: high overall claim count with rejections
    try:
        past_claim_count = int(user_history.get("past_claim_count", 0) or 0)
        rejected_claim = int(user_history.get("rejected_claim", 0) or 0)
        if past_claim_count > 5 and rejected_claim > 1:
            flags.append("user_history_risk")
    except (ValueError, TypeError):
        pass

    # Rule 2: high claim frequency in last 90 days
    try:
        last_90_days = int(user_history.get("last_90_days_claim_count", 0) or 0)
        if last_90_days > 3:
            flags.append("manual_review_required")
    except (ValueError, TypeError):
        pass

    # Rule 3: explicit history flags set
    try:
        history_flags = str(user_history.get("history_flags", "") or "").strip()
        if history_flags and history_flags.lower() != "none":
            flags.append("user_history_risk")
    except (ValueError, TypeError):
        pass

    # Rule 4: prompt injection in claim text
    if user_claim and detect_claim_injection(user_claim):
        flags.append("text_instruction_present")
        flags.append("manual_review_required")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    return deduped
