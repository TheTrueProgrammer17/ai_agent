"""
risk.py — Pure-logic risk assessment based on user claim history.
No LLM or API calls. Returns list of risk flag strings.
"""


def assess_risk(user_history: dict) -> list:
    """
    Assess risk flags based on user history.

    Rules:
      - if past_claim_count > 5 and rejected_claim > 1 → "user_history_risk"
      - if last_90_days_claim_count > 3 → "manual_review_required"
      - if history_flags is not empty and not "none" → "user_history_risk"
      - deduplicate before returning

    Returns [] if user_history is empty.
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

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            deduped.append(f)

    return deduped
