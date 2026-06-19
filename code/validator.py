"""
validator.py — Pure-logic output validation and value fixing.
No LLM or API calls. Ensures all output values are within allowed sets.
"""

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPE = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part",
    "damage_not_visible", "claim_mismatch", "possible_manipulation",
    "non_original_image", "text_instruction_present",
    "user_history_risk", "manual_review_required",
}

OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    },
}

CORRECTION_COUNTS = {
    "wrong_object_corrected": 0,
    "issue_type_corrected": 0,
    "injection_corrected": 0,
    "severity_corrected": 0,
    "evidence_corrected": 0,
}


def _to_bool_str(value) -> str:
    """Normalise any value to lowercase 'true' or 'false' string."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if value.strip().lower() == "true":
            return "true"
        if value.strip().lower() == "false":
            return "false"
    return "false"


def validate_and_fix(result: dict, claim_object: str) -> dict:
    """
    Takes raw VLM output dict, fixes any invalid values in-place.
    Returns fixed dict.
    """
    fixed = dict(result)

    # --- claim_status ---
    cs = str(fixed.get("claim_status", "")).strip().lower()
    if cs not in ALLOWED_CLAIM_STATUS:
        fixed["claim_status"] = "not_enough_information"
    else:
        fixed["claim_status"] = cs

    # --- issue_type ---
    it = str(fixed.get("issue_type", "")).strip().lower()
    
    # Normalization mapping
    norm_map = {
        "fracture": "crack",
        "split": "crack",
        "broken hinge": "broken_part",
        "detached": "broken_part",
        "tear": "torn_packaging",
        "wet": "water_damage",
        "water stained": "water_damage",
    }
    if it in norm_map:
        it = norm_map[it]
        
    if it not in ALLOWED_ISSUE_TYPE:
        fixed["issue_type"] = "unknown"
    else:
        fixed["issue_type"] = it

    # --- severity ---
    sev = str(fixed.get("severity", "")).strip().lower()
    if sev not in ALLOWED_SEVERITY:
        fixed["severity"] = "unknown"
    else:
        fixed["severity"] = sev

    # --- object_part ---
    obj_key = str(claim_object).strip().lower()
    allowed_parts = OBJECT_PARTS.get(obj_key, set())
    op = str(fixed.get("object_part", "")).strip().lower()
    if not allowed_parts or op not in allowed_parts:
        fixed["object_part"] = "unknown"
    else:
        fixed["object_part"] = op

    # --- risk_flags ---
    rf = fixed.get("risk_flags", None)
    if isinstance(rf, list):
        filtered = [f.strip().lower() for f in rf if f.strip().lower() in ALLOWED_RISK_FLAGS]
        fixed["risk_flags"] = ";".join(filtered) if filtered else "none"
    elif isinstance(rf, str):
        if not rf.strip():
            fixed["risk_flags"] = "none"
        else:
            parts = [f.strip().lower() for f in rf.split(";") if f.strip().lower() in ALLOWED_RISK_FLAGS]
            fixed["risk_flags"] = ";".join(parts) if parts else "none"
    else:
        fixed["risk_flags"] = "none"

    # --- supporting_image_ids ---
    sid = fixed.get("supporting_image_ids", None)
    if isinstance(sid, list):
        joined = ";".join(str(s).strip() for s in sid if str(s).strip())
        fixed["supporting_image_ids"] = joined if joined else "none"
    elif isinstance(sid, str):
        fixed["supporting_image_ids"] = sid.strip() if sid.strip() else "none"
    else:
        fixed["supporting_image_ids"] = "none"

    # --- evidence_standard_met ---
    fixed["evidence_standard_met"] = _to_bool_str(fixed.get("evidence_standard_met"))

    # --- valid_image ---
    fixed["valid_image"] = _to_bool_str(fixed.get("valid_image"))

    # --- POST-PROCESSING CONSISTENCY RULES ---
    risk_flags_list = [
        f.strip() 
        for f in fixed.get("risk_flags", "none").split(";") 
        if f.strip() and f.strip() != "none"
    ]

    # Rule 1: wrong_object -> contradicted
    if (
        "wrong_object" in risk_flags_list
        and fixed.get("claim_status") == "not_enough_information"
    ):
        fixed["claim_status"] = "contradicted"
        fixed["claim_status_justification"] = (
            fixed.get("claim_status_justification", "") +
            " [AUTO-CORRECTED: wrong object detected, claim contradicted.]"
        )
        CORRECTION_COUNTS["wrong_object_corrected"] += 1

    # Rule 2: known issue_type + NEI -> supported
    if (
        fixed.get("claim_status") == "not_enough_information"
        and fixed.get("issue_type") not in ["unknown", "none"]
        and "damage_not_visible" not in risk_flags_list
        and "wrong_object" not in risk_flags_list
        and "wrong_object_part" not in risk_flags_list
    ):
        fixed["claim_status"] = "supported"
        fixed["claim_status_justification"] = (
            fixed.get("claim_status_justification", "") +
            " [AUTO-CORRECTED: damage type identified, status upgraded to supported.]"
        )
        # Also fix severity if it is unknown
        if fixed.get("severity") == "unknown":
            fixed["severity"] = "low"
        CORRECTION_COUNTS["issue_type_corrected"] += 1

    # Rule 3: text_instruction_present + NEI -> contradicted
    if (
        "text_instruction_present" in risk_flags_list
        and fixed.get("claim_status") == "not_enough_information"
    ):
        fixed["claim_status"] = "contradicted"
        fixed["claim_status_justification"] = (
            fixed.get("claim_status_justification", "") +
            " [AUTO-CORRECTED: instruction injection detected.]"
        )
        CORRECTION_COUNTS["injection_corrected"] += 1

    # Rule 4: severity unknown + supported/contradicted -> fix
    if (
        fixed.get("claim_status") in ["supported", "contradicted"]
        and fixed.get("severity") == "unknown"
    ):
        if fixed.get("claim_status") == "contradicted":
            fixed["severity"] = "none"
        else:
            fixed["severity"] = "low"
        CORRECTION_COUNTS["severity_corrected"] += 1

    # Rule 5: evidence_standard_met + valid_image + NEI fix
    if (
        fixed.get("claim_status") == "not_enough_information"
        and fixed.get("evidence_standard_met") == "true"
        and fixed.get("valid_image") == "true"
        and fixed.get("issue_type") not in ["unknown", "none"]
    ):
        fixed["claim_status"] = "supported"
        if fixed.get("severity") == "unknown":
            fixed["severity"] = "low"
        CORRECTION_COUNTS["evidence_corrected"] += 1

    return fixed
