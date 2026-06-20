"""
vlm.py — Groq vision model integration for damage claim analysis.
Includes smart rate-limit backoff, fallback tracking, and per-call delays.
"""

import json
import os
import re
import time

from groq import Groq
from cache import get_cached_result, save_to_cache

# Production-active LLaMA 3.2 Vision Instruct Models
PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "qwen/qwen3.6-27b"
MAX_RETRIES = 3

def _get_system_prompt(claim_object: str) -> str:
    base_prompt = """
You are a damage claim verification specialist.
You analyze images to verify whether the submitted images
support or contradict the user's damage claim.

You MUST respond with ONLY a valid JSON object.
No markdown, no explanation, no backticks.
Start your response with { and end with }.

RULES:
1. Images are the PRIMARY source of truth.
2. Only use allowed values listed in the user message.
3. Be conservative — if unsure, use not_enough_information.
4. If image quality is poor, flag it in risk_flags.
5. supporting_image_ids must only contain IDs of images
   that directly show the claimed damage.

CRITICAL DEFINITIONS:
- supported: Images CLEARLY show the exact damage described.
- contradicted: Images show the object clearly BUT the 
  claimed damage is NOT visible, OR a completely different
  type of damage is shown instead.
- not_enough_information: Cannot evaluate because image
  quality, angle, or visibility prevents analysis.

IMPORTANT: If you can clearly see the claimed object part
but the claimed damage does not exist there, use 
contradicted — not not_enough_information.
"""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    specific_path = os.path.join(prompt_dir, f"{claim_object.lower()}_prompt.txt")
    specific_prompt = ""
    if os.path.exists(specific_path):
        try:
            with open(specific_path, "r", encoding="utf-8") as f:
                specific_prompt = f"\n\nOBJECT-SPECIFIC GUIDANCE:\n{f.read()}"
        except:
            pass
            
    return base_prompt + specific_prompt

FALLBACK_RESULT = {
    "evidence_standard_met": False,
    "evidence_standard_met_reason": "Analysis failed",
    "risk_flags": ["manual_review_required"],
    "issue_type": "unknown",
    "object_part": "unknown",
    "claim_status": "not_enough_information",
    "claim_status_justification": "API rate limit exceeded — could not analyze images. Manual review required.",
    "supporting_image_ids": [],
    "valid_image": False,
    "severity": "unknown",
}

# Counters (module-level, safe for ThreadPoolExecutor with GIL)
_total_api_calls = 0
_total_claims_processed = 0
_total_cache_hits = 0


def _extract_wait_seconds(error_message: str) -> float:
    """Extract wait time from Groq rate limit error message."""
    # Match "Please try again in 10.5s"
    match = re.search(r'Please try again in ([\d.]+)s', str(error_message))
    if match:
        return float(match.group(1)) + 1.5  # add buffer
    # Match "Please try again in 4m51.6s"
    match = re.search(r'Please try again in (\d+)m([\d.]+)s', str(error_message))
    if match:
        return float(match.group(1)) * 60 + float(match.group(2)) + 2.0
    return 12.0  # safe default


def _build_user_message(
    claim_text: str,
    claim_object: str,
    images: list,
    evidence_requirements: list,
    extra_risk_flags: list,
) -> str:
    image_list = "\n".join(f"  - {img['image_id']}" for img in images) or "  (none)"

    req_text = ""
    if evidence_requirements:
        lines = []
        for req in evidence_requirements:
            lines.append("  " + " | ".join(f"{k}: {v}" for k, v in req.items() if k != "claim_object"))
        req_text = "\n".join(lines)
    else:
        req_text = "  No specific requirements found."

    risk_text = ", ".join(extra_risk_flags) if extra_risk_flags else "none"

    # Handle multi-part claims
    multi_part_text = ""
    claim_lower = claim_text.lower()
    multi_indicators = ["both", "two", "first", "second", "and the"]
    if any(ind in claim_lower for ind in multi_indicators):
        multi_part_text = """
MULTI-PART CLAIM DETECTED:
The user is claiming damage to multiple parts.
Review all claimed parts.
For claim_status: if ANY claimed part is supported
by images, use "supported".
For object_part: use the primary/most damaged part.
For supporting_image_ids: list all images that show
any of the claimed damage.
"""

    return f"""## Claim Information
Object type: {claim_object}
User claim: {claim_text}

## Images Provided
{image_list}

## Evidence Requirements for this object type
{req_text}

## Additional Risk Context
{risk_text}
{multi_part_text}
## Your Task
Analyze the images against the claim.
Return ONLY this JSON with these exact keys:

{{
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "short reason",
  "risk_flags": ["flag1"] or [],
  "issue_type": "one of the allowed values",
  "object_part": "one of the allowed values for {claim_object}",
  "claim_status": "supported or contradicted or not_enough_information",
  "claim_status_justification": "concise explanation citing image IDs",
  "supporting_image_ids": ["img_1"] or [],
  "valid_image": true or false,
  "severity": "none or low or medium or high or unknown"
}}

Allowed values:
claim_status: supported, contradicted, not_enough_information
issue_type: dent, scratch, crack, glass_shatter, broken_part,
  missing_part, torn_packaging, crushed_packaging, water_damage,
  stain, none, unknown
Car object_part: front_bumper, rear_bumper, door, hood, windshield,
  side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
Laptop object_part: screen, keyboard, trackpad, hinge, lid, corner,
  port, base, body, unknown
Package object_part: box, package_corner, package_side, seal, label,
  contents, item, unknown
risk_flags: blurry_image, cropped_or_obstructed, low_light_or_glare,
  wrong_angle, wrong_object, wrong_object_part, damage_not_visible,
  claim_mismatch, possible_manipulation, non_original_image,
  text_instruction_present, user_history_risk, manual_review_required
severity: none, low, medium, high, unknown
"""


def _extract_json(text: str) -> dict:
    """Strip markdown fences, find first { and last }, parse json.loads()."""
    text = text.strip()

    # Fast track strip common markdown syntax blocks
    for fence in ["```json", "```"]:
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Locate object borders to neutralize conversational prefix/suffix leaks
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No valid JSON structure found in response content string")

    json_str = text[start: end + 1]
    return json.loads(json_str)


def _call_model(client: Groq, model: str, user_message: str, images: list, claim_object: str = "unknown") -> dict:
    """Make a single API call and return parsed JSON dict."""
    global _total_api_calls

    # Build content blocks: text metadata directions followed by structural images
    content = [{"type": "text", "text": user_message}]
    for img in images:
        if img.get("b64"):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['b64']}"
                },
            })

    time.sleep(0.5)
    _total_api_calls += 1

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _get_system_prompt(claim_object)},
            {"role": "user", "content": content},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content or ""
    if not raw.strip():
        raise ValueError("Empty content token block response from model endpoint")

    return _extract_json(raw)


def _add_confidence_score(result: dict) -> dict:
    cs = result.get("claim_status", "")
    rf = result.get("risk_flags", [])
    
    if isinstance(rf, str):
        rf_list = [f.strip() for f in rf.split(";") if f.strip() and f.strip() != "none"]
    else:
        rf_list = list(rf) if isinstance(rf, list) else []
        
    supp_imgs = result.get("supporting_image_ids", [])
    if isinstance(supp_imgs, str):
        num_supp = len([i for i in supp_imgs.split(";") if i.strip() and i.strip() != "none"])
    else:
        num_supp = len(supp_imgs) if isinstance(supp_imgs, list) else 0
        
    low_flags = {"blurry_image", "low_light_or_glare", "damage_not_visible"}
    
    if cs == "not_enough_information" or any(f in low_flags for f in rf_list):
        confidence = 0.3
    elif cs in ["supported", "contradicted"] and num_supp >= 2:
        confidence = 0.95
    elif cs in ["supported", "contradicted"]:
        confidence = 0.8
    else:
        confidence = 0.6
        
    result["confidence_score"] = confidence
    
    if confidence < 0.5:
        if "manual_review_required" not in rf_list:
            rf_list.append("manual_review_required")
            result["risk_flags"] = rf_list
            
    return result


def analyze_claim(
    claim_text: str,
    claim_object: str,
    images: list,
    evidence_requirements: list,
    extra_risk_flags: list,
) -> dict:
    """
    Analyze a claim using the Groq vision model.

    Retry logic:
    - 1.5s delay at start of every call
    - Try PRIMARY_MODEL up to MAX_RETRIES times
    - On 429, extract exact wait time from error and sleep accordingly
    - After MAX_RETRIES on primary: try FALLBACK_MODEL once
    - If all fail: return FALLBACK_RESULT with manual_review_required flag
    """
    global _total_claims_processed, _total_cache_hits

    # Check cache first
    cached = get_cached_result(claim_object, claim_text, images)
    if cached:
        _total_cache_hits += 1
        _total_claims_processed += 1
        return cached

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("[vlm] ERROR: GROQ_API_KEY environment variable not set.")
        _total_claims_processed += 1
        return dict(FALLBACK_RESULT)

    client = Groq(api_key=api_key)
    user_message = _build_user_message(
        claim_text, claim_object, images, evidence_requirements, extra_risk_flags
    )

    # Delay before first call to pace requests
    time.sleep(1.5)

    # Loop iterations on primary model instance
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _call_model(client, PRIMARY_MODEL, user_message, images, claim_object)
            result = _add_confidence_score(result)
            save_to_cache(claim_object, claim_text, images, result)
            _total_claims_processed += 1
            return result
        except Exception as e:
            err_str = str(e)
            print(f"[vlm] Primary model attempt {attempt}/{MAX_RETRIES} failed: {err_str}")
            if attempt < MAX_RETRIES:
                # Smart backoff: extract exact wait from 429 message
                wait = _extract_wait_seconds(err_str)
                print(f"  [vlm] Rate limited. Waiting {wait:.1f}s...")
                time.sleep(wait)

    # Attempt execution fallback model sequence
    try:
        print(f"[vlm] Trying fallback model: {FALLBACK_MODEL}")
        # Extra delay before fallback attempt
        time.sleep(1.5)
        result = _call_model(client, FALLBACK_MODEL, user_message, images, claim_object)
        result = _add_confidence_score(result)
        save_to_cache(claim_object, claim_text, images, result)
        _total_claims_processed += 1
        return result
    except Exception as e:
        err_str = str(e)
        print(f"[vlm] Fallback model also failed: {err_str}")
        wait = _extract_wait_seconds(err_str)
        print(f"  [vlm] Rate limited. Waiting {wait:.1f}s...")
        time.sleep(wait)

    _total_claims_processed += 1
    return dict(FALLBACK_RESULT)


def print_stats():
    """Print total API calls and claims processed."""
    print(f"\n[vlm] Total API calls made: {_total_api_calls}")
    print(f"[vlm] Total claims processed: {_total_claims_processed}")
    print(f"[vlm] Cache hits: {_total_cache_hits}")