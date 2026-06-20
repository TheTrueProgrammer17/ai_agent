"""
cache.py — File-based caching for VLM responses.
Generates a cache key based on claim_object, user_claim, and image hashes.
"""

import hashlib
import json
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

def _get_cache_key(claim_object: str, user_claim: str, images: list) -> str:
    key_components = [str(claim_object).strip().lower(), str(user_claim).strip().lower()]
    for img in sorted(images, key=lambda x: x.get("image_id", "")):
        if img.get("b64"):
            img_hash = hashlib.sha256(img["b64"].encode("utf-8")).hexdigest()
            key_components.append(img_hash)
    
    raw_key = "|".join(key_components)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def get_cached_result(claim_object: str, user_claim: str, images: list) -> dict:
    if not os.path.exists(CACHE_DIR):
        return None
    key = _get_cache_key(claim_object, user_claim, images)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_to_cache(claim_object: str, user_claim: str, images: list, result: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _get_cache_key(claim_object, user_claim, images)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f)
    except Exception:
        pass
