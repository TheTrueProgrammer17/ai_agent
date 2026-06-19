"""
loader.py — Data loading and enrichment for the damage claim pipeline.
"""

import csv
import os


def load_claims(path: str) -> list:
    """Read claims CSV, return each row as a dict."""
    claims = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                claims.append(dict(row))
    except FileNotFoundError:
        print(f"[loader] ERROR: Claims file not found: {path}")
    except Exception as e:
        print(f"[loader] ERROR reading claims: {e}")
    return claims


def load_user_history(path: str) -> dict:
    """Read user_history.csv, index by user_id for fast lookup."""
    history = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id", "").strip()
                if uid:
                    history[uid] = dict(row)
    except FileNotFoundError:
        print(f"[loader] WARNING: User history file not found: {path}")
    except Exception as e:
        print(f"[loader] ERROR reading user history: {e}")
    return history


def load_evidence_requirements(path: str) -> list:
    """Read evidence_requirements.csv, return list of dicts."""
    requirements = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                requirements.append(dict(row))
    except FileNotFoundError:
        print(f"[loader] WARNING: Evidence requirements file not found: {path}")
    except Exception as e:
        print(f"[loader] ERROR reading evidence requirements: {e}")
    return requirements


def enrich_claims(claims: list, user_history: dict, evidence_requirements: list) -> list:
    """
    For each claim, attach:
      - user_history dict for that user_id (or empty dict if not found)
      - evidence_requirements list filtered for that claim_object
    Return enriched list ready for processing.
    """
    enriched = []
    for claim in claims:
        uid = claim.get("user_id", "").strip()
        claim_object = claim.get("claim_object", "").strip().lower()

        # Attach user history
        claim["_user_history"] = user_history.get(uid, {})

        # Attach filtered evidence requirements for this object type
        filtered_reqs = [
            req for req in evidence_requirements
            if req.get("claim_object", "").strip().lower() == claim_object
        ]
        claim["_evidence_requirements"] = filtered_reqs

        enriched.append(claim)
    return enriched
