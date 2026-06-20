"""
test_validator.py — Tests for pure-logic output validation and value fixing.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code"))

from validator import validate_and_fix

class TestValidator(unittest.TestCase):
    def test_case_1_known_issue_type_nei_to_supported(self):
        # Case 1: issue_type=dent, claim_status=not_enough_information. Expected: supported.
        result = {
            "issue_type": "dent",
            "claim_status": "not_enough_information",
            "risk_flags": "none",
            "severity": "unknown",
            "evidence_standard_met": "true",
            "valid_image": "true"
        }
        fixed = validate_and_fix(result, claim_object="car")
        self.assertEqual(fixed["claim_status"], "supported")

    def test_case_2_wrong_object_nei_to_contradicted(self):
        # Case 2: wrong_object flag, claim_status=not_enough_information. Expected: contradicted.
        result = {
            "issue_type": "unknown",
            "claim_status": "not_enough_information",
            "risk_flags": "wrong_object",
            "severity": "unknown",
            "evidence_standard_met": "true",
            "valid_image": "true"
        }
        fixed = validate_and_fix(result, claim_object="car")
        self.assertEqual(fixed["claim_status"], "contradicted")

    def test_case_3_supported_unknown_severity_to_low(self):
        # Case 3: supported, severity=unknown. Expected: low.
        result = {
            "issue_type": "dent",
            "claim_status": "supported",
            "risk_flags": "none",
            "severity": "unknown",
            "evidence_standard_met": "true",
            "valid_image": "true"
        }
        fixed = validate_and_fix(result, claim_object="car")
        self.assertEqual(fixed["claim_status"], "supported")
        self.assertEqual(fixed["severity"], "low")

    def test_case_4_contradicted_unknown_severity_to_none(self):
        # Case 4: contradicted, severity=unknown. Expected: none.
        result = {
            "issue_type": "unknown",
            "claim_status": "contradicted",
            "risk_flags": "none",
            "severity": "unknown",
            "evidence_standard_met": "true",
            "valid_image": "true"
        }
        fixed = validate_and_fix(result, claim_object="car")
        self.assertEqual(fixed["claim_status"], "contradicted")
        self.assertEqual(fixed["severity"], "none")

    def test_case_5_evidence_met_valid_image_nei_to_supported(self):
        # Case 5: evidence_standard_met=true, valid_image=true, claim_status=not_enough_information. Expected: supported.
        result = {
            "issue_type": "scratch",
            "claim_status": "not_enough_information",
            "risk_flags": "none",
            "severity": "unknown",
            "evidence_standard_met": "true",
            "valid_image": "true"
        }
        fixed = validate_and_fix(result, claim_object="car")
        self.assertEqual(fixed["claim_status"], "supported")

if __name__ == '__main__':
    unittest.main()
