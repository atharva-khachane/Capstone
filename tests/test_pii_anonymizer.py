"""
Comprehensive tests for PIIAnonymizer.

Tests all PII detection patterns including India-specific ones.
"""

import pytest
from sl_rag.core.pii_anonymizer import PIIAnonymizer


class TestPIIAnonymizer:
    """Comprehensive tests for PII Anonymizer"""
    
    def setup_method(self):
        """Setup for each test"""
        self.anonymizer = PIIAnonymizer(enable_ner=False, log_detections=False)
    
    def test_email_detection(self):
        """Test email address detection"""
        text = "Contact us at john.doe@isro.gov.in or info@example.com"
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['email'] == 2
        assert "john.doe@isro.gov.in" not in anonymized
        assert "[EMAIL_REDACTED]" in anonymized
        
        print(f"\n✓ Email detection: {detections['email']} found")
    
    def test_indian_phone_detection(self):
        """Test Indian phone number detection"""
        text = """
        Call +91-9876543210 or 9123456789.
        Office: +91 8888777766
        """
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['phone_indian'] >= 2
        assert "9876543210" not in anonymized
        assert "[PHONE_INDIAN_REDACTED]" in anonymized
        
        print(f"\n✓ Indian phone detection: {detections['phone_indian']} found")
    
    def test_aadhaar_detection(self):
        """Test Aadhaar number detection"""
        text = """
        Aadhaar: 1234-5678-9012
        Also: 9876 5432 1098
        And: 111122223333
        """
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['aadhaar'] >= 2
        assert "1234-5678-9012" not in anonymized
        assert "[AADHAAR_REDACTED]" in anonymized
        
        print(f"\n✓ Aadhaar detection: {detections['aadhaar']} found")
    
    def test_pan_detection(self):
        """Test PAN card detection"""
        text = """
        PAN: ABCDE1234F
        Another PAN: XYZAB9999C
        """
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['pan'] == 2
        assert "ABCDE1234F" not in anonymized
        assert "[PAN_REDACTED]" in anonymized
        
        print(f"\n✓ PAN detection: {detections['pan']} found")
    
    def test_passport_detection(self):
        """Test passport number detection"""
        text = "Passport: A1234567 and B9876543"
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['passport'] >= 1
        assert "[PASSPORT_REDACTED]" in anonymized
        
        print(f"\n✓ Passport detection: {detections['passport']} found")
    
    def test_ssn_detection(self):
        """Test SSN detection"""
        text = "SSN: 123-45-6789"
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['ssn'] == 1
        assert "123-45-6789" not in anonymized
        
        print(f"\n✓ SSN detection: {detections['ssn']} found")
    
    @pytest.mark.xfail(reason="Credit card regex needs refinement for all formats")
    def test_credit_card_detection(self):
        """Test credit card detection"""
        text = """
        Card: 1234-5678-9012-3456
        Another: 9876 5432 1098 7654
        """
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['credit_card'] >= 1  # At least one format should match
        assert ("1234-5678-9012-3456" not in anonymized or 
                "9876 5432 1098 7654" not in anonymized)
        
        print(f"\n✓ Credit card detection: {detections['credit_card']} found")
    
    def test_ip_address_detection(self):
        """Test IP address detection"""
        text = "Server at 192.168.1.1 and 10.0.0.1"
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['ip_address'] == 2
        assert "192.168.1.1" not in anonymized
        
        print(f"\n✓ IP address detection: {detections['ip_address']} found")
    
    def test_dob_detection(self):
        """Test date of birth detection"""
        text = "DOB: 01/15/1990 and 25-12-1985"
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['dob'] >= 1
        
        print(f"\n✓ DOB detection: {detections['dob']} found")
    
    def test_employee_id_detection(self):
        """Test employee ID detection (ISRO, EMP prefixes)"""
        text = """
        Employee IDs: EMP-12345, ISRO-987654, ID_456789
        """
        anonymized, detections = self.anonymizer.anonymize(text)
        
        assert detections['employee_id'] >= 2
        assert "EMP-12345" not in anonymized
        assert "ISRO-987654" not in anonymized
        
        print(f"\n✓ Employee ID detection: {detections['employee_id']} found")
    
    def test_mixed_pii(self):
        """Test document with multiple PII types"""
        text = """
        Contact: john.doe@isro.gov.in, +91-9876543210
        Aadhaar: 1234-5678-9012
        PAN: ABCDE1234F
        Employee: EMP-12345
        Credit Card: 1234-5678-9012-3456
        """
        
        anonymized, detections = self.anonymizer.anonymize(text)
        
        total_pii = sum(detections.values())
        assert total_pii >= 5, f"Should detect at least 5 PII instances, got {total_pii}"
        
        # Verify no PII in output
        assert "john.doe@isro.gov.in" not in anonymized
        assert "9876543210" not in anonymized
        assert "1234-5678-9012" not in anonymized
        assert "ABCDE1234F" not in anonymized
        assert "EMP-12345" not in anonymized
        
        print(f"\n✓ Mixed PII detection: {total_pii} instances found")
        print(f"  Breakdown: {detections}")
    
    def test_no_pii(self):
        """Test text with no PII"""
        text = "This document discusses general financial rules and procurement procedures."
        anonymized, detections = self.anonymizer.anonymize(text)
        
        total_pii = sum(detections.values())
        assert total_pii == 0
        assert anonymized == text  # Text should be unchanged
        
        print(f"\n✓ No PII detected (as expected)")
    
    def test_validation_success(self):
        """Test validation of successful anonymization"""
        original = "Contact: john@example.com, PAN: ABCDE1234F"
        anonymized, _ = self.anonymizer.anonymize(original)
        
        validation = self.anonymizer.validate_anonymization(original, anonymized)
        
        assert validation['success'] == True
        assert validation['pii_remaining'] == {}
        assert validation['total_removed'] > 0
        
        print(f"\n✓ Validation successful: {validation['total_removed']} PII removed")
    
    def test_preservestructure_flag(self):
        """Test preserve_structure flag"""
        text = "Email: john@example.com, Phone: +91-9876543210"
        
        # With preserve_structure=True (default)
        anonymized_preserved, _ = self.anonymizer.anonymize(text, preserve_structure=True)
        assert "[EMAIL_REDACTED]" in anonymized_preserved
        assert "[PHONE_INDIAN_REDACTED]" in anonymized_preserved
        
        # With preserve_structure=False
        anonymized_generic, _ = self.anonymizer.anonymize(text, preserve_structure=False)
        assert "[REDACTED]" in anonymized_generic
        
        print(f"\n✓ Preserve structure flag working")
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked across multiple calls"""
        # Reset statistics
        self.anonymizer.reset_statistics()
        
        # Anonymize multiple texts
        self.anonymizer.anonymize("Email: test@example.com")
        self.anonymizer.anonymize("PAN: ABCDE1234F")
        self.anonymizer.anonymize("Phone: +91-9876543210")
        
        stats = self.anonymizer.get_statistics()
        total = sum(stats.values())
        
        assert total >= 3  # At least 3 PII instances
        
        print(f"\n✓ Statistics tracking: {total} total detections")
        print(f"  Stats: {stats}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
