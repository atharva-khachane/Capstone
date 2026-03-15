from pathlib import Path

from sl_rag.core.document_loader import DocumentLoader


def test_validate_file_rejects_non_pdf_extension(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("not a pdf")
    loader = DocumentLoader(ocr_enabled=False)

    result = loader.validate_file(str(txt_path))
    assert result["is_valid"] is False
    assert "invalid_extension" in result["errors"]


def test_validate_file_reports_malware_scan_failure(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nfake content")

    def scanner(_path, _bytes):
        return False, "eicar_signature"

    loader = DocumentLoader(ocr_enabled=False, malware_scanner=scanner)
    result = loader.validate_file(str(pdf_path))

    assert result["is_valid"] is False
    assert any(e.startswith("malware_detected") for e in result["errors"])
    assert result["checks"]["malware_scan_passed"] is False


def test_validate_file_returns_explicit_checks(tmp_path):
    pdf_path = tmp_path / "ok.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nok")

    loader = DocumentLoader(ocr_enabled=False)
    result = loader.validate_file(str(pdf_path))

    assert "checks" in result
    assert result["checks"]["exists"] is True
    assert result["checks"]["extension_pdf"] is True
    assert result["checks"]["readable"] is True
