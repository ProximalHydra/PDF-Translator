from collections import Counter

import pymupdf as fitz
import pytest

from pdf_translator.domain import DocumentError, Run
from pdf_translator.pdf_engine import FontResolver, inspect_pdf, render_pdf
from pdf_translator.pipeline import execute, protected_tokens


def test_roundtrip_keeps_pages_images_vectors_links_and_turkish(source_pdf, tmp_path, gateway):
    output = tmp_path / "translated.pdf"
    document = inspect_pdf(source_pdf)
    stages = []
    report = execute(source_pdf, output, document, gateway, lambda *v: stages.append(v[0]), "openai", "test-model")
    assert report["semantic_passed"] == len(document.runs) == 2
    assert {"translating", "reviewing", "rendering", "verifying"} <= set(stages)
    with fitz.open(source_pdf) as before, fitz.open(output) as after:
        assert before.page_count == after.page_count == 1
        assert before[0].rect == after[0].rect
        text = after[0].get_text()
        assert "Yıllık rapor" in text
        assert "Toplam gelir 1250 USD oldu." in text
        assert "Annual report" not in text
        assert "Total revenue" not in text
        assert after[0].get_images()
        assert len(after[0].get_drawings()) >= len(before[0].get_drawings())
        assert after.get_toc() == before.get_toc()
        assert any(link.get("uri") == "https://example.com" for link in after[0].get_links())
        assert after[0].get_pixmap().pixel(40, 40) == before[0].get_pixmap().pixel(40, 40)
    assert gateway.review_calls, "Meaning checks must actually run."
    assert all(detail["rendered"] for detail in report["details"])


def test_semantic_correction_is_applied_and_rechecked(source_pdf, tmp_path, gateway):
    original_review = gateway.review
    calls = []

    def review(entries):
        calls.append(entries)
        checks = original_review(entries)
        if len(calls) == 1:
            checks[-1].update(status="needs_correction", corrected_text="Toplam gelir 1250 USD idi.", reason="Zaman düzeltmesi.")
        return checks

    gateway.review = review
    output = tmp_path / "corrected.pdf"
    report = execute(source_pdf, output, inspect_pdf(source_pdf), gateway, lambda *a: None, "gemini", "test")
    assert len(calls) == 2
    assert len(calls[1]) == 1
    assert calls[1][0]["translation"] == "Toplam gelir 1250 USD idi."
    assert report["corrected"] == 1
    assert report["details"][-1]["corrected"] is True
    with fitz.open(output) as result:
        assert "1250 USD idi." in result[0].get_text()


def test_numbers_are_checked_even_when_ai_approves(source_pdf, tmp_path, gateway):
    gateway.mapping = {**gateway.mapping, "Total revenue was 1250 USD.": "Toplam gelir 2500 USD oldu."}
    report = execute(source_pdf, tmp_path / "bad-number.pdf", inspect_pdf(source_pdf), gateway, lambda *a: None, "deepseek", "test")
    assert report["status"] == "needs_review"
    assert any(issue["code"] == "protected_token_mismatch" for issue in report["issues"])
    assert report["semantic_passed"] == 1


def test_unresolved_second_review_is_never_reported_pass(source_pdf, tmp_path, gateway):
    def review(entries):
        return [{"id": e["id"], "status": "needs_correction", "corrected_text": e["translation"], "reason": "Anlam belirsiz."} for e in entries]

    gateway.review = review
    report = execute(source_pdf, tmp_path / "unresolved.pdf", inspect_pdf(source_pdf), gateway, lambda *a: None, "openrouter", "test")
    assert report["status"] == "needs_review"
    assert report["semantic_passed"] == 0
    assert any(issue["code"] == "semantic_review" for issue in report["issues"])


def test_impossible_layout_keeps_source_and_reports_it(source_pdf, tmp_path, gateway):
    gateway.mapping = {**gateway.mapping, "Annual report": "Türkçeye çevrilmiş çok uzun bir belge başlığı " * 150}
    output = tmp_path / "overflow.pdf"
    report = execute(source_pdf, output, inspect_pdf(source_pdf), gateway, lambda *a: None, "openai", "test")
    assert any(issue["code"] == "text_did_not_fit" for issue in report["issues"])
    assert report["status"] == "needs_review"
    assert report["details"][0]["rendered"] is False
    with fitz.open(output) as pdf:
        assert "Annual report" in pdf[0].get_text()


def test_missing_font_uses_unicode_fallback_in_bold(source_pdf, tmp_path):
    with fitz.open(source_pdf) as pdf:
        resolver = FontResolver(pdf, tmp_path / "fonts")
        run = Run("r", "test", "MissingSubsetFont", 12, 0, 16)
        family, fallback = resolver.resolve(0, run, "İğüşöç ŞİĞ ı")
        assert fallback
        font = fitz.Font(fontfile=str(tmp_path / "fonts" / f"{family}.ttf"))
        assert font.is_bold
        assert all(font.has_glyph(ord(c), fallback=False) for c in "İğüşöçŞı")


def test_embedded_font_is_reused_when_it_has_turkish(tmp_path):
    source = tmp_path / "embedded.pdf"
    font = fitz.Font("notos")
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_font(fontname="OriginalNoto", fontbuffer=font.buffer)
        page.insert_text((60, 70), "A short source line.", fontname="OriginalNoto", fontsize=12)
        pdf.save(source)
    document = inspect_pdf(source)
    output = tmp_path / "embedded-tr.pdf"
    issues = render_pdf(source, output, document, {document.runs[0].id: "Kısa kaynak satırı."}, tmp_path / "fonts")
    assert not any(issue.code == "font_substituted" for issue in issues)
    with fitz.open(output) as pdf:
        assert "Kısa kaynak satırı." in pdf[0].get_text()


def test_rotated_page_retains_page_rotation_and_text(source_pdf, tmp_path, gateway):
    rotated = tmp_path / "rotated.pdf"
    with fitz.open(source_pdf) as pdf:
        pdf[0].set_rotation(90)
        pdf.save(rotated)
    output = tmp_path / "rotated-tr.pdf"
    execute(rotated, output, inspect_pdf(rotated), gateway, lambda *a: None, "openai", "test")
    with fitz.open(output) as pdf:
        assert pdf[0].rotation == 90
        assert "Yıllık rapor" in pdf[0].get_text()


def test_mixed_styles_preserve_color_bold_and_punctuation_spacing(tmp_path):
    source = tmp_path / "styled.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        x = 50
        for text, font, color in [("The ", "helv", (0, 0, 0)),
                                   ("result", "hebo", (0.8, 0.1, 0.1)),
                                   (".", "helv", (0, 0, 0))]:
            page.insert_text((x, 70), text, fontsize=12, fontname=font, color=color)
            x += fitz.get_text_length(text, fontname=font, fontsize=12)
        pdf.save(source)
    document = inspect_pdf(source)
    assert ''.join(run.join_before + run.text for run in document.runs) == "The result."
    mapping = {"The ": "Bu ", "result": "sonuç", ".": "."}
    output = tmp_path / "styled-tr.pdf"
    render_pdf(source, output, document,
               {run.id: mapping[run.text] for run in document.runs}, tmp_path / "fonts")
    with fitz.open(output) as pdf:
        assert ''.join(pdf[0].get_text().split()) == "Busonuç."
        spans = [s for b in pdf[0].get_text("dict")["blocks"] for line in b.get("lines", []) for s in line["spans"]]
        assert any(s["flags"] & 16 and s["color"] != 0 for s in spans)


def test_scan_and_encryption_are_rejected(tmp_path):
    scan = tmp_path / "scan.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50))
        pix.clear_with(255)
        page.insert_image(page.rect, pixmap=pix)
        pdf.save(scan)
    with pytest.raises(DocumentError, match="OCR"):
        inspect_pdf(scan)
    secret = tmp_path / "secret.pdf"
    with fitz.open() as pdf:
        pdf.new_page().insert_text((40, 40), "Hidden")
        pdf.save(secret, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="password")
    with pytest.raises(DocumentError, match="Şifreli"):
        inspect_pdf(secret)


def test_protected_tokens_preserve_numbers_urls_and_email():
    assert protected_tokens("Total 1,250.50 USD; 2025/04/20; https://example.com a@b.org") == Counter({"1,250.50": 1, "2025/04/20": 1, "https://example.com": 1, "a@b.org": 1})
