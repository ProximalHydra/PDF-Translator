from pathlib import Path

import pymupdf as fitz
import pytest


@pytest.fixture
def source_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(35, 35, 560, 145), fill=(0.92, 0.9, 0.98), color=None)
        page.insert_text((55, 75), "Annual report", fontsize=20, fontname="hebo", color=(0.2, 0.15, 0.4))
        page.insert_text((55, 120), "Total revenue was 1250 USD.", fontsize=12)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40))
        pix.clear_with(170)
        page.insert_image(fitz.Rect(55, 170, 95, 210), pixmap=pix)
        page.draw_rect(fitz.Rect(50, 230, 300, 300), color=(0.3, 0.3, 0.3))
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(55, 60, 200, 80), "uri": "https://example.com"})
        doc.set_toc([[1, "Report", 1]])
        doc.save(path)
    return path


class StubGateway:
    """A deterministic AI boundary; PDF parsing, layout and reports remain real."""
    mapping = {"Annual report": "Yıllık rapor", "Total revenue was 1250 USD.": "Toplam gelir 1250 USD oldu."}

    def __init__(self, *args, **kwargs):
        self.review_calls = []

    def translate(self, blocks):
        return {run["id"]: self.mapping.get(run["text"], run["text"])
                for block in blocks for run in block["runs"]}

    def review(self, entries):
        self.review_calls.append(entries)
        return [{"id": entry["id"], "status": "pass", "reason": "Kaynakla uyumlu.", "corrected_text": None}
                for entry in entries]

    def close(self):
        pass


@pytest.fixture
def gateway():
    return StubGateway()
