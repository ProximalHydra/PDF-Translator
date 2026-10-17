import json
import time

import pymupdf as fitz
from fastapi.testclient import TestClient

from conftest import StubGateway
from pdf_translator.app import create_app
from pdf_translator.domain import TranslationError


def wait_for_finish(client, id_):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        document = client.get(f"/api/documents/{id_}").json()
        if document["status"] in {"completed", "needs_review", "failed"}:
            return document
        time.sleep(0.02)
    raise AssertionError("Background translation did not finish")


def test_upload_translate_review_download_and_delete(source_pdf, tmp_path):
    data = tmp_path / "data"
    app = create_app(data, gateway_factory=StubGateway)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["status"] == "ok"
        upload = client.post("/api/documents", files={"file": ("report.pdf", source_pdf.read_bytes(), "application/pdf")})
        assert upload.status_code == 201, upload.text
        id_ = upload.json()["id"]
        assert client.get(f"/api/documents/{id_}/preview/source").headers["content-type"] == "image/png"
        start = client.post(f"/api/documents/{id_}/translate", json={"provider": "gemini", "model": "test", "api_key": "private-test-key", "consent": True})
        assert start.status_code == 202, start.text
        final = wait_for_finish(client, id_)
        assert final["status"] in {"completed", "needs_review"}, final
        report = client.get(f"/api/documents/{id_}/report").json()
        assert report["semantic_passed"] == 2
        assert report["provider"] == "gemini"
        translated = client.get(f"/api/documents/{id_}/download/translated")
        assert translated.status_code == 200
        with fitz.open(stream=translated.content, filetype="pdf") as pdf:
            assert "1250 USD" in pdf[0].get_text()
            assert "Yıllık rapor" in pdf[0].get_text()
        assert client.get(f"/api/documents/{id_}/download/report").json()["segments"] == 2
        assert "private-test-key" not in client.get("/api/documents").text
        for path in data.rglob("*"):
            if path.is_file():
                assert b"private-test-key" not in path.read_bytes()
        assert client.delete(f"/api/documents/{id_}").status_code == 204
        assert client.get(f"/api/documents/{id_}").status_code == 404
        assert not (data / id_).exists()


def test_invalid_upload_missing_key_and_consent(source_pdf, tmp_path, monkeypatch):
    for name in ("TRANSLATOR_OPENAI_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with TestClient(create_app(tmp_path / "data", StubGateway)) as client:
        assert client.post("/api/documents", files={"file": ("fake.pdf", b"not pdf", "application/pdf")}).status_code == 422
        assert not list((tmp_path / "data").glob("*/source.pdf"))
        id_ = client.post("/api/documents", files={"file": ("valid.pdf", source_pdf.read_bytes(), "application/pdf")}).json()["id"]
        assert client.post(f"/api/documents/{id_}/translate", json={"api_key": "abc"}).status_code == 422
        assert client.post(f"/api/documents/{id_}/translate", json={"consent": True}).status_code == 422
        assert client.get(f"/api/documents/{id_}/download/translated").status_code == 409
        assert client.get(f"/api/documents/{id_}/download/evil").status_code == 422
        assert client.post("/api/documents", headers={"Origin": "https://evil.example"}, files={"file": ("a.pdf", b"bad")}).status_code == 403


def test_failed_api_request_is_visible_and_can_retry(source_pdf, tmp_path):
    class FailingGateway(StubGateway):
        def translate(self, blocks):
            raise TranslationError("API anahtarı kabul edilmedi.")

    with TestClient(create_app(tmp_path / "data", FailingGateway)) as client:
        id_ = client.post("/api/documents", files={"file": ("valid.pdf", source_pdf.read_bytes())}).json()["id"]
        client.post(f"/api/documents/{id_}/translate", json={"api_key": "test", "consent": True})
        final = wait_for_finish(client, id_)
        assert final["status"] == "failed"
        assert "anahtarı" in final["message"]
        assert client.get(f"/api/documents/{id_}/download/translated").status_code == 409
        assert client.post(f"/api/documents/{id_}/translate", json={"api_key": "test", "consent": True}).status_code == 202
        assert wait_for_finish(client, id_)["status"] == "failed"


def test_restart_recovers_documents_and_interrupts_unfinished_jobs(tmp_path):
    directory = tmp_path / "data" / "abc"
    directory.mkdir(parents=True)
    (directory / "document.json").write_text(json.dumps({"id": "abc", "status": "reviewing", "created_at": "2026-01-01", "progress": 50}))
    with TestClient(create_app(tmp_path / "data", StubGateway)) as client:
        doc = client.get("/api/documents/abc").json()
        assert doc["status"] == "failed"
        assert "yeniden" in doc["message"]


def test_config_reports_only_key_presence(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSLATOR_GEMINI_KEY", "do-not-expose")
    with TestClient(create_app(tmp_path / "data", StubGateway)) as client:
        response = client.get("/api/config")
        assert response.json()["providers"]["gemini"]["configured"]
        assert "do-not-expose" not in response.text
