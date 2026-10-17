"""Real browser + actual PDF pipeline, replacing only the billable AI boundary.

Run: uv run python tests/browser_smoke.py --browser /usr/bin/chromium
No credentials, external requests or development-only HTTP routes are used.
"""
import argparse
import multiprocessing
import socket
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pymupdf as fitz
from playwright.sync_api import sync_playwright

from conftest import StubGateway

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def serve(data, port):
    import uvicorn
    from pdf_translator.app import create_app
    uvicorn.run(create_app(Path(data), StubGateway), host="127.0.0.1", port=port, log_level="error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default=None, help="Optional Chromium executable")
    parser.add_argument("--artifacts", default="/tmp/forma-browser")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "Annual report.pdf"
        with fitz.open() as pdf:
            page = pdf.new_page()
            page.insert_text((55, 75), "Annual report", fontsize=20, fontname="hebo")
            page.insert_text((55, 120), "Total revenue was 1250 USD.", fontsize=12)
            pdf.save(source)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        process = multiprocessing.Process(target=serve, args=(str(Path(temp) / "data"), port))
        process.start()
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    if httpx.get(base + "/api/health").is_success:
                        break
                except httpx.ConnectError:
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError("Test server did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=args.browser, headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"])
                page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                page.goto(base)
                page.wait_for_function("() => !document.querySelector('#connection-button').disabled")
                page.get_by_role("heading", name="İçerik Türkçe.").wait_for()
                page.screenshot(path=str(artifacts / "desktop.png"), full_page=True)
                page.locator("#connection-button").click()
                page.locator("#provider-select").select_option("gemini")
                assert page.locator("#model-input").input_value() == "gemini-2.5-flash"
                page.locator("#api-key-input").fill("browser-test-key")
                page.get_by_role("button", name="Bağlantıyı uygula").click()
                assert "Gemini" in page.locator("#connection-label").inner_text()
                assert "browser-test-key" not in page.evaluate("JSON.stringify(localStorage)")
                page.locator("#file-input").set_input_files(str(source))
                page.locator(".file-card").wait_for()
                with page.expect_response(lambda response: response.url.endswith("/translate")):
                    page.locator("#translate-button").click()
                page.wait_for_function("() => document.querySelector('#consent') === document.activeElement")
                assert page.locator("#consent").evaluate("element => element === document.activeElement")
                page.locator("#consent").check()
                page.locator("#translate-button").click()
                page.locator(".download-link").wait_for(timeout=20000)
                page.screenshot(path=str(artifacts / "completed.png"), full_page=True)
                page.get_by_role("button", name="Kontrol raporu", exact=True).click()
                page.locator(".comparison-row").first.wait_for()
                assert "Yıllık rapor" in page.locator("#review-body").inner_text()
                page.screenshot(path=str(artifacts / "review.png"), full_page=True)
                with page.expect_download() as info:
                    page.locator("#review-body").get_by_role("link", name="Türkçe PDF’i indir", exact=True).click()
                downloaded = artifacts / "translated.pdf"
                info.value.save_as(downloaded)
                with fitz.open(downloaded) as pdf:
                    assert "Yıllık rapor" in pdf[0].get_text()
                    assert "1250 USD" in pdf[0].get_text()
                page.locator("#review-approved").check()
                page.locator("#teach-document").click()
                page.get_by_role("button", name="Belleğe kaydedildi").wait_for()
                page.get_by_role("button", name="İncelemeyi kapat").click()
                page.locator('.main-nav [data-view="learning"]').click()
                page.wait_for_function("() => document.querySelector('#stat-approved').textContent === '2'")
                page.locator('[data-memory-filter="approved"]').click()
                page.locator(".memory-pair").first.wait_for()
                page.screenshot(path=str(artifacts / "learning.png"), full_page=True)
                page.locator('.main-nav [data-view="connections"]').click()
                page.locator("#provider-select").select_option("opencode")
                assert page.locator("#base-url-input").input_value() == "https://opencode.ai/zen/v1"
                page.screenshot(path=str(artifacts / "connections.png"), full_page=True)
                # Refresh drops the API key. A repeated PDF still translates offline.
                page.reload()
                page.locator(".file-card").wait_for()
                page.locator("#execution-mode").select_option("offline")
                assert page.locator("#provider-label").inner_text() == "Yerel öğrenci + bellek"
                with page.expect_response(lambda response: response.url.endswith("/translate")) as replay:
                    page.locator("#translate-button").click()
                assert replay.value.status == 202
                page.wait_for_function("() => document.querySelector('#document-status').textContent.includes('2 bellek')", timeout=20000)
                page.set_viewport_size({"width": 390, "height": 844})
                page.locator("#toast").wait_for(state="hidden", timeout=10000)
                page.screenshot(path=str(artifacts / "mobile.png"), full_page=True)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Horizontal overflow on mobile"
                page.on("dialog", lambda dialog: dialog.accept())
                page.get_by_role("button", name="Annual report.pdf belgesini sil", exact=True).click()
                page.locator(".file-card").wait_for(state="detached")
                # The deliberate missing-consent request above is an expected 422.
                assert not [error for error in errors if '422' not in error], errors
                browser.close()
            print("PASS: desktop/mobile, configurable providers including OpenCode, upload, consent, translation, review, PDF download, teaching, offline replay after key reset, deletion.")
            print(f"Artifacts: {artifacts}")
        finally:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
