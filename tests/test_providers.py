import json

import httpx
import pytest
from openai import OpenAI

from pdf_translator.domain import TranslationError
from pdf_translator.providers import AIGateway, PROVIDERS, validate_ids, validate_base_url


@pytest.mark.parametrize("provider", [id_ for id_, spec in PROVIDERS.items() if spec.get("url") and spec.get("protocol", "openai_chat") == "openai_chat"])
def test_provider_uses_compatible_json_api_and_parses_response(provider):
    requests = []

    def respond(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "untrusted" in body["messages"][0]["content"].lower()
        content = {"translations": [{"id": "r1", "text": "Türkçe metin"}]}
        return httpx.Response(200, json={"id": "chat-test", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]})

    gateway = AIGateway(provider, "test", "test-key")
    gateway.client.close()
    gateway.client = OpenAI(api_key="test-key", base_url=PROVIDERS[provider]["url"], http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        assert gateway.translate([{"id": "b1", "runs": [{"id": "r1", "text": "Source text"}]}]) == {"r1": "Türkçe metin"}
        assert requests[0].url.host == httpx.URL(PROVIDERS[provider]["url"]).host
        assert requests[0].url.path.endswith("/chat/completions")
    finally:
        gateway.close()


@pytest.mark.parametrize("rows", [[], [{"id": "a"}, {"id": "a"}], [{"id": "wrong"}], None])
def test_missing_duplicate_and_unknown_ids_stop_translation(rows):
    with pytest.raises(TranslationError):
        validate_ids(rows, ["a"], "çeviri")


def test_truncated_output_never_becomes_a_pdf():
    gateway = AIGateway("openai", "test", "test-key")
    gateway.client.close()
    gateway.client = OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "id": "test", "created": 1, "model": "test", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]}))))
    try:
        with pytest.raises(TranslationError, match="tamamlanmadı"):
            gateway.request("JSON", {})
    finally:
        gateway.close()


def test_provider_errors_never_expose_credentials():
    gateway = AIGateway("openai", "test", "test-secret")
    gateway.client.close()
    gateway.client = OpenAI(api_key="test-secret", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {"message": "Bad token test-secret"}}))))
    try:
        with pytest.raises(TranslationError) as error:
            gateway.request("JSON", {})
        assert "test-secret" not in str(error.value)
        assert "anahtarı" in str(error.value)
    finally:
        gateway.close()


@pytest.mark.parametrize("protocol,base,path", [
    ("anthropic", "https://vendor.example", "/v1/messages"),
    ("anthropic", "https://opencode.ai/zen/v1", "/zen/v1/messages"),
    ("gemini", "https://vendor.example/v1beta", "/v1beta/models/test:generateContent"),
    ("ollama", "http://127.0.0.1:11434", "/api/chat"),
])
def test_native_protocols_use_their_own_wire_formats(protocol, base, path):
    result = {"translations": [{"id": "r1", "text": "Türkçe metin"}]}
    def respond(request):
        assert request.url.path == path
        assert "api_key" not in request.url.params
        body = json.loads(request.content)
        if protocol == "anthropic":
            assert request.headers["x-api-key"] == "test"
            assert body["system"] and body["max_tokens"] == 8192
            return httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(result)}]})
        if protocol == "gemini":
            assert request.headers["x-goog-api-key"] == "test"
            assert body["systemInstruction"] and body["generationConfig"]["responseMimeType"] == "application/json"
            return httpx.Response(200, json={"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(result)}]}}]})
        assert body["stream"] is False and body["format"] == "json"
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(result)}})
    gateway = AIGateway("custom", "test", "test", base_url=base, protocol=protocol)
    gateway.http.close()
    gateway.http = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        assert gateway.translate([{"runs": [{"id": "r1", "text": "Source"}]}]) == {"r1": "Türkçe metin"}
    finally:
        gateway.close()


def test_custom_responses_api():
    result = {"translations": [{"id": "r1", "text": "Türkçe"}]}
    def respond(request):
        assert request.url.path == "/v1/responses"
        assert json.loads(request.content)["text"]["format"]["type"] == "json_object"
        return httpx.Response(200, json={"id": "resp_test", "created_at": 1, "status": "completed", "model": "test", "object": "response",
            "output": [{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": json.dumps(result), "annotations": []}]}]})
    gateway = AIGateway("custom", "test", "test", base_url="https://vendor.example/v1", protocol="openai_responses")
    gateway.client.close()
    gateway.client = OpenAI(api_key="test", base_url=gateway.base_url, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        assert gateway.translate([{"runs": [{"id": "r1", "text": "Source"}]}]) == {"r1": "Türkçe"}
    finally:
        gateway.close()


def test_custom_chat_without_json_mode_still_validates_json():
    def respond(request):
        assert "response_format" not in json.loads(request.content)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '```json\n{"ok":true}\n```'}}]})
    gateway = AIGateway("my-provider", "test", "", base_url="https://vendor.example/v1", json_mode=False)
    gateway.client.close()
    gateway.client = OpenAI(api_key="test", base_url=gateway.base_url, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        assert gateway.request("JSON", {}) == {"ok": True}
    finally:
        gateway.close()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "https://user:password@api.example/v1", "https://api.example?key=secret", "http://169.254.169.254", "http://public.example/v1"])
def test_invalid_or_secret_bearing_endpoint_urls_are_rejected(url):
    with pytest.raises(TranslationError):
        validate_base_url(url)


def test_keyless_local_client_never_inherits_openai_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send")
    monkeypatch.setenv("OPENAI_ORG_ID", "private-org")
    gateway = AIGateway("custom", "local", "", base_url="http://127.0.0.1:1234/v1")
    try:
        assert gateway.client.api_key == ""
        assert not gateway.client.auth_headers
        assert gateway.client.organization == ""
        assert gateway.local
    finally:
        gateway.close()
