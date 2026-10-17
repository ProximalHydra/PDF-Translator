import json

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from conftest import StubGateway
from test_app import wait_for_finish
from pdf_translator.app import create_app, TranslationRequest, connection_key
from pdf_translator.domain import TranslationError
from pdf_translator.memory import TranslationMemory
from pdf_translator.learner import split_corpus


def test_teaching_a_pdf_enables_real_offline_translation_without_a_gateway(source_pdf, tmp_path):
    calls = []
    def factory(*args, **kwargs):
        calls.append(args)
        return StubGateway()
    with TestClient(create_app(tmp_path / "data", factory)) as client:
        def upload():
            return client.post("/api/documents", files={"file": ("a.pdf", source_pdf.read_bytes())}).json()["id"]
        first = upload()
        client.post(f"/api/documents/{first}/translate", json={"api_key": "test", "consent": True})
        assert wait_for_finish(client, first)["status"] in {"completed", "needs_review"}
        assert client.get("/api/learning").json()["memory"]["candidates"] == 2
        assert client.post(f"/api/documents/{first}/translate", json={"mode": "offline"}).status_code == 422
        assert client.post(f"/api/documents/{first}/learn", json={"confirm": False}).status_code == 422
        assert client.post(f"/api/documents/{first}/learn", json={"confirm": True}).json()["approved"] == 2
        second = upload()
        started = client.post(f"/api/documents/{second}/translate", json={"mode": "offline", "consent": False})
        assert started.status_code == 202, started.text
        final = wait_for_finish(client, second)
        assert final["status"] in {"completed", "needs_review"}, final
        report = client.get(f"/api/documents/{second}/report").json()
        assert report["engines"] == {"memory": 2}
        assert len(calls) == 1, "Offline replay must never even construct an API gateway."
        with fitz.open(stream=client.get(f"/api/documents/{second}/download/translated").content, filetype="pdf") as pdf:
            assert "Yıllık rapor" in pdf[0].get_text()
        export = client.get("/api/learning/export")
        assert len(export.text.splitlines()) == 2
        assert "test" not in export.text
        assert client.get("/api/learning").json()["memory"]["reused"] == 2
        client.delete(f"/api/documents/{first}")
        assert client.get("/api/learning").json()["memory"]["approved"] == 2


def test_memory_requires_approval_and_same_context_and_persists(tmp_path):
    memory = TranslationMemory(tmp_path)
    id_ = memory.put("Charge", "Ücret", "Charge to the bank", quality="candidate")
    assert memory.lookup("Charge", "Charge to the bank") is None
    memory.approve(id_)
    assert memory.lookup("Charge", "Electrical charge") is None
    assert memory.lookup("Charge", "Charge to the bank")["target"] == "Ücret"
    memory.put("Charge", "Hücum", "Charge to the bank", quality="candidate")
    restored = TranslationMemory(tmp_path)
    assert restored.lookup("Charge", "Charge to the bank")["target"] == "Ücret"


def test_memory_corrections_keep_protected_tokens_and_delete(tmp_path):
    app = create_app(tmp_path / "data", StubGateway)
    id_ = app.state.memory.put("Total 1250 USD", "Toplam 2500 USD", "Total 1250 USD")
    with TestClient(app) as client:
        assert client.patch(f"/api/memory/{id_}", json={}).status_code == 422
        assert client.patch(f"/api/memory/{id_}", json={"target": "Toplam 1250 USD"}).status_code == 200
        assert client.get("/api/memory?quality=approved").json()["pairs"][0]["target"] == "Toplam 1250 USD"
        assert client.delete(f"/api/memory/{id_}").status_code == 204
        assert client.get("/api/memory").json()["pairs"] == []


def test_training_requires_distinct_examples_and_has_no_holdout_leakage():
    rows = [{"source": f"Source {i}", "target": f"Çeviri {i}", "context": str(i)} for i in range(20)]
    rows.append({**rows[0], "context": "another context"})
    training, heldout = split_corpus(rows)
    assert {r["source"] for r in training}.isdisjoint({r["source"] for r in heldout})
    with pytest.raises(TranslationError, match="20 farklı"):
        split_corpus([rows[0]] * 20)


def test_server_credentials_never_follow_custom_urls(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_OPENAI_KEY", "server-secret")
    assert connection_key(TranslationRequest()) == "server-secret"
    assert connection_key(TranslationRequest(base_url="https://another.example/v1")) == ""
    assert connection_key(TranslationRequest(base_url="https://another.example/v1", api_key="user-selected-key")) == "user-selected-key"


def test_training_and_automation_status_are_honest_without_model(tmp_path):
    with TestClient(create_app(tmp_path / "data", StubGateway)) as client:
        status = client.get("/api/learning").json()
        assert not status["student"]["ready"]
        assert not status["student"]["trained"]
        assert client.post("/api/learning/train", json={}).status_code == 422
        assert client.patch("/api/learning/automation?enabled=false").json()["auto_train"] is False


def test_real_gradient_updates_and_reload_of_lora_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from safetensors.torch import load_file
    from peft import PeftModel
    from pdf_translator.learner import train_lora
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokens = {token: i for i, token in enumerate(["<pad>", "<unk>", "<eos>", "source", "Turkish", "hello", "merhaba", "system", "user", "assistant"])}
    raw = Tokenizer(models.WordLevel(tokens, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>",
        chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    base = Qwen3ForCausalLM(config)
    rows = [{"source": "hello", "target": "merhaba", "context": "hello", "language": "en"} for _ in range(8)]
    progress = []
    trained, used = train_lora(base, tokenizer, rows, tmp_path, lambda step, total, loss: progress.append(loss), steps=3)
    assert used == 8 and len(progress) == 3
    assert all(torch.isfinite(torch.tensor(progress)))
    weights = load_file(tmp_path / "adapter_model.safetensors")
    assert any("lora_B" in key and value.abs().sum() > 0 for key, value in weights.items())
    restored = PeftModel.from_pretrained(Qwen3ForCausalLM(config), tmp_path, local_files_only=True)
    assert restored.peft_config["default"].r == 8
    assert any("lora_B" in name and value.abs().sum() > 0 for name, value in restored.named_parameters())
