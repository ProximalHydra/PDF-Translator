import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from starlette.concurrency import run_in_threadpool

from .domain import DocumentError, TranslationError
from .pdf_engine import MAX_CHARACTERS, MAX_PAGES, inspect_pdf, preview
from .pipeline import execute, protected_tokens
from .providers import AIGateway, PROVIDERS, PROTOCOLS, configured_key, validate_base_url
from .memory import TranslationMemory, AdaptiveGateway
from .learner import LocalLearner, LocalGateway, split_corpus

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD = 25 * 1024 * 1024
ACTIVE = {"queued", "translating", "reviewing", "rendering", "verifying"}


class TranslationRequest(BaseModel):
    provider: str = Field(default="openai", min_length=1, max_length=60, pattern=r"^[a-zA-Z0-9_-]+$")
    model: str = Field(default="gpt-4.1-mini", min_length=1, max_length=120, pattern=r"^[\w./:@+-]+$")
    api_key: SecretStr | None = None
    consent: bool = False
    base_url: str | None = Field(default=None, max_length=500)
    protocol: Literal["openai_chat", "openai_responses", "anthropic", "gemini", "ollama"] | None = None
    max_tokens: int = Field(default=8192, ge=512, le=32768)
    json_mode: bool = True
    mode: Literal["hybrid", "cloud", "offline"] = "hybrid"
    source_language: str = Field(default="auto", max_length=20, pattern=r"^[a-zA-Z-]+$")
    collect_examples: bool = True


class ApprovalRequest(BaseModel):
    confirm: bool = False
    ids: list[str] | None = Field(default=None, max_length=2000)


class PairApproval(BaseModel):
    target: str | None = Field(default=None, min_length=1, max_length=20000)


class TrainingRequest(BaseModel):
    steps: int = Field(default=32, ge=1, le=128)


def connection_key(body):
    if body.api_key and body.api_key.get_secret_value().strip():
        return body.api_key.get_secret_value().strip()
    preset = PROVIDERS.get(body.provider, {})
    # A server binding must never follow an overridden URL to a different host.
    destination = body.base_url or preset.get("url", "")
    if not preset.get("url") or urlsplit(destination).netloc != urlsplit(preset["url"]).netloc:
        return ""
    return configured_key(body.provider)


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.documents = {}
        for path in root.glob("*/document.json"):
            try:
                value = json.loads(path.read_text())
                if value["status"] in ACTIVE:
                    value.update(status="failed", progress=0, message="Sunucu yeniden başladı. Çeviriyi yeniden başlatabilirsiniz.")
                self.documents[value["id"]] = value
            except (ValueError, KeyError, OSError):
                continue

    def get(self, id_: str):
        with self.lock:
            if id_ not in self.documents:
                raise HTTPException(404, "Belge bulunamadı.")
            return dict(self.documents[id_])

    def update(self, id_: str, **fields):
        with self.lock:
            self.documents[id_].update(fields)
            path = self.root / id_ / "document.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.documents[id_], ensure_ascii=False), encoding="utf-8")
            temp.replace(path)


def create_app(data_dir: Path | None = None, gateway_factory=AIGateway):
    store = Store(data_dir or Path(os.getenv("PDF_TRANSLATOR_DATA_DIR", str(ROOT / ".data"))))
    memory = TranslationMemory(store.root / "learning")
    learner = LocalLearner(store.root / "learning" / "student", memory)

    @asynccontextmanager
    async def lifespan(application):
        application.state.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translation")
        application.state.learning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="student")
        yield
        application.state.executor.shutdown(wait=True, cancel_futures=True)
        application.state.learning_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Forma · PDF Çeviri", lifespan=lifespan)
    app.state.store = store
    app.state.memory, app.state.learner = memory, learner

    @app.middleware("http")
    async def headers(request: Request, call_next):
        # No CORS. Block cross-origin browser mutations, including local network CSRF.
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE", "PUT", "PATCH"} and origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Bu kaynaktan gelen istek kabul edilmedi."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'"
        )
        return response

    @app.exception_handler(DocumentError)
    async def document_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(TranslationError)
    async def translation_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "forma", "version": "0.2.0"}

    @app.get("/api/config")
    def config():
        return {"providers": {id_: {"label": spec["label"], "model": spec["model"],
                                   "base_url": spec["url"], "protocol": spec.get("protocol", "openai_chat"),
                                   "local": spec.get("local", False),
                                   "configured": bool(configured_key(id_))}
                              for id_, spec in PROVIDERS.items()},
                "max_upload_mb": MAX_UPLOAD // 1024 // 1024,
                "max_pages": MAX_PAGES, "max_characters": MAX_CHARACTERS,
                "protocols": PROTOCOLS}

    @app.post("/api/connections/models")
    def connection_models(body: TranslationRequest):
        key = connection_key(body)
        gateway = AIGateway(body.provider, body.model, key, base_url=body.base_url, protocol=body.protocol)
        try:
            return {"models": gateway.models()}
        finally:
            gateway.close()

    @app.get("/api/learning")
    def learning():
        return {"memory": memory.stats(), "student": learner.status()}

    def maybe_train():
        with learner.status_lock:
            count = memory.stats()["approved"]
            if not learner.state.get("auto_train", True) or not learner.ready or not learner.installed:
                return
            if learner.state["phase"] in {"downloading", "training", "evaluating"} or count - learner.state.get("last_train_count", 0) < 20:
                return
            try:
                split_corpus(memory.approved())
            except TranslationError:
                return
            learner.update(phase="evaluating", last_train_count=count, message="20 yeni onaylı örnek birikti; otomatik eğitim hazırlanıyor")
            app.state.learning_executor.submit(learner.train)

    @app.patch("/api/learning/automation")
    def automation(enabled: bool):
        learner.update(auto_train=enabled)
        maybe_train()
        return learner.status()

    @app.get("/api/memory")
    def memory_list(quality: Literal["candidate", "approved"] | None = None, offset: int = 0):
        return {"pairs": memory.list(quality, offset=max(0, offset)), "stats": memory.stats()}

    @app.patch("/api/memory/{id_}")
    def approve_pair(id_: str, body: PairApproval):
        # Find the exact pair without changing it before validation.
        with memory.connect() as db:
            row = db.execute("SELECT * FROM pairs WHERE id=?", (id_,)).fetchone()
        if not row:
            raise HTTPException(404, "Metin çifti bulunamadı.")
        if protected_tokens(row["source"]) != protected_tokens(body.target if body.target is not None else row["target"]):
            raise HTTPException(422, "Sayı ve bağlantılar kaynakla eşleşmiyor; onaylamadan önce düzelt.")
        memory.approve(id_, body.target)
        maybe_train()
        return {"status": "approved", "stats": memory.stats()}

    @app.delete("/api/memory/{id_}", status_code=204)
    def delete_pair(id_: str):
        if not memory.delete(id_):
            raise HTTPException(404, "Metin çifti bulunamadı.")

    @app.get("/api/learning/export")
    def export_corpus():
        from fastapi.responses import Response
        rows = memory.approved()
        content = '\n'.join(json.dumps({"source": r["source"], "target": r["target"], "context": r["context"], "language": r["language"]}, ensure_ascii=False) for r in rows)
        return Response(content, media_type="application/x-ndjson", headers={"Content-Disposition": 'attachment; filename="forma-approved.jsonl"'})

    @app.post("/api/learning/download", status_code=202)
    def download_student():
        with learner.status_lock:
            if learner.state["phase"] in {"downloading", "training", "evaluating"}:
                raise HTTPException(409, "Bir model işlemi zaten çalışıyor.")
            if not learner.installed:
                raise HTTPException(422, "Yerel bağımlılıkları kur: uv sync --frozen --extra local")
            learner.update(phase="downloading", message="Model indirme işlemi başlatılıyor")
            def prepare_and_train():
                learner.prepare()
                maybe_train()
            app.state.learning_executor.submit(prepare_and_train)
        return learner.status()

    @app.post("/api/learning/train", status_code=202)
    def train_student(body: TrainingRequest):
        with learner.status_lock:
            if learner.state["phase"] in {"downloading", "training", "evaluating"}:
                raise HTTPException(409, "Bir model işlemi zaten çalışıyor.")
            split_corpus(memory.approved())
            if not learner.ready or not learner.installed:
                raise HTTPException(422, "Eğitimden önce yerel temel modeli indir.")
            learner.update(phase="evaluating", last_train_count=memory.stats()["approved"], message="Eğitim hazırlanıyor")
            app.state.learning_executor.submit(learner.train, body.steps)
        return learner.status()

    @app.get("/api/documents")
    def documents():
        with store.lock:
            return {"documents": sorted([dict(v) for v in store.documents.values()], key=lambda d: d["created_at"], reverse=True)}

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        id_ = uuid.uuid4().hex
        directory = store.root / id_
        directory.mkdir(mode=0o700)
        source = directory / "source.pdf"
        total = 0
        try:
            with source.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "Dosya en fazla 25 MB olabilir.")
                    handle.write(chunk)
            if total == 0 or not source.read_bytes()[:1024].lstrip().startswith(b"%PDF-"):
                raise DocumentError("Geçerli bir PDF dosyası seçin.")
            document = await run_in_threadpool(inspect_pdf, source)
            await run_in_threadpool(preview, source, directory / "source.png")
            filename = (file.filename or "belge.pdf").replace("\\", "/").split("/")[-1]
            filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)[:180] or "belge.pdf"
            info = {"id": id_, "filename": filename, "pages": document.pages,
                    "characters": document.characters, "segments": len(document.runs), "size": total,
                    "status": "uploaded", "progress": 0, "message": "Çeviriye hazır",
                    "created_at": datetime.now(timezone.utc).isoformat(), "report": None}
            with store.lock:
                store.documents[id_] = info
                store.update(id_)
            return info
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            with store.lock:
                store.documents.pop(id_, None)
            raise
        finally:
            await file.close()

    def run_job(id_: str, body: TranslationRequest, key: str):
        directory = store.root / id_
        gateway = None
        try:
            document = inspect_pdf(directory / "source.pdf")
            if body.mode == "offline":
                fallback = LocalGateway(learner, body.source_language) if learner.ready else None
            else:
                fallback = gateway_factory(body.provider, body.model, key, base_url=body.base_url,
                    protocol=body.protocol, max_tokens=body.max_tokens, json_mode=body.json_mode)
            gateway = AdaptiveGateway(memory, fallback, body.source_language, use_memory=body.mode != "cloud")

            def progress(stage, amount, message):
                store.update(id_, status=stage, progress=amount, message=message)

            report = execute(directory / "source.pdf", directory / "translated.pdf", document,
                             gateway, progress, body.provider if body.mode != "offline" else "offline",
                             body.model if body.mode != "offline" else "Qwen3 + onaylı bellek")
            report["source_language"], report["mode"] = body.source_language, body.mode
            if body.collect_examples:
                for detail in report["details"]:
                    if detail["status"] == "pass" and detail["rendered"] and detail["origin"] == "api":
                        memory.put(detail["source"], detail["translation"], detail["context"],
                                   body.source_language, document_id=id_)
            preview(directory / "translated.pdf", directory / "translated.png")
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {k: v for k, v in report.items() if k not in {"details", "limitations"}}
            store.update(id_, status=report["status"], progress=100, report=summary,
                         message="Kontrol tamamlandı" if report["status"] == "completed" else "Tamamlandı · İncelemeniz gereken noktalar var")
        except (TranslationError, DocumentError) as exc:
            store.update(id_, status="failed", progress=0, message=str(exc), report=None)
        except Exception:
            # Never echo SDK responses, headers, keys or document content into errors/logs.
            store.update(id_, status="failed", progress=0, message="Belge işlenirken beklenmeyen bir hata oluştu. Başka bir modelle veya daha küçük PDF ile yeniden deneyin.", report=None)
        finally:
            if gateway is not None:
                gateway.close()

    @app.get("/api/documents/{id_}")
    def document(id_: str):
        return store.get(id_)

    @app.post("/api/documents/{id_}/translate", status_code=202)
    def translate(id_: str, body: TranslationRequest):
        document = store.get(id_)
        pdf = inspect_pdf(store.root / id_ / "source.pdf")
        unknown = sum(not memory.lookup(r.text, " ".join(x.text for x in b.runs), body.source_language)
                      for b in pdf.blocks for r in b.runs) if body.mode != "cloud" else len(pdf.runs)
        if body.mode != "offline" and unknown and not body.consent:
            raise HTTPException(422, "Belgenin seçtiğiniz AI sağlayıcısına gönderilmesini onaylayın.")
        key = connection_key(body)
        if body.mode != "offline":
            validate_base_url(body.base_url or PROVIDERS.get(body.provider, {}).get("url", ""))
        if not key and body.mode != "offline" and unknown and body.provider not in {"custom", "ollama", "lmstudio"}:
            raise HTTPException(422, "Seçtiğiniz sağlayıcının API anahtarını Bağlantı ayarlarına girin.")
        if len(key) > 4096:
            raise HTTPException(422, "API anahtarı çok uzun.")
        if body.mode == "offline" and unknown and not learner.ready:
            raise HTTPException(422, f"{unknown} metin bellekte yok. Yeni metinleri API’siz çevirmek için yerel modeli indir.")
        with store.lock:
            document = store.get(id_)
            if document["status"] in ACTIVE:
                raise HTTPException(409, "Bu belge zaten işleniyor.")
            if sum(d["status"] in ACTIVE for d in store.documents.values()) >= 8:
                raise HTTPException(429, "Kuyruk dolu. Çalışan belgeler tamamlandıktan sonra yeniden deneyin.")
            store.update(id_, status="queued", progress=1, report=None,
                         provider=body.provider, model=body.model, message="İşlem sırasında")
            # Do not keep an extra copy of secrets in job configuration.
            clean = body.model_copy(update={"api_key": None})
            if unknown == 0 and body.mode != "cloud":
                clean = clean.model_copy(update={"mode": "offline"})
            app.state.executor.submit(run_job, id_, clean, key)
        return store.get(id_)

    @app.post("/api/documents/{id_}/learn")
    def learn_document(id_: str, body: ApprovalRequest):
        document = store.get(id_)
        if document["status"] not in {"completed", "needs_review"}:
            raise HTTPException(409, "Belge tamamlanmadan belleğe eklenemez.")
        if not body.confirm:
            raise HTTPException(422, "Çevirileri inceleyip onayladığını belirt.")
        report = json.loads((store.root / id_ / "report.json").read_text())
        selected = [d for d in report["details"] if body.ids is None or d["id"] in body.ids]
        eligible = [d for d in selected if d.get("rendered") and protected_tokens(d["source"]) == protected_tokens(d["translation"])]
        if not eligible:
            raise HTTPException(422, "Onaylanabilir metin yok. Sayı veya yerleşim sorunlarını önce çöz.")
        for detail in eligible:
            memory.put(detail["source"], detail["translation"], detail.get("context", detail["source"]),
                       report.get("source_language", "auto"), "approved", id_)
        maybe_train()
        return {"approved": len(eligible), "skipped": len(selected) - len(eligible), "stats": memory.stats()}

    @app.delete("/api/documents/{id_}", status_code=204)
    def delete(id_: str):
        with store.lock:
            document = store.get(id_)
            if document["status"] in ACTIVE:
                raise HTTPException(409, "İşlenen belge silinemez. İşlemin tamamlanmasını bekleyin.")
            shutil.rmtree(store.root / id_)
            del store.documents[id_]

    @app.get("/api/documents/{id_}/preview/{kind}")
    def get_preview(id_: str, kind: Literal["source", "translated"]):
        document = store.get(id_)
        if kind == "translated" and document["status"] not in {"completed", "needs_review"}:
            raise HTTPException(409, "Çıktı henüz hazır değil.")
        path = store.root / id_ / f"{kind}.png"
        if not path.exists():
            raise HTTPException(404, "Önizleme bulunamadı.")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/documents/{id_}/download/{kind}")
    def download(id_: str, kind: Literal["source", "translated", "report"]):
        document = store.get(id_)
        if kind != "source" and document["status"] not in {"completed", "needs_review"}:
            raise HTTPException(409, "Çıktı henüz hazır değil.")
        stem = Path(document["filename"]).stem
        filename = f"{stem}_kontrol.json" if kind == "report" else f"{stem}{'_TR' if kind == 'translated' else ''}.pdf"
        path = store.root / id_ / ("report.json" if kind == "report" else f"{kind}.pdf")
        if not path.is_file():
            raise HTTPException(404, "Dosya bulunamadı.")
        return FileResponse(path, media_type="application/json" if kind == "report" else "application/pdf", filename=filename)

    @app.get("/api/documents/{id_}/report")
    def report(id_: str):
        document = store.get(id_)
        if document["status"] not in {"completed", "needs_review"}:
            raise HTTPException(409, "Kontrol raporu henüz hazır değil.")
        return json.loads((store.root / id_ / "report.json").read_text(encoding="utf-8"))

    app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
    return app


app = create_app()
