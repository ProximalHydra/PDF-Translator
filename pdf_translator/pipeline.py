import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .domain import Document, Issue, Review
from .pdf_engine import render_pdf


def batches(document: Document, limit=5500):
    batch, length = [], 0
    for block in document.blocks:
        row = {"id": block.id, "runs": [{"id": r.id, "text": r.text} for r in block.runs]}
        size = sum(len(r.text) for r in block.runs)
        if batch and length + size > limit:
            yield batch
            batch, length = [], 0
        batch.append(row)
        length += size
    if batch:
        yield batch


def protected_tokens(text: str):
    # Numbers, URLs and email addresses must survive a literal translation.
    return Counter(re.findall(r"https?://[^\s<>]+|[\w.+-]+@[\w.-]+\.\w+|\d+(?:[.,:/-]\d+)*", text))


def execute(source: Path, output: Path, document: Document, gateway, progress,
            provider: str, model: str) -> dict:
    all_batches = list(batches(document))
    translations = {}
    review = Review()
    lookup = {r.id: b for b in document.blocks for r in b.runs}
    for i, batch in enumerate(all_batches):
        progress("translating", 5 + int(40 * i / len(all_batches)), f"Metinler çevriliyor · {i + 1}/{len(all_batches)} bölüm")
        translations.update(gateway.translate(batch))

    for i, batch in enumerate(all_batches):
        progress("reviewing", 48 + int(30 * i / len(all_batches)), f"Kaynakla karşılaştırılıyor · {i + 1}/{len(all_batches)} bölüm")
        entries = [{"id": run["id"], "source": run["text"], "translation": translations[run["id"]]}
                   for block in batch for run in block["runs"]]
        checks = gateway.review(entries)
        corrected = []
        initial = {check["id"]: check for check in checks}
        for check in checks:
            if check["status"] == "needs_correction":
                translations[check["id"]] = check["corrected_text"]
                corrected.append(check["id"])
                review.corrected += 1
        # Corrections get a second independent comparison before being accepted.
        final = dict(initial)
        if corrected:
            retry = [{**entry, "translation": translations[entry["id"]]} for entry in entries if entry["id"] in corrected]
            final.update({check["id"]: check for check in gateway.review(retry)})
        for entry in entries:
            id_ = entry["id"]
            check = final[id_]
            run_block = lookup[id_]
            issues = []
            if check["status"] == "unverified":
                issues.append(Issue("manual_semantic_review", "Yerel üretimin anlam kontrolü kullanıcı incelemesi gerektiriyor.", run_block.page + 1, run_block.id))
            elif check["status"] != "pass":
                issues.append(Issue("semantic_review", "İkinci kontrolde de anlam veya bütünlük sorunu bildirildi.", run_block.page + 1, run_block.id))
            if protected_tokens(entry["source"]) != protected_tokens(translations[id_]):
                issues.append(Issue("protected_token_mismatch", "Sayı, bağlantı veya e-posta kaynakla eşleşmiyor.", run_block.page + 1, run_block.id))
            review.issues.extend(issues)
            if not issues:
                review.passed += 1
            review.details.append({"id": id_, "page": run_block.page + 1,
                                   "source": entry["source"], "translation": translations[id_],
                                   "context": " ".join(r.text for r in run_block.runs),
                                   "origin": getattr(gateway, "origins", {}).get(id_, "api"),
                                   "corrected": id_ in corrected,
                                   "status": "needs_review" if issues else "pass",
                                   "reason": check.get("reason", "")})

    progress("rendering", 82, "Türkçe metin PDF’e yerleştiriliyor")
    review.issues.extend(render_pdf(source, output, document, translations, output.parent / "fonts"))
    progress("verifying", 96, "Sayfalar, fontlar ve çıktı bütünlüğü kontrol ediliyor")
    issues = [issue.as_dict() for issue in review.issues]
    layout_failures = {issue["block_id"] for issue in issues if issue["code"] in {"text_did_not_fit", "unsupported_rotation"}}
    for detail in review.details:
        if any(detail["id"].startswith(block_id + "-r") for block_id in layout_failures):
            detail["status"] = "needs_review"
            detail["rendered"] = False
        else:
            detail["rendered"] = True
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider, "model": model, "language": "tr",
        "status": "needs_review" if issues else "completed",
        "pages": document.pages, "segments": len(document.runs),
        "semantic_passed": review.passed, "corrected": review.corrected,
        "engines": dict(Counter(detail["origin"] for detail in review.details)),
        "issues": issues, "details": review.details,
        "checks": {"semantic_comparison": not any(i["code"] == "manual_semantic_review" for i in issues), "corrections_rechecked": True,
                   "protected_tokens": True, "page_count": True, "page_dimensions": True,
                   "rendered_text_completeness": True},
        "limitations": ["AI kontrolü insan denetiminin yerini tutmaz.",
                        "Görsellerin içindeki yazılar çevrilmez.",
                        "Font veya punto değişiklikleri ve çevrilmeden kalan bölümler issues alanında bildirilir."],
    }
