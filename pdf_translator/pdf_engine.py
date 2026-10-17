import hashlib
import html
import math
import re
import threading
import unicodedata
from functools import wraps
from pathlib import Path

import pymupdf as fitz

from .domain import Block, Document, DocumentError, Issue, Run

MAX_PAGES = 60
MAX_CHARACTERS = 120_000
PDF_LOCK = threading.RLock()  # MuPDF calls must not run concurrently in threads.


def serialized_pdf(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with PDF_LOCK:
            return function(*args, **kwargs)
    return wrapped


@serialized_pdf
def inspect_pdf(path: Path) -> Document:
    try:
        with fitz.open(path) as doc:
            if not doc.is_pdf or doc.needs_pass:
                raise DocumentError("Şifreli veya PDF olmayan dosyalar desteklenmiyor. Şifreyi kaldırıp yeniden yükleyin.")
            if not 1 <= doc.page_count <= MAX_PAGES:
                raise DocumentError(f"Belge 1–{MAX_PAGES} sayfa arasında olmalı.")
            blocks = []
            characters = 0
            image_only = []
            for page_number, page in enumerate(doc):
                page_blocks = []
                data = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
                for index, block in enumerate(data["blocks"]):
                    if block.get("type") != 0:
                        continue
                    block_id = f"p{page_number + 1}-b{index}"
                    runs = []
                    directions = []
                    for line in block.get("lines", []):
                        directions.append(line.get("dir", (1, 0)))
                        first_in_line = True
                        previous_end = None
                        for span in line.get("spans", []):
                            text = span["text"]
                            if not text.strip():
                                continue
                            style = (span["font"], round(span["size"], 2), span["color"], span["flags"])
                            separator = ""
                            if runs and not runs[-1].text.endswith(" ") and not text.startswith(" "):
                                if first_in_line:
                                    separator = "" if runs[-1].text.endswith("-") else " "
                                elif previous_end is not None and span["bbox"][0] - previous_end > span["size"] * 0.15:
                                    separator = " "
                            if runs and style == (runs[-1].font, runs[-1].size, runs[-1].color, runs[-1].flags):
                                runs[-1].text += separator + text
                            else:
                                runs.append(Run(f"{block_id}-r{len(runs)}", text, *style, join_before=separator))
                            first_in_line = False
                            previous_end = span["bbox"][2]
                    if not runs:
                        continue
                    angle = (round(math.degrees(math.atan2(-directions[0][1], directions[0][0]))) % 360)
                    supported = angle in (0, 90, 180, 270) and all(
                        abs(d[0] - directions[0][0]) < 0.01 and abs(d[1] - directions[0][1]) < 0.01
                        for d in directions
                    )
                    bbox = tuple(block["bbox"])
                    if not all(math.isfinite(v) for v in bbox) or fitz.Rect(bbox).is_empty:
                        raise DocumentError("PDF geçersiz metin koordinatları içeriyor.")
                    line_boxes = [fitz.Rect(line["bbox"]) for line in block["lines"] if line.get("spans")]
                    alignment = "left"
                    if len(line_boxes) > 1 and angle == 0:
                        starts = [r.x0 for r in line_boxes]
                        ends = [r.x1 for r in line_boxes]
                        centers = [(r.x0 + r.x1) / 2 for r in line_boxes]
                        if max(starts) - min(starts) > 3:
                            if max(centers) - min(centers) < 2:
                                alignment = "center"
                            elif max(ends) - min(ends) < 2:
                                alignment = "right"
                    page_blocks.append(Block(block_id, page_number, bbox, runs, angle, supported, alignment))
                    characters += sum(len(run.text) for run in runs)
                if not page_blocks and page.get_images():
                    image_only.append(page_number + 1)
                blocks.extend(page_blocks)
                if characters > MAX_CHARACTERS:
                    raise DocumentError(f"Belge {MAX_CHARACTERS:,} karakter sınırını aşıyor. Daha küçük parçalara bölün.")
            if image_only:
                pages = ", ".join(map(str, image_only[:8]))
                raise DocumentError(f"{pages}. sayfada yalnızca görsel var; OCR gerekiyor. Bu sürüm metni seçilebilen PDF’leri destekler.")
            if not blocks:
                raise DocumentError("PDF içinde çevrilebilir metin bulunamadı. Taranmış belgeler için önce OCR uygulayın.")
            return Document(doc.page_count, blocks, characters)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("PDF okunamadı. Dosya bozuk veya desteklenmeyen bir yapıda olabilir.") from exc


@serialized_pdf
def preview(path: Path, output: Path):
    with fitz.open(path) as doc:
        page = doc[0]
        scale = min(1.25, 900 / max(page.rect.width, page.rect.height))
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(output)


def clean_font_name(name: str) -> str:
    name = re.sub(r"^[A-Z]{6}\+", "", name)
    return re.sub(r"[^a-z0-9]", "", name.lower())


BASE_FONTS = {
    "helvetica": "helv", "helvetica-bold": "hebo",
    "helvetica-oblique": "heit", "helvetica-boldoblique": "hebi",
    "times-roman": "tiro", "times-bold": "tibo",
    "times-italic": "tiit", "times-bolditalic": "tibi",
    "courier": "cour", "courier-bold": "cobo",
    "courier-oblique": "coit", "courier-boldoblique": "cobi",
}
BASE_FONTS = {clean_font_name(name): value for name, value in BASE_FONTS.items()}


class FontResolver:
    def __init__(self, doc, directory: Path):
        self.doc = doc
        self.directory = directory
        directory.mkdir(exist_ok=True)
        self.cache = {}
        self.styles = {}

    def resolve(self, page: int, run: Run, text: str):
        key = (page, run.font)
        if key not in self.cache:
            font = None
            target = clean_font_name(run.font)
            for info in self.doc.get_page_fonts(page, full=True):
                if clean_font_name(info[3]) == target:
                    try:
                        buffer = self.doc.extract_font(info[0])[3]
                        if buffer:
                            font = fitz.Font(fontbuffer=buffer)
                    except Exception:
                        pass
                    break
            if font is None and target in BASE_FONTS:
                font = fitz.Font(BASE_FONTS[target])
            self.cache[key] = font
        font = self.cache[key]
        fallback = font is None or not all(
            c.isspace() or font.has_glyph(ord(c), fallback=False) for c in text
        )
        if fallback:
            # Noto includes Turkish. Match the original bold/italic style.
            bold, italic = bool(run.flags & 16), bool(run.flags & 2)
            family = "notosbi" if bold and italic else "notosbo" if bold else "notosit" if italic else "notos"
            font = fitz.Font(family)
        digest = hashlib.sha256(font.buffer).hexdigest()[:16]
        name = "f" + digest
        if name not in self.styles:
            (self.directory / f"{name}.ttf").write_bytes(font.buffer)
            self.styles[name] = f"@font-face {{font-family:{name};src:url('{name}.ttf');}}"
        return name, fallback


@serialized_pdf
def render_pdf(source: Path, output: Path, document: Document, translations: dict[str, str],
               font_dir: Path) -> list[Issue]:
    issues = []
    expected_text = {}
    with fitz.open(source) as doc, fitz.open() as probe:
        resolver = FontResolver(doc, font_dir)
        for page_number, page in enumerate(doc):
            planned = []
            # Test placement BEFORE removing any original text.
            scratch = probe.new_page(width=page.mediabox.width, height=page.mediabox.height)
            for block in [b for b in document.blocks if b.page == page_number]:
                if not block.supported:
                    issues.append(Issue("unsupported_rotation", "Eğik veya karışık yönlü metin kaynak dilinde bırakıldı.", page_number + 1, block.id))
                    continue
                parts = []
                used_fallback = False
                for run in block.runs:
                    text = translations[run.id]
                    font, fallback = resolver.resolve(page_number, run, text)
                    used_fallback |= fallback
                    parts.append(html.escape(run.join_before) + f'<span style="font-family:{font};font-size:{run.size}pt;color:#{run.color:06x}">{html.escape(text)}</span>')
                markup = '<div>' + ''.join(parts) + '</div>'
                css = "body,div{margin:0;padding:0;}div{line-height:1.05;text-align:" + block.alignment + ";}" + "".join(resolver.styles.values())
                rect = fitz.Rect(block.bbox)
                kwargs = dict(css=css, archive=fitz.Archive(str(font_dir)), scale_low=0.65, rotate=block.rotation)
                spare, scale = scratch.insert_htmlbox(rect, markup, **kwargs)
                if spare < 0:
                    issues.append(Issue("text_did_not_fit", "Türkçe metin alana sığmadı; bu bölüm kaynak dilinde bırakıldı.", page_number + 1, block.id))
                    continue
                planned.append((rect, markup, kwargs, block.id))
                expected_text.setdefault(page_number, []).append(''.join(run.join_before + translations[run.id] for run in block.runs))
                if used_fallback:
                    issues.append(Issue("font_substituted", "Orijinal font kullanılamadı veya Türkçe karakter içermedi; Noto Sans kullanıldı.", page_number + 1, block.id))
                if scale < 0.90:
                    issues.append(Issue("font_reduced", f"Metni sığdırmak için punto %{round((1 - scale) * 100)} küçültüldü.", page_number + 1, block.id))
            # Transparent redactions remove only text, retaining images and vector backgrounds.
            links = page.get_links()
            for rect, *_ in planned:
                page.add_redact_annot(rect, fill=False, cross_out=False)
            if planned:
                page.apply_redactions(images=0, graphics=0, text=0)
                current_links = {str(link.get("from")) + str(link.get("uri")) + str(link.get("page")) for link in page.get_links()}
                for link in links:
                    signature = str(link.get("from")) + str(link.get("uri")) + str(link.get("page"))
                    if signature not in current_links:
                        page.insert_link({k: v for k, v in link.items() if k not in {"xref", "id"}})
                for rect, markup, kwargs, block_id in planned:
                    spare, _ = page.insert_htmlbox(rect, markup, **kwargs)
                    if spare < 0:
                        raise DocumentError("Doğrulanan metin yerleşimi PDF’e uygulanamadı.")
            probe.delete_page(-1)
        metadata = dict(doc.metadata)
        metadata["subject"] = "Türkçe çeviri · Forma · Kontrol raporuyla birlikte değerlendirin"
        doc.set_metadata(metadata)
        doc.save(output, garbage=4, deflate=True)
    with fitz.open(source) as original, fitz.open(output) as translated:
        if translated.page_count != original.page_count:
            raise DocumentError("Çıktının sayfa sayısı kaynakla eşleşmiyor.")
        for index in range(original.page_count):
            if translated[index].rect != original[index].rect:
                raise DocumentError("Çıktının sayfa ölçüleri kaynakla eşleşmiyor.")
            if original[index].get_text().strip() and not translated[index].get_text().strip():
                raise DocumentError("Çıktıda metin kaybı tespit edildi.")
            def compact(text):
                return ''.join(unicodedata.normalize("NFKC", text).replace("\u00ad", "").split())
            actual = compact(translated[index].get_text())
            if any(compact(text) not in actual for text in expected_text.get(index, [])):
                raise DocumentError(f"{index + 1}. çıktı sayfasında beklenen Türkçe metnin tamamı doğrulanamadı. Eksik çıktı sunmamak için işlem durduruldu.")
    return issues
