from __future__ import annotations

import contextlib
import csv
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pymupdf as fitz
from bs4 import BeautifulSoup, NavigableString, Tag

from .util import digest, stable_id

MAX_INPUT_BYTES = 50 * 1024 * 1024


def _new_segment(segments: list[dict], document_id: str, *, text: str, kind: str, section_path: list[str], locator: str, page: int | None = None, bbox: list[float] | None = None, representation: str = "normalized_text", metadata: dict | None = None) -> None:
    text = text.strip()
    if not text:
        return
    path = list(section_path) or ["Document"]
    segment_id = stable_id("seg", document_id, locator, text)
    segments.append({
        "segment_id": segment_id,
        "document_id": document_id,
        "ordinal": len(segments),
        "kind": kind,
        "section_path": path,
        "section_id": stable_id("sec", document_id, path),
        "locator": locator,
        "page": page,
        "bbox": bbox,
        "representation": representation,
        "text": text,
        "text_sha256": digest(text),
        "metadata": metadata or {},
    })


def _update_heading(headings: list[tuple[int, str]], level: int, title: str) -> list[tuple[int, str]]:
    # A heading closes every open heading at its own or a deeper level; skipped
    # levels are not invented.
    headings = [h for h in headings if h[0] < level]
    headings.append((level, title.strip()))
    return headings


def _path(headings: list[tuple[int, str]]) -> list[str]:
    return [title for _, title in headings]


def _is_footnote(s: str) -> bool:
    return bool(re.match(r"^(?:\[\^[^\]]+\]:|(?:footnote|note)\s+\d+\s*[:.])", s, re.I))


def _parse_text(raw: bytes, document_id: str) -> tuple[list[dict], list[dict], int | None]:
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text files must be UTF-8") from exc
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    has_pages = "\f" in content
    pages = content.split("\f")
    page_count = len(pages) if has_pages else None
    headings: list[tuple[int, str]] = []
    segments: list[dict] = []
    line_base = 1
    for page_idx, page_text in enumerate(pages):
        lines = page_text.splitlines()
        page = page_idx + 1 if has_pages else None
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                i += 1
                continue
            locator = f"page:{page}/line:{line_base + i}" if page else f"line:{line_base + i}"
            heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
            if heading:
                headings = _update_heading(headings, len(heading.group(1)), heading.group(2))
                _new_segment(segments, document_id, text=stripped, kind="heading", section_path=_path(headings), locator=locator, page=page)
                i += 1
                continue
            if "|" in stripped and stripped.count("|") >= 2:
                table_lines = []
                while i < len(lines) and lines[i].strip() and lines[i].count("|") >= 2:
                    table_lines.append(lines[i].strip())
                    i += 1
                _new_segment(segments, document_id, text="\n".join(table_lines), kind="table", section_path=_path(headings), locator=locator, page=page, representation="normalized_markdown_table")
                continue
            if _is_footnote(stripped):
                _new_segment(segments, document_id, text=stripped, kind="footnote", section_path=_path(headings), locator=locator, page=page)
                i += 1
                continue
            para = [stripped]
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or re.match(r"^#{1,6}\s+", nxt) or ("|" in nxt and nxt.count("|") >= 2) or _is_footnote(nxt):
                    break
                para.append(nxt)
                i += 1
            _new_segment(segments, document_id, text="\n".join(para), kind="paragraph", section_path=_path(headings), locator=locator, page=page)
        line_base += len(lines)
    issues: list[dict] = []
    if not segments:
        issues.append({"code": "EMPTY_DOCUMENT", "message": "No text could be parsed.", "page": None})
    return segments, issues, page_count


_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi", re.I)


def _block_fonts(page: Any) -> dict[int, dict]:
    """Dominant font size and bold share of each text block, keyed by block number."""
    fonts: dict[int, dict] = {}
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        sizes: Counter = Counter()
        chars = bold = 0
        for line in block["lines"]:
            for span in line["spans"]:
                n = len(span["text"].strip())
                if not n:
                    continue
                chars += n
                sizes[round(float(span["size"]) * 2) / 2] += n
                if span["flags"] & 16 or _BOLD_FONT.search(span["font"]):
                    bold += n
        if chars:
            fonts[block["number"]] = {"bbox": [round(float(x), 1) for x in block["bbox"]], "size": sizes.most_common(1)[0][0], "sizes": sizes, "bold": bold / chars}
    return fonts


def _furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", " ".join(text.split()).casefold())


def _in_margin(block: tuple, height: float) -> bool:
    return float(block[3]) <= height * 0.09 or float(block[1]) >= height * 0.91


def _font_heading(text: str, font: dict | None, body_size: float, page_width: float, left_margin: float) -> int | None:
    """Heading level from typography and position, or None.

    Headings are bold or larger than body text, short, left-aligned or centred,
    and not sentences, signatures, captions or rows of figures.
    """
    clean = " ".join(text.split())
    if not font or len(clean) > 120 or len(clean.split()) > 14 or sum(c.isalpha() for c in clean) < 4:
        return None
    if re.search(r"[.,;:]$", clean) or clean.casefold().startswith("/s/"):
        return None
    if clean.startswith("(") and not re.match(r"^\(\d+\)\s+[A-Z]", clean):
        return None
    if len(re.findall(r"\d[\d,.]*", clean)) >= 2:
        return None
    large = font["size"] >= body_size + 1.5
    if not large and not (font["bold"] >= 0.9 and font["size"] >= body_size - 0.5):
        return None
    x0, _, x1, _ = font["bbox"]
    if x0 > left_margin + page_width * 0.04 and abs((x0 + x1) / 2 - page_width / 2) > page_width * 0.06:
        return None
    if re.match(r"^PART\s+[IVX]+\b", clean, re.I):
        return 1
    if re.match(r"^ITEM\s+\d+[A-C]?\b", clean, re.I):
        return 2
    level = 3 if large else 4
    numbered = re.match(r"^\d+(?:\.\d+)+", clean)
    if numbered:
        level += numbered.group().count(".")
    return min(level, 6)


def _pdf_heading(text: str) -> tuple[int, str] | None:
    """Fallback for PDFs whose fonts carry no size or weight distinctions."""
    clean = " ".join(text.split())
    if len(clean) > 100 or len(clean.split()) > 12:
        return None
    if re.match(r"^\d+(?:\.\d+){0,3}\.?\s+[A-Z]", clean):
        return min(6, clean.split()[0].count(".") + 1), clean
    letters = [c for c in clean if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        return 1, clean
    return None


def _table_text(rows: list[list[Any]]) -> str:
    result = []
    for row in rows:
        cells = [" ".join(str(cell or "").split()) for cell in row]
        if any(cells):
            result.append("| " + " | ".join(cells) + " |")
    return "\n".join(result)


def _layout_tables(page: Any, text_blocks: list[tuple], existing_tables: list[Any]) -> list[dict]:
    """Conservative fallback for borderless, aligned annual-report tables.

    The reconstructed row strings are NOT raw PDF quotes. Never infer missing
    columns; mark every resulting table for visual review.
    """
    if existing_tables:
        return []
    blocks = sorted(text_blocks, key=lambda b: (float(b[1]), float(b[0])))
    found: list[dict] = []
    used_until = -1
    for index, head in enumerate(blocks):
        if index <= used_until:
            continue
        title = " ".join(str(head[4]).split())
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", title))
        if len(title) > 200 or len(years) < 2 or float(head[3] - head[1]) > 32:
            continue
        previous_bottom = float(head[3])
        rows: list[tuple] = []
        nonrows = 0
        last_index = index
        for j in range(index + 1, min(len(blocks), index + 90)):
            block = blocks[j]
            gap = float(block[1]) - previous_bottom
            if gap > 34:
                break
            text = " ".join(str(block[4]).split())
            numbers = re.findall(r"(?<![\w.])\(?-?\d[\d,]*(?:\.\d+)?\)?(?![\w.])", text)
            width = float(block[2] - block[0])
            height = float(block[3] - block[1])
            row_like = bool(re.search(r"[A-Za-z]", text)) and len(text) <= 220 and len(numbers) >= 2 and width >= float(page.rect.width) * 0.55 and height <= 24
            if row_like:
                rows.append(block)
                nonrows = 0
                last_index = j
            else:
                nonrows += 1
                if nonrows > 2:
                    break
            previous_bottom = max(previous_bottom, float(block[3]))
        if len(rows) < 3:
            continue
        box = [min(float(b[0]) for b in [head] + rows), min(float(b[1]) for b in [head] + rows), max(float(b[2]) for b in [head] + rows), max(float(b[3]) for b in [head] + rows)]
        found.append({"text": "\n".join([title] + [" ".join(str(b[4]).split()) for b in rows]), "bbox": [round(x, 2) for x in box], "block_numbers": [b[5] for b in [head] + rows]})
        used_until = last_index
    return found


def _parse_pdf(raw: bytes, document_id: str) -> tuple[list[dict], list[dict], int, list[int]]:
    segments: list[dict] = []
    issues: list[dict] = []
    blank_pages: list[int] = []
    headings: list[tuple[int, str]] = []
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise ValueError("PDF could not be opened") from exc
    try:
        if doc.needs_pass:
            raise ValueError("Encrypted PDFs are not supported; supply a decrypted copy")
        page_count = len(doc)
        # First pass: document-wide typography and repeated page headers/footers,
        # which a single page cannot reveal.
        pages: list[tuple[list[tuple], dict[int, dict]]] = []
        body_sizes: Counter = Counter()
        margin_pages: Counter = Counter()
        for page in doc:
            blocks = page.get_text("blocks", sort=True)
            text_blocks = [b for b in blocks if len(b) > 6 and b[6] == 0 and str(b[4]).strip()]
            try:
                fonts = _block_fonts(page)
            except Exception:
                fonts = {}
            # Use font data only where the two extractions describe the same block.
            fonts = {b[5]: fonts[b[5]] for b in text_blocks if b[5] in fonts and fonts[b[5]]["bbox"] == [round(float(x), 1) for x in b[:4]]}
            for font in fonts.values():
                body_sizes.update(font["sizes"])
            margin_pages.update({_furniture_key(str(b[4])) for b in text_blocks if _in_margin(b, float(page.rect.height))})
            pages.append((text_blocks, fonts))
        furniture = {key for key, count in margin_pages.items() if count >= 3}
        body_size = body_sizes.most_common(1)[0][0] if body_sizes else 0.0
        typographic = body_size > 0 and any(f["size"] >= body_size + 1.5 or f["bold"] >= 0.9 for _, fonts in pages for f in fonts.values())
        seen_furniture: set[str] = set()
        for index, page in enumerate(doc):
            page_no = index + 1
            text_blocks, fonts = pages[index]
            text_length = sum(len(str(b[4]).strip()) for b in text_blocks)
            try:
                images = page.get_image_info()
                if images:
                    issues.append({"code": "PDF_IMAGES_NOT_OCR", "message": "Page contains image content that was not OCR'd, including potentially material text or a name.", "page": page_no, "image_count": len(images)})
            except Exception:
                images = None
                issues.append({"code": "PDF_IMAGE_INSPECTION_FAILED", "message": "Could not inspect page images for missing text.", "page": page_no})
            if text_length < 20:
                try:
                    structurally_blank = text_length == 0 and images == [] and not page.get_drawings() and not list(page.annots() or [])
                except Exception:
                    structurally_blank = False
                if structurally_blank:
                    blank_pages.append(page_no)
                else:
                    issues.append({"code": "PDF_PAGE_TEXT_INSUFFICIENT", "message": "Page has little or no extractable text and nonblank or unknown content; manual review or OCR is required.", "page": page_no})
            events = []
            for b in text_blocks:
                events.append((float(b[1]), float(b[0]), "block", b))
            tables: list[Any] = []
            inferred: list[dict] = []
            try:
                # PyMuPDF's table diagnostics should not corrupt CLI JSON output.
                with contextlib.redirect_stdout(io.StringIO()):
                    tables = list(page.find_tables().tables)
                for ti, table in enumerate(tables):
                    events.append((float(table.bbox[1]), float(table.bbox[0]), "table", (ti, table)))
                inferred = _layout_tables(page, text_blocks, tables)
                for ti, table in enumerate(inferred):
                    events.append((float(table["bbox"][1]), float(table["bbox"][0]), "layout_table", (ti, table)))
                if inferred:
                    issues.append({"code": "PDF_LAYOUT_TABLE_NEEDS_REVIEW", "message": "A borderless table was inferred from text positions; verify its rows, columns, units, and header against the original page.", "page": page_no, "table_count": len(inferred)})
            except Exception:
                issues.append({"code": "PDF_TABLE_INSPECTION_FAILED", "message": "Table detection failed; raw text remains available but table context needs review.", "page": page_no})
            events.sort(key=lambda e: (e[0], e[1], 0 if e[2] == "table" else 1))
            table_boxes = [[float(x) for x in t.bbox] for t in tables] + [t["bbox"] for t in inferred]
            prose = [float(b[0]) for b in text_blocks if len(str(b[4]).strip()) >= 40]
            left_margin = min(prose or [float(b[0]) for b in text_blocks] or [0.0])
            for _, _, typ, item in events:
                if typ == "block":
                    b = item
                    # Normalize PDF line wrapping for stable quote matching. Offsets remain
                    # explicitly relative to this normalized block, not source file bytes.
                    text = " ".join(str(b[4]).split())
                    key = _furniture_key(text)
                    is_furniture = _in_margin(b, float(page.rect.height)) and key in furniture
                    # A section that starts at the top of a page is later repeated as its
                    # running header: only the first occurrence may open a heading.
                    may_open = not is_furniture or key not in seen_furniture
                    if is_furniture:
                        seen_furniture.add(key)
                    cx, cy = (float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2
                    in_table = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in table_boxes)
                    level = None
                    if may_open and not in_table:
                        if typographic:
                            level = _font_heading(text, fonts.get(b[5]), body_size, float(page.rect.width), left_margin)
                        else:
                            fallback = _pdf_heading(text)
                            level = fallback[0] if fallback else None
                    if level:
                        headings = _update_heading(headings, level, text)
                    kind = "heading" if level else "page_furniture" if is_furniture else "footnote" if _is_footnote(text) else "paragraph"
                    _new_segment(segments, document_id, text=text, kind=kind, section_path=_path(headings), locator=f"page:{page_no}/block:{b[5]}", page=page_no, bbox=[round(float(x), 2) for x in b[:4]], representation="pymupdf_whitespace_normalized_text_block")
                elif typ == "table":
                    ti, table = item
                    text = _table_text(table.extract())
                    _new_segment(segments, document_id, text=text, kind="table", section_path=_path(headings), locator=f"page:{page_no}/table:{ti}", page=page_no, bbox=[round(float(x), 2) for x in table.bbox], representation="reconstructed_pdf_table", metadata={"note": "Table cell text and separators are reconstructed; visually verify against the source PDF."})
                else:
                    ti, table = item
                    _new_segment(segments, document_id, text=table["text"], kind="table", section_path=_path(headings), locator=f"page:{page_no}/inferred-layout-table:{ti}", page=page_no, bbox=table["bbox"], representation="reconstructed_pdf_layout_table", metadata={"source_block_numbers": table["block_numbers"], "note": "Rows were reconstructed from PDF text-block geometry; visually verify labels, period alignment, units and values."})
    finally:
        doc.close()
    if not segments:
        issues.append({"code": "EMPTY_DOCUMENT", "message": "No extractable PDF text was found.", "page": None})
    return segments, issues, page_count, blank_pages


def _html_table(tag: Tag) -> str:
    rows = []
    for tr in tag.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"], recursive=False)]
        if cells:
            rows.append(cells)
    return _table_text(rows)


def _parse_html(raw: bytes, document_id: str) -> tuple[list[dict], list[dict], None]:
    soup = BeautifulSoup(raw, "html.parser")
    scripts_present = any(tag.get("src") or tag.get_text(strip=True) for tag in soup.find_all("script"))
    embedded_present = bool(soup.find(["iframe", "object", "embed"]))
    images_present = bool(soup.find("img"))
    for element in soup.find_all(["script", "style", "template"]):
        element.decompose()
    # Keep noscript fallback text as data; scripts are never executed.
    for element in soup.find_all("noscript"):
        element.unwrap()
    root = soup.body or soup
    selected = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "blockquote", "pre"}
    containers = {"div", "section", "article"}
    def is_unit(tag: Tag) -> bool:
        if tag.name in selected:
            return True
        return bool(tag.name in containers and not tag.find(list(selected | containers)))
    segments: list[dict] = []
    headings: list[tuple[int, str]] = []
    node_no = 0
    orphan_text_count = 0
    merged_table_count = 0
    for node in root.descendants:
        if isinstance(node, Tag):
            if not is_unit(node) or any(isinstance(p, Tag) and is_unit(p) for p in node.parents if p is not root):
                continue
            node_no += 1
            locator = f"html-id:{node.get('id')}/node:{node_no}" if node.get("id") else f"html-node:{node_no}"
            text = _html_table(node) if node.name == "table" else node.get_text(" ", strip=True)
            if node.name and re.fullmatch(r"h[1-6]", node.name):
                headings = _update_heading(headings, int(node.name[1]), text)
                kind = "heading"
            elif node.name == "table":
                kind = "table"
                if any(cell.get("rowspan", "1") != "1" or cell.get("colspan", "1") != "1" for cell in node.find_all(["th", "td"])):
                    merged_table_count += 1
            else:
                kind = "footnote" if _is_footnote(text) or str(node.get("id", "")).lower().startswith(("fn", "footnote")) else "paragraph"
            links = [a.get("href") for a in node.find_all("a", href=True) if str(a.get("href")).startswith("#")]
            _new_segment(segments, document_id, text=text, kind=kind, section_path=_path(headings), locator=locator, representation="reconstructed_html_table" if kind == "table" else "normalized_html_text", metadata={"internal_links": links})
        elif isinstance(node, NavigableString):
            if any(isinstance(p, Tag) and is_unit(p) for p in node.parents if p is not root):
                continue
            text = str(node).strip()
            if text:
                node_no += 1
                orphan_text_count += 1
                _new_segment(segments, document_id, text=text, kind="paragraph", section_path=_path(headings), locator=f"html-text:{node_no}", representation="normalized_html_text")
    issues = [] if segments else [{"code": "EMPTY_DOCUMENT", "message": "No visible static HTML content was found; JavaScript is not executed.", "page": None}]
    if scripts_present:
        issues.append({"code": "HTML_DYNAMIC_CONTENT_NOT_RENDERED", "message": "HTML contains scripts that were not executed; dynamically rendered content may be missing.", "page": None})
    if embedded_present:
        issues.append({"code": "HTML_EMBEDDED_CONTENT_NOT_LOADED", "message": "Embedded HTML content was not fetched; review the original page and embedded sources.", "page": None})
    if images_present:
        issues.append({"code": "HTML_IMAGES_NOT_OCR", "message": "HTML images were not fetched or OCR'd; they may contain material evidence.", "page": None})
    if orphan_text_count:
        issues.append({"code": "HTML_UNSCOPED_TEXT", "message": "Some visible HTML text fell outside recognized structural blocks and may be fragmented; review the original page.", "page": None, "count": orphan_text_count})
    if merged_table_count:
        issues.append({"code": "HTML_MERGED_TABLE_CELLS", "message": "Merged HTML table cells were not expanded; verify table headers and row relationships manually.", "page": None, "count": merged_table_count})
    return segments, issues, None


def _parse_csv(raw: bytes, document_id: str) -> tuple[list[dict], list[dict], None]:
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV files must be UTF-8") from exc
    reader = csv.reader(io.StringIO(content, newline=""))
    rows = list(reader)
    segments: list[dict] = []
    if rows:
        header = rows[0]
        for i, row in enumerate(rows[1:], 2):
            table_text = _table_text([header, row])
            _new_segment(segments, document_id, text=table_text, kind="table", section_path=["Records"], locator=f"csv-row:{i}", representation="reconstructed_csv_record", metadata={"row_number": i, "note": "Headers are repeated as context; separators are reconstructed."})
    issues = [] if segments else [{"code": "EMPTY_DOCUMENT", "message": "CSV has no data rows.", "page": None}]
    return segments, issues, None


def parse_file(path: str, layout: Callable[[bytes, str], tuple[dict, dict]] | None = None) -> tuple[dict, list[dict], list[dict]]:
    """Parse with the native parsers, or with ``layout`` (raw bytes, sha256) -> (analyzeResult, parser metadata)."""
    p = Path(path)
    if not p.is_file():
        raise ValueError("Input document does not exist or is not a regular file")
    if p.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Input exceeds the 50 MiB proof-of-concept limit")
    raw = p.read_bytes()
    sha = digest(raw)
    document_id = "doc_" + sha[:24]
    suffix = p.suffix.lower()
    blank_pages: list[int] = []
    parser: dict = {"name": "native"}
    if layout is not None:
        from .azure_di import SUPPORTED_SUFFIXES, segments_from_layout
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("Unsupported format for Azure Document Intelligence. Use PDF, an image (PNG/JPEG/TIFF/BMP/HEIF), DOCX, XLSX, PPTX or HTML")
        result, parser = layout(raw, sha)
        segments, issues, page_count = segments_from_layout(result, document_id)
    elif suffix in {".txt", ".md", ".markdown"}:
        segments, issues, page_count = _parse_text(raw, document_id)
    elif suffix in {".html", ".htm"}:
        segments, issues, page_count = _parse_html(raw, document_id)
    elif suffix == ".csv":
        segments, issues, page_count = _parse_csv(raw, document_id)
    elif suffix == ".pdf":
        segments, issues, page_count, blank_pages = _parse_pdf(raw, document_id)
    else:
        raise ValueError("Unsupported format. Use UTF-8 TXT/Markdown, static HTML, CSV, or a text-based PDF")
    document = {"document_id": document_id, "filename": p.name, "sha256": sha, "bytes": len(raw), "format": suffix.lstrip("."), "page_count": page_count, "structurally_blank_pdf_pages": blank_pages, "parser": parser}
    return document, segments, issues
