"""Azure AI Document Intelligence (prebuilt-layout) parser adapter.

Maps an ``analyzeResult`` onto the same segment shape as the native parsers, so
chunking, extraction, citation verification and coverage are unchanged. Quotes
still have to match the stored normalized text, but that text is now Azure's
OCR/layout output rather than the file's own text layer: an OCR misread becomes
the "source". Low-confidence words and handwriting are therefore raised as
parse issues on their pages.
"""
from __future__ import annotations

import base64
import os
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from typing import Any

import httpx

from .parsing import _new_segment, _path, _table_text, _update_heading
from .util import digest, json_text

MODEL_ID = "prebuilt-layout"
API_VERSION = "2024-11-30"
SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".heif", ".docx", ".xlsx", ".pptx", ".html", ".htm"}
LOW_CONFIDENCE = 0.8

_HEADING_ROLES = {"title", "sectionHeading"}
_FURNITURE_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


class DocumentIntelligenceError(ValueError):
    """Sanitized service failure: never includes document content, keys or response bodies."""


def endpoint_from_env() -> str:
    return os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")


def layout_fingerprint(endpoint_sha256: str | None) -> dict:
    """Everything that can change the layout output for identical source bytes."""
    return {"parser": "azure_di", "model_id": MODEL_ID, "api_version": API_VERSION, "endpoint_sha256": endpoint_sha256}


class AzureLayoutClient:
    def __init__(self, *, allow_remote: bool, endpoint: str | None = None, key: str | None = None, client: httpx.Client | None = None, poll_interval: float = 2.0, timeout: float = 600.0):
        if not allow_remote:
            raise ValueError("Remote processing is disabled. Explicitly pass --allow-remote-sensitive-data after assessing your data-sharing and provider controls.")
        self.endpoint = (endpoint or endpoint_from_env()).rstrip("/")
        self.key = key or os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        if not self.endpoint.startswith("https://"):
            raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT must be an https:// resource endpoint")
        if not self.key:
            raise ValueError("AZURE_DOCUMENT_INTELLIGENCE_KEY is required for the Azure Document Intelligence parser")
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=60.0)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = {"Ocp-Apim-Subscription-Key": self.key}
        for attempt in range(3):
            try:
                response = self.client.request(method, url, headers=headers, **kwargs)
            except httpx.HTTPError:
                if attempt < 2:
                    time.sleep(1.0)
                    continue
                raise DocumentIntelligenceError("Azure Document Intelligence request failed") from None
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                time.sleep(self._retry_after(response, 1.0))
                continue
            return response
        raise DocumentIntelligenceError("Azure Document Intelligence request failed")

    @staticmethod
    def _retry_after(response: httpx.Response, default: float) -> float:
        try:
            return min(30.0, max(0.0, float(response.headers.get("Retry-After", default))))
        except ValueError:
            return default

    def analyze(self, raw: bytes) -> dict:
        url = f"{self.endpoint}/documentintelligence/documentModels/{MODEL_ID}:analyze"
        response = self._request("POST", url, params={"api-version": API_VERSION}, json={"base64Source": base64.b64encode(raw).decode("ascii")})
        if response.status_code != 202:
            raise DocumentIntelligenceError(f"Azure Document Intelligence returned HTTP {response.status_code}")
        location = response.headers.get("Operation-Location", "")
        # The subscription key is sent to the polling URL, so it must stay on the resource endpoint.
        if not location.startswith(self.endpoint + "/"):
            raise DocumentIntelligenceError("Azure Document Intelligence returned an unexpected operation location")
        deadline = time.monotonic() + self.timeout
        while True:
            poll = self._request("GET", location)
            if not poll.is_success:
                raise DocumentIntelligenceError(f"Azure Document Intelligence returned HTTP {poll.status_code} while polling")
            try:
                body = poll.json()
            except ValueError:
                raise DocumentIntelligenceError("Azure Document Intelligence returned invalid JSON") from None
            status = body.get("status") if isinstance(body, dict) else None
            if status == "succeeded":
                result = body.get("analyzeResult")
                if not isinstance(result, dict):
                    raise DocumentIntelligenceError("Azure Document Intelligence returned no analyzeResult")
                return result
            if status not in {"notStarted", "running"}:
                code = (body.get("error") or {}).get("code", "unknown") if isinstance(body, dict) else "unknown"
                raise DocumentIntelligenceError(f"Azure Document Intelligence analysis {status or 'failed'} ({code})")
            if time.monotonic() > deadline:
                raise DocumentIntelligenceError("Azure Document Intelligence analysis timed out")
            time.sleep(self._retry_after(poll, self.poll_interval))


def _element(ref: str) -> tuple[str, int] | None:
    parts = ref.strip("/").split("/")
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None


def _section_depths(sections: list[dict]) -> dict[int, int]:
    """Depth of the section that directly contains each paragraph (root section = 0)."""
    children = {e[1] for s in sections for e in map(_element, s.get("elements", [])) if e and e[0] == "sections"}
    depths: dict[int, int] = {}
    stack = [(i, 0) for i in range(len(sections)) if i not in children]
    seen: set[int] = set()
    while stack:
        index, depth = stack.pop()
        if index in seen or index >= len(sections):
            continue
        seen.add(index)
        for ref in sections[index].get("elements", []):
            e = _element(ref)
            if e and e[0] == "paragraphs":
                depths.setdefault(e[1], depth)
            elif e and e[0] == "sections":
                stack.append((e[1], depth + 1))
    return depths


def _region(item: dict, scales: dict[int, float]) -> tuple[int | None, list[float] | None, list[int]]:
    regions = item.get("boundingRegions") or []
    pages = sorted({r["pageNumber"] for r in regions if "pageNumber" in r})
    if not regions:
        return None, None, pages
    first = regions[0]
    page = first.get("pageNumber")
    poly = first.get("polygon") or []
    if len(poly) < 8:
        return page, None, pages
    scale = scales.get(page, 1.0)
    xs, ys = poly[0::2], poly[1::2]
    return page, [round(min(xs) * scale, 2), round(min(ys) * scale, 2), round(max(xs) * scale, 2), round(max(ys) * scale, 2)], pages


def _offset(item: dict) -> int:
    spans = item.get("spans") or []
    return min((s["offset"] for s in spans), default=-1)


def _within(spans: list[dict], containers: list[tuple[int, int]]) -> bool:
    return bool(spans) and all(any(a <= s["offset"] and s["offset"] + s["length"] <= b for a, b in containers) for s in spans)


def _table(table: dict) -> tuple[str, int, int]:
    """Grid text, nonempty column-header rows, and merged body cells."""
    rows = [["" for _ in range(table.get("columnCount", 0))] for _ in range(table.get("rowCount", 0))]
    header_rows: set[int] = set()
    merged = 0
    for cell in table.get("cells", []):
        r, c = cell.get("rowIndex", 0), cell.get("columnIndex", 0)
        if r < len(rows) and c < len(rows[r]):
            # Merged cells keep their content at the origin; covered positions stay empty.
            rows[r][c] = " ".join(str(cell.get("content", "")).split())
        if cell.get("kind") == "columnHeader":
            header_rows.update(range(r, r + cell.get("rowSpan", 1)))
        elif cell.get("rowSpan", 1) > 1 or cell.get("columnSpan", 1) > 1:
            merged += 1
    kept_headers = sum(1 for r in sorted(header_rows) if r < len(rows) and any(rows[r]))
    return _table_text(rows), kept_headers, merged


def segments_from_layout(result: dict, document_id: str) -> tuple[list[dict], list[dict], int]:
    pages = result.get("pages") or []
    scales = {p["pageNumber"]: 72.0 if p.get("unit") == "inch" else 1.0 for p in pages}
    page_spans = sorted((s["offset"], s["offset"] + s["length"], p["pageNumber"]) for p in pages for s in p.get("spans", []))
    words = sorted((w["span"]["offset"], w.get("confidence", 1.0)) for p in pages for w in p.get("words", []) if "span" in w)
    word_offsets = [w[0] for w in words]

    def page_of(offset: int) -> int | None:
        for start, end, number in page_spans:
            if start <= offset < end:
                return number
        return None

    def min_confidence(spans: list[dict]) -> float | None:
        values = [words[i][1] for s in spans for i in range(bisect_left(word_offsets, s["offset"]), bisect_right(word_offsets, s["offset"] + s["length"] - 1))]
        return round(min(values), 3) if values else None

    paragraphs = result.get("paragraphs") or []
    tables = result.get("tables") or []
    figures = result.get("figures") or []
    depths = _section_depths(result.get("sections") or [])
    table_spans = [(s["offset"], s["offset"] + s["length"]) for t in tables for s in t.get("spans", [])]
    figure_of = {e[1]: f.get("id", str(i)) for i, f in enumerate(figures) for e in map(_element, f.get("elements", [])) if e and e[0] == "paragraphs"}

    items = [(_offset(p), 1, "paragraph", i, p) for i, p in enumerate(paragraphs) if not _within(p.get("spans") or [], table_spans)]
    items += [(_offset(t), 0, "table", i, t) for i, t in enumerate(tables)]
    items.sort(key=lambda x: (x[0], x[1], x[3]))

    segments: list[dict] = []
    headings: list[tuple[int, str]] = []
    low_confidence: Counter = Counter()
    merged_pages: Counter = Counter()
    for _, _, typ, index, item in items:
        page, bbox, item_pages = _region(item, scales)
        confidence = min_confidence(item.get("spans") or [])
        if confidence is not None and confidence < LOW_CONFIDENCE and page is not None:
            low_confidence[page] += 1
        metadata: dict[str, Any] = {"min_word_confidence": confidence}
        if len(item_pages) > 1:
            metadata["pages"] = item_pages
        if typ == "table":
            text, header_rows, merged = _table(item)
            if merged and page is not None:
                merged_pages[page] += merged
            caption = (item.get("caption") or {}).get("content")
            metadata.update({"header_rows": header_rows, "caption": " ".join(caption.split()) if caption else None, "note": "Cell text comes from Azure Document Intelligence; merged cells are placed at their first row and column."})
            _new_segment(segments, document_id, text=text, kind="table", section_path=_path(headings), locator=f"page:{page}/di-table:{index}", page=page, bbox=bbox, representation="azure_di_layout_table", metadata=metadata)
            continue
        text = " ".join(str(item.get("content", "")).split())
        role = item.get("role")
        if role in _HEADING_ROLES and text:
            # Section nesting is Azure's; without a section tree, title outranks sectionHeading.
            level = depths[index] + 1 if index in depths else 1 if role == "title" else 2
            headings = _update_heading(headings, min(level, 6), text)
            kind = "heading"
        elif role in _FURNITURE_ROLES:
            kind = "page_furniture"
        elif role == "footnote":
            kind = "footnote"
        else:
            kind = "paragraph"
        if index in figure_of:
            metadata["figure_id"] = figure_of[index]
        if role:
            metadata["di_role"] = role
        _new_segment(segments, document_id, text=text, kind=kind, section_path=_path(headings), locator=f"page:{page}/di-paragraph:{index}", page=page, bbox=bbox, representation="azure_di_layout_text", metadata=metadata)

    issues: list[dict] = []
    for page in pages:
        if not page.get("words"):
            issues.append({"code": "DI_PAGE_NO_TEXT", "message": "Azure Document Intelligence found no text on this page; check whether it is blank or unreadable.", "page": page["pageNumber"]})
    for page, count in sorted(low_confidence.items()):
        issues.append({"code": "OCR_LOW_CONFIDENCE", "message": f"Some recognized words have confidence below {LOW_CONFIDENCE}; check names, numbers and dates against the original page.", "page": page, "segment_count": count})
    handwritten: dict[int, int] = defaultdict(int)
    for style in result.get("styles") or []:
        if style.get("isHandwritten"):
            for span in style.get("spans", []):
                page = page_of(span["offset"])
                if page is not None:
                    handwritten[page] += 1
    for page, count in sorted(handwritten.items()):
        issues.append({"code": "HANDWRITTEN_CONTENT", "message": "Page contains handwriting; recognized text needs visual verification.", "page": page, "span_count": count})
    for figure in figures:
        page, _, _ = _region(figure, scales)
        issues.append({"code": "DI_FIGURE_NOT_INTERPRETED", "message": "Text inside figures was recognized, but charts, photos and graphics were not interpreted.", "page": page, "figure_id": figure.get("id")})
    for page, count in sorted(merged_pages.items()):
        issues.append({"code": "DI_TABLE_MERGED_CELLS", "message": "A table has merged body cells; verify row and column relationships against the original page.", "page": page, "count": count})
    if not segments:
        issues.append({"code": "EMPTY_DOCUMENT", "message": "Azure Document Intelligence returned no text.", "page": None})
    return segments, issues, len(pages)


def parser_metadata(result: dict, endpoint_sha256: str | None) -> dict:
    return {"name": "azure_di_layout", "model_id": result.get("modelId", MODEL_ID), "api_version": result.get("apiVersion", API_VERSION), "endpoint_sha256": endpoint_sha256, "result_sha256": digest(json_text(result))}
