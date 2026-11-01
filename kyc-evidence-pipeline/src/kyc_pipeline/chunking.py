from __future__ import annotations

import re
from collections import defaultdict
from typing import Any
from .util import stable_id


# A sentence ends at . ! or ? (optionally closed by a quote or bracket) before a
# capital or digit, or at a line break. Abbreviations can cause false ends; that
# only moves a split point.
_SENTENCE_END = re.compile(r"[.!?][\"')\]”’]?[ \t]+(?=[\"'(\[“‘]?[A-Z0-9])|\n+")
# A clause starts a line with (a), 1., 2.1, iv), Section 3, Article 4 or Clause 5.
_CLAUSE_START = re.compile(r"(?<=\n)[ \t]*(?:\(?(?:[a-z]|[ivx]{1,4}|\d{1,3})[.)]|\d+(?:\.\d+)+\.?|(?:Section|Article|Clause)\s+\d+)\s", re.I)
_FIELD_LINE = re.compile(r"^\s*[A-Z][\w ()/&.'-]{0,60}:[ \t]+\S")


def _last(pattern: re.Pattern, text: str, low: int, high: int, use_start: bool) -> int | None:
    found = None
    for m in pattern.finditer(text, low):
        position = m.start() if use_start else m.end()
        if position > high:
            break
        if low < position <= high:
            found = position
    return found


def _windows(text: str, maximum: int, overlap: int, boundary: str = "whitespace") -> list[tuple[int, int]]:
    if not text:
        return []
    spans = []
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            low = start + maximum // 2
            cut = _last(_CLAUSE_START, text, low, end, True) if boundary == "clause" else None
            if cut is None and boundary in {"sentence", "clause"}:
                cut = _last(_SENTENCE_END, text, low, end, False)
            if cut is None:
                last_space = text.rfind(" ", low, end)
                last_line = text.rfind("\n", low, end)
                if max(last_space, last_line) > start:
                    cut = max(last_space, last_line) + 1
            end = cut or end
        spans.append((start, end))
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        if boundary != "whitespace":
            # Start the overlap at a sentence so no chunk opens mid-sentence; a
            # sentence longer than the overlap gives no overlap at all.
            starts = [m.end() for m in _SENTENCE_END.finditer(text, next_start, end) if m.end() < end]
            next_start = starts[0] if starts else end
        start = next_start
    return spans


def is_field_record(text: str) -> bool:
    """Two or more 'Label: value' lines making up at least half the lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    fields = sum(bool(_FIELD_LINE.match(line)) for line in lines)
    return fields >= 2 and fields * 2 >= len(lines)


def _excerpt(text: str, limit: int, tail: bool) -> str:
    """Up to `limit` characters cut at a space, so any quote from it is still in the segment."""
    if len(text) <= limit:
        return text
    if tail:
        cut = text.find(" ", len(text) - limit)
        return text[cut + 1:] if cut >= 0 else text[-limit:]
    cut = text.rfind(" ", 0, limit)
    return text[:cut] if cut > 0 else text[:limit]


def _context_ref(seg: dict, role: str, tail: bool = False) -> dict:
    return {"segment_id": seg["segment_id"], "locator": seg["locator"], "page": seg["page"], "role": role, "text": _excerpt(seg["text"], 800, tail)}


def _section_context(seg: dict, unit: str, by_section: dict[str, list[dict]], header: list[dict]) -> list[dict]:
    """Nearby text the model may use to resolve 'the Company', units or a title, and may cite."""
    refs: list[dict] = []
    if unit == "parent_section":
        body = [s for s in by_section[seg["section_id"]] if s["kind"] not in {"heading", "footnote"}]
        if body and body[0]["segment_id"] != seg["segment_id"]:
            refs.append(_context_ref(body[0], "section_lead"))
        # A table's context is the text introducing it, not an earlier table.
        before = [s for s in body if s["ordinal"] < seg["ordinal"] and not (seg["kind"] == "table" and s["kind"] == "table")]
        if before and before[-1]["segment_id"] != seg["segment_id"]:
            refs.append(_context_ref(before[-1], "preceding_text", tail=True))
    elif unit == "document_header":
        refs.extend(_context_ref(s, "document_header") for s in header if s["segment_id"] != seg["segment_id"])
    unique: dict[str, dict] = {}
    for ref in refs:
        unique.setdefault(ref["segment_id"], ref)
    return list(unique.values())


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[\s|:\-]+", line))


def build_chunks(segments: list[dict], profile: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    maximum, overlap = profile["max_chars"], profile["overlap_chars"]
    # Profiles stored before strategies took effect lack these; the defaults are the old behaviour.
    boundary = profile.get("boundary", "whitespace")
    keep_records = profile.get("keep_records_whole", False)
    context_unit = profile.get("context_unit", "none")
    by_section: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        by_section[seg["section_id"]].append(seg)
    header = [s for s in segments if s["kind"] != "footnote"][:3]
    chunks: list[dict] = []
    warnings: list[dict] = []
    notes_by_section: dict[str, list[dict]] = defaultdict(list)
    sections: dict[str, dict] = {}
    by_page: dict[int, list[dict]] = defaultdict(list)
    for seg in segments:
        if seg.get("page") is not None:
            by_page[seg["page"]].append(seg)
        if seg["kind"] == "footnote":
            notes_by_section[seg["section_id"]].append(seg)
        sec = sections.setdefault(seg["section_id"], {"section_id": seg["section_id"], "section_path": seg["section_path"], "segment_count": 0, "chunk_count": 0, "pages": []})
        sec["segment_count"] += 1
        if seg["page"] is not None and seg["page"] not in sec["pages"]:
            sec["pages"].append(seg["page"])
    for seg in segments:
        text = seg["text"]
        table_header = None
        spans = []
        if seg["kind"] == "table":
            lines = list(re.finditer(r"[^\n]+", text))
            content_lines = [m for m in lines if not _is_separator(m.group())]
            if content_lines:
                table_header = content_lines[0].group()
                header_rows = seg.get("metadata", {}).get("header_rows") or 0
                if header_rows > 1:
                    last = content_lines[min(header_rows, len(content_lines)) - 1]
                    table_header = text[content_lines[0].start():last.end()]
                elif seg["representation"] == "reconstructed_pdf_table" and len(content_lines) > 1:
                    second = content_lines[1]
                    years = set(re.findall(r"\b(?:19|20)\d{2}\b", second.group()))
                    if len(years) >= 2:
                        table_header = text[content_lines[0].start():second.end()]
                # Header is indexed too; rows retain the exact header as contextual evidence.
                for match in content_lines:
                    line = match.group()
                    if len(line) > maximum * 4:
                        warnings.append({"code": "OVERSIZED_TABLE_ROW_SPLIT", "message": "A table row exceeded the size limit and was split; review the complete table manually.", "page": seg["page"], "segment_id": seg["segment_id"]})
                        spans.extend((match.start() + s, match.start() + e) for s, e in _windows(line, maximum * 4, overlap))
                    else:
                        spans.append((match.start(), match.end()))
        elif keep_records and is_field_record(text) and len(text) <= maximum * 4:
            spans = [(0, len(text))]
        else:
            if keep_records and is_field_record(text):
                warnings.append({"code": "OVERSIZED_FIELD_RECORD_SPLIT", "message": "A field record exceeded the size limit and was split; its fields may be separated from the record's name.", "page": seg["page"], "segment_id": seg["segment_id"]})
            spans = _windows(text, maximum, overlap, boundary)
        section_context = _section_context(seg, context_unit, by_section, header)
        for start, end in spans:
            body = text[start:end]
            # Footnote text is context, not silently incorporated into the quoted source span.
            refs = set(re.findall(r"\[\^?([\w-]+)\]", body))
            related_notes = []
            for note in notes_by_section.get(seg["section_id"], []):
                m = re.match(r"\[\^?([^\]]+)\]:", note["text"])
                if m and m.group(1) in refs:
                    related_notes.append({"segment_id": note["segment_id"], "locator": note["locator"], "text": note["text"]})
            context_refs = []
            if seg["kind"] == "table" and seg.get("page") is not None and seg.get("bbox"):
                for context in by_page[seg["page"]]:
                    if context["segment_id"] == seg["segment_id"] or not context.get("bbox") or context["bbox"][3] > seg["bbox"][1] + 2:
                        continue
                    value = context["text"]
                    if len(value) <= 250 and (re.search(r"(?i)\b(?:incorporated|inc|corp|corporation|plc|limited|ltd)\b", value) or re.search(r"(?i)consolidated|(?:dollars|\$|USD)\s+in\s+millions", value)):
                        context_refs.append({"segment_id": context["segment_id"], "locator": context["locator"], "page": context["page"], "text": value})
                context_refs = context_refs[-6:]
            chunk = {
                "chunk_id": stable_id("chk", seg["document_id"], seg["segment_id"], start, end),
                "document_id": seg["document_id"], "segment_id": seg["segment_id"],
                "section_id": seg["section_id"], "section_path": seg["section_path"],
                "kind": seg["kind"], "page": seg["page"], "locator": seg["locator"],
                "start": start, "end": end, "text": body,
                "table_header": table_header, "footnote_context": related_notes, "context_refs": context_refs,
                "section_context": section_context,
                "embedding": None, "embedding_model": None,
            }
            chunks.append(chunk)
            sections[seg["section_id"]]["chunk_count"] += 1
    return chunks, list(sections.values()), warnings
