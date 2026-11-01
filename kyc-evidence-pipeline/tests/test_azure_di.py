from __future__ import annotations

import json

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from kyc_pipeline.azure_di import AzureLayoutClient, DocumentIntelligenceError, segments_from_layout
from kyc_pipeline.chunking import build_chunks
from kyc_pipeline.config import load_contract
from kyc_pipeline.graph import build_graph
from kyc_pipeline.store import EvidenceStore
from kyc_pipeline.util import utc_now

ENDPOINT = "https://kyc-test.cognitiveservices.azure.com"


def _layout_result() -> dict:
    """Hand-built analyzeResult in the prebuilt-layout 2024-11-30 shape."""
    parts: list[str] = []
    words: list[dict] = []

    def add(text: str, page: int, *, confidence: float = 0.99) -> dict:
        offset = len("\n".join(parts)) + (1 if parts else 0)
        parts.append(text)
        cursor = offset
        for word in text.split(" "):
            words.append({"page": page, "content": word, "span": {"offset": cursor, "length": len(word)}, "confidence": confidence})
            cursor += len(word) + 1
        return {"offset": offset, "length": len(text)}

    def region(page: int, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
        return [{"pageNumber": page, "polygon": [x0, y0, x1, y0, x1, y1, x0, y1]}]

    header = add("Aster Holdings Ltd Annual Return", 1)
    title = add("Annual Return 2024", 1)
    own_head = add("Ownership", 1)
    own_text = add("Aster Holdings Ltd owns 65% of Birch Ltd.", 1)
    note = add("1 Shares held through a nominee.", 1)
    dir_head = add("Directors", 1)
    dir_text = add("Jane Roe was appointed director on 1 March 2024.", 1, confidence=0.42)
    # | Holder | Shares   | Class |   two header rows; "Class" spans both
    # |        | (number) |       |
    # | Aster Holdings Ltd | 650 | A |
    # | Unallocated (spans 2 columns) | B |
    grid = [("Holder", 0, 0, "columnHeader", 1, 1), ("Shares", 0, 1, "columnHeader", 1, 1), ("Class", 0, 2, "columnHeader", 2, 1), ("(number)", 1, 1, "columnHeader", 1, 1), ("Aster Holdings Ltd", 2, 0, "content", 1, 1), ("650", 2, 1, "content", 1, 1), ("A", 2, 2, "content", 1, 1), ("Unallocated", 3, 0, "content", 1, 2), ("B", 3, 2, "content", 1, 1)]
    cell_spans = [add(g[0], 1) for g in grid]
    table_span = {"offset": cell_spans[0]["offset"], "length": cell_spans[-1]["offset"] + cell_spans[-1]["length"] - cell_spans[0]["offset"]}
    figure_text = add("Group structure chart", 1)
    signature = add("Signed: J. Roe", 1)
    page_number = add("1", 1)
    content = "\n".join(parts)

    def para(span: dict, role: str | None = None, box=(1, 1, 7, 1.3), page: int = 1) -> dict:
        item = {"spans": [span], "boundingRegions": region(page, *box), "content": content[span["offset"]:span["offset"] + span["length"]]}
        if role:
            item["role"] = role
        return item

    paragraphs = [
        para(header, "pageHeader", (1, 0.3, 7, 0.5)),
        para(title, "title", (1, 1.0, 7, 1.4)),
        para(own_head, "sectionHeading", (1, 2.0, 3, 2.2)),
        para(own_text, None, (1, 2.4, 7, 2.6)),
        para(note, "footnote", (1, 10.2, 7, 10.4)),
        para(dir_head, "sectionHeading", (1, 3.0, 3, 3.2)),
        para(dir_text, None, (1, 3.4, 7, 3.6)),
    ]
    # Cell content is also reported as paragraphs; the adapter must not duplicate it.
    paragraphs += [para(span, None, (1, 4.0, 2, 4.2)) for span in cell_spans]
    paragraphs += [para(figure_text, None, (2, 6.0, 5, 6.2)), para(signature, None, (1, 8.0, 3, 8.2)), para(page_number, "pageNumber", (4, 10.6, 4.2, 10.8))]
    figure_index, signature_index = 7 + len(grid), 8 + len(grid)
    table_cells = [{"kind": kind, "rowIndex": r, "columnIndex": c, "rowSpan": rs, "columnSpan": cs, "content": text, "spans": [span], "boundingRegions": region(1, 1, 4, 7, 4.2)} for (text, r, c, kind, rs, cs), span in zip(grid, cell_spans)]

    sections = [
        {"spans": [], "elements": ["/paragraphs/1", "/sections/1", "/sections/2"]},
        {"spans": [], "elements": ["/paragraphs/2", "/paragraphs/3", "/tables/0"]},
        {"spans": [], "elements": ["/paragraphs/5", "/paragraphs/6", "/figures/0", f"/paragraphs/{signature_index}"]},
    ]
    pages = [
        {"pageNumber": 1, "unit": "inch", "width": 8.5, "height": 11, "spans": [{"offset": 0, "length": len(content)}], "words": [{k: v for k, v in w.items() if k != "page"} for w in words]},
        {"pageNumber": 2, "unit": "inch", "width": 8.5, "height": 11, "spans": [], "words": []},
    ]
    return {
        "apiVersion": "2024-11-30", "modelId": "prebuilt-layout", "contentFormat": "text", "content": content,
        "pages": pages, "paragraphs": paragraphs,
        "tables": [{"rowCount": 4, "columnCount": 3, "cells": table_cells, "spans": [table_span], "boundingRegions": region(1, 1, 4, 7, 5.2)}],
        "figures": [{"id": "1.1", "elements": [f"/paragraphs/{figure_index}"], "spans": [figure_text], "boundingRegions": region(1, 2, 6, 5, 7)}],
        "sections": sections,
        "styles": [{"isHandwritten": True, "confidence": 0.9, "spans": [signature]}],
    }


def test_layout_result_maps_to_sections_kinds_and_review_issues():
    segments, issues, page_count = segments_from_layout(_layout_result(), "doc_test")
    assert page_count == 2
    by_text = {s["text"]: s for s in segments}
    assert by_text["Annual Return 2024"]["kind"] == "heading"
    assert by_text["Aster Holdings Ltd owns 65% of Birch Ltd."]["section_path"] == ["Annual Return 2024", "Ownership"]
    # Sibling sections replace each other rather than nesting.
    assert by_text["Jane Roe was appointed director on 1 March 2024."]["section_path"] == ["Annual Return 2024", "Directors"]
    assert by_text["Aster Holdings Ltd Annual Return"]["kind"] == "page_furniture"
    assert by_text["1"]["kind"] == "page_furniture"
    assert by_text["1 Shares held through a nominee."]["kind"] == "footnote"
    assert by_text["Group structure chart"]["metadata"]["figure_id"] == "1.1"
    own = by_text["Aster Holdings Ltd owns 65% of Birch Ltd."]
    assert own["locator"] == "page:1/di-paragraph:3" and own["bbox"] == [72.0, 172.8, 504.0, 187.2]
    assert own["representation"] == "azure_di_layout_text"
    tables = [s for s in segments if s["kind"] == "table"]
    assert len(tables) == 1
    assert tables[0]["text"].splitlines() == ["| Holder | Shares | Class |", "|  | (number) |  |", "| Aster Holdings Ltd | 650 | A |", "| Unallocated |  | B |"]
    assert tables[0]["metadata"]["header_rows"] == 2
    # Table cell text is not repeated as loose paragraphs.
    assert not any(s["kind"] == "paragraph" and s["text"] in {"Holder", "650", "Unallocated"} for s in segments)
    codes = {(i["code"], i["page"]) for i in issues}
    assert ("OCR_LOW_CONFIDENCE", 1) in codes
    assert ("HANDWRITTEN_CONTENT", 1) in codes
    assert ("DI_FIGURE_NOT_INTERPRETED", 1) in codes
    assert ("DI_PAGE_NO_TEXT", 2) in codes
    assert ("DI_TABLE_MERGED_CELLS", 1) in codes
    assert by_text["Jane Roe was appointed director on 1 March 2024."]["metadata"]["min_word_confidence"] == 0.42


def test_multi_row_table_header_is_kept_as_row_context():
    segments, _, _ = segments_from_layout(_layout_result(), "doc_test")
    chunks, _, _ = build_chunks(segments, {"max_chars": 1200, "overlap_chars": 100})
    rows = [c for c in chunks if c["kind"] == "table" and "Aster Holdings Ltd" in c["text"]]
    assert rows and rows[0]["table_header"] == "| Holder | Shares | Class |\n|  | (number) |  |"


def test_sections_absent_falls_back_to_roles():
    result = _layout_result()
    result.pop("sections")
    segments, _, _ = segments_from_layout(result, "doc_test")
    by_text = {s["text"]: s for s in segments}
    assert by_text["Aster Holdings Ltd owns 65% of Birch Ltd."]["section_path"] == ["Annual Return 2024", "Ownership"]


def _transport(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_client_submits_polls_and_returns_analyze_result(monkeypatch):
    monkeypatch.setattr("kyc_pipeline.azure_di.time.sleep", lambda _: None)
    seen = []
    polls = iter([{"status": "running"}, {"status": "succeeded", "analyzeResult": {"pages": [], "content": ""}}])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.headers.get("Ocp-Apim-Subscription-Key")))
        if request.method == "POST":
            assert request.url.params["api-version"] == "2024-11-30"
            assert json.loads(request.content)["base64Source"] == "JVBERg=="
            return httpx.Response(202, headers={"Operation-Location": f"{ENDPOINT}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/abc?api-version=2024-11-30"})
        return httpx.Response(200, json=next(polls))

    client = AzureLayoutClient(allow_remote=True, endpoint=ENDPOINT, key="test-key", client=_transport(handler))
    assert client.analyze(b"%PDF") == {"pages": [], "content": ""}
    assert seen[0][0] == "POST" and "prebuilt-layout:analyze" in seen[0][1]
    assert [m for m, _, _ in seen] == ["POST", "GET", "GET"] and all(k == "test-key" for _, _, k in seen)


def test_client_refuses_to_send_key_to_another_host():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, headers={"Operation-Location": "https://attacker.example/steal"})
    client = AzureLayoutClient(allow_remote=True, endpoint=ENDPOINT, key="test-key", client=_transport(handler))
    with pytest.raises(DocumentIntelligenceError, match="unexpected operation location"):
        client.analyze(b"%PDF")


def test_client_failure_is_sanitized(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, headers={"Operation-Location": f"{ENDPOINT}/op/1"})
        return httpx.Response(200, json={"status": "failed", "error": {"code": "InvalidContent", "message": "Jane Roe passport scan unreadable"}})
    client = AzureLayoutClient(allow_remote=True, endpoint=ENDPOINT, key="test-key", client=_transport(handler))
    with pytest.raises(DocumentIntelligenceError) as exc:
        client.analyze(b"%PDF")
    assert "InvalidContent" in str(exc.value) and "Jane" not in str(exc.value)


def test_client_requires_remote_permission_and_https():
    with pytest.raises(ValueError, match="Remote processing is disabled"):
        AzureLayoutClient(allow_remote=False, endpoint=ENDPOINT, key="k")
    with pytest.raises(ValueError, match="https"):
        AzureLayoutClient(allow_remote=True, endpoint="http://insecure.example", key="k")


class FakeLayout:
    def __init__(self):
        self.calls = 0

    def analyze(self, raw: bytes) -> dict:
        self.calls += 1
        return _layout_result()


def test_graph_extracts_from_layout_and_retry_reuses_cached_analysis(tmp_path):
    scan = tmp_path / "annual_return.png"
    scan.write_bytes(b"\x89PNG synthetic scan")
    store = EvidenceStore(tmp_path / "db.sqlite")
    fake = FakeLayout()
    graph = build_graph(store, InMemorySaver(), layout_factory=lambda _: fake)
    base = {"case_id": "test", "doc_path": str(scan), "requested_profile": "ownership_control", "contract": load_contract(), "parser": "azure_di", "parser_endpoint_sha256": "e" * 64, "extractor": "rules", "embeddings": "lexical", "allow_remote": True, "verify_material": False, "model": "unused", "embedding_model": "unused", "expected_document_sha256": None}
    first = {**base, "run_id": "run_first", "started_at": utc_now(), "retry_from_run": None}
    store.start_run("run_first", "test", first["started_at"])
    graph.invoke(first, {"configurable": {"thread_id": "run_first"}})
    state = graph.get_state({"configurable": {"thread_id": "run_first"}}).values
    assert state["document"]["parser"]["name"] == "azure_di_layout" and state["document"]["page_count"] == 2
    fact = next(f for f in state["facts"] if f["category"] == "ownership")
    assert fact["subject"] == "Aster Holdings Ltd" and fact["object"] == "Birch Ltd"
    assert any(i["code"] == "PARSE_OR_STRUCTURE_INCOMPLETE" and i["details"]["source_code"] == "OCR_LOW_CONFIDENCE" for i in state["issues"])
    retry = {**base, "run_id": "run_retry", "started_at": utc_now(), "retry_from_run": "run_first", "expected_document_sha256": state["document"]["sha256"], "allow_remote": False}
    store.start_run("run_retry", "test", retry["started_at"])
    graph.invoke(retry, {"configurable": {"thread_id": "run_retry"}})
    assert fake.calls == 1
    units = store.get_extraction_units("run_retry")
    assert units and all(u.get("reused_from_run") == "run_first" for u in units.values())


def test_azure_parser_rejects_unsupported_formats(tmp_path):
    from kyc_pipeline.parsing import parse_file
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Unsupported format for Azure"):
        parse_file(str(path), lambda raw, sha: ({}, {}))
