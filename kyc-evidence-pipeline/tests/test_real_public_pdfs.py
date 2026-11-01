"""Regression checks against actual, pinned primary-source PDF files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import pytest

from kyc_pipeline.chunking import build_chunks
from kyc_pipeline.config import load_contract, requirements_for
from kyc_pipeline.evidence import verify_candidates
from kyc_pipeline.extraction import rules_extract
from kyc_pipeline.parsing import parse_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "examples/public_reports"


@pytest.fixture(scope="module")
def real_reports():
    manifest = json.loads((REPORT_DIR / "sources.json").read_text())
    contract = load_contract()
    profile = contract["profiles"]["financial_annual_report"]
    requirements = requirements_for(contract, profile)
    results = {}
    for entry in manifest:
        file = REPORT_DIR / entry["filename"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"]
        doc, segments, issues = parse_file(str(file))
        chunks, _, _ = build_chunks(segments, profile)
        results[entry["filename"].split("_")[0]] = (doc, segments, chunks, issues, requirements)
    return results


def extract_page(data, page):
    doc, segments, chunks, _, req = data
    chosen = [c for c in chunks if c["page"] == page]
    logs = [{"chunk_id": c["chunk_id"], "status": "OK", "method": "rules", "candidates": rules_extract(c, req)} for c in chosen]
    return verify_candidates(doc, segments, chunks, logs, req, None, False)


def test_berkshire_real_150_page_pdf_ownership_and_financial_table(real_reports):
    doc, _, _, issues, _ = real_reports["berkshire"]
    assert doc["page_count"] == 150
    assert doc["structurally_blank_pdf_pages"] == [2, 4, 148]
    ev, facts, warnings, _ = extract_page(real_reports["berkshire"], 45)
    matches = [f for f in facts if f["predicate"] == "owns_equity_in" and f["subject"] == "Berkshire or a subsidiary"]
    assert {(f["object"], f["value"]) for f in matches} >= {("The Kraft Heinz Company", 27.0), ("Occidental Petroleum Corporation", 28.0)}
    assert all(f["effective_date"] == "2024-12-31" and f["entity_resolution"] == "UNRESOLVED" for f in matches)
    assert any(i["code"] == "ENTITY_IDENTITY_UNRESOLVED" for i in warnings)
    ev, facts, warnings, _ = extract_page(real_reports["berkshire"], 90)
    revenue = next(f for f in facts if f["predicate"] == "reported_revenue" and f["effective_date"] == "2024")
    assert revenue["value"] == 371433.0 and revenue["unit"] == "millions_of_dollars"
    assert revenue["date_role"] == "reporting_period"
    assert {r["role"] for r in revenue["evidence_roles"]} >= {"primary_quote", "supporting_quote", "supporting_reference"}
    by_id = {e["evidence_id"]: e for e in ev}
    assert by_id[revenue["evidence_ids"][0]]["representation"] == "reconstructed_pdf_table"
    assert all(x is not False for x in revenue["source_support"]["literal_checks"].values())


def test_disney_real_132_page_pdf_inferred_table_and_figure_is_unresolved(real_reports):
    doc, _, _, issues, _ = real_reports["disney"]
    assert doc["page_count"] == 132
    assert any(i["code"] == "PDF_IMAGES_NOT_OCR" and i["page"] == 3 for i in issues)
    assert any(i["code"] == "PDF_LAYOUT_TABLE_NEEDS_REVIEW" and i["page"] == 38 for i in issues)
    _, facts, issues, _ = extract_page(real_reports["disney"], 13)
    espn = next(f for f in facts if f["object"] == "ESPN" and f["predicate"] == "owns_equity_in")
    assert espn["value"] == 80.0 and espn["qualifier"] == "generally"
    assert espn["subject"] == "the Company" and espn["entity_resolution"] == "UNRESOLVED"
    venu = next(f for f in facts if f["predicate"] == "launch_enjoined_by_court")
    assert venu["subject"] == "Venu Sports" and venu["effective_date"] == "2024-08" and venu["claim_type"] == "order"
    ev, facts, issues, _ = extract_page(real_reports["disney"], 38)
    revenue = next(f for f in facts if f["predicate"] == "reported_revenue" and f["effective_date"] == "2024")
    assert revenue["value"] == 91361.0
    assert revenue["subject"] == "UNRESOLVED_REPORTING_ENTITY"
    assert any(i["code"] == "CLAIM_LITERAL_SUPPORT_INCOMPLETE" for i in issues)
    assert next(e for e in ev if e["evidence_id"] == revenue["evidence_ids"][0])["representation"] == "reconstructed_pdf_layout_table"


def test_real_reports_resolve_registrants_and_group_subjects(real_reports):
    from kyc_pipeline.entities import resolve_document
    doc, segments, chunks, _, req = real_reports["disney"]
    _, facts, issues, _ = extract_page(real_reports["disney"], 3)
    _, more, more_issues, _ = extract_page(real_reports["disney"], 13)
    entities, linked, _, issues = resolve_document(doc, segments, facts + more, issues + more_issues, req)
    # The cover-page name is an image; the document's own "Company" definitions name it.
    registrant = next(e for e in entities if "sec_registrant" in e["roles"])
    assert registrant["name"] == "The Walt Disney Company"
    assert {(i["scheme"], i["value"]) for i in registrant["identifiers"]} == {("SEC_FILE_NUMBER", "001-38842"), ("US_EIN", "83-0940635")}
    espn = next(f for f in linked if f.get("object") == "ESPN")
    group = next(e for e in entities if e["entity_id"] == espn["entity_links"]["subject"]["entity_id"])
    assert espn["entity_resolution"] == "RESOLVED_TO_GROUP" and group["group_of"] == registrant["entity_id"]
    doc, segments, chunks, _, req = real_reports["berkshire"]
    _, facts, issues, _ = extract_page(real_reports["berkshire"], 21)
    _, more, more_issues, _ = extract_page(real_reports["berkshire"], 45)
    entities, linked, _, issues = resolve_document(doc, segments, facts + more, issues + more_issues, req)
    berkshire = next(e for e in entities if "sec_registrant" in e["roles"])
    assert {"Berkshire", "Berkshire Hathaway Inc."} <= {n["name"] for n in berkshire["names"]}
    kraft = next(f for f in linked if f.get("object") == "The Kraft Heinz Company" and f["subject"] == "Berkshire or a subsidiary")
    assert kraft["entity_resolution"] == "RESOLVED_TO_GROUP"
    assert any(i["code"] == "DEFINED_TERM_AMBIGUOUS" and i["details"]["term"] == "company" for i in issues)
