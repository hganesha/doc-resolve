from __future__ import annotations

import json
import pymupdf
import pytest

from kyc_pipeline.chunking import build_chunks
from kyc_pipeline.config import load_contract, requirements_for, select_profile
from kyc_pipeline.evidence import verify_candidates
from kyc_pipeline.extraction import rules_extract
from kyc_pipeline.parsing import parse_file
from kyc_pipeline.reconciliation import reconcile
from kyc_pipeline.util import date_interval, temporal_overlap


def contract_for(profile_id: str):
    c = load_contract()
    p = c["profiles"][profile_id]
    return c, p, requirements_for(c, p)


def test_pdf_coordinates_and_blank_page_are_reported(tmp_path):
    p = tmp_path / "sample.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Aster Holdings Ltd owns 65% of Birch Ltd.")
    pdf.new_page()
    pdf.save(str(p))
    pdf.close()
    doc, segments, issues = parse_file(str(p))
    assert doc["page_count"] == 2
    assert any(s["page"] == 1 and s["bbox"] and "65%" in s["text"] for s in segments)
    assert doc["structurally_blank_pdf_pages"] == [2]
    assert not any(i["code"] == "PDF_PAGE_TEXT_INSUFFICIENT" and i["page"] == 2 for i in issues)


def test_html_inline_text_table_and_complex_table_warning(tmp_path):
    p = tmp_path / "article.html"
    p.write_text('''<html><body><h1>Annual Report</h1><div><span>Aster Holdings Ltd</span> owns 65% of Birch Ltd.</div>
<table><tr><th colspan="2">Financials</th></tr><tr><td>Revenue</td><td>200</td></tr></table>
<script>IGNORED MALICIOUS SCRIPT</script></body></html>''')
    _, segments, issues = parse_file(str(p))
    assert any(s["text"] == "Aster Holdings Ltd owns 65% of Birch Ltd." for s in segments)
    assert any(s["kind"] == "table" and "Financials" in s["text"] for s in segments)
    assert all("MALICIOUS SCRIPT" not in s["text"] for s in segments)
    assert any(i["code"] == "HTML_MERGED_TABLE_CELLS" for i in issues)


def test_whole_document_dispersed_facts_with_correct_page_anchors(tmp_path):
    pages = [f"# Other Information {i}\nThis page contains ordinary fictional background information.\n" for i in range(1, 181)]
    pages[11] = "# Corporate Structure\nAs of 2023-12-31, Aster Holdings Ltd owns 65% of Birch Ltd.\n"
    pages[46] = "# Control\nCedar Group Ltd has contractual voting rights over Birch Ltd as of 2023-12-31.\n"
    pages[102] = "# Legal\nBirch Ltd is subject to a regulatory investigation as of 2024-10-15.\n"
    pages[160] = "# Subsequent Events\nAster Holdings Ltd disposed of part of its interest in Birch Ltd on 2024-02-15.\n"
    p = tmp_path / "large_report.txt"
    p.write_text("\f".join(pages))
    doc, segments, issues = parse_file(str(p))
    _, profile, reqs = contract_for("financial_annual_report")
    chunks, _, warnings = build_chunks(segments, profile)
    logs = [{"chunk_id": c["chunk_id"], "status": "OK", "method": "rules", "candidates": rules_extract(c, reqs)} for c in chunks]
    evidence, facts, support_issues, logs = verify_candidates(doc, segments, chunks, logs, reqs, None, False)
    assert not issues and not warnings and not support_issues
    assert len(facts) == 4
    ev_by_id = {e["evidence_id"]: e for e in evidence}
    pages_by_predicate = {f["predicate"]: ev_by_id[f["evidence_ids"][0]]["page"] for f in facts}
    assert pages_by_predicate == {"owns_equity_in": 12, "has_voting_rights_over": 47, "subject_to_regulatory_investigation": 103, "disposed_interest_in": 161}
    _, recon_issues = reconcile(doc, facts, [], [])
    assert any(i["code"] == "STALE_OWNERSHIP_AFTER_DISPOSAL" for i in recon_issues)


def test_hallucinated_quote_is_rejected_not_turned_into_evidence(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("# Ownership\nAster Holdings Ltd owns 65% of Birch Ltd.\n")
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    chunk = next(c for c in chunks if "65%" in c["text"])
    bad = {"category": "ownership", "subject": "Aster Holdings Ltd", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": 99, "unit": "percent", "source_quote": "Aster Holdings Ltd owns 99% of Birch Ltd."}
    logs = [{"chunk_id": chunk["chunk_id"], "status": "OK", "method": "openai", "candidates": [bad]}]
    evidence, facts, issues, updated = verify_candidates(doc, segments, chunks, logs, reqs, None, False)
    assert evidence == [] and facts == []
    # The chunk was processed; only the fabricated claim is discarded and flagged.
    assert updated[0]["status"] == "OK"
    assert "citation" in updated[0]["rejected_candidates"][0]["reason"]
    rejected = next(i for i in issues if i["code"] == "CANDIDATE_CITATION_REJECTED")
    assert rejected["categories"] == ["ownership"] and rejected["severity"] == "high"


def test_literals_do_not_make_up_semantic_verification(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("Aster Holdings Ltd owns 65% of Birch Ltd.\n")
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    chunk = chunks[0]
    candidate = {"category": "ownership", "subject": "Aster Holdings Ltd", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": 45, "unit": "percent", "source_quote": chunk["text"]}
    logs = [{"chunk_id": chunk["chunk_id"], "status": "OK", "method": "openai", "candidates": [candidate]}]
    _, facts, issues, _ = verify_candidates(doc, segments, chunks, logs, reqs, None, False)
    assert facts[0]["verification_status"] == "PENDING_HUMAN"
    assert facts[0]["source_support"]["literal_checks"]["value_literal_present"] is False
    assert issues[0]["code"] == "CLAIM_LITERAL_SUPPORT_INCOMPLETE"


def test_year_level_temporal_uncertainty_and_no_false_identity_merge():
    assert temporal_overlap("2024", "2024-12-31")
    assert not temporal_overlap("2023", "2024")
    assert temporal_overlap(None, "2024")
    with pytest.raises(ValueError):
        date_interval("2024-13-01")


def test_auto_profile_fails_closed_when_ambiguous():
    contract = load_contract()
    with pytest.raises(ValueError, match="ambiguous"):
        select_profile(contract, None, "unknown.txt", "An unlabeled document")


def test_scanned_pdf_causes_extraction_failed_coverage(cli, paths):
    p = paths["tmp"] / "scan.pdf"
    pdf = pymupdf.open()
    pdf.new_page()
    pdf.save(str(p))
    pdf.close()
    start = cli("ingest", p, "--case-id", "scan", "--profile", "financial_annual_report", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert start["status"] == "PAUSED_FOR_REVIEW"
    assert any(i["code"] == "PARSE_OR_STRUCTURE_INCOMPLETE" for i in start["review_request"]["issues"])
    response = {"reviewer_id": "TEST", "decisions": [{"issue_id": i["issue_id"], "action": "defer", "reason": "OCR and review are needed."} for i in start["review_request"]["issues"]]}
    decision = paths["tmp"] / "scan-decision.json"
    decision.write_text(json.dumps(response))
    cli("resume", "--run-id", start["run_id"], "--decision", decision, "--db", paths["db"], "--checkpoint-db", paths["cp"])
    package = cli("report", "--case-id", "scan", "--db", paths["db"])
    assert all(c["status"] == "EXTRACTION_FAILED" for c in package["coverage"])


def test_human_readable_dates_and_semantic_scope_are_not_silently_certified(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text('As of December 31, 2024, Berkshire or a subsidiary owned approximately 27% of Kraft Heinz.\n')
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    chunk = chunks[0]
    candidate = {"category": "ownership", "subject": "Berkshire", "predicate": "owns_equity_in", "object": "Kraft Heinz", "value": 27, "unit": "%", "effective_date": "December 31, 2024", "source_quote": chunk["text"]}
    logs = [{"chunk_id": chunk["chunk_id"], "status": "OK", "method": "openai", "candidates": [candidate]}]
    _, facts, issues, _ = verify_candidates(doc, segments, chunks, logs, reqs, None, False)
    assert facts[0]["effective_date"] == "2024-12-31"
    assert facts[0]["unit"] == "percent"
    assert facts[0]["source_support"]["literal_checks"]["date_literal_present"] is True
    assert {i["code"] for i in issues} >= {"SUBJECT_SCOPE_NARROWED", "QUANTITATIVE_QUALIFIER_DROPPED"}


def test_cross_segment_support_must_be_in_the_allowed_context(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text('Acme owns 65% of Beta.\n\nRegistration context: Acme.\n')
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    primary = next(c for c in chunks if "65%" in c["text"])
    context = next(s for s in segments if "Registration context" in s["text"])
    candidate = {"category": "ownership", "subject": "Acme", "predicate": "owns_equity_in", "object": "Beta", "value": 65, "unit": "percent", "source_quote": primary["text"], "supporting_refs": [{"segment_id": context["segment_id"], "quote": context["text"]}]}
    def verify(chunk):
        logs = [{"chunk_id": chunk["chunk_id"], "status": "OK", "method": "rules", "candidates": [candidate]}]
        return verify_candidates(doc, segments, [chunk], logs, reqs, None, False)
    _, facts, _, logs = verify(primary)
    assert not facts and logs[0]["rejected_candidates"]
    primary["context_refs"] = [{"segment_id": context["segment_id"], "text": context["text"]}]
    ev, facts, _, logs = verify(primary)
    assert len(facts) == 1 and any(r["role"] == "supporting_reference" for r in facts[0]["evidence_roles"])


def test_literal_overlap_does_not_silently_treat_tax_as_revenue_or_cost_as_stake(tmp_path):
    p = tmp_path / "misclassified.txt"
    p.write_text('Berkshire made payments to the IRS totaling $26.8 billion.\n\nLate in the year we increased our ownership of the utility operation from about 92% to 100% at a cost of roughly $3.9 billion.\n')
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("financial_annual_report")
    chunks, _, _ = build_chunks(segments, profile)
    tax = next(c for c in chunks if "IRS" in c["text"])
    utility = next(c for c in chunks if "utility operation" in c["text"])
    logs = [
        {"chunk_id": tax["chunk_id"], "status": "OK", "method": "openai", "candidates": [{"category": "financial_condition", "subject": "Berkshire", "predicate": "reported_revenue", "value": 26.8, "unit": "billion dollars", "source_quote": tax["text"]}]},
        {"chunk_id": utility["chunk_id"], "status": "OK", "method": "openai", "candidates": [{"category": "ownership", "subject": "utility operation", "predicate": "owns_equity_in", "value": 3.9, "unit": "billion dollars", "source_quote": utility["text"]}]},
    ]
    _, facts, issues, _ = verify_candidates(doc, segments, chunks, logs, reqs, None, False)
    codes = {i["code"] for i in issues}
    assert len(facts) == 2
    assert "PREDICATE_NOT_LITERAL_IN_EVIDENCE" in codes
    # The ownership schema catches the missing owned entity and the cost given as a stake.
    violations = next(i for i in issues if i["code"] == "FACT_SCHEMA_VIOLATION" and i["details"]["predicate"] == "owns_equity_in")["details"]["violations"]
    assert "object is required" in violations and any(v.startswith("unit must be percent") for v in violations)
    assert all(f["verification_status"] == "PENDING_HUMAN" for f in facts)


def test_numeric_string_does_not_hide_dropped_approximation(tmp_path):
    p = tmp_path / "amount.txt"
    p.write_text('As of December 31, 2024, Aster owned approximately 27% of Birch.\n')
    doc, segments, _ = parse_file(str(p))
    _, profile, reqs = contract_for("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    c = chunks[0]
    log = [{"chunk_id": c["chunk_id"], "status": "OK", "method": "openai", "candidates": [{"category": "ownership", "subject": "Aster", "predicate": "owns_equity_in", "object": "Birch", "value": "27", "unit": "percent", "source_quote": c["text"]}]}]
    _, facts, issues, _ = verify_candidates(doc, segments, chunks, log, reqs, None, False)
    assert len(facts) == 1
    assert any(i["code"] == "QUANTITATIVE_QUALIFIER_DROPPED" for i in issues)


def test_pdf_headings_follow_typography_not_capitals(tmp_path):
    p = tmp_path / "typeset.pdf"
    pdf = pymupdf.open()
    body = "Aster Holdings Ltd owns 65% of Birch Ltd as described in this section of the report."
    for n in range(1, 5):
        page = pdf.new_page()
        page.insert_text((72, 30), "ACME HOLDINGS ANNUAL REPORT", fontname="helv", fontsize=10)
        page.insert_text((72, 820), f"Page {n}", fontname="helv", fontsize=10)
        lines = {
            1: [("PART I", "hebo", 10), ("ITEM 1. Business", "hebo", 10), (body, "helv", 10), ("SUMMARY OF TERMS", "helv", 10), ("/S/ JANE DOE", "hebo", 10)],
            2: [("Ownership Structure", "helv", 16), (body, "helv", 10)],
            3: [("Voting Arrangements", "hebo", 10), (body, "helv", 10)],
            4: [(body, "helv", 10), (body, "helv", 10)],
        }[n]
        for i, (text, font, size) in enumerate(lines):
            page.insert_text((72, 120 + 50 * i), text, fontname=font, fontsize=size)
    pdf.save(str(p))
    pdf.close()
    _, segments, _ = parse_file(str(p))
    assert [s["text"] for s in segments if s["kind"] == "heading"] == ["PART I", "ITEM 1. Business", "Ownership Structure", "Voting Arrangements"]
    by_page = {}
    for s in segments:
        if s["kind"] == "paragraph" and s["text"] == body:
            by_page.setdefault(s["page"], s["section_path"])
    assert by_page[1] == ["PART I", "ITEM 1. Business"]
    assert by_page[2] == ["PART I", "ITEM 1. Business", "Ownership Structure"]
    # Body text on a later page stays in the section that was last opened.
    assert by_page[3] == by_page[4] == ["PART I", "ITEM 1. Business", "Ownership Structure", "Voting Arrangements"]
    kinds = {s["text"]: s["kind"] for s in segments}
    assert kinds["SUMMARY OF TERMS"] == "paragraph" and kinds["/S/ JANE DOE"] == "paragraph"
    assert all(s["kind"] == "page_furniture" for s in segments if s["text"] == "ACME HOLDINGS ANNUAL REPORT" or s["text"].startswith("Page "))


def test_pdf_without_typographic_cues_falls_back_to_capitalised_headings(tmp_path):
    p = tmp_path / "plain.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 120), "OWNERSHIP", fontname="helv", fontsize=10)
    page.insert_text((72, 170), "Aster Holdings Ltd owns 65% of Birch Ltd.", fontname="helv", fontsize=10)
    pdf.save(str(p))
    pdf.close()
    _, segments, _ = parse_file(str(p))
    assert next(s for s in segments if "65%" in s["text"])["section_path"] == ["OWNERSHIP"]


def test_parse_gaps_qualify_coverage_without_voiding_extracted_categories():
    from kyc_pipeline.coverage import compute_coverage
    from kyc_pipeline.evidence import make_issue
    reqs = [{"category": "ownership", "description": "x", "material": True, "applicable": True}, {"category": "control", "description": "x", "material": True, "applicable": True}]
    chunks = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
    logs = [{"chunk_id": "c1", "status": "OK"}, {"chunk_id": "c2", "status": "OK"}]
    facts = [{"category": "ownership"}]
    gap = make_issue("PARSE_OR_STRUCTURE_INCOMPLETE", "Images not OCR'd", document_id="d", categories=["*"], details={"pages": [3]})
    clean = {c["category"]: c for c in compute_coverage(reqs, chunks, logs, facts, [], "openai")}
    assert clean["ownership"]["status"] == "FOUND" and clean["control"]["status"] == "NOT_FOUND_AFTER_REVIEW"
    gapped = {c["category"]: c for c in compute_coverage(reqs, chunks, logs, facts, [gap], "openai")}
    assert gapped["ownership"]["status"] == "FOUND"
    # Absence cannot be asserted while part of the source is unparsed.
    assert gapped["control"]["status"] == "NOT_REVIEWED"
    assert all(not c["source_fully_parsed"] and c["parse_gap_pages"] == [3] and "pages 3" in c["note"] for c in gapped.values())
    empty = compute_coverage(reqs, [], [], [], [gap], "openai")
    assert all(c["status"] == "EXTRACTION_FAILED" for c in empty)
