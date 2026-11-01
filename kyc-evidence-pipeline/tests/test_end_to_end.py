from __future__ import annotations
import json
from pathlib import Path

from conftest import ROOT, defer_all


def test_annual_pause_resume_footnote_and_coverage(cli, paths):
    doc = ROOT / "examples/synthetic_annual_report.txt"
    result = cli("ingest", doc, "--case-id", "case-annual", "--profile", "financial_annual_report", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert result["status"] == "PAUSED_FOR_REVIEW"
    request = result["review_request"]
    assert {i["code"] for i in request["issues"]} == {"STALE_OWNERSHIP_AFTER_DISPOSAL"}
    assert any("65%" in e["quote"] and e["page"] == 1 for e in request["evidence"])
    assert any("disposed" in e["quote"] and e["page"] == 5 for e in request["evidence"])
    status = cli("status", "--run-id", result["run_id"], "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert status["review_request"]["issues"][0]["issue_id"] == request["issues"][0]["issue_id"]
    decision = defer_all(result, paths["tmp"])
    done = cli("resume", "--run-id", result["run_id"], "--decision", decision, "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert done["status"] == "PERSISTED"
    assert done["run_status"] == "REVIEW_REQUIRED"
    package = cli("report", "--case-id", "case-annual", "--db", paths["db"])
    assert len(package["facts"]) == 5
    assert all(f["verification_status"] == "PENDING_HUMAN" for f in package["facts"])
    coverage = {x["category"]: x["status"] for x in package["coverage"]}
    assert coverage["ownership"] == "FOUND"
    assert coverage["legal_identity"] == "NOT_REVIEWED"
    assert "NOT_FOUND_AFTER_REVIEW" not in coverage.values()
    assert package["issues"][0]["status"] == "OPEN"
    table_fact = next(f for f in package["facts"] if f["predicate"] == "reported_revenue")
    evidence = {e["evidence_id"]: e for e in package["evidence"]}
    roles = {r["role"]: evidence[r["evidence_id"]] for r in table_fact["evidence_roles"]}
    assert "200" in roles["primary_quote"]["quote"]
    assert "Revenue (USD millions)" in roles["supporting_quote"]["quote"]
    assert "excludes discontinued" in roles["referenced_footnote"]["quote"]
    for ev in package["evidence"]:
        segment = cli("segment", "--case-id", "case-annual", "--segment-id", ev["segment_id"], "--db", paths["db"])
        assert segment["text"][ev["start"]:ev["end"]] == ev["quote"]
    query = cli("query", "--case-id", "case-annual", "--query", "Who controls Birch Ltd?", "--db", paths["db"])
    assert any("voting rights" in r["parent_context"] for r in query["results"])
    assert all(r["child"]["text"] in r["parent_context"] for r in query["results"])
    assert any(x["action"] == "REVIEW_RECORDED" for x in package["audit_events"])


def test_registry_is_local_and_discovery_does_not_assert_complete(cli, paths):
    result = cli("ingest", ROOT / "examples/synthetic_registry.txt", "--case-id", "reg", "--profile", "corporate_registry", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert result["status"] == "PERSISTED"
    assert result["run_status"] == "REVIEW_REQUIRED"
    package = cli("report", "--case-id", "reg", "--db", paths["db"])
    assert {f["predicate"] for f in package["facts"]} == {"has_legal_name", "has_registration_number", "has_jurisdiction"}
    assert {c["category"]: c["status"] for c in package["coverage"]}["ownership"] == "NOT_REVIEWED"
    assert all(not c["completeness_asserted"] for c in package["coverage"])


def test_cross_document_conflict_human_resolution_preserves_both_sources(cli, paths):
    a, b = paths["tmp"] / "a.txt", paths["tmp"] / "b.txt"
    a.write_text("# Ownership\nAs of 2024-12-31, Aster Holdings Ltd owns 65% of Birch Ltd.\n")
    b.write_text("# Ownership\nAs of 2024-12-31, Aster Holdings Ltd owns 45% of Birch Ltd.\n")
    cli("ingest", a, "--case-id", "conflict", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    second = cli("ingest", b, "--case-id", "conflict", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert second["status"] == "PAUSED_FOR_REVIEW"
    issue = next(i for i in second["review_request"]["issues"] if i["code"] == "POSSIBLE_OWNERSHIP_CONFLICT")
    assert len(second["review_request"]["evidence"]) == 2
    facts = {f["fact_id"]: f for f in second["review_request"]["facts"]}
    chosen = next(fid for fid in issue["fact_ids"] if facts[fid]["value"] == 45)
    # Invalid human input cannot silently clear the conflict or consume the checkpoint.
    invalid = paths["tmp"] / "invalid.json"
    invalid.write_text(json.dumps({"reviewer_id": "analyst", "decisions": [{"issue_id": issue["issue_id"], "action": "resolve", "reason": "reviewed", "selected_fact_ids": ["invented"]}]}))
    error = cli("resume", "--run-id", second["run_id"], "--decision", invalid, "--db", paths["db"], "--checkpoint-db", paths["cp"], expected=1)
    assert "selecting exactly one" in error["message"]
    decision = paths["tmp"] / "valid.json"
    decision.write_text(json.dumps({"reviewer_id": "analyst", "decisions": [{"issue_id": issue["issue_id"], "action": "resolve", "reason": "Reviewed both fictional statements.", "selected_fact_ids": [chosen]}]}))
    cli("resume", "--run-id", second["run_id"], "--decision", decision, "--db", paths["db"], "--checkpoint-db", paths["cp"])
    package = cli("report", "--case-id", "conflict", "--db", paths["db"])
    assert sorted(f["value"] for f in package["facts"] if f["predicate"] == "owns_equity_in") == [45, 65]
    assert next(i for i in package["issues"] if i["issue_id"] == issue["issue_id"])["status"] == "RESOLVED_BY_HUMAN"
    review = package["review_decisions"][0]
    assert review["selected_fact_ids"] == [chosen]
    assert review["reviewer_identity_authenticated"] is False
    assert all(f["verification_status"] == "PENDING_HUMAN" for f in package["facts"])


def test_distinct_dated_ownership_snapshots_do_not_automatically_conflict(cli, paths):
    doc = paths["tmp"] / "different_dates.txt"
    doc.write_text("# Ownership\nAs of 2023-12-31, Aster Holdings Ltd owns 65% of Birch Ltd.\n\nAs of 2024-12-31, Aster Holdings Ltd owns 45% of Birch Ltd.\n")
    result = cli("ingest", doc, "--case-id", "timeline", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert result["status"] == "PERSISTED"
    package = cli("report", "--case-id", "timeline", "--db", paths["db"])
    assert not any(i["code"] == "POSSIBLE_OWNERSHIP_CONFLICT" for i in package["issues"])


def test_remote_processing_requires_explicit_permission_even_with_key(cli, paths):
    result = cli("ingest", ROOT / "examples/synthetic_registry.txt", "--case-id", "remote", "--profile", "corporate_registry", "--extractor", "openai", "--db", paths["db"], "--checkpoint-db", paths["cp"], expected=1)
    assert "explicitly pass --allow-remote-sensitive-data" in result["message"].lower()


def test_concurrent_case_change_requires_reconciliation_again(cli, paths):
    a, b, c = (paths["tmp"] / name for name in ("old.txt", "conflict.txt", "other.txt"))
    a.write_text("As of 2024-12-31, Aster Holdings Ltd owns 65% of Birch Ltd.\n")
    b.write_text("As of 2024-12-31, Aster Holdings Ltd owns 45% of Birch Ltd.\n")
    c.write_text("As of 2024-12-31, Elm Ltd owns 20% of Fir Ltd.\n")
    base = ("--case-id", "concurrent", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    cli("ingest", a, *base)
    paused = cli("ingest", b, *base)
    assert paused["status"] == "PAUSED_FOR_REVIEW"
    cli("ingest", c, *base)
    decision = defer_all(paused, paths["tmp"])
    failed = cli("resume", "--run-id", paused["run_id"], "--decision", decision, "--db", paths["db"], "--checkpoint-db", paths["cp"], expected=1)
    assert "Case evidence changed" in failed["message"]
    retry = cli("retry", "--run-id", paused["run_id"], "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert retry["status"] == "PAUSED_FOR_REVIEW"
    assert any(i["code"] == "POSSIBLE_OWNERSHIP_CONFLICT" for i in retry["review_request"]["issues"])


def test_page_limited_extraction_cannot_imply_document_wide_review(cli, paths):
    result = cli("ingest", ROOT / "examples/synthetic_annual_report.txt", "--case-id", "partial", "--profile", "financial_annual_report", "--extract-pages", "2", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert result["status"] == "PERSISTED"
    package = cli("report", "--case-id", "partial", "--db", paths["db"])
    control = next(c for c in package["coverage"] if c["category"] == "control")
    assert control["status"] == "FOUND"
    assert control["chunks_not_reviewed"] > 0
    assert not control["all_parsed_chunks_processed"]
    assert any(c["category"] == "control" for c in package["unresolved"]["material_requirement_gaps"])
    assert not any(c["status"] == "NOT_FOUND_AFTER_REVIEW" for c in package["coverage"])
