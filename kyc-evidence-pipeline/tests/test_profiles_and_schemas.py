from __future__ import annotations

import json
import re

import pytest

from conftest import ROOT
from kyc_pipeline.chunking import build_chunks
from kyc_pipeline.config import CONFIG_DIR, load_contract, requirements_for
from kyc_pipeline.coverage import compute_coverage
from kyc_pipeline.evidence import verify_candidates
from kyc_pipeline.extraction import requirements_payload, rules_extract
from kyc_pipeline.parsing import parse_file
from kyc_pipeline.reconciliation import reconcile


def _contract(profile_id):
    contract = load_contract()
    profile = contract["profiles"][profile_id]
    return contract, profile, requirements_for(contract, profile)


def _parse(tmp_path, text, name="source.txt"):
    path = tmp_path / name
    path.write_text(text)
    return parse_file(str(path))


def _verify(doc, segments, chunk, candidates, reqs):
    logs = [{"chunk_id": chunk["chunk_id"], "status": "OK", "method": "openai", "candidates": candidates}]
    return verify_candidates(doc, segments, [chunk], logs, reqs, None, False)


def test_every_rules_predicate_is_declared_in_the_typed_schemas():
    source = (ROOT / "src" / "kyc_pipeline" / "extraction.py").read_text()
    # Predicates passed to _mk, to the cover-page add(), and in the registry field patterns.
    emitted = set(re.findall(r'_mk\("[a-z_]+",\s*[^,]+,\s*"([a-z_]+)"', source)) | set(re.findall(r'add\("([a-z_]+)"', source)) | set(re.findall(r'\(r"[^"]+", "([a-z_]+)"\)', source))
    schemas = load_contract()["schemas"]
    assert {"owns_equity_in", "has_legal_name", "reported_revenue", "accused_of"} <= emitted
    assert emitted <= set(schemas), sorted(emitted - set(schemas))


def test_a_field_record_is_kept_whole_so_its_fields_keep_the_records_name(tmp_path):
    filler = "".join(f"Registered office line {i}: {'Unit 4, Example Business Park, Sample Road ' * 2}\n" for i in range(20))
    doc, segments, _ = _parse(tmp_path, "# Corporate Registry Extract\nLegal name: Aster Holdings Ltd\n" + filler + "Registration number: TX-123456\nJurisdiction: Texas\n")
    _, profile, reqs = _contract("corporate_registry")
    assert 1800 < len(segments[-1]["text"]) < 4 * 1800
    chunks, _, _ = build_chunks(segments, profile)
    record = [c for c in chunks if "TX-123456" in c["text"]]
    assert len(record) == 1 and "Legal name: Aster Holdings Ltd" in record[0]["text"]
    numbers = [f for f in rules_extract(record[0], reqs) if f["predicate"] == "has_registration_number"]
    assert [f["subject"] for f in numbers] == ["Aster Holdings Ltd"]
    # Without the setting the record is split and the number loses its subject.
    split, _, _ = build_chunks(segments, {**profile, "keep_records_whole": False})
    tail = next(c for c in split if "TX-123456" in c["text"])
    assert "Legal name" not in tail["text"]
    assert [f["subject"] for f in rules_extract(tail, reqs) if f["predicate"] == "has_registration_number"] == ["UNRESOLVED_REGISTRY_SUBJECT"]


def test_sentence_and_clause_boundaries_do_not_split_mid_sentence(tmp_path):
    sentences = " ".join(f"Holding company number {i} reported a routine change in its board composition during the year." for i in range(40))
    doc, segments, _ = _parse(tmp_path, "# Ownership\n" + sentences + " As of 2024-12-31, Aster Holdings Ltd owns 60% of Birch Ltd.\n")
    _, profile, _ = _contract("ownership_control")
    assert profile["boundary"] == "sentence"
    chunks, _, _ = build_chunks(segments, profile)
    body = [c for c in chunks if c["kind"] == "paragraph"]
    assert len(body) > 2 and all(c["text"][0].isupper() and c["text"].rstrip().endswith(".") for c in body)
    assert any("As of 2024-12-31, Aster Holdings Ltd owns 60% of Birch Ltd." in c["text"] for c in body)
    loose, _, _ = build_chunks(segments, {**profile, "boundary": "whitespace"})
    assert any(not c["text"].rstrip().endswith(".") for c in loose if c["kind"] == "paragraph")
    clauses = "\n".join(f"{i}. The licensee shall comply with condition {i}. " + "It applies throughout the term. " * 12 for i in range(1, 8))
    doc, segments, _ = _parse(tmp_path, "# Licence conditions\n" + clauses + "\n", "licence.txt")
    _, profile, _ = _contract("regulatory_licensing")
    body = [c for c in build_chunks(segments, profile)[0] if c["kind"] == "paragraph"]
    text = segments[-1]["text"]
    # Each split falls just before a numbered clause; the overlap then starts at a sentence.
    assert len(body) >= 2 and all(re.match(r"\d\. The licensee", text[c["end"]:]) for c in body[:-1])
    assert all(c["text"][0].isupper() or c["text"][0].isdigit() for c in body)


def test_section_context_lets_the_model_name_the_company_with_a_citation(tmp_path):
    text = '# Ownership\nAster Holdings Ltd ("the Company") is the parent of the group.\n\nThe group reports its holdings annually.\n\nThe Company owns 60% of Birch Ltd.\n\n# Other\nUnrelated text.\n'
    doc, segments, _ = _parse(tmp_path, text)
    _, profile, reqs = _contract("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    chunk = next(c for c in chunks if "60%" in c["text"])
    roles = {ref["role"]: ref["text"] for ref in chunk["section_context"]}
    assert roles == {"section_lead": 'Aster Holdings Ltd ("the Company") is the parent of the group.', "preceding_text": "The group reports its holdings annually."}
    lead = next(ref for ref in chunk["section_context"] if ref["role"] == "section_lead")
    candidate = {"category": "ownership", "subject": "Aster Holdings Ltd", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": 60, "unit": "percent", "source_quote": chunk["text"], "supporting_refs": [{"segment_id": lead["segment_id"], "quote": 'Aster Holdings Ltd ("the Company")'}]}
    _, facts, _, _ = _verify(doc, segments, chunk, [candidate], reqs)
    assert len(facts) == 1 and facts[0]["source_support"]["literal_checks"]["subject_literal_present"]
    outside = next(s for s in segments if s["text"] == "Unrelated text.")
    _, facts, _, logs = _verify(doc, segments, chunk, [{**candidate, "supporting_refs": [{"segment_id": outside["segment_id"], "quote": "Unrelated text."}]}], reqs)
    assert not facts and logs[0]["rejected_candidates"]
    # An article's chunks see its header: publisher and date.
    doc, segments, _ = _parse(tmp_path, "# Example Daily News\nPublished 2024-03-01 by Example Daily News.\n\nA paragraph.\n\nAster Holdings Ltd was accused of bribery.\n", "article.txt")
    chunks, _, _ = build_chunks(segments, _contract("adverse_media")[1])
    chunk = next(c for c in chunks if "accused" in c["text"])
    assert [ref["role"] for ref in chunk["section_context"]] == ["document_header"] * 3
    assert "Published 2024-03-01 by Example Daily News." in [ref["text"] for ref in chunk["section_context"]]


def test_typed_categories_reject_undeclared_predicates_and_flag_violations(tmp_path):
    doc, segments, _ = _parse(tmp_path, "# Ownership\nAster Holdings Ltd owns 27.5% of Birch Ltd. Aster Holdings Ltd was accused of bribery by a critic.\n")
    _, profile, reqs = _contract("ownership_control")
    chunks, _, _ = build_chunks(segments, profile)
    chunk = chunks[-1]
    base = {"subject": "Aster Holdings Ltd", "source_quote": "Aster Holdings Ltd owns 27.5% of Birch Ltd."}
    _, facts, issues, logs = _verify(doc, segments, chunk, [{**base, "category": "ownership", "predicate": "owns_shares_of", "object": "Birch Ltd", "value": 27.5, "unit": "percent"}], reqs)
    assert not facts and logs[0]["rejected_candidates"][0]["predicate"] == "owns_shares_of"
    rejected = next(i for i in issues if i["code"] == "CANDIDATE_SCHEMA_REJECTED")
    assert rejected["details"]["predicates"] == ["owns_shares_of"] and rejected["severity"] == "high"
    coverage = {c["category"]: c["status"] for c in compute_coverage(reqs, chunks, logs, facts, issues, "openai")}
    assert coverage["ownership"] == "EXTRACTION_FAILED" and coverage["control"] != "EXTRACTION_FAILED"
    # A numeric string is coerced where the schema expects a number.
    _, facts, issues, _ = _verify(doc, segments, chunk, [{**base, "category": "ownership", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": "27.5", "unit": "%"}], reqs)
    assert facts[0]["value"] == 27.5 and facts[0]["schema_check"] == {"status": "CONFORMS", "violations": []} and not issues
    # The adverse-media schema keeps an allegation an allegation.
    _, _, media_reqs = _contract("adverse_media")
    claim = {"category": "allegations_and_outcomes", "subject": "Aster Holdings Ltd", "predicate": "accused_of", "object": "bribery", "claim_type": "finding", "source_quote": "Aster Holdings Ltd was accused of bribery by a critic."}
    _, facts, issues, _ = _verify(doc, segments, chunk, [claim], media_reqs)
    assert facts[0]["schema_check"]["violations"] == ["claim_type must be one of allegation"]
    assert [i["code"] for i in issues] == ["FACT_SCHEMA_VIOLATION"]


def test_open_categories_keep_facts_but_mark_them_untyped(tmp_path):
    doc, segments, _ = _parse(tmp_path, "# KYC Review\nNote: The customer's stated turnover differs from the accounts.\n")
    _, profile, reqs = _contract("internal_kyc")
    assert {r["category"]: bool(r["predicates"]) for r in reqs}["discrepancies"] is False
    chunk = build_chunks(segments, profile)[0][-1]
    candidate = {"category": "discrepancies", "subject": "UNRESOLVED_CUSTOMER", "predicate": "has_turnover_discrepancy", "object": "stated turnover", "source_quote": "The customer's stated turnover differs from the accounts."}
    _, facts, _, _ = _verify(doc, segments, chunk, [candidate], reqs)
    assert facts[0]["schema_check"]["status"] == "UNTYPED"
    payload = {r["category"]: r for r in requirements_payload(reqs)}
    assert payload["discrepancies"]["typed"] is False
    ownership = {r["category"]: r for r in requirements_payload(_contract("ownership_control")[2])}["ownership"]
    owns = next(p for p in ownership["predicates"] if p["predicate"] == "owns_equity_in")
    assert ownership["typed"] and "evidence_terms" not in owns


def test_single_valued_attributes_conflict_only_when_they_really_differ():
    def fact(fid, predicate, value, subject="Acme Ltd", date=None, resolution="TEXT_MATCH_ONLY"):
        return {"fact_id": fid, "document_id": "d", "category": "legal_identity", "subject": subject, "predicate": predicate, "object": value, "value": None, "unit": None, "effective_date": date, "claim_type": "reported_statement", "entity_resolution": resolution}
    schemas = load_contract()["schemas"]
    def codes(previous, current):
        _, issues = reconcile({"document_id": "d2"}, current, previous, [], {}, schemas)
        return [(i["code"], i["details"]["values"]) for i in issues]
    assert codes([fact("a", "has_jurisdiction", "Texas")], [fact("b", "has_jurisdiction", "Delaware")]) == [("POSSIBLE_ATTRIBUTE_CONFLICT", ["Delaware", "Texas"])]
    # Formatting and legal-form differences are not conflicts.
    assert not codes([fact("a", "has_legal_name", "ACME LTD.")], [fact("b", "has_legal_name", "Acme Limited")])
    assert not codes([fact("a", "has_us_employer_identification_number", "12-3456789")], [fact("b", "has_us_employer_identification_number", "123456789")])
    # Different periods are a change over time; unresolved subjects are not compared; multi-valued predicates are left alone.
    assert not codes([fact("a", "has_legal_name", "Acme Ltd", date="2019")], [fact("b", "has_legal_name", "Acme Group Ltd", date="2024")])
    assert not codes([fact("a", "has_jurisdiction", "Texas", subject="UNRESOLVED_REGISTRY_SUBJECT")], [fact("b", "has_jurisdiction", "Delaware", subject="UNRESOLVED_REGISTRY_SUBJECT")])
    assert not codes([fact("a", "has_former_name", "Old Acme Ltd")], [fact("b", "has_former_name", "Older Acme Ltd")])
    # Without typed schemas (a contract stored before they existed) nothing is checked.
    _, issues = reconcile({"document_id": "d2"}, [fact("b", "has_jurisdiction", "Delaware")], [fact("a", "has_jurisdiction", "Texas")], [], {})
    assert not issues


@pytest.mark.parametrize("edit, message", [
    (lambda s: s.replace("    value: percent\n    date_roles: [as_of", "    value: shares\n    date_roles: [as_of", 1), "value must be one of"),
    (lambda s: s.replace("category: control\n", "category: controll\n", 1), "unknown policy category"),
    (lambda s: s.replace("    single_valued: true\n", "    single_valued: yes please\n", 1), "single_valued"),
    (lambda s: s.replace("    object: required\n", "    object: required\n    cardinality: one\n", 1), "Unknown fields"),
])
def test_invalid_schemas_are_rejected(tmp_path, edit, message):
    path = tmp_path / "schemas.yaml"
    path.write_text(edit((CONFIG_DIR / "extraction_schemas.yaml").read_text()))
    with pytest.raises(ValueError, match=message):
        load_contract(schemas_path=path)


def test_invalid_profile_strategy_settings_are_rejected(tmp_path):
    path = tmp_path / "profiles.yaml"
    path.write_text((CONFIG_DIR / "document_profiles.yaml").read_text().replace("boundary: sentence", "boundary: paragraph", 1))
    with pytest.raises(ValueError, match="boundary must be one of"):
        load_contract(profiles_path=path)


def test_cli_pauses_on_a_legal_name_conflict_and_records_the_resolution(cli, paths):
    base = ("--case-id", "names", "--profile", "corporate_registry", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    for name, legal in (("first.txt", "Aster Holdings Ltd"), ("second.txt", "Aster Group Ltd")):
        (paths["tmp"] / name).write_text(f"# Corporate Registry Extract\nLegal name: {legal}\nRegistration number: TX-123456\nJurisdiction: Texas\n")
    assert cli("ingest", paths["tmp"] / "first.txt", *base)["status"] == "PERSISTED"
    second = cli("ingest", paths["tmp"] / "second.txt", *base)
    assert second["status"] == "PAUSED_FOR_REVIEW"
    issue = next(i for i in second["review_request"]["issues"] if i["code"] == "POSSIBLE_ATTRIBUTE_CONFLICT")
    assert issue["details"] == {"predicate": "has_legal_name", "values": ["Aster Group Ltd", "Aster Holdings Ltd"]} and issue["allowed_actions"] == ["resolve", "defer"]
    selected = next(f["fact_id"] for f in second["review_request"]["facts"] if f["object"] == "Aster Group Ltd")
    decisions = [{"issue_id": i["issue_id"], "action": "resolve", "reason": "Registry shows a rename.", "selected_fact_ids": [selected]} if i is issue else {"issue_id": i["issue_id"], "action": "defer", "reason": "Fictional test case."} for i in second["review_request"]["issues"]]
    decision = paths["tmp"] / "decision.json"
    decision.write_text(json.dumps({"reviewer_id": "TEST_ANALYST", "decisions": decisions}))
    cli("resume", "--run-id", second["run_id"], "--decision", decision, "--db", paths["db"], "--checkpoint-db", paths["cp"])
    package = cli("report", "--case-id", "names", "--db", paths["db"])
    stored = next(i for i in package["issues"] if i["issue_id"] == issue["issue_id"])
    assert stored["status"] == "RESOLVED_BY_HUMAN" and stored["selected_fact_ids"] == [selected]
    assert all(f["schema_check"]["status"] == "CONFORMS" for f in package["facts"])
    # The profiles command shows the strategy settings and each category's predicates.
    listing = {p["profile_id"]: p for p in cli("profiles")["profiles"]}
    registry = listing["corporate_registry"]
    assert registry["keep_records_whole"] is True and registry["context_unit"] == "document_header"
    assert "has_legal_name" in next(r for r in registry["requirements"] if r["category"] == "legal_identity")["predicates"]
