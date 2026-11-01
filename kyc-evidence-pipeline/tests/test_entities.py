from __future__ import annotations

from conftest import ROOT, defer_all
from kyc_pipeline.chunking import build_chunks
from kyc_pipeline.config import load_contract, requirements_for
from kyc_pipeline.entities import legal_key, resolve_case, resolve_document
from kyc_pipeline.evidence import verify_candidates
from kyc_pipeline.extraction import rules_extract
from kyc_pipeline.parsing import parse_file


def _run(path, profile_id="financial_annual_report"):
    contract = load_contract()
    profile = contract["profiles"][profile_id]
    requirements = requirements_for(contract, profile)
    document, segments, _ = parse_file(str(path))
    chunks, _, _ = build_chunks(segments, profile)
    logs = [{"chunk_id": c["chunk_id"], "status": "OK", "method": "rules", "candidates": rules_extract(c, requirements)} for c in chunks]
    evidence, facts, issues, _ = verify_candidates(document, segments, chunks, logs, requirements, None, False)
    entities, facts, definition_evidence, issues = resolve_document(document, segments, facts, issues, requirements)
    return document, segments, entities, facts, evidence + definition_evidence, issues


def test_legal_key_normalizes_forms_and_leading_article():
    assert legal_key("The Kraft Heinz Company") == legal_key("Kraft Heinz Co.") == "kraft heinz co"
    assert legal_key("BERKSHIRE HATHAWAY INC.") == legal_key("Berkshire Hathaway Incorporated")
    assert legal_key("Aster Holdings Limited") == legal_key("Aster Holdings Ltd")


def test_cover_page_definitions_and_pronouns_resolve_to_entity_and_group(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text(
        "# Form 10-K\n"
        "Commission file number 001-99999 ASTER HOLDINGS LTD (Exact name of Registrant as specified in its charter) Delaware 12-3456789 State or other jurisdiction of incorporation or organization\n\n"
        "# Business\n"
        "Aster Holdings Ltd (“Aster”) is a holding company. In these notes the terms “we,” “us,” or “our” refer to Aster and its consolidated subsidiaries.\n\n"
        "Birch Industries Limited (“Birch”) makes widgets.\n\n"
        "As of 2024-12-31, Aster owns 65% of Birch.\n\n"
        "As of 2024-12-31, we owned 20% of Cedar Ltd.\n\n"
        "As of 2024-12-31, Aster or a subsidiary owns 30% of Elm Ltd.\n"
    )
    document, _, entities, facts, evidence, issues = _run(doc)
    by_id = {e["entity_id"]: e for e in entities}
    aster = next(e for e in entities if e["name"] == "ASTER HOLDINGS LTD")
    assert {(i["scheme"], i["value"]) for i in aster["identifiers"]} == {("SEC_FILE_NUMBER", "001-99999"), ("US_EIN", "12-3456789")}
    assert [j["value"] for j in aster["jurisdictions"]] == ["Delaware"] and aster["roles"] == ["sec_registrant"]
    assert {"Aster", "Aster Holdings Ltd"} <= {n["name"] for n in aster["names"]}
    owns = {f["object"]: f for f in facts if f["predicate"] == "owns_equity_in"}
    # "Aster" is the defined term for the registrant; "Birch" for Birch Industries Limited.
    birch = owns["Birch"]
    assert birch["entity_resolution"] == "RESOLVED"
    assert by_id[birch["entity_links"]["subject"]["entity_id"]] is aster
    assert by_id[birch["entity_links"]["object"]["entity_id"]]["name"] == "Birch Industries Limited"
    link = birch["entity_links"]["subject"]
    assert link["basis"] == "DEFINED_TERM" and all(e in {x["evidence_id"] for x in evidence} for e in link["evidence_ids"])
    # "we" and "X or a subsidiary" are the group, which is a separate entity from Aster.
    for obj in ("Cedar Ltd", "Elm Ltd"):
        fact = owns[obj]
        group = by_id[fact["entity_links"]["subject"]["entity_id"]]
        assert fact["entity_resolution"] == "RESOLVED_TO_GROUP"
        assert group["entity_type"] == "group" and group["group_of"] == aster["entity_id"]
    group_issues = [i for i in issues if i["code"] == "ENTITY_RESOLVED_TO_GROUP"]
    assert {owns["Cedar Ltd"]["fact_id"], owns["Elm Ltd"]["fact_id"]} <= {f for i in group_issues for f in i["fact_ids"]}
    assert all(i["severity"] == "high" for i in group_issues)
    assert not any(i["code"] == "ENTITY_IDENTITY_UNRESOLVED" for i in issues)


def test_unnamed_registrant_is_named_by_a_shared_defined_term(tmp_path):
    # The Disney pattern: the cover-page name is an image; a generic definition and
    # the auditor's named definition both define "Company".
    doc = tmp_path / "report.md"
    doc.write_text(
        "Commission File Number 001-11111\n\n"
        "Delaware 98-7654321 State or Other Jurisdiction of I.R.S. Employer Identification Incorporation or Organization\n\n"
        "The terms “Company”, “we”, “our” and “us” are used in this report to refer collectively to the parent company and the subsidiaries through which businesses are conducted.\n\n"
        "We have audited the consolidated balance sheets of Juniper Media Corporation and its subsidiaries (the “Company”) as of 2024-09-28.\n\n"
        "As of 2024-09-28, the Company owns 80% of Oak Sports Ltd.\n"
    )
    _, _, entities, facts, _, issues = _run(doc)
    registrant = next(e for e in entities if "sec_registrant" in e["roles"])
    assert registrant["name"] == "Juniper Media Corporation"
    assert {(i["scheme"], i["value"]) for i in registrant["identifiers"]} == {("SEC_FILE_NUMBER", "001-11111"), ("US_EIN", "98-7654321")}
    assert registrant["resolution_evidence_ids"]
    cover = [f for f in facts if f["subject"] == "UNRESOLVED_REGISTRANT"]
    assert cover and all(f["entity_resolution"] == "RESOLVED" and f["entity_links"]["subject"]["basis"] == "REGISTRANT_COVER" for f in cover)
    # The placeholder subject no longer counts as unsupported once resolved.
    assert not any(i["code"] in {"ENTITY_IDENTITY_UNRESOLVED", "CLAIM_LITERAL_SUPPORT_INCOMPLETE", "REGISTRANT_NAME_NOT_IN_TEXT"} and set(i["fact_ids"]) & {f["fact_id"] for f in cover} for i in issues)
    owns = next(f for f in facts if f["predicate"] == "owns_equity_in")
    assert owns["entity_resolution"] == "RESOLVED_TO_GROUP"


def test_registrant_without_name_or_definitions_is_flagged(tmp_path):
    doc = tmp_path / "cover.md"
    doc.write_text("Commission File Number 001-22222\n\nDelaware 11-2222222 State or Other Jurisdiction of I.R.S. Employer Identification Incorporation or Organization\n")
    _, _, entities, facts, _, issues = _run(doc)
    assert any(i["code"] == "REGISTRANT_NAME_NOT_IN_TEXT" for i in issues)
    assert next(e for e in entities if "sec_registrant" in e["roles"])["name"] is None


def test_conflicting_definitions_and_former_names_are_not_linked(tmp_path):
    doc = tmp_path / "terms.md"
    doc.write_text(
        "Maple Holdings Inc. (“Maple”) owns stores.\n\n"
        "Maple Logistics LLC (“Maple”) runs trucks.\n\n"
        "Fir Enterprises 18 Corp. (formerly known as Fir Group Inc.) (“Legacy Fir”) issued notes.\n\n"
        "As of 2024-12-31, Maple owns 40% of Pine Ltd.\n"
    )
    _, _, entities, facts, _, issues = _run(doc, "ownership_control")
    ambiguous = next(i for i in issues if i["code"] == "DEFINED_TERM_AMBIGUOUS")
    assert ambiguous["details"]["term"] == "maple" and ambiguous["severity"] == "medium"
    owns = next(f for f in facts if f["predicate"] == "owns_equity_in")
    assert owns["entity_links"]["subject"]["basis"] == "MENTION" and owns["entity_resolution"] == "TEXT_MATCH_ONLY"
    assert not any("Legacy Fir" in {n["name"] for n in e["names"]} for e in entities)


def _org(eid, doc, name, identifiers=(), jurisdiction=None):
    return {"entity_id": eid, "document_id": doc, "entity_type": "organization", "name": name, "names": [{"name": name, "kind": "legal_name", "evidence_ids": []}],
            "identifiers": [{"scheme": s, "value": v, "jurisdiction": j, "evidence_ids": [], "fact_id": f"fact_{eid}_{s}"} for s, v, j in identifiers],
            "jurisdictions": [{"value": jurisdiction, "evidence_ids": []}] if jurisdiction else [], "roles": [], "group_of": None, "strength": "IDENTIFIED", "resolution_evidence_ids": []}


def test_case_merges_on_identifier_and_name_but_not_across_conflicts():
    entities = [
        _org("e1", "d1", "Aster Holdings Limited", [("REGISTRATION_NUMBER", "TX-1", "Texas")], "Texas"),
        _org("e2", "d2", "Aster Holdings Ltd"),  # same legal name, nothing contradicts it
        _org("e3", "d3", "Aster Group Plc", [("REGISTRATION_NUMBER", "TX-1", "Texas")]),  # renamed, same number
        _org("e4", "d4", "Aster Holdings Ltd", [], "Ontario"),  # same name, other jurisdiction
        _org("e5", "d5", "Birch Inc.", [("US_EIN", "11-1111111", None)]),
        _org("e6", "d6", "Birch Inc.", [("US_EIN", "22-2222222", None)]),
    ]
    entities.append({**_org("g1", "d2", "Aster Holdings Ltd and subsidiaries"), "entity_type": "group", "group_of": "e2", "identifiers": [], "names": [{"name": "we", "kind": "pronoun", "evidence_ids": []}]})
    case, mapping, issues = resolve_case(entities, "d6")
    assert mapping["e1"] == mapping["e2"] == mapping["e3"] != mapping["e4"]
    aster = next(c for c in case if c["entity_id"] == mapping["e1"])
    assert aster["match_basis"] == "IDENTIFIER" and aster["documents"] == ["d1", "d2", "d3"]
    assert mapping["e5"] != mapping["e6"]
    collisions = [i for i in issues if i["code"] == "ENTITY_NAME_COLLISION"]
    assert {tuple(i["details"]["entity_ids"]) for i in collisions} >= {("e1", "e4"), ("e5", "e6")}
    group = next(c for c in case if c["entity_id"] == mapping["g1"])
    assert group["entity_type"] == "group" and group["group_of"] == mapping["e2"]


def test_shared_identifier_with_different_values_is_a_conflict():
    entities = [
        _org("e1", "d1", "Cedar Corp", [("SEC_FILE_NUMBER", "001-1", None), ("US_EIN", "11-1111111", None)]),
        _org("e2", "d2", "Cedar Corporation", [("SEC_FILE_NUMBER", "001-1", None), ("US_EIN", "99-9999999", None)]),
    ]
    _, mapping, issues = resolve_case(entities, "d2")
    assert mapping["e1"] == mapping["e2"]
    conflict = next(i for i in issues if i["code"] == "ENTITY_IDENTIFIER_CONFLICT")
    assert conflict["details"]["conflicts"] == {"US_EIN": ["11-1111111", "99-9999999"]} and conflict["severity"] == "high"
    assert set(conflict["fact_ids"]) == {"fact_e1_US_EIN", "fact_e2_US_EIN", "fact_e1_SEC_FILE_NUMBER", "fact_e2_SEC_FILE_NUMBER"}


def test_conflict_detected_across_documents_that_name_the_owner_differently(cli, paths):
    registry, defined, plain = (paths["tmp"] / n for n in ("registry.txt", "defined.txt", "plain.txt"))
    registry.write_text("# Corporate Registry Extract\nLegal name: Aster Holdings Limited\nRegistration number: TX-123456\nJurisdiction: Texas\n")
    defined.write_text("# Ownership\nAster Holdings Ltd (“Aster”) is a holding company.\n\nAs of 2024-12-31, Aster owns 45% of Birch Ltd.\n")
    plain.write_text("# Ownership\nAs of 2024-12-31, Aster Holdings Limited owns 65% of Birch Ltd.\n")
    base = ("--case-id", "entities", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    cli("ingest", registry, "--profile", "corporate_registry", *base)
    first = cli("ingest", defined, "--profile", "ownership_control", *base)
    assert first["status"] == "PERSISTED"
    second = cli("ingest", plain, "--profile", "ownership_control", *base)
    # Before the entity layer "Aster" and "Aster Holdings Limited" were never compared.
    assert second["status"] == "PAUSED_FOR_REVIEW"
    assert any(i["code"] == "POSSIBLE_OWNERSHIP_CONFLICT" for i in second["review_request"]["issues"])
    cli("resume", "--run-id", second["run_id"], "--decision", defer_all(second, paths["tmp"]), "--db", paths["db"], "--checkpoint-db", paths["cp"])
    package = cli("report", "--case-id", "entities", "--db", paths["db"])
    aster = next(e for e in package["entities"] if e["name"] and legal_key(e["name"]) == "aster holdings ltd")
    assert aster["match_basis"] == "LEGAL_NAME" and len(aster["documents"]) == 3
    assert aster["identifiers"] == [{"scheme": "REGISTRATION_NUMBER", "value": "TX-123456", "jurisdiction": "Texas"}]
    owners = {f["case_entity_ids"]["subject"] for f in package["facts"] if f["predicate"] == "owns_equity_in"}
    assert owners == {aster["entity_id"]}


def test_approximate_and_precise_figures_are_not_a_conflict(cli, paths):
    doc = paths["tmp"] / "approx.txt"
    doc.write_text("# Investments\nAs of 2024-12-31, Aster Holdings Ltd owned approximately 27% of the outstanding common stock of Birch Ltd.\n\nAs of 2024-12-31, Aster Holdings Ltd owns 27.2% of Birch Ltd.\n")
    result = cli("ingest", doc, "--case-id", "approx", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    package = cli("report", "--case-id", "approx", "--db", paths["db"])
    values = sorted(f["value"] for f in package["facts"] if f["predicate"] == "owns_equity_in")
    assert values == [27.0, 27.2]
    assert not any(i["code"] == "POSSIBLE_OWNERSHIP_CONFLICT" for i in package["issues"])
    assert any(f["type"] == "OWNERSHIP_FIGURES_DIFFER_BY_ROUNDING" for f in package["findings"])


def test_rounded_figure_without_qualifier_is_consistent_but_different_stakes_conflict():
    from kyc_pipeline.reconciliation import reconcile
    def fact(fid, value):
        return {"fact_id": fid, "category": "ownership", "subject": "Aster Holdings Ltd", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": value, "unit": "percent", "effective_date": "2024-12-31", "claim_type": "reported_statement", "qualifier": None}
    findings, issues = reconcile({"document_id": "d"}, [fact("a", 28.0), fact("b", 28.2)], [], [])
    assert not issues and any(f["type"] == "OWNERSHIP_FIGURES_DIFFER_BY_ROUNDING" for f in findings)
    _, issues = reconcile({"document_id": "d"}, [fact("a", 28.0), fact("b", 28.6)], [], [])
    assert [i["code"] for i in issues] == ["POSSIBLE_OWNERSHIP_CONFLICT"]
