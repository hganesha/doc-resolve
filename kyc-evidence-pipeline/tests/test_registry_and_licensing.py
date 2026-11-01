from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from conftest import ROOT
from kyc_pipeline.config import load_contract
from kyc_pipeline.entities import resolve_case
from kyc_pipeline.graph import build_graph
from kyc_pipeline.reconciliation import reconcile
from kyc_pipeline.registry import GleifClient, jurisdiction_code, status_issues
from kyc_pipeline.schemas import validate_schemas
from kyc_pipeline.store import EvidenceStore
from kyc_pipeline.util import lei_check_digits_valid, utc_now

LEI = "5299000ASTERHOLD0035"  # synthetic, with valid check digits


def _ingest(tmp_path, source: Path | str, *, case="reg", profile=None, registry_factory=None, registry_check=None, run_id="run_1", store=None, defer=False, **extra):
    """Run one document; with ``defer``, a paused run is resumed by deferring every review item."""
    if isinstance(source, Path):
        path = source
    else:
        path = tmp_path / f"{run_id}.txt"
        path.write_text(source)
    store = store or EvidenceStore(tmp_path / "db.sqlite")
    state = {"run_id": run_id, "case_id": case, "started_at": utc_now(), "doc_path": str(path), "requested_profile": profile, "contract": load_contract(), "extractor": "rules", "embeddings": "lexical", "allow_remote": False, "verify_material": False, "model": "unused", "embedding_model": "unused", "retry_from_run": None, "expected_document_sha256": None, "registry_check": registry_check, **extra}
    store.start_run(run_id, case, state["started_at"])
    graph = build_graph(store, InMemorySaver(), registry_factory=registry_factory or (lambda *_: pytest.fail("registry looked up without --registry-check")))
    config = {"configurable": {"thread_id": run_id}}
    result = graph.invoke(state, config)
    if defer and "__interrupt__" in result:
        decisions = [{"issue_id": i["issue_id"], "action": "defer", "reason": "Deferred by the test."} for i in result["__interrupt__"][0].value["issues"]]
        graph.invoke(Command(resume={"reviewer_id": "TEST", "decisions": decisions}), config)
    return store, result, graph.get_state(config).values


def _facts(values):
    return {(f["predicate"], f["subject"], f.get("object"), f.get("value"), f.get("effective_date")) for f in values["facts"]}


def test_registry_extract_fields_officers_and_dates(tmp_path):
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_registry_extract.txt")
    assert values["profile_id"] == "corporate_registry" and "__interrupt__" not in result
    facts = _facts(values)
    company = "Aster Holdings Ltd"
    assert {
        ("has_legal_name", company, company, None, None),
        ("has_registration_number", company, "TX-123456", None, None),
        ("has_lei", company, LEI, None, None),
        ("has_incorporation_date", company, None, None, "2015-03-12"),
        ("has_former_name", company, "Aster Ventures Ltd", None, None),
        # Status, address, form and officers are as of the extract date.
        ("has_legal_status", company, "Active", None, "2026-01-15"),
        ("has_registered_address", company, "100 Main Street, Austin, Texas 78701", None, "2026-01-15"),
        ("has_legal_form", company, "Private limited company", None, "2026-01-15"),
        ("is_director_of", "Jane Smith", company, None, "2026-01-15"),
        ("is_director_of", "Omar Haddad", company, None, "2026-01-15"),
        ("is_officer_of", "Lena Park", company, "Company secretary", "2026-01-15"),
    } <= facts
    assert all(f["schema_check"]["status"] == "CONFORMS" for f in values["facts"])
    # Every fact cites its own line; the name and extract-date lines are cited as support.
    director = next(f for f in values["facts"] if f["subject"] == "Jane Smith")
    quotes = {e["evidence_id"]: e["quote"] for e in values["evidence"]}
    assert [quotes[e] for e in director["evidence_ids"]] == ["Directors: Jane Smith (appointed 2019-01-01); Omar Haddad", "Company name: Aster Holdings Ltd", "Date of extract: 15 January 2026"]
    entity = next(e for e in values["entities"] if e["name"] == company)
    assert {(i["scheme"], i["value"]) for i in entity["identifiers"]} == {("LEI", LEI), ("REGISTRATION_NUMBER", "TX-123456")}


def test_licence_register_entry_raises_status_and_expiry_issues(tmp_path):
    store, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_licence_register.txt")
    assert values["profile_id"] == "regulatory_licensing"
    firm, licence = "Aster Payments Ltd", "Electronic money institution authorisation"
    assert {
        ("holds_license", firm, licence, "Suspended", "2026-02-02"),
        ("has_license_number", firm, "912345", None, "2026-02-02"),
        ("license_issued_by", firm, "Texas Financial Services Commission", None, "2026-02-02"),
        ("license_granted_on", firm, licence, None, "2019-06-01"),
        ("license_expires_on", firm, licence, None, "2025-05-31"),
        ("license_restricted_by", firm, "No new customers may be onboarded", None, "2026-02-02"),
        ("license_restricted_by", firm, "Safeguarding audit required by 30 June 2026", None, "2026-02-02"),
    } <= _facts(values)
    codes = {i["code"]: i for i in values["issues"]}
    assert codes["LICENSE_STATUS_ADVERSE"]["details"]["status"] == "Suspended"
    assert codes["LICENSE_EXPIRY_PASSED"]["details"]["expires"] == "2025-05-31"
    assert store.export_case("reg")["status"] == "REVIEW_REQUIRED"


def test_status_checks_on_facts():
    def fact(predicate, obj=None, value=None, when=None, category="legal_identity"):
        return {"fact_id": predicate + str(obj), "category": category, "predicate": predicate, "object": obj, "value": value, "effective_date": when}
    facts = [fact("has_lei", "5299000ASTERHOLD0036"), fact("has_lei", LEI), fact("has_legal_status", "Dissolved"), fact("has_legal_status", "Active"),
             fact("holds_license", "Banking licence", "Authorised", category="license_status"), fact("license_expires_on", when="2030-01-01", category="license_status")]
    issues = status_issues("d", facts, {"legal_identity": True}, today=date(2026, 6, 1))
    assert sorted(i["code"] for i in issues) == ["LEI_CHECK_DIGITS_INVALID", "REGISTRY_STATUS_ADVERSE"]
    assert lei_check_digits_valid("HWUPKR0MPOU8FGXBT394") and not lei_check_digits_valid("HWUPKR0MPOU8FGXBT395")


def test_a_misprinted_lei_does_not_identify_or_merge_entities(tmp_path):
    bad = LEI[:-1] + "6"
    documents = []
    for n, name in enumerate(["Aster Holdings Ltd", "Birch Holdings Ltd"]):
        _, result, values = _ingest(tmp_path, f"# Companies Registry Extract\nCompany name: {name}\nLEI: {bad}\n", run_id=f"run_{n}", store=EvidenceStore(tmp_path / f"db{n}.sqlite"))
        # Legal identity is material, so a check-digit failure pauses the run for review.
        assert any(i["code"] == "LEI_CHECK_DIGITS_INVALID" for i in result["__interrupt__"][0].value["issues"])
        documents += values["entities"]
    entities, _, _ = resolve_case(documents, "case")
    assert len(entities) == 2 and not any(e["identifiers"] for e in entities)


def test_one_status_per_licence_not_per_firm():
    schemas = load_contract()["schemas"]
    def fact(fid, licence, status):
        return {"fact_id": fid, "document_id": "d", "category": "license_status", "subject": "Aster Payments Ltd", "predicate": "holds_license", "object": licence, "value": status, "effective_date": None, "entity_links": {}}
    current = [fact("a", "E-money authorisation", "Suspended"), fact("b", "Payment services licence", "Authorised")]
    _, issues = reconcile({"document_id": "d"}, current, [], [], {}, schemas)
    assert not [i for i in issues if i["code"] == "POSSIBLE_ATTRIBUTE_CONFLICT"]
    _, issues = reconcile({"document_id": "d"}, current, [fact("c", "E-money authorisation", "Authorised")], [], {}, schemas)
    conflict = next(i for i in issues if i["code"] == "POSSIBLE_ATTRIBUTE_CONFLICT")
    assert conflict["details"] == {"predicate": "holds_license", "object": "E-money authorisation", "values": ["Authorised", "Suspended"]}


def test_per_object_schema_needs_an_object_and_a_value():
    with pytest.raises(ValueError, match="per_object"):
        validate_schemas({"predicates": {"has_status": {"category": "x", "description": "d", "single_valued": "per_object"}}})


def test_jurisdiction_codes():
    assert [jurisdiction_code(x) for x in ("Delaware", "State of New York", "England and Wales", "US-CA", "Atlantis")] == ["US-DE", "US-NY", "GB", "US-CA", None]


def _gleif(lei, name="Aster Holdings Ltd", jurisdiction="US-TX", registered_as="TX-123456", other=(), status="ACTIVE", registration="ISSUED", created="2015-03-12T00:00:00Z"):
    return {"data": {"type": "lei-records", "id": lei, "attributes": {"lei": lei, "entity": {
        "legalName": {"name": name}, "otherNames": [{"name": n, "type": "PREVIOUS_LEGAL_NAME"} for n in other], "jurisdiction": jurisdiction,
        "registeredAt": {"id": "RA000999"}, "registeredAs": registered_as, "legalAddress": {"addressLines": ["100 Main Street"], "city": "Austin", "region": "US-TX", "country": "US", "postalCode": "78701"},
        "status": status, "creationDate": created}, "registration": {"status": registration, "corroborationLevel": "FULLY_CORROBORATED"}}}}


def _client(responses):
    calls = []
    def handle(request):
        calls.append(request.url.path)
        reply = responses(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(404) if reply is None else reply if isinstance(reply, httpx.Response) else httpx.Response(200, json=reply)
    return GleifClient(client=httpx.Client(transport=httpx.MockTransport(handle))), calls


def test_gleif_record_matches_and_is_cached(tmp_path):
    client, calls = _client(lambda lei: _gleif(lei, other=["Aster Ventures Ltd"]))
    store, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_registry_extract.txt", registry_check="gleif", registry_factory=lambda *_: client)
    check, = values["registry_checks"]
    assert check["status"] == "CHECKED" and not check["served_from_cache"] and calls == [f"/api/v1/lei-records/{LEI}"]
    assert {c["field"]: c["result"] for c in check["comparisons"]} == {"legal_name": "MATCH", "former_names": "MATCH", "jurisdiction": "MATCH", "registration_number": "MATCH", "incorporation_date": "MATCH"}
    assert not [i for i in values["issues"] if i["code"].startswith(("REGISTRY_", "LEI_"))]
    assert store.export_case("reg")["registry_checks"][0]["lei"] == LEI
    # A second document about the same entity reuses the stored record.
    client2, calls2 = _client(lambda lei: pytest.fail("cached record not reused"))
    _, _, values = _ingest(tmp_path, f"# Companies Registry Extract\nCompany name: Aster Holdings Ltd\nLEI: {LEI}\n", registry_check="gleif", registry_factory=lambda *_: client2, run_id="run_2", store=store)
    assert values["registry_checks"][0]["served_from_cache"] and not calls2


def test_gleif_mismatches_raise_issues(tmp_path):
    record = _gleif(LEI, name="Birch Group Ltd", jurisdiction="GB", registered_as="00999999", status="INACTIVE", registration="LAPSED", created="2001-01-01T00:00:00Z")
    client, _ = _client(lambda lei: record)
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_registry_extract.txt", registry_check="gleif", registry_factory=lambda *_: client)
    codes = {i["code"]: i for i in values["issues"]}
    assert codes["REGISTRY_NAME_MISMATCH"]["severity"] == "high" and codes["REGISTRY_JURISDICTION_MISMATCH"]["details"]["registry"] == "GB"
    assert {"REGISTRY_NUMBER_MISMATCH", "REGISTRY_INCORPORATION_DATE_MISMATCH", "REGISTRY_ENTITY_INACTIVE", "LEI_NOT_CURRENT"} <= set(codes)
    # High-severity mismatches stop the run for review.
    assert {i["code"] for i in result["__interrupt__"][0].value["issues"]} >= {"REGISTRY_NAME_MISMATCH", "REGISTRY_JURISDICTION_MISMATCH"}


def test_gleif_not_found_and_lookup_failure(tmp_path):
    client, _ = _client(lambda lei: None)
    _, _, values = _ingest(tmp_path, ROOT / "examples/synthetic_registry_extract.txt", registry_check="gleif", registry_factory=lambda *_: client)
    assert values["registry_checks"][0]["status"] == "NOT_FOUND" and any(i["code"] == "REGISTRY_RECORD_NOT_FOUND" for i in values["issues"])
    client, _ = _client(lambda lei: httpx.Response(400, json={"errors": [{"detail": "secret"}]}))
    _, _, values = _ingest(tmp_path, ROOT / "examples/synthetic_registry_extract.txt", case="other", registry_check="gleif", registry_factory=lambda *_: client, run_id="run_9", store=EvidenceStore(tmp_path / "db2.sqlite"))
    failed = next(i for i in values["issues"] if i["code"] == "REGISTRY_LOOKUP_FAILED")
    assert failed["details"]["error"] == "GLEIF returned HTTP 400" and values["registry_checks"][0]["status"] == "LOOKUP_FAILED"
