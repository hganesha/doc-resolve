from __future__ import annotations

import base64

import httpx
import pytest

from conftest import ROOT
from kyc_pipeline.graph import _registry_client
from kyc_pipeline.registry import CompaniesHouseClient, check_companies_house, companies_house_number, same_person
from kyc_pipeline.store import EvidenceStore
from test_registry_and_licensing import LEI, _client as _gleif_client, _gleif, _ingest

EXTRACT = ROOT / "examples/synthetic_uk_registry_extract.txt"
NUMBER = "99000123"


def _company(name="Birchwood Trading Limited", status="active", type_="ltd", jurisdiction="england-wales", created="2018-06-04", postcode="BS1 4QA", previous=("Birchwood Imports Limited",), **extra):
    return {"company_number": NUMBER, "company_name": name, "company_status": status, "type": type_, "jurisdiction": jurisdiction, "date_of_creation": created,
            "registered_office_address": {"address_line_1": "5 Harbour Street", "locality": "Bristol", "postal_code": postcode},
            "previous_company_names": [{"name": n, "effective_from": "2018-06-04", "ceased_on": "2020-02-01"} for n in previous], **extra}


OFFICERS = [
    {"name": "SMITH, Jane Elizabeth", "officer_role": "director", "appointed_on": "2019-01-01"},
    {"name": "HADDAD, Omar", "officer_role": "director", "appointed_on": "2018-06-04", "resigned_on": "2026-03-31"},
    {"name": "PARK, Lena", "officer_role": "secretary", "appointed_on": "2018-06-04"},
    {"name": "PATEL, Ravi", "officer_role": "director", "appointed_on": "2025-11-01"},
]


def _ch(company=None, officers=OFFICERS, pscs=(), status=200, page_size=100):
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status != 200:
            return httpx.Response(status)
        path = request.url.path
        if company is None:
            return httpx.Response(404, json={"errors": [{"error": "company-profile-not-found"}]})
        if path.endswith("/officers"):
            start = int(request.url.params.get("start_index", 0))
            return httpx.Response(200, json={"items": list(officers)[start:start + page_size], "total_results": len(officers)})
        if path.endswith("/persons-with-significant-control"):
            return httpx.Response(200, json={"items": list(pscs), "total_results": len(pscs)})
        return httpx.Response(200, json=company)

    client = httpx.Client(base_url="https://api.company-information.service.gov.uk", transport=httpx.MockTransport(handle), auth=httpx.BasicAuth("test-key", ""))
    return CompaniesHouseClient(client=client, max_retries=0), calls


def _ingest_ch(tmp_path, client, source=EXTRACT, **kw):
    return _ingest(tmp_path, source, case=kw.pop("case", "ch"), registry_check=["companies-house"], registry_factory=lambda _, src: client if src == "companies-house" else pytest.fail(src), **kw)


def _codes(values):
    return sorted(i["code"] for i in values["issues"] if i["code"].startswith("REGISTRY_"))


def test_matching_record_officers_and_cache(tmp_path):
    client, calls = _ch(_company())
    store, result, values = _ingest_ch(tmp_path, client)
    assert "__interrupt__" not in result and _codes(values) == []
    check = values["registry_checks"][0]
    assert (check["source"], check["company_number"], check["status"], check["served_from_cache"]) == ("COMPANIES_HOUSE", NUMBER, "CHECKED", False)
    assert {c["field"]: c["result"] for c in check["comparisons"]} == {"legal_name": "MATCH", "former_names": "MATCH", "jurisdiction": "MATCH", "legal_status": "MATCH", "legal_form": "MATCH", "incorporation_date": "MATCH", "registered_address": "MATCH"}
    # Jane Smith is 'SMITH, Jane Elizabeth'; Omar Haddad resigned after the extract's date; Ravi Patel is only on the register.
    assert {o["name"]: o["result"] for o in check["officers"]["stated"]} == {"Jane Smith": "CURRENT", "Omar Haddad": "RESIGNED_SINCE", "Lena Park": "CURRENT"}
    assert [o["name"] for o in check["officers"]["registry_only"]] == ["Ravi PATEL"]
    assert base64.b64decode(calls[0].headers["authorization"].split()[1]) == b"test-key:"
    assert [c.url.path for c in calls] == [f"/company/{NUMBER}", f"/company/{NUMBER}/officers", f"/company/{NUMBER}/persons-with-significant-control"]
    # A second document about the same company reuses the stored record.
    client2, calls2 = _ch(_company())
    _, _, values = _ingest_ch(tmp_path, client2, source=f"# Companies Registry Extract\nCompany name: Birchwood Trading Limited\nCompany number: {NUMBER}\nJurisdiction: England and Wales\n", run_id="run_2", store=store)
    assert values["registry_checks"][0]["served_from_cache"] is True and not calls2


def test_differences_from_the_register_raise_issues(tmp_path):
    record = _company(name="Elmstead Holdings Limited", status="dissolved", type_="plc", jurisdiction="scotland", created="2017-01-01", postcode="EH1 1AA", previous=(), has_insolvency_history=True, accounts={"overdue": True})
    officers = [{"name": "SMITH, Jane", "officer_role": "director", "appointed_on": "2018-06-04", "resigned_on": "2025-06-30"}]
    client, _ = _ch(record, officers=officers)
    _, result, values = _ingest_ch(tmp_path, client)
    assert "__interrupt__" in result
    assert _codes(values) == ["REGISTRY_ADDRESS_MISMATCH", "REGISTRY_ENTITY_INACTIVE", "REGISTRY_INCORPORATION_DATE_MISMATCH", "REGISTRY_JURISDICTION_MISMATCH", "REGISTRY_LEGAL_FORM_MISMATCH", "REGISTRY_NAME_MISMATCH", "REGISTRY_OFFICER_NOT_FOUND", "REGISTRY_OFFICER_NOT_FOUND", "REGISTRY_OFFICER_RESIGNED", "REGISTRY_RECORD_FLAGGED"]
    issues = {i["code"]: i for i in values["issues"]}
    assert issues["REGISTRY_ENTITY_INACTIVE"]["severity"] == "high"
    assert issues["REGISTRY_RECORD_FLAGGED"]["details"]["flags"] == ["accounts overdue", "insolvency history"]
    assert issues["REGISTRY_OFFICER_RESIGNED"]["details"] == {"source": "COMPANIES_HOUSE", "company_number": NUMBER, "name": "Jane Smith", "as_of": "2026-01-15", "resigned_on": ["2025-06-30"]}
    check = values["registry_checks"][0]
    assert {c["field"]: c["result"] for c in check["comparisons"]}["legal_status"] == "MISMATCH"


def test_a_previous_name_is_reported_as_a_rename(tmp_path):
    client, _ = _ch(_company(name="Birchwood Group Limited", previous=("Birchwood Trading Limited", "Birchwood Imports Limited")))
    _, _, values = _ingest_ch(tmp_path, client)
    assert _codes(values) == ["REGISTRY_NAME_IS_OTHER_NAME"]


def test_not_found_failure_and_missing_key(tmp_path, monkeypatch):
    client, _ = _ch(None)
    _, _, values = _ingest_ch(tmp_path, client)
    assert _codes(values) == ["REGISTRY_RECORD_NOT_FOUND"] and values["registry_checks"][0]["status"] == "NOT_FOUND"
    client, _ = _ch(_company(), status=500)
    _, _, values = _ingest_ch(tmp_path, client, case="other", run_id="run_2", store=EvidenceStore(tmp_path / "db2.sqlite"))
    assert values["registry_checks"][0]["error"] == "Companies House returned HTTP 500" and _codes(values) == ["REGISTRY_LOOKUP_FAILED"]
    # Without a key every lookup fails visibly instead of the check being skipped.
    monkeypatch.delenv("COMPANIES_HOUSE_API_KEY", raising=False)
    _, _, values = _ingest(tmp_path, EXTRACT, case="third", run_id="run_3", store=EvidenceStore(tmp_path / "db3.sqlite"), registry_check=["companies-house"], registry_factory=_registry_client)
    assert values["registry_checks"][0]["error"] == "COMPANIES_HOUSE_API_KEY is not set"


def test_only_uk_companies_are_looked_up(tmp_path):
    client, calls = _ch(_company())
    # A Texas company is never sent; a number with no stated jurisdiction is listed but not sent.
    _, _, values = _ingest_ch(tmp_path, client, source=ROOT / "examples/synthetic_registry_extract.txt")
    assert values["registry_checks"] == []
    _, _, values = _ingest_ch(tmp_path, client, source=f"# Companies Registry Extract\nCompany name: Birchwood Trading Limited\nCompany number: {NUMBER}\n", case="other", run_id="run_2")
    assert values["registry_checks"][0]["status"] == "NOT_LOOKED_UP" and values["registry_checks"][0]["reason"] == "no UK jurisdiction stated"
    assert not calls


def test_gleif_and_companies_house_together(tmp_path):
    ch, _ = _ch(_company())
    gleif, _ = _gleif_client(lambda lei: _gleif(lei, name="Birchwood Trading Limited", jurisdiction="GB", registered_as=NUMBER, created="2018-06-04T00:00:00Z", other=["Birchwood Imports Limited"]))
    text = EXTRACT.read_text() + f"LEI: {LEI}\n"
    _, result, values = _ingest(tmp_path, text, case="both", registry_check=["companies-house", "gleif"], registry_factory=lambda _, src: ch if src == "companies-house" else gleif)
    assert sorted(c["source"] for c in values["registry_checks"]) == ["COMPANIES_HOUSE", "GLEIF"] and _codes(values) == []


def test_officer_pages_are_followed():
    officers = [{"name": f"PERSON, Number{n}", "officer_role": "director"} for n in range(150)]
    client, calls = _ch(_company(), officers=officers, page_size=100)
    record = client.lookup(NUMBER)
    assert len(record["officers"]) == 150 and len([c for c in calls if c.url.path.endswith("/officers")]) == 2


def test_reported_stakes_against_psc_bands():
    entity = {"entity_id": "e1", "name": "Birchwood Trading Limited", "names": [{"name": "Birchwood Trading Limited", "kind": "legal_name"}], "identifiers": [{"scheme": "REGISTRATION_NUMBER", "value": NUMBER, "jurisdiction": "England and Wales"}], "jurisdictions": ["England and Wales"]}
    pscs = [{"name": "Mrs Jane Elizabeth Smith", "kind": "individual-person-with-significant-control", "natures_of_control": ["ownership-of-shares-50-to-75-percent", "voting-rights-50-to-75-percent"], "notified_on": "2019-01-01"},
            {"name": "Aster Holdings Ltd", "kind": "corporate-entity-person-with-significant-control", "natures_of_control": ["ownership-of-shares-25-to-50-percent"]},
            {"name": "Old Owner Ltd", "kind": "corporate-entity-person-with-significant-control", "natures_of_control": ["ownership-of-shares-75-to-100-percent"], "ceased_on": "2019-01-01"}]
    record = CompaniesHouseClient(client=_ch(_company(), officers=[], pscs=pscs)[0].client).lookup(NUMBER)
    stake = lambda fid, holder, value: {"fact_id": fid, "predicate": "owns_equity_in", "subject": holder, "object": "Birchwood Trading Limited", "value": value, "effective_date": "2025-12-31"}
    held = [stake("f1", "Jane Smith", 60.0), stake("f2", "Aster Holdings Ltd", 55.0), stake("f3", "Old Owner Ltd", 30.0), stake("f4", "Small Holder Ltd", 5.0)]
    checks, issues = check_companies_house("d", [entity], {}, {"e1": held}, [], {"e1"}, lambda n: (record, "2026-01-01T00:00:00+00:00", False))
    assert {r["holder"]: r["result"] for r in checks[0]["persons_with_significant_control"]["stated"]} == {"Jane Smith": "WITHIN_BAND", "Aster Holdings Ltd": "OUTSIDE_BAND", "Old Owner Ltd": "MISSING_FROM_REGISTER", "Small Holder Ltd": "NOT_A_PSC"}
    assert sorted(i["code"] for i in issues) == ["REGISTRY_PSC_BAND_MISMATCH", "REGISTRY_PSC_MISSING"]
    assert next(p for p in record["persons_with_significant_control"] if p["name"] == "Mrs Jane Elizabeth Smith")["bands"] == {"ownership-of-shares": [50, 75], "voting-rights": [50, 75]}


def test_company_numbers_and_names():
    assert [companies_house_number(v) for v in ("9876543", "SC 123456", "09876543", "OC-301234", "123456789", "TX123", None)] == ["09876543", "SC123456", "09876543", "OC301234", None, "TX000123", None]
    assert same_person("Jane Smith", "Mrs Jane Elizabeth Smith") and not same_person("J Smith", "Jane Smith") and not same_person("Jane Smyth", "Jane Smith")
