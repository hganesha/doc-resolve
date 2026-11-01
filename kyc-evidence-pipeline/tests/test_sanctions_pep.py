from __future__ import annotations

from datetime import date

from conftest import ROOT
from kyc_pipeline.screening import load_list, report_checks, screen_entities
from test_registry_and_licensing import _ingest

LIST = str(ROOT / "examples/synthetic_screening_list.csv")


def _codes(values):
    return sorted(i["code"] for i in values["issues"] if i["code"].startswith(("SANCTIONS_", "PEP_", "SCREENING_")))


def test_screening_report_results(tmp_path):
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_screening_report.txt", defer=True)
    assert values["profile_id"] == "sanctions_pep" and "__interrupt__" in result
    facts = {(f["predicate"], f["subject"]): f for f in values["facts"]}
    hit = facts[("sanctions_screening_result", "Birchwood Trading Limited")]
    assert (hit["value"], hit["qualifier"], hit["effective_date"]) == ("Potential match", "BIRCHWOOD TRADING LLC (synthetic list entry SYN-0042)", "2026-10-20")
    assert facts[("holds_public_function", "Jane Elizabeth Smith")]["object"] == "Member of Bristol City Council"
    assert all(f["schema_check"]["status"] == "CONFORMS" for f in values["facts"])
    assert _codes(values) == ["PEP_IDENTIFIED", "SANCTIONS_SCREENING_HIT"]
    issue = next(i for i in values["issues"] if i["code"] == "SANCTIONS_SCREENING_HIT")
    assert issue["severity"] == "high" and issue["details"]["name"] == "Birchwood Trading Limited"
    # A "no match" is recorded as a finding, never as a clearance.
    clear = next(f for f in values["findings"] if f["type"] == "SANCTIONS_SCREENING_NO_MATCH")
    assert clear["details"]["name"] == "Jane Elizabeth Smith" and "not a clearance" in clear["description"]


def test_screening_age(tmp_path):
    _, _, values = _ingest(tmp_path, ROOT / "examples/synthetic_screening_report.txt", defer=True)
    _, issues = report_checks("d", values["facts"], [], {"sanctions_pep": True}, {"max_age_days": 90}, today=date(2027, 3, 1))
    stale = [i for i in issues if i["code"] == "SCREENING_OUT_OF_DATE"]
    assert len(stale) == 4 and {i["details"]["age_days"] for i in stale} == {132}


def test_pep_declaration_contradicted_either_order(tmp_path):
    for order, case in (((ROOT / "examples/synthetic_pep_declaration.txt", ROOT / "examples/synthetic_screening_report.txt"), "a"), ((ROOT / "examples/synthetic_screening_report.txt", ROOT / "examples/synthetic_pep_declaration.txt"), "b")):
        store = None
        for n, path in enumerate(order):
            store, _, values = _ingest(tmp_path, path, case=case, run_id=f"{case}_{n}", store=store, defer=True)
        contradiction = next(i for i in values["issues"] if i["code"] == "PEP_DECLARATION_CONTRADICTED")
        # "Jane Smith" declared; the report screened "Jane Elizabeth Smith": resolution joins them.
        assert contradiction["severity"] == "high" and contradiction["details"]["positions"] == ["Member of Bristol City Council"]
    declaration = _ingest(tmp_path, ROOT / "examples/synthetic_pep_declaration.txt", case="alone", run_id="alone")[2]
    assert declaration["facts"][0]["predicate"] == "declared_pep_status" and _codes(declaration) == []


def test_list_screening(tmp_path):
    # Same name and date of birth: a possible match for review.
    _, result, card = _ingest(tmp_path, ROOT / "examples/synthetic_identity_card.txt", case="card", run_id="card", screening_lists=[LIST])
    assert "__interrupt__" in result and _codes(card) == ["SANCTIONS_LIST_POSSIBLE_MATCH"]
    check, = card["screening_checks"]
    assert (check["name"], check["result"]) == ("OMAR HADDAD", "POSSIBLE_MATCH")
    assert (check["matches"][0]["entry_id"], check["matches"][0]["date_of_birth"]) == ("SYN-0230", "MATCH")
    assert check["lists"][0]["entries"] == 4 and len(check["lists"][0]["sha256"]) == 64
    # Same name, different date of birth: discounted, at medium severity.
    _, result, passport = _ingest(tmp_path, ROOT / "examples/synthetic_passport.txt", case="passport", run_id="passport", screening_lists=[LIST])
    assert "__interrupt__" not in result and _codes(passport) == ["SANCTIONS_LIST_MATCH_DIFFERENT_BIRTH_DATE"]
    assert passport["screening_checks"][0]["result"] == "MATCH_DISCOUNTED_BY_DATE_OF_BIRTH"
    # Legal forms are ignored for organizations; without a date of birth a name match cannot be discounted.
    store, _, report = _ingest(tmp_path, ROOT / "examples/synthetic_screening_report.txt", case="report", run_id="report", screening_lists=[LIST], defer=True)
    results = {c["name"]: c["result"] for c in report["screening_checks"]}
    assert results == {"Birchwood Trading Limited": "POSSIBLE_MATCH", "Jane Elizabeth Smith": "POSSIBLE_MATCH"}
    assert store.export_case("report")["screening_checks"] == report["screening_checks"]


def test_name_matching_rules():
    lists = [load_list(LIST)]
    entity = lambda name, kind, births=(): {"entity_id": name, "entity_type": kind, "name": name, "names": [{"name": name, "kind": "mention"}], "birth_dates": list(births)}
    results = {}
    for e in (entity("Jane Smith", "person"), entity("Janet Smith", "person"), entity("Omar Hadad", "person"), entity("Northgate Shipping S.A.", "organization"), entity("Northgate Shipping", "person"), entity("Smith Jane Elizabeth Holdings", "organization")):
        checks, _ = screen_entities("d", [e], {e["entity_id"]}, lists)
        results[e["name"]] = checks[0]["result"]
    assert results == {
        "Jane Smith": "POSSIBLE_MATCH",  # the list's middle name is extra; no birth date to compare
        "Janet Smith": "NO_MATCH",  # a different given name
        "Omar Hadad": "NO_MATCH",  # the surname must match
        "Northgate Shipping S.A.": "POSSIBLE_MATCH",
        "Northgate Shipping": "NO_MATCH",  # a person is not screened against an entity entry
        "Smith Jane Elizabeth Holdings": "NO_MATCH",  # an organization is not screened against a person entry
    }


def test_ofac_sdn_format(tmp_path):
    (tmp_path / "sdn.csv").write_text('36,"AERO-CARIBBEAN",-0- ,"CUBA",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- \n'
                                      '7001,"QUINTERO, Rafael",individual,"SDNT",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,"DOB 04 Mar 1961; POB Cali, Colombia."\n')
    (tmp_path / "alt.csv").write_text('7001,501,"aka","QUINTERO GOMEZ, Rafael",-0- \n')
    loaded = load_list(str(tmp_path / "sdn.csv"))
    assert loaded["format"] == "OFAC_SDN_CSV" and loaded["entry_count"] == 2
    entry = loaded["entries"][1]
    assert (entry["person"], entry["birth_dates"], entry["aliases"]) == (True, ["1961-03-04"], ["QUINTERO GOMEZ, Rafael"])
    person = {"entity_id": "p", "entity_type": "person", "name": "Rafael Quintero Gomez", "names": [{"name": "Rafael Quintero Gomez", "kind": "mention"}], "birth_dates": ["1961-03-04"]}
    checks, issues = screen_entities("d", [person], {"p"}, [loaded])
    assert checks[0]["matches"][0]["listed_name"] == "QUINTERO GOMEZ, Rafael" and [i["code"] for i in issues] == ["SANCTIONS_LIST_POSSIBLE_MATCH"]
