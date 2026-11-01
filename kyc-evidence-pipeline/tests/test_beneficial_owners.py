from __future__ import annotations

from conftest import ROOT
from kyc_pipeline.ownership import to_mermaid
from kyc_pipeline.store import EvidenceStore
from test_companies_house import NUMBER, _ch, _company
from test_registry_and_licensing import _ingest

DECLARATION = (ROOT / "examples/synthetic_beneficial_ownership_declaration.txt").read_text()
REGISTER = ROOT / "examples/synthetic_shareholder_register.txt"


def _case(tmp_path, *sources, **kw):
    store = EvidenceStore(tmp_path / "db.sqlite")
    results = []
    for n, source in enumerate(sources):
        _, result, values = _ingest(tmp_path, source, case="ubo", run_id=f"run_{n}", store=store, profile="ownership_control", defer=True, **kw)
        results.append((result, values))
    return store, results, store.export_case("ubo")["ownership_graph"]


def _report(graph, name):
    entry = next(b for b in graph["beneficial_owners"] if b["name"] == name)
    return entry, {o["name"]: o for o in entry["owners"]}


def test_declared_owners_match_the_holdings(tmp_path):
    _, results, graph = _case(tmp_path, DECLARATION, REGISTER)
    assert all("__interrupt__" not in r for r, _ in results)
    entry, owners = _report(graph, "Birchwood Trading Limited")
    assert entry["status"] == "IDENTIFIED" and entry["direct_holdings_percent"] == {"low": 100.0, "high": 100.0}
    # Omar Haddad holds 60% of Aster, which holds 60% of Birchwood: 36%, and control through Aster's majority.
    assert (owners["Omar Haddad"]["interest"], owners["Omar Haddad"]["bases"]) == ({"low": 36.0, "high": 36.0}, ["HOLDINGS", "CONTROL", "DECLARED"])
    assert owners["Omar Haddad"]["control"]["basis"] == "CONTROL_CHAIN"
    # Jane Smith's 40% of the votes is not control; Lena Park's 24% look-through is below the threshold.
    assert owners["Jane Smith"]["control"] is None and owners["Jane Smith"]["bases"] == ["HOLDINGS", "DECLARED"]
    assert "Lena Park" not in owners
    assert {n["name"]: n["kind"] for n in graph["nodes"]}["Lena Park"] == "person"
    mermaid = to_mermaid(graph)
    assert '(["Omar Haddad"]):::person' in mermaid and "40% votes" in mermaid


def test_an_omitted_owner_and_a_wrong_percentage_are_raised(tmp_path):
    declaration = DECLARATION.replace("| Omar Haddad | 36 | Indirect, through Aster Holdings Ltd |\n", "").replace("| Jane Smith | 40 | Direct shareholding |", "| Jane Smith | 45 | Direct shareholding |")
    _, results, graph = _case(tmp_path, declaration, REGISTER)
    first, second = ([(i["code"], i["details"]) for i in v["issues"] if i["code"].startswith("UBO_")] for _, v in results)
    assert first == [("UBO_DECLARED_PERCENT_DIFFERS", {"company": first[0][1]["company"], "company_name": "Birchwood Trading Limited", "person": first[0][1]["person"], "name": "Jane Smith", "declared": [45.0], "basis": "HOLDINGS", "low": 40.0, "high": 40.0})]
    # Omar Haddad's stake only appears once Aster's register is in the case, so that run raises the omission.
    assert [(code, d["name"], d["bases"]) for code, d in second] == [("UBO_NOT_DECLARED", "Omar Haddad", ["HOLDINGS", "CONTROL"])]
    assert "__interrupt__" in results[1][0]
    # Deferring the issues completes the run, so the exported graph includes the register.
    entry, owners = _report(graph, "Birchwood Trading Limited")
    assert owners["Omar Haddad"]["declared"] is None and owners["Jane Smith"]["declared"]["percent"] == [45.0]


def test_register_pscs_fill_gaps_and_raise_undeclared_people(tmp_path):
    pscs = [{"name": "Mrs Jane Elizabeth Smith", "kind": "individual-person-with-significant-control", "natures_of_control": ["ownership-of-shares-25-to-50-percent"]},
            {"name": "Aster Holdings Ltd", "kind": "corporate-entity-person-with-significant-control", "natures_of_control": ["ownership-of-shares-50-to-75-percent", "voting-rights-50-to-75-percent"]},
            {"name": "Mr Ravi Patel", "kind": "individual-person-with-significant-control", "natures_of_control": ["significant-influence-or-control"]}]
    client, _ = _ch(_company(), officers=[], pscs=pscs)
    # Only the beneficial-owner table: the shareholdings come from the register.
    declaration = DECLARATION.split("| Shareholder |")[0] + "| Beneficial owner |" + DECLARATION.split("| Beneficial owner |")[1]
    _, results, graph = _case(tmp_path, declaration, registry_check=["companies-house"], registry_factory=lambda *_: client)
    issues = [(i["code"], i["details"]["name"]) for i in results[0][1]["issues"] if i["code"].startswith("UBO_")]
    assert issues == [("UBO_NOT_DECLARED", "Mr Ravi Patel")]
    entry, owners = _report(graph, "Birchwood Trading Limited")
    assert owners["Jane Smith"]["register_psc"]["share_band"] == [25.0, 50.0] and owners["Jane Smith"]["interest"] == {"low": 25.0, "high": 50.0}
    assert owners["Mr Ravi Patel"]["control"]["basis"] == "REGISTER_PSC"
    # Aster is a corporate PSC whose owners the case doesn't show; Omar Haddad is declared but not corroborated.
    assert [c["name"] for c in entry["chain_ends_at"]] == ["Aster Holdings Ltd"]
    assert owners["Omar Haddad"]["flags"] == ["NOT_CORROBORATED"]
    assert any(e["type"] == "REGISTER_PSC" and e["source"] == "COMPANIES_HOUSE" for e in graph["edges"])


def test_no_identified_owner_lists_directors(tmp_path):
    text = """# Shareholder Register
Company name: Elm Ventures Ltd
Register date: 31 December 2025
Directors: Ms Jane Smith; Omar Haddad

| Shareholder | Type | Shares held (%) |
| --- | --- | --- |
| Cedar Nominees Ltd | Company | 100 |
"""
    _, _, graph = _case(tmp_path, text)
    entry, owners = _report(graph, "Elm Ventures Ltd")
    assert entry["status"] == "NOT_IDENTIFIED" and owners == {}
    assert [c["name"] for c in entry["chain_ends_at"]] == ["Cedar Nominees Ltd"]
    assert entry["senior_managing_officials"] == ["Ms Jane Smith", "Omar Haddad"]


def test_one_person_across_documents(tmp_path):
    other = """# Shareholder Register
Company name: Elm Ventures Ltd
Register date: 31 December 2025

| Shareholder | Type | Shares held (%) |
| --- | --- | --- |
| Ms Jane Smith | Individual | 30 |
"""
    _, _, graph = _case(tmp_path, DECLARATION, other)
    people = [n for n in graph["nodes"] if n["kind"] == "person" and "Smith" in (n["name"] or "")]
    assert len(people) == 1
    assert {b["name"] for b in graph["beneficial_owners"] if any(o["person"] == people[0]["node_id"] for o in b["owners"])} == {"Birchwood Trading Limited", "Elm Ventures Ltd"}
