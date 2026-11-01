from __future__ import annotations

import os
import subprocess
import sys

import pytest

from conftest import ROOT, defer_all
from kyc_pipeline.config import CONFIG_DIR, load_contract
from kyc_pipeline.ownership import build_ownership_graph, to_mermaid


def _entity(eid, name, entity_type="organization", group_of=None, identifiers=()):
    return {"entity_id": eid, "entity_type": entity_type, "name": name, "group_of": group_of, "identifiers": list(identifiers), "roles": [], "documents": ["d"]}


ENTITIES = [_entity(x, f"{x} Ltd", identifiers=[{"scheme": "LEI", "value": x * 20, "jurisdiction": None}] if x == "A" else ()) for x in "ABCDX"] + [_entity("G", "A Ltd and subsidiaries", "group", group_of="A")]
MAP = {f"doc_{e['entity_id']}": e["entity_id"] for e in ENTITIES}


def _fact(fid, holder, held, value=None, date="2024-12-31", predicate="owns_equity_in", qualifier=None):
    return {"fact_id": fid, "document_id": "d", "category": "ownership", "subject": f"{holder} Ltd", "object": f"{held} Ltd", "predicate": predicate, "value": value, "unit": "percent" if value is not None else None, "effective_date": date, "qualifier": qualifier, "entity_links": {"subject": {"entity_id": f"doc_{holder}"}, "object": {"entity_id": f"doc_{held}"}}}


def _graph(facts, issues=(), **kwargs):
    return build_ownership_graph(facts, ENTITIES, MAP, list(issues), **kwargs)


def _interest(graph, owner, owned):
    return next(i for i in graph["interests"] if (i["owner"], i["owned"]) == (owner, owned))


def test_look_through_multiplies_along_paths_and_adds_across_them():
    graph, issues = _graph([_fact("f1", "A", "B", 60), _fact("f2", "B", "C", 50), _fact("f3", "A", "C", 10), _fact("f4", "B", "C", predicate="has_voting_rights_over")])
    assert not issues
    a_in_c = _interest(graph, "A", "C")
    assert a_in_c["percent"] == {"low": 40.0, "high": 40.0} and a_in_c["direct"] and a_in_c["meets_reporting_threshold"] == "YES"
    assert sorted(p["nodes"] for p in a_in_c["paths"]) == [["A", "B", "C"], ["A", "C"]]
    control = {(c["controller"], c["controlled"]): c for c in graph["control"]}
    assert control[("A", "B")]["basis"] == "MAJORITY_STAKE" and control[("B", "C")]["basis"] == "VOTING_RIGHTS"
    assert control[("A", "C")]["basis"] == "CONTROL_CHAIN" and control[("A", "C")]["path"] == ["A", "B", "C"]
    # A has an identifier but no reported owners; absence is listed, not treated as "none".
    assert graph["identified_entities_without_reported_owners"] == ["A"]
    assert "flowchart TD" in to_mermaid(graph)


def test_conflicting_figures_become_a_range_until_a_reviewer_selects_one():
    facts = [_fact("f1", "A", "B", 40), _fact("f2", "A", "B", 60)]
    conflict = {"issue_id": "i1", "code": "POSSIBLE_OWNERSHIP_CONFLICT", "status": "OPEN", "fact_ids": ["f1", "f2"]}
    graph, _ = _graph(facts, [conflict])
    edge = graph["edges"][0]
    assert edge["status"] == "CONFLICTED" and edge["percent"] == {"low": 40.0, "high": 60.0} and edge["conflict_issue_ids"] == ["i1"]
    assert graph["control"][0]["certainty"] == "POSSIBLE"
    resolved = {**conflict, "status": "RESOLVED_BY_HUMAN", "selected_fact_ids": ["f2"]}
    graph, _ = _graph(facts, [resolved])
    edge = graph["edges"][0]
    assert edge["status"] == "REPORTED" and edge["percent"] == {"low": 60.0, "high": 60.0} and edge["fact_ids"] == ["f2"]
    assert graph["control"][0]["certainty"] == "REPORTED"


def test_latest_period_supersedes_and_a_later_disposal_makes_the_stake_stale():
    facts = [_fact("old", "A", "B", 70, "2022-12-31"), _fact("new", "A", "B", 55, "2024-12-31"), _fact("sale", "A", "B", date="2025-03-01", predicate="disposed_interest_in")]
    edge = _graph(facts)[0]["edges"][0]
    assert edge["fact_ids"] == ["new"] and edge["superseded_fact_ids"] == ["old"] and edge["disposal_fact_ids"] == ["sale"]
    assert edge["status"] == "STALE_AFTER_DISPOSAL" and edge["percent"] == {"low": 0.0, "high": 55.0}
    graph, _ = _graph(facts[:2] + [_fact("sale", "A", "B", date=None, predicate="disposed_interest_in")])
    assert graph["edges"][0]["status"] == "CHRONOLOGY_UNRESOLVED"


def test_group_holding_is_attributed_to_its_parent_without_double_counting():
    graph, issues = _graph([_fact("p", "A", "X", 27.2), _fact("g", "G", "X", 27.0, qualifier="approximately")])
    assert not issues
    interest = _interest(graph, "A", "X")
    assert interest["percent"] == {"low": 27.2, "high": 27.2} and "ATTRIBUTED_FROM_GROUP" in interest["flags"]
    assert any(e["type"] == "GROUP_PARENT" and (e["holder"], e["held"]) == ("A", "G") for e in graph["edges"])
    _, issues = _graph([_fact("p", "A", "X", 40), _fact("g", "G", "X", 20)])
    assert [i["code"] for i in issues] == ["GROUP_STAKE_BELOW_PARENT_STAKE"]


def test_distinct_holders_cannot_own_more_than_all_of_an_entity():
    _, issues = _graph([_fact("f1", "A", "X", 60), _fact("f2", "B", "X", 55)])
    assert [(i["code"], i["severity"], i["fact_ids"]) for i in issues] == [("OWNERSHIP_TOTAL_EXCEEDS_100", "high", ["f1", "f2"])]
    # A's figure may include the stake it holds through B, so A and B are not added.
    assert not _graph([_fact("f1", "A", "X", 60), _fact("f2", "B", "X", 55), _fact("f3", "A", "B", 100)])[1]
    # Different periods are a chronology, not a contradiction.
    assert not _graph([_fact("f1", "A", "X", 60, "2022-12-31"), _fact("f2", "B", "X", 55)])[1]
    # Only issues touching the current document's facts are raised.
    assert not _graph([_fact("f1", "A", "X", 60), _fact("f2", "B", "X", 55)], current_fact_ids={"other"})[1]


def test_cycles_are_reported_and_not_followed():
    graph, issues = _graph([_fact("f1", "A", "B", 30), _fact("f2", "B", "A", 20)])
    assert graph["cycles"] == [["A", "B"]]
    assert [(i["code"], i["severity"]) for i in issues] == [("OWNERSHIP_CYCLE", "medium")]
    assert _interest(graph, "A", "B")["percent"] == {"low": 30.0, "high": 30.0}


def test_policy_rejects_invalid_ownership_settings(tmp_path):
    policy = (CONFIG_DIR / "kyc_policy.yaml").read_text().replace("reporting_threshold_percent: 25", "reporting_threshold_percent: 250")
    path = tmp_path / "policy.yaml"
    path.write_text(policy)
    with pytest.raises(ValueError, match="reporting_threshold_percent"):
        load_contract(policy_path=path)


def test_cli_builds_a_chain_across_documents_and_flags_over_100(cli, paths):
    docs = {"aster.txt": "As of 2024-12-31, Aster Holdings Ltd owns 60% of Birch Ltd.", "birch.txt": "As of 2024-12-31, Birch Ltd owns 50% of Cedar Ltd.", "dune.txt": "As of 2024-12-31, Dune Ltd owns 70% of Birch Ltd."}
    base = ("--case-id", "chain", "--profile", "ownership_control", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    for name in ("aster.txt", "birch.txt"):
        (paths["tmp"] / name).write_text("# Ownership\n" + docs[name] + "\n")
        assert cli("ingest", paths["tmp"] / name, *base)["status"] == "PERSISTED"
    result = cli("ownership", "--case-id", "chain", "--db", paths["db"])
    graph = result["ownership_graph"]
    names = {n["node_id"]: n["name"] for n in graph["nodes"]}
    chain = next(i for i in graph["interests"] if (names[i["owner"]], names[i["owned"]]) == ("Aster Holdings Ltd", "Cedar Ltd"))
    assert chain["percent"] == {"low": 30.0, "high": 30.0} and not chain["direct"] and chain["meets_reporting_threshold"] == "YES"
    (paths["tmp"] / "dune.txt").write_text("# Ownership\n" + docs["dune.txt"] + "\n")
    third = cli("ingest", paths["tmp"] / "dune.txt", *base)
    assert third["status"] == "PAUSED_FOR_REVIEW"
    assert any(i["code"] == "OWNERSHIP_TOTAL_EXCEEDS_100" for i in third["review_request"]["issues"])
    cli("resume", "--run-id", third["run_id"], "--decision", defer_all(third, paths["tmp"]), "--db", paths["db"], "--checkpoint-db", paths["cp"])
    out = paths["tmp"] / "graph.mmd"
    # The fixture parses JSON; Mermaid is plain text, so run it directly.
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "kyc_pipeline", "ownership", "--case-id", "chain", "--format", "mermaid", "--output", str(out), "--db", paths["db"]], cwd=ROOT, env=env, check=True, timeout=90)
    mermaid = out.read_text()
    assert mermaid.startswith("flowchart TD") and "Dune Ltd" in mermaid and "60% (2024-12-31" in mermaid
