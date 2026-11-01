from __future__ import annotations

from conftest import ROOT
from kyc_pipeline.media import topic_of
from test_registry_and_licensing import _ingest

CHARGED, FOLLOWUP, NOTICE = (ROOT / "examples" / n for n in ("synthetic_media_article_charged.txt", "synthetic_media_article_followup.txt", "synthetic_enforcement_notice.txt"))


def _facts(values):
    return {(f["predicate"], f["subject"], f.get("object"), f.get("qualifier"), f.get("effective_date")) for f in values["facts"]}


def _events(values):
    return {(f["details"]["subject"], f["details"]["topic"]): f["details"] for f in values["findings"] if f["type"] == "ADVERSE_MEDIA_EVENT"}


def test_article_metadata_and_outcome_stages(tmp_path):
    _, result, values = _ingest(tmp_path, CHARGED, defer=True)
    assert values["profile_id"] == "adverse_media" and "__interrupt__" in result
    headline = "Trading firm director charged over bribery scheme"
    assert {
        ("published_by", headline, "The Harbour Gazette", None, None),
        ("published_on", headline, None, None, "2026-06-03"),
        ("charged_with", "Omar Haddad", "bribery of public officials", None, "2026-06-02"),
        ("denied", "Omar Haddad", "the allegations", None, None),
        ("under_investigation_for", "Northgate Shipping SA", "suspected money laundering", "Serious Fraud Office", None),
    } == _facts(values)
    assert all(f["schema_check"]["status"] == "CONFORMS" and f["source_support"]["literal_checks"]["subject_literal_present"] for f in values["facts"])
    assert next(f for f in values["facts"] if f["predicate"] == "denied")["claim_type"] == "denial"
    # The headline is not a party.
    assert sorted(e["name"] for e in values["case_entities"]) == ["Northgate Shipping SA", "Omar Haddad"]
    issues = {i["details"]["subject"]: (i["code"], i["severity"], i["details"]["status"]) for i in values["issues"] if i["code"].startswith("ADVERSE_")}
    # A charge for a financial crime is high; an investigation is medium.
    assert issues == {"Omar Haddad": ("ADVERSE_MEDIA_OPEN_MATTER", "high", "CHARGED"), "Northgate Shipping SA": ("ADVERSE_MEDIA_OPEN_MATTER", "medium", "UNDER_INVESTIGATION")}
    assert _events(values)[("Omar Haddad", "bribery and corruption")]["single_source"] is True


def test_events_follow_the_story_across_documents(tmp_path):
    store, _, _ = _ingest(tmp_path, CHARGED, case="story", run_id="a", defer=True)
    _, _, values = _ingest(tmp_path, FOLLOWUP, case="story", run_id="b", store=store, defer=True)
    events = _events(values)
    cleared = events[("Omar Haddad", "bribery and corruption")]
    assert (cleared["status"], [s["stage"] for s in cleared["stages"]], cleared["distinct_sources"]) == ("CLEARED", ["CHARGED", "CLEARED"], 2)
    assert cleared["denied"][0]["text"] == "the allegations"
    fined = events[("Northgate Shipping SA", "money laundering")]
    assert fined["status"] == "PENALIZED" and fined["sources"] == ["The Harbour Gazette", "Western Business Daily"]
    penalty = next(f for f in values["facts"] if f["predicate"] == "received_regulatory_penalty")
    assert (penalty["object"], penalty["value"], penalty["unit"], penalty["qualifier"]) == ("Coastal Markets Authority", 2.4, "GBP_millions", "failures in anti-money laundering controls")
    # A clearance raises nothing; the penalty is established.
    assert sorted((i["code"], i["details"]["subject"]) for i in values["issues"] if i["code"].startswith("ADVERSE_")) == [("ADVERSE_MEDIA_ESTABLISHED", "Northgate Shipping SA")]


def test_enforcement_notice(tmp_path):
    store, _, _ = _ingest(tmp_path, FOLLOWUP, case="notice", run_id="a", defer=True)
    _, _, values = _ingest(tmp_path, NOTICE, case="notice", run_id="b", store=store, defer=True)
    assert values["profile_id"] == "regulatory_licensing"
    facts = {(f["predicate"], f.get("object")): f for f in values["facts"]}
    penalty = facts[("received_regulatory_penalty", "Coastal Markets Authority")]
    assert (penalty["value"], penalty["unit"], penalty["claim_type"], penalty["effective_date"]) == (2400000.0, "GBP", "order", "2026-10-09")
    assert {o for p, o in facts if p == "subject_to_enforcement_action"} == {"Public censure", "Restriction on onboarding new customers for 12 months"}
    # The notice's "Regulator:" line is not read as a licence issuer.
    assert not any(f["category"] == "license_status" for f in values["facts"])
    event, = [i["details"] for i in values["issues"] if i["code"] == "ADVERSE_MEDIA_ESTABLISHED"]
    # The censure has no stated conduct, so it takes the notice's topic; the notice counts as the regulator's own source.
    assert event["topic"] == "money laundering" and event["sources"] == ["Coastal Markets Authority", "Western Business Daily"]


def test_outcome_sentence_variants(tmp_path):
    text = "\n".join([
        "Headline: Roundup", "Publisher: Example Wire", "Published: 1 July 2026", "",
        "Kestrel Mining Plc pleaded guilty to bribery of officials in Ghana.",
        "The charges of tax evasion against Lena Park were dropped.",
        "Delta Freight Ltd agreed to pay USD 12 million to settle sanctions violations claims.",
        "The Serious Fraud Office opened an investigation into Aster Holdings Ltd over suspected false accounting.",
        "Aster Holdings Ltd was accused of bribery in 2024 by a named critic.",
    ])
    _, _, values = _ingest(tmp_path, text, profile="adverse_media", defer=True)
    facts = _facts(values)
    assert ("convicted_of", "Kestrel Mining Plc", "bribery of officials", None, None) in facts
    assert ("cleared_of", "Lena Park", "charges of tax evasion", None, None) in facts
    assert ("under_investigation_for", "Aster Holdings Ltd", "suspected false accounting", "Serious Fraud Office", None) in facts
    settled = next(f for f in values["facts"] if f["predicate"] == "settled")
    assert (settled["object"], settled["value"], settled["unit"]) == ("sanctions violations claims", 12.0, "USD_millions")
    statuses = {(d["subject"], d["topic"]): d["status"] for d in _events(values).values()}
    assert statuses == {("Kestrel Mining Plc", "bribery and corruption"): "CONVICTED", ("Lena Park", "tax"): "CLEARED", ("Delta Freight Ltd", "sanctions"): "SETTLED", ("Aster Holdings Ltd", "fraud"): "UNDER_INVESTIGATION", ("Aster Holdings Ltd", "bribery and corruption"): "ALLEGED"}


def test_topics():
    assert [topic_of(t) for t in ("anti-money laundering controls", "bribing a minister", "evading US sanctions", "insider dealing", "mis-selling", "workplace safety")] == ["money laundering", "bribery and corruption", "sanctions", "market abuse", "consumer protection", "other"]
