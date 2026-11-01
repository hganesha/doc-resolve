from __future__ import annotations

from datetime import date

from conftest import ROOT
from kyc_pipeline.prior import prior_checks
from test_registry_and_licensing import _ingest

REVIEW = (ROOT / "examples/synthetic_prior_kyc_review.txt").read_text()


def _codes(values):
    return sorted(i["code"] for i in values["issues"] if i["code"].startswith(("PRIOR_", "REVIEW_", "MATERIAL_")))


def test_prior_review_record(tmp_path):
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_prior_kyc_review.txt")
    assert values["profile_id"] == "internal_kyc" and "__interrupt__" not in result
    facts = {f["predicate"]: f for f in values["facts"]}
    assert (facts["kyc_decision"]["value"], facts["kyc_decision"]["qualifier"], facts["kyc_decision"]["effective_date"]) == ("Approved", "Ownership and source of funds evidenced", "2024-03-10")
    assert (facts["assigned_risk_rating"]["value"], facts["next_review_due"]["effective_date"]) == ("Medium", "2026-03-10")
    # The review date dates the company record and the owners on file.
    assert facts["has_registered_address"]["effective_date"] == "2024-03-10"
    assert (facts["is_beneficial_owner_of"]["subject"], facts["is_beneficial_owner_of"]["value"]) == ("Jane Smith", 40.0)
    assert all(f["schema_check"]["status"] == "CONFORMS" for f in values["facts"])
    summary = next(f for f in values["findings"] if f["type"] == "PRIOR_KYC_REVIEW")["details"]
    assert summary == {"name": "Birchwood Trading Limited", "decision": "Approved", "risk_rating": "Medium", "reviewed_on": "2024-03-10", "next_review_due": "2026-03-10"}
    overdue, = [i for i in values["issues"] if i["code"] == "REVIEW_OVERDUE"]
    assert overdue["details"]["review_due"] == "2026-03-10" and overdue["severity"] == "medium"


def test_adverse_decision_and_high_rating(tmp_path):
    text = REVIEW.replace("KYC decision: Approved", "KYC decision: Exited").replace("Decision reason: Ownership and source of funds evidenced", "Decision reason: Source of funds not evidenced").replace("Customer risk rating: Medium", "Customer risk rating: High")
    _, result, values = _ingest(tmp_path, text, defer=True)
    assert "__interrupt__" in result and _codes(values) == ["PRIOR_DECISION_ADVERSE", "PRIOR_HIGH_RISK_RATING", "REVIEW_OVERDUE"]
    adverse = next(i for i in values["issues"] if i["code"] == "PRIOR_DECISION_ADVERSE")
    assert adverse["severity"] == "high" and adverse["details"]["reason"] == "Source of funds not evidenced"


def test_review_not_overdue_after_a_later_decision(tmp_path):
    _, _, values = _ingest(tmp_path, REVIEW)
    assert [i["code"] for i in prior_checks("d", values["facts"], [], today=date(2026, 3, 9))[1]] == []
    later = REVIEW.replace("Review date: 10 March 2024", "Review date: 12 March 2026").replace("Next review due: 10 March 2026", "Next review due: 12 March 2027")
    store, _, _ = _ingest(tmp_path, REVIEW, case="c", run_id="old")
    _, _, values = _ingest(tmp_path, later, case="c", run_id="new", store=store)
    assert _codes(values) == []


def test_material_changes_since_last_review(tmp_path):
    store = None
    for n, name in enumerate(["synthetic_prior_kyc_review.txt", "synthetic_uk_registry_extract.txt", "synthetic_beneficial_ownership_declaration.txt"]):
        store, _, values = _ingest(tmp_path, ROOT / "examples" / name, run_id=f"r{n}", store=store, defer=True)
        changes = [i["details"] for i in values["issues"] if i["code"] == "MATERIAL_CHANGE_SINCE_LAST_REVIEW"]
        if n == 1:
            assert [(c["field"], c["on_file"], c["now"]) for c in changes] == [("registered address", "22 Quay Road, Bristol BS1 6AB", "5 Harbour Street, Bristol BS1 4QA")]
        if n == 2:
            assert [(c["field"], c["added"], c["no_longer_declared"]) for c in changes] == [("beneficial owners", ["Omar Haddad"], [])]
    # A document dated before the review is not a change since it.
    store, _, _ = _ingest(tmp_path, ROOT / "examples/synthetic_prior_kyc_review.txt", case="early", run_id="e0")
    old_extract = (ROOT / "examples/synthetic_uk_registry_extract.txt").read_text().replace("15 January 2026", "15 January 2023")
    _, _, values = _ingest(tmp_path, old_extract, case="early", run_id="e1", store=store)
    assert _codes(values) == []
