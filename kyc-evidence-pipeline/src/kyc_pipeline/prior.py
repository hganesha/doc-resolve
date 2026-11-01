"""Checks against the bank's own prior KYC file.

* A next-review date that has passed with no later decision on file.
* A prior decision that was adverse (declined, exited, escalated), or a high
  risk rating: a returning customer needs the earlier reasoning reviewed.
* Material change since the last review: a registered or residential address,
  legal name, or beneficial owner that a later-dated document states
  differently from the prior file.

A "review file" is any document that records a decision, risk rating or
next-review date; its review date is that decision's date. Facts are compared
by case entity, so different spellings resolved as one entity agree.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

from .evidence import make_issue
from .util import identity_key, stable_id

_ADVERSE = re.compile(r"\b(?:declin|reject|refus|exit|terminat|offboard|clos|suspend|escalat|sar\b|suspicious)", re.I)
_HIGH_RISK = re.compile(r"\b(?:high|very\s+high|prohibited|unacceptable)\b", re.I)
_REVIEW_PREDICATES = {"kyc_decision", "assigned_risk_rating", "next_review_due"}
_CHANGEABLE = {"has_registered_address": "registered address", "has_residential_address": "residential address", "has_legal_name": "legal name"}


def _finding(kind: str, fact_ids: list[str], description: str, details: dict) -> dict:
    ids = sorted(set(fact_ids))
    return {"finding_id": stable_id("finding", kind, ids), "type": kind, "status": "INFORMATIONAL", "category": "prior_decisions", "fact_ids": ids, "description": description, "details": details}


def _full_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value and len(value) == 10 else None


def prior_checks(document_id: str, facts: list[dict], previous: list[dict], entity_map: dict[str, str] | None = None, today: date | None = None) -> tuple[list[dict], list[dict]]:
    """Prior-file checks for one document against the case. Returns (findings, issues)."""
    today = today or datetime.now(timezone.utc).date()
    entity_map = entity_map or {}
    current = {f["fact_id"] for f in facts}
    everything = previous + facts

    def entity(f: dict, role: str = "subject") -> str:
        link = (f.get("entity_links") or {}).get(role)
        name = f["subject"] if role == "subject" else f.get("object") or ""
        return entity_map.get(link["entity_id"], identity_key(name)) if link else identity_key(name)

    findings, issues = [], []
    # Each review file and its review date.
    reviews: dict[str, date | None] = {}
    for f in everything:
        if f["predicate"] in _REVIEW_PREDICATES:
            when = _full_date(f.get("effective_date")) if f["predicate"] != "next_review_due" else None
            if f["document_id"] not in reviews or (when and (reviews[f["document_id"]] is None or when > reviews[f["document_id"]])):
                reviews[f["document_id"]] = when or reviews.get(f["document_id"])

    for f in facts:
        if f["predicate"] == "kyc_decision" and _ADVERSE.search(f.get("value") or ""):
            issues.append(make_issue("PRIOR_DECISION_ADVERSE", "The bank's own file records an adverse decision on this customer; review its reasoning before proceeding.", document_id=document_id, categories=["prior_decisions"], severity="high", fact_ids=[f["fact_id"]], details={"name": f["subject"], "decision": f["value"], "reason": f.get("qualifier"), "decided_on": f.get("effective_date")}))
        if f["predicate"] == "assigned_risk_rating" and _HIGH_RISK.search(f.get("value") or ""):
            issues.append(make_issue("PRIOR_HIGH_RISK_RATING", "The bank's own file rated this customer high risk; enhanced due diligence usually applies.", document_id=document_id, categories=["prior_decisions"], severity="medium", fact_ids=[f["fact_id"]], details={"name": f["subject"], "rating": f["value"], "rated_on": f.get("effective_date")}))
    review_facts = [f for f in facts if f["predicate"] in _REVIEW_PREDICATES]
    if review_facts:
        by = {f["predicate"]: f for f in review_facts}
        findings.append(_finding("PRIOR_KYC_REVIEW", [f["fact_id"] for f in review_facts], "The bank's own KYC file, as recorded. It is the bank's earlier judgment, not evidence about the customer.", {"name": review_facts[0]["subject"], "decision": by.get("kyc_decision", {}).get("value"), "risk_rating": by.get("assigned_risk_rating", {}).get("value"), "reviewed_on": reviews.get(document_id) and reviews[document_id].isoformat(), "next_review_due": by.get("next_review_due", {}).get("effective_date")}))

    # Overdue: the due date passed and no decision on file is dated on or after it.
    for f in everything:
        due = _full_date(f.get("effective_date")) if f["predicate"] == "next_review_due" else None
        if not due or due >= today:
            continue
        later = [d for d in everything if d["predicate"] == "kyc_decision" and entity(d) == entity(f) and (_full_date(d.get("effective_date")) or date.min) >= due]
        if later or not ({f["fact_id"]} & current or f["document_id"] == document_id):
            continue
        issues.append(make_issue("REVIEW_OVERDUE", "The customer's periodic review was due before this run and no later decision is on file.", document_id=document_id, categories=["prior_decisions"], severity="medium", fact_ids=[f["fact_id"]], details={"name": f["subject"], "review_due": f["effective_date"], "days_overdue": (today - due).days}))

    # Material change: a later-dated document states an address, name or owner differently.
    for review_doc, reviewed in reviews.items():
        if not reviewed:
            continue
        on_file = [f for f in everything if f["document_id"] == review_doc]
        later = [f for f in everything if f["document_id"] != review_doc and (_full_date(f.get("effective_date")) or date.min) > reviewed]
        if not later or not (review_doc == document_id or {f["fact_id"] for f in later} & current):
            continue
        for predicate, label in _CHANGEABLE.items():
            for old in (f for f in on_file if f["predicate"] == predicate):
                for new in (f for f in later if f["predicate"] == predicate and entity(f) == entity(old)):
                    if identity_key(new.get("object") or "") != identity_key(old.get("object") or "") and (old["fact_id"] in current or new["fact_id"] in current):
                        issues.append(make_issue("MATERIAL_CHANGE_SINCE_LAST_REVIEW", f"The {label} differs from the one on the bank's file at the last review; confirm the change and whether it triggers a review.", document_id=document_id, categories=["material_changes", "prior_decisions"], severity="medium", fact_ids=[old["fact_id"], new["fact_id"]], details={"name": old["subject"], "field": label, "on_file": old.get("object"), "reviewed_on": reviewed.isoformat(), "now": new.get("object"), "stated_on": new.get("effective_date")}))
        # Beneficial owners of each entity the file lists owners for.
        owned = {entity(f, "object") for f in on_file if f["predicate"] == "is_beneficial_owner_of"}
        for company in owned:
            before = {entity(f): f for f in on_file if f["predicate"] == "is_beneficial_owner_of" and entity(f, "object") == company}
            after = {entity(f): f for f in later if f["predicate"] == "is_beneficial_owner_of" and entity(f, "object") == company}
            if not after:
                continue
            added = [after[k] for k in sorted(set(after) - set(before))]
            removed = [before[k] for k in sorted(set(before) - set(after))]
            involved = [f["fact_id"] for f in list(before.values()) + list(after.values())]
            if (added or removed) and set(involved) & current:
                name = next(iter(before.values()))["object"]
                issues.append(make_issue("MATERIAL_CHANGE_SINCE_LAST_REVIEW", "The beneficial owners declared since the last review differ from those on the bank's file; confirm the change in ownership.", document_id=document_id, categories=["material_changes", "prior_decisions", "ownership"], severity="medium", fact_ids=sorted(involved), details={"name": name, "field": "beneficial owners", "reviewed_on": reviewed.isoformat(), "on_file": sorted(f["subject"] for f in before.values()), "now": sorted(f["subject"] for f in after.values()), "added": [f["subject"] for f in added], "no_longer_declared": [f["subject"] for f in removed]}))
    return findings, issues
