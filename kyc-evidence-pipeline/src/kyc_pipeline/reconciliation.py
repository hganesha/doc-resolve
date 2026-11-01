from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from .entities import legal_key
from .evidence import make_issue
from .util import date_interval, identity_key, json_text, stable_id, temporal_overlap


APPROXIMATE = re.compile(r"\b(?:approximately|about|roughly|nearly|almost|around|generally)\b", re.I)


def party_key(fact: dict, role: str, entity_map: dict[str, str]) -> str:
    """Case entity id where the entity layer linked one, else the conservative text key."""
    link = fact.get("entity_links", {}).get(role)
    if link and link["entity_id"] in entity_map:
        return entity_map[link["entity_id"]]
    return "text:" + identity_key(fact.get(role)) if fact.get(role) else ""


def _decimals(value: float) -> int:
    text = repr(float(value)).rstrip("0").rstrip(".")
    return len(text.split(".")[1]) if "." in text else 0


def rounding_consistent(a: dict, b: dict) -> bool:
    """'28%' and '28.2%' agree at the coarser precision; so do 'approximately 27%' and '27.2%'."""
    x, y = float(a["value"]), float(b["value"])
    places = min(_decimals(x), _decimals(y))
    if round(x, places) == round(y, places):
        return True
    approximate = any(APPROXIMATE.search(f.get("qualifier") or "") for f in (a, b))
    return approximate and round(x) == round(y)


def _attribute_value(fact: dict, per_object: bool = False):
    return fact.get("value") if per_object or fact.get("object") is None else fact.get("object")


def _attribute_key(fact: dict, per_object: bool = False) -> str:
    """'BERKSHIRE HATHAWAY INC.' and 'Berkshire Hathaway Incorporated' agree; so do '12-3456789' and '123456789'."""
    value = _attribute_value(fact, per_object)
    return re.sub(r"\s+", "", legal_key(value)) if isinstance(value, str) else json_text(value)


def issue_finding(issue: dict) -> dict:
    return {"finding_id": stable_id("finding", issue["issue_id"]), "type": issue["code"], "status": "OPEN", "category": issue["categories"][0] if issue["categories"] else "general", "issue_id": issue["issue_id"], "fact_ids": issue["fact_ids"], "description": issue["message"]}


def reconcile(document: dict, current_facts: list[dict], previous_facts: list[dict], existing_issues: list[dict], entity_map: dict[str, str] | None = None, schemas: dict[str, dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Conservative matching on resolved entities (or text, where unresolved) and chronology. No UBO calculation."""
    entity_map = entity_map or {}
    single_valued = {name: spec["single_valued"] == "per_object" for name, spec in (schemas or {}).items() if spec.get("single_valued")}

    def _key(fact: dict) -> tuple[str, str]:
        return party_key(fact, "subject", entity_map), party_key(fact, "object", entity_map)

    document_id = document["document_id"]
    current_ids = {f["fact_id"] for f in current_facts}
    all_facts = previous_facts + current_facts
    issues = list(existing_issues)
    findings: list[dict] = []

    # Preserve each source claim; a grouped finding is not proof of independent corroboration.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for f in all_facts:
        sig = (_key(f), f["predicate"], json_text(f.get("value")), f.get("unit"), f.get("effective_date"), f.get("claim_type"))
        groups[sig].append(f)
    for group in groups.values():
        ids = sorted({f["fact_id"] for f in group})
        if len(ids) >= 2 and any(i in current_ids for i in ids):
            findings.append({"finding_id": stable_id("finding", "similar_claims", ids), "type": "SIMILAR_SOURCE_CLAIMS", "status": "INFORMATIONAL", "category": group[0]["category"], "fact_ids": ids, "description": "Multiple source claims have the same normalized fields; independence and source reliability are not established."})

    ownership = [f for f in all_facts if f["predicate"] == "owns_equity_in" and f.get("object") and isinstance(f.get("value"), (int, float))]
    by_pair: dict[tuple, list[dict]] = defaultdict(list)
    for f in ownership:
        by_pair[_key(f)].append(f)
    for pair, snapshots in by_pair.items():
        if not all(pair):
            continue
        for a, b in combinations(snapshots, 2):
            if not ({a["fact_id"], b["fact_id"]} & current_ids):
                continue
            if a.get("unit") != b.get("unit") or a["value"] == b["value"]:
                continue
            if rounding_consistent(a, b):
                ids = sorted([a["fact_id"], b["fact_id"]])
                findings.append({"finding_id": stable_id("finding", "rounding", ids), "type": "OWNERSHIP_FIGURES_DIFFER_BY_ROUNDING", "status": "INFORMATIONAL", "category": "ownership", "fact_ids": ids, "description": "The same parties' stakes differ only by rounding or a stated approximation; treated as consistent, not as a conflict."})
                continue
            if not temporal_overlap(a.get("effective_date"), b.get("effective_date")):
                continue
            issue = make_issue("POSSIBLE_OWNERSHIP_CONFLICT", "The same parties have different ownership percentages in overlapping or unknown time periods. This may be a contradiction or an unresolved chronology.", document_id=document_id, categories=["ownership"], fact_ids=[a["fact_id"], b["fact_id"]], details={"dates": sorted([a.get("effective_date") or "unknown", b.get("effective_date") or "unknown"])}, allowed_actions=["resolve", "defer"])
            issues.append(issue)

    sales = [f for f in all_facts if f["predicate"] == "disposed_interest_in" and f.get("object")]
    for sale in sales:
        pair = _key(sale)
        candidates = by_pair.get(pair, [])
        for snapshot in candidates:
            if not ({sale["fact_id"], snapshot["fact_id"]} & current_ids):
                continue
            si, oi = date_interval(sale.get("effective_date")), date_interval(snapshot.get("effective_date"))
            if si and oi and si[0] > oi[1]:
                # A later reported ownership snapshot may address staleness; do not calculate a new stake.
                has_newer = any((date_interval(f.get("effective_date")) or (None, None))[0] is not None and date_interval(f["effective_date"])[0] > si[1] for f in candidates)
                if not has_newer:
                    issues.append(make_issue("STALE_OWNERSHIP_AFTER_DISPOSAL", "An ownership snapshot predates a disposal; the current percentage cannot be inferred from these claims.", document_id=document_id, categories=["ownership", "material_changes"], fact_ids=[sale["fact_id"], snapshot["fact_id"]], details={"ownership_date": snapshot.get("effective_date"), "disposal_date": sale.get("effective_date")}))
            elif not si or not oi:
                issues.append(make_issue("OWNERSHIP_CHRONOLOGY_UNRESOLVED", "A disposal and ownership claim match by text, but missing dates prevent a reliable chronology.", document_id=document_id, categories=["ownership", "material_changes"], fact_ids=[sale["fact_id"], snapshot["fact_id"]]))

    # A typed schema marks some predicates single-valued: one legal name,
    # jurisdiction or identifier per entity at a time. Differing values for the
    # same entity are either a change over time or a wrong merge; both need review.
    attributes: dict[tuple, list[dict]] = defaultdict(list)
    for f in all_facts:
        if f["predicate"] in single_valued and f.get("entity_resolution") != "UNRESOLVED" and not f["subject"].startswith("UNRESOLVED_"):
            subject = party_key(f, "subject", entity_map)
            per_object = single_valued[f["predicate"]]
            if subject and (not per_object or f.get("value") is not None):
                # Per-object attributes (a licence's status) are keyed by the object too.
                attributes[(subject, f["predicate"], legal_key(f.get("object") or "") if per_object else "")].append(f)
    for (_, predicate, _), claims in attributes.items():
        per_object = single_valued[predicate]
        for a, b in combinations(claims, 2):
            if not ({a["fact_id"], b["fact_id"]} & current_ids) or _attribute_key(a, per_object) == _attribute_key(b, per_object):
                continue
            if not temporal_overlap(a.get("effective_date"), b.get("effective_date")):
                continue
            issues.append(make_issue("POSSIBLE_ATTRIBUTE_CONFLICT", f"The same entity has different values for {predicate} in overlapping or unknown periods. This may be an error, a change over time such as a rename, or two entities wrongly merged.", document_id=document_id, categories=sorted({a["category"], b["category"]}), fact_ids=[a["fact_id"], b["fact_id"]], details={"predicate": predicate, **({"object": a.get("object")} if per_object else {}), "values": sorted({str(_attribute_value(a, per_object)), str(_attribute_value(b, per_object))})}, allowed_actions=["resolve", "defer"]))

    unique = {issue["issue_id"]: issue for issue in issues}
    issues = list(unique.values())
    findings += [issue_finding(issue) for issue in issues]
    return findings, issues
