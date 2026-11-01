"""Adverse media and enforcement: one event per subject and topic, across documents.

Each outcome-stage fact (accused, under investigation, charged, settled,
fined or other enforcement, convicted or found responsible, cleared, denied)
is filed under its subject's case entity and a topic read from the conduct
(bribery, fraud, money laundering, ...). An event's status is the furthest
stage reached, unless a later clearance ends it. Sources are counted by
publisher (or by document where none is named); several publishers are not
proof of independent reporting, since outlets often repeat one another.
"""
from __future__ import annotations

import re
from collections import defaultdict

from .evidence import make_issue
from .util import identity_key, stable_id

TOPICS: list[tuple[str, str]] = [
    ("money laundering", r"launder|\baml\b|anti-money"),
    ("sanctions", r"sanction|embargo|export\s+control"),
    ("terrorist financing", r"terroris"),
    ("bribery and corruption", r"brib|corrupt|kickback"),
    ("fraud", r"fraud|deceiv|decept|misrepresent|ponzi|embezzl|false\s+accounting"),
    ("tax", r"\btax"),
    ("market abuse", r"insider|market\s+(?:abuse|manipulation)|spoofing"),
    ("environmental", r"pollut|environment|emission"),
    ("labour and human rights", r"forced\s+labou?r|modern\s+slavery|human\s+rights|child\s+labou?r"),
    ("consumer protection", r"mis-?sell|consumer"),
]
FINANCIAL_CRIME = {"money laundering", "sanctions", "terrorist financing", "bribery and corruption", "fraud", "tax"}
# Furthest stage first wins the status; CLEARED is handled separately.
STAGES = {"accused_of": ("ALLEGED", 1), "under_investigation_for": ("UNDER_INVESTIGATION", 2), "subject_to_regulatory_investigation": ("UNDER_INVESTIGATION", 2), "charged_with": ("CHARGED", 3), "settled": ("SETTLED", 4), "received_regulatory_penalty": ("PENALIZED", 5), "subject_to_enforcement_action": ("PENALIZED", 5), "convicted_of": ("CONVICTED", 5), "found_responsible_for": ("CONVICTED", 5)}
ESTABLISHED = {"PENALIZED", "CONVICTED"}


def topic_of(text: str | None) -> str:
    for name, pattern in TOPICS:
        if re.search(pattern, text or "", re.I):
            return name
    return "other"


def media_checks(document_id: str, facts: list[dict], previous: list[dict], entity_map: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Group the case's outcome facts into events and raise those this document adds to. Returns (findings, issues)."""
    entity_map = entity_map or {}
    current = {f["fact_id"] for f in facts}
    everything = previous + facts

    def who(f: dict) -> str:
        link = (f.get("entity_links") or {}).get("subject")
        return entity_map.get(link["entity_id"], identity_key(f["subject"])) if link else identity_key(f["subject"])

    publisher = {f["document_id"]: f["object"] for f in everything if f["predicate"] == "published_by" and f.get("object")}
    published = {f["document_id"]: f["effective_date"] for f in everything if f["predicate"] == "published_on" and f.get("effective_date")}
    events: dict[tuple[str, str], list[dict]] = defaultdict(list)
    denials: dict[str, list[dict]] = defaultdict(list)
    outcomes = [f for f in everything if f["predicate"] in STAGES or f["predicate"] == "cleared_of"]
    topics = {f["fact_id"]: topic_of(" ".join(str(x) for x in (f.get("object"), f.get("qualifier")) if x)) for f in outcomes}
    # An action stated without its conduct (a censure) takes the topic the same document gives its subject.
    in_document: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in outcomes:
        if topics[f["fact_id"]] != "other":
            in_document[(f["document_id"], who(f))].add(topics[f["fact_id"]])
    for f in everything:
        if f["predicate"] == "denied":
            denials[who(f)].append(f)
    for f in outcomes:
        topic = topics[f["fact_id"]]
        known = in_document.get((f["document_id"], who(f)), set())
        events[(who(f), next(iter(known)) if topic == "other" and len(known) == 1 else topic)].append(f)

    def source(f: dict) -> str:
        # An official notice is its regulator's own statement.
        regulator = f.get("object") if f["predicate"] == "received_regulatory_penalty" else f.get("qualifier") if f["predicate"] == "subject_to_enforcement_action" else None
        return publisher.get(f["document_id"]) or (regulator if f.get("claim_type") == "order" and regulator else f"document {f['document_id']}")

    findings, issues = [], []
    for (entity, topic), group in sorted(events.items()):
        when = lambda f: f.get("effective_date") or published.get(f["document_id"]) or ""
        group = sorted(group, key=lambda f: (when(f), f["fact_id"]))
        adverse = [f for f in group if f["predicate"] in STAGES]
        furthest = max(adverse, key=lambda f: STAGES[f["predicate"]][1], default=None)
        cleared = [f for f in group if f["predicate"] == "cleared_of"]
        status = STAGES[furthest["predicate"]][0] if furthest else "CLEARED"
        if cleared and furthest and status not in ESTABLISHED and when(cleared[-1]) >= when(furthest):
            status = "CLEARED"
        sources = sorted({source(f) for f in group})
        stages = [{"stage": STAGES[f["predicate"]][0] if f["predicate"] in STAGES else "CLEARED", "date": when(f) or None, "conduct": f.get("object") if f["predicate"] != "received_regulatory_penalty" else f.get("qualifier"), "source": source(f), "document_id": f["document_id"], "fact_id": f["fact_id"], "claim_type": f.get("claim_type")} for f in group]
        denied = [f for f in denials.get(entity, [])]
        ids = [f["fact_id"] for f in group] + [f["fact_id"] for f in denied]
        details = {"subject": group[0]["subject"], "entity_id": entity, "topic": topic, "status": status, "stages": stages, "sources": sources, "distinct_sources": len(sources), "single_source": len(sources) == 1, "denied": [{"text": f.get("object"), "document_id": f["document_id"], "fact_id": f["fact_id"]} for f in denied], "first_reported": min((s["date"] for s in stages if s["date"]), default=None), "latest": max((s["date"] for s in stages if s["date"]), default=None)}
        findings.append({"finding_id": stable_id("finding", "ADVERSE_MEDIA_EVENT", entity, topic), "type": "ADVERSE_MEDIA_EVENT", "status": "INFORMATIONAL", "category": "allegations_and_outcomes", "fact_ids": sorted(set(ids)), "description": f"Reported {topic} matter: {status.lower().replace('_', ' ')}. Several sources are not proof of independent reporting; a report is not a finding.", "details": details})
        if not {f["fact_id"] for f in group} & current:
            continue
        if status == "CLEARED":
            continue
        if status in ESTABLISHED:
            issues.append(make_issue("ADVERSE_MEDIA_ESTABLISHED", "A source reports a conviction, penalty or enforcement action against this subject; review the matter and its relevance.", document_id=document_id, categories=["allegations_and_outcomes", "legal_regulatory"], severity="high", fact_ids=sorted(set(ids)), details=details))
        else:
            severe = status == "CHARGED" and topic in FINANCIAL_CRIME
            issues.append(make_issue("ADVERSE_MEDIA_OPEN_MATTER", "A source reports an allegation, investigation, charge or settlement concerning this subject that is not resolved on the evidence here.", document_id=document_id, categories=["allegations_and_outcomes"], severity="high" if severe else "medium", fact_ids=sorted(set(ids)), details=details))
    return findings, issues
