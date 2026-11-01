"""Sanctions and PEP: screening reports, declarations, and opt-in list screening.

* Report checks: a positive sanctions result, a PEP result or public function,
  a customer's "not a PEP" declaration that other evidence contradicts, and a
  screening older than the policy allows.
* List screening (``--screening-list``): the case's people and organizations
  named by the current document are compared with a local list file: a
  simple CSV, or OFAC's SDN CSV (with ``alt.csv`` aliases beside it). Nothing
  is downloaded; the file's SHA-256 and entry count are recorded with every
  check. Matching is by name only, with dates of birth used to separate
  people where both sides have one.

A name match is a lead for an analyst, not a determination; a "no match" is
only as good as the list file, its date, and name matching. Neither clears
anyone.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from .entities import legal_key, person_key
from .evidence import make_issue
from .extraction import is_positive
from .util import normalize_date, stable_id, temporal_overlap

_FORM_WORDS = {"inc", "corp", "co", "llc", "lp", "llp", "ltd", "plc", "sa", "ag", "gmbh", "nv", "bv", "se", "spa", "sas", "pty", "na", "the"}
_DOB = re.compile(r"\bDOB\s+((?:\d{1,2}\s+)?(?:[A-Z][a-z]{2}\s+)?\d{4})", re.I)


def _finding(kind: str, fact_ids: list[str], description: str, details: dict) -> dict:
    ids = sorted(set(fact_ids))
    return {"finding_id": stable_id("finding", kind, ids), "type": kind, "status": "INFORMATIONAL", "category": "sanctions_pep", "fact_ids": ids, "description": description, "details": details}


def report_checks(document_id: str, facts: list[dict], previous: list[dict], material: dict[str, bool], settings: dict | None = None, today: date | None = None, entity_map: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Checks on screening-report and declaration facts. Returns (findings, issues).

    People are matched across documents by case entity where ``entity_map`` resolves them
    (so "Jane Smith" and "Jane Elizabeth Smith" agree once resolution joined them), else by name.
    """
    entity_map = entity_map or {}

    def who(f: dict) -> str:
        link = (f.get("entity_links") or {}).get("subject")
        return entity_map.get(link["entity_id"], person_key(f["subject"])) if link else person_key(f["subject"])

    today = today or datetime.now(timezone.utc).date()
    max_age = int((settings or {}).get("max_age_days", 90))
    severity = "high" if material.get("sanctions_pep", True) else "medium"
    findings, issues = [], []
    current = {f["fact_id"] for f in facts}
    for f in facts:
        if f["predicate"] == "sanctions_screening_result":
            if is_positive(f.get("value")):
                issues.append(make_issue("SANCTIONS_SCREENING_HIT", "The screening report records a sanctions match for this name; an analyst must decide whether it is the same person or entity.", document_id=document_id, categories=["sanctions_pep"], severity=severity, fact_ids=[f["fact_id"]], details={"name": f["subject"], "result": f["value"], "matched_entry": f.get("qualifier"), "lists": f.get("object"), "screened_on": f.get("effective_date")}))
            else:
                findings.append(_finding("SANCTIONS_SCREENING_NO_MATCH", [f["fact_id"]], "The screening report records no sanctions match for this name, on the lists and date it states. It is not a clearance.", {"name": f["subject"], "result": f["value"], "lists": f.get("object"), "screened_on": f.get("effective_date")}))
        if f["predicate"] in {"sanctions_screening_result", "pep_screening_result"} and f.get("effective_date") and len(f["effective_date"]) == 10:
            age = (today - date.fromisoformat(f["effective_date"])).days
            if age > max_age:
                issues.append(make_issue("SCREENING_OUT_OF_DATE", f"The screening is more than {max_age} days old; lists change, so screen again.", document_id=document_id, categories=["sanctions_pep"], severity="medium", fact_ids=[f["fact_id"]], details={"name": f["subject"], "screened_on": f["effective_date"], "age_days": age, "max_age_days": max_age}))
    # PEP evidence is gathered across the case, so a declaration and a screening report can arrive in either order.
    pep_evidence: dict[str, list[dict]] = defaultdict(list)
    for f in previous + facts:
        if f["predicate"] == "holds_public_function" or (f["predicate"] == "pep_screening_result" and is_positive(f.get("value"))):
            pep_evidence[who(f)].append(f)
    for key, evidence in pep_evidence.items():
        if not {e["fact_id"] for e in evidence} & current:
            continue
        issues.append(make_issue("PEP_IDENTIFIED", "The evidence identifies this person as a politically exposed person (or a family member or close associate); enhanced due diligence usually applies.", document_id=document_id, categories=["sanctions_pep"], severity="medium", fact_ids=sorted(e["fact_id"] for e in evidence), details={"name": evidence[0]["subject"], "positions": sorted({e["object"] for e in evidence if e.get("object")}), "qualifiers": sorted({e["qualifier"] for e in evidence if e.get("qualifier")})}))
    for f in previous + facts:
        if f["predicate"] != "declared_pep_status" or is_positive(f.get("value")):
            continue
        evidence = pep_evidence.get(who(f), [])
        if evidence and ({f["fact_id"]} | {e["fact_id"] for e in evidence}) & current:
            issues.append(make_issue("PEP_DECLARATION_CONTRADICTED", "The customer declared this person is not a politically exposed person, but other evidence in the case says they are.", document_id=document_id, categories=["sanctions_pep", "discrepancies"], severity=severity, fact_ids=sorted({f["fact_id"]} | {e["fact_id"] for e in evidence}), details={"name": f["subject"], "declared": f["value"], "declared_on": f.get("effective_date"), "positions": sorted({e["object"] for e in evidence if e.get("object")})}))
    return findings, issues


# List screening -----------------------------------------------------------------

def _name_tokens(name: str, person: bool) -> list[str]:
    if person:
        # Lists often print "SURNAME, Given Names".
        surname, comma, given = name.partition(",")
        name = f"{given} {surname}" if comma else name
        return person_key(name).split()
    return [t for t in legal_key(name).split() if t not in _FORM_WORDS]


def _dates(text: str | None) -> list[str]:
    out = []
    for raw in re.split(r"[;|]", text or ""):
        try:
            out.append(normalize_date(raw.strip())) if raw.strip() else None
        except ValueError:
            continue
    return out


@lru_cache(maxsize=8)
def _load(path: str, mtime: float, size: int) -> dict:
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    entries = []
    header = [c.strip().lower() for c in rows[0]] if rows else []
    if "name" in header:
        col = {name: header.index(name) for name in header}
        get = lambda row, key: row[col[key]].strip() if key in col and col[key] < len(row) else ""
        for row in rows[1:]:
            if not row or not get(row, "name"):
                continue
            kind = get(row, "type").lower()
            entries.append({"id": get(row, "id") or None, "name": get(row, "name"), "aliases": [a.strip() for a in get(row, "aliases").split(";") if a.strip()], "person": kind in {"person", "individual"} if kind else None, "list": get(row, "list") or Path(path).stem, "program": get(row, "program") or get(row, "programme") or None, "birth_dates": _dates(get(row, "date_of_birth"))})
        source_format = "CSV"
    elif rows and len(rows[0]) >= 12 and rows[0][0].strip().isdigit():
        # OFAC SDN: ent_num, SDN_Name, SDN_Type, Program, Title, ..., Remarks (12 columns, no header).
        aliases: dict[str, list[str]] = defaultdict(list)
        alt = Path(path).with_name("alt.csv")
        if alt.exists():
            for row in csv.reader(alt.read_text(encoding="utf-8-sig", errors="replace").splitlines()):
                if len(row) >= 4 and row[3].strip() not in {"", "-0-"}:
                    aliases[row[0].strip()].append(row[3].strip())
        for row in rows:
            if len(row) < 12 or not row[0].strip().isdigit():
                continue
            kind = row[2].strip().lower()
            remarks = row[11] if row[11].strip() != "-0-" else ""
            entries.append({"id": row[0].strip(), "name": row[1].strip(), "aliases": aliases.get(row[0].strip(), []), "person": kind == "individual" if kind not in {"", "-0-"} else False, "list": "OFAC SDN", "program": row[3].strip() or None, "birth_dates": _dates(";".join(m.group(1) for m in _DOB.finditer(remarks)))})
        source_format = "OFAC_SDN_CSV"
    else:
        raise ValueError(f"{path}: not a screening CSV (needs a 'name' column) or an OFAC SDN CSV")
    index: dict[str, set[int]] = defaultdict(set)
    for n, e in enumerate(entries):
        for name in [e["name"]] + e["aliases"]:
            for token in _name_tokens(name, bool(e["person"])):
                if len(token) >= 3:
                    index[token].add(n)
    return {"path": path, "sha256": sha, "format": source_format, "entries": entries, "index": dict(index), "entry_count": len(entries)}


def load_list(path: str) -> dict:
    p = Path(path)
    stat = p.stat()
    return _load(str(p.resolve()), stat.st_mtime, stat.st_size)


def _score(a: list[str], b: list[str], person: bool) -> float:
    if not a or not b:
        return 0.0
    if person and a[-1] not in b:
        return 0.0  # the surname must appear
    if set(a) == set(b):
        return 1.0
    if person and len(a) >= 2 and a[0] == b[0] and set(a) <= set(b):
        return 0.95  # a listed middle name we lack ("Jane Smith" / "Jane Elizabeth Smith")
    return SequenceMatcher(None, " ".join(sorted(a)), " ".join(sorted(b))).ratio()


def screen_entities(document_id: str, case_entities: list[dict], entity_ids: set[str], lists: list[dict], settings: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Screen each case entity in ``entity_ids`` against the loaded lists. Returns (checks, issues)."""
    threshold = float((settings or {}).get("name_match_threshold", 0.88))
    checks, issues = [], []
    sources = [{"list_file": Path(l["path"]).name, "format": l["format"], "sha256": l["sha256"], "entries": l["entry_count"]} for l in lists]
    for entity in case_entities:
        if entity["entity_id"] not in entity_ids or entity["entity_type"] == "group" or not entity["name"]:
            continue
        person = entity["entity_type"] == "person"
        names = sorted({n["name"] for n in entity["names"] if n["kind"] in {"legal_name", "full_name", "mention"}} | {entity["name"]})
        matches: dict[tuple, dict] = {}
        for loaded in lists:
            for name in names:
                tokens = _name_tokens(name, person)
                candidates = set().union(*[loaded["index"].get(t, set()) for t in tokens if len(t) >= 3]) if tokens else set()
                for n in candidates:
                    e = loaded["entries"][n]
                    if e["person"] is not None and e["person"] != person and entity["entity_type"] != "unknown":
                        continue
                    for listed in [e["name"]] + e["aliases"]:
                        score = _score(tokens, _name_tokens(listed, bool(e["person"])), person)
                        key = (loaded["sha256"], n)
                        if score >= threshold and score > matches.get(key, {}).get("score", 0):
                            births = entity.get("birth_dates") or []
                            dob = "NOT_COMPARED" if not (births and e["birth_dates"]) else "MATCH" if any(temporal_overlap(x, y) for x in births for y in e["birth_dates"]) else "DIFFERENT"
                            matches[key] = {"list": e["list"], "entry_id": e["id"], "listed_name": listed, "primary_name": e["name"], "program": e["program"], "our_name": name, "score": round(score, 3), "date_of_birth": dob, "listed_birth_dates": e["birth_dates"]}
        found = sorted(matches.values(), key=lambda m: -m["score"])
        checks.append({"entity_id": entity["entity_id"], "name": entity["name"], "entity_type": entity["entity_type"], "names_screened": names, "lists": sources, "result": "POSSIBLE_MATCH" if any(m["date_of_birth"] != "DIFFERENT" for m in found) else "MATCH_DISCOUNTED_BY_DATE_OF_BIRTH" if found else "NO_MATCH", "matches": found})
        for m in found:
            if m["date_of_birth"] == "DIFFERENT":
                issues.append(make_issue("SANCTIONS_LIST_MATCH_DIFFERENT_BIRTH_DATE", "A list entry matches this person's name but its date of birth differs; an analyst should confirm it is a different person.", document_id=document_id, categories=["sanctions_pep"], severity="medium", details={"entity_id": entity["entity_id"], **m}))
            else:
                issues.append(make_issue("SANCTIONS_LIST_POSSIBLE_MATCH", "A sanctions-list entry matches this name; an analyst must decide whether it is the same person or entity before proceeding.", document_id=document_id, categories=["sanctions_pep"], severity="high", details={"entity_id": entity["entity_id"], **m}))
    return checks, issues
