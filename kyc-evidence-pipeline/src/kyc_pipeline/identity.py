"""Identity-document checks: the machine-readable zone, expiry, and proof-of-address age.

* MRZ: the two-line passport zone (TD3, 2 x 44 characters) and the three-line
  identity-card zone (TD1, 3 x 30) are read from the document text. Each field
  has an ICAO 9303 check digit, recomputed here. The MRZ's document number,
  date of birth, expiry date and names are compared with the printed (visual)
  fields the rules extracted. Valid check digits show the zone is internally
  consistent, not that the document is genuine: a forger can compute them too.
* Expiry: a document whose expiry date is before the run date is raised.
* Proof of address: a bill or statement older than the policy's maximum age.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone

from .entities import person_key
from .evidence import make_issue
from .util import stable_id

_TD3 = re.compile(r"^(P[A-Z<][A-Z<]{3}[A-Z<]{39})\s*\n\s*([A-Z0-9<]{44})$", re.M)
_TD1 = re.compile(r"^([AIC][A-Z<][A-Z<]{3}[A-Z0-9<]{25})\s*\n\s*([A-Z0-9<]{30})\s*\n\s*([A-Z<]{30})$", re.M)


def check_digit(field: str) -> int:
    """ICAO 9303: weights 7, 3, 1; digits as themselves, A-Z as 10-35, '<' as 0."""
    total = 0
    for i, ch in enumerate(field):
        value = int(ch) if ch.isdigit() else ord(ch) - 55 if ch.isalpha() else 0
        total += value * (7, 3, 1)[i % 3]
    return total % 10


def _mrz_date(yymmdd: str, future: bool) -> str | None:
    """Two-digit years: a birth date is in the past; an expiry date is within 50 years from now (else last century)."""
    if not re.fullmatch(r"\d{6}", yymmdd):
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    today = datetime.now(timezone.utc).date()
    year = 2000 + yy if (2000 + yy <= today.year + 50 if future else 2000 + yy <= today.year) else 1900 + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _names(field: str) -> tuple[str, str]:
    surname, _, given = field.strip("<").partition("<<")
    return surname.replace("<", " ").strip(), given.replace("<", " ").strip()


def parse_mrz(text: str) -> list[dict]:
    """Every TD3 or TD1 zone in the text, with its fields and check-digit results."""
    zones = []
    for m in _TD3.finditer(text):
        l1, l2 = m.group(1), m.group(2)
        checks = {"document_number": (l2[0:9], l2[9]), "date_of_birth": (l2[13:19], l2[19]), "expiry_date": (l2[21:27], l2[27]), "personal_number": (l2[28:42], l2[42])}
        composite = (l2[0:10] + l2[13:20] + l2[21:43], l2[43])
        surname, given = _names(l1[5:])
        zones.append({"format": "TD3", "lines": [l1, l2], "start": m.start(), "document_code": l1[0:2].strip("<"), "issuing_state": l1[2:5].strip("<"), "surname": surname, "given_names": given, "document_number": l2[0:9].strip("<"), "nationality": l2[10:13].strip("<"), "date_of_birth": _mrz_date(l2[13:19], False), "sex": l2[20], "expiry_date": _mrz_date(l2[21:27], True), "checks": checks, "composite": composite})
    for m in _TD1.finditer(text):
        l1, l2, l3 = m.group(1), m.group(2), m.group(3)
        checks = {"document_number": (l1[5:14], l1[14]), "date_of_birth": (l2[0:6], l2[6]), "expiry_date": (l2[8:14], l2[14])}
        composite = (l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29], l2[29])
        surname, given = _names(l3)
        zones.append({"format": "TD1", "lines": [l1, l2, l3], "start": m.start(), "document_code": l1[0:2].strip("<"), "issuing_state": l1[2:5].strip("<"), "surname": surname, "given_names": given, "document_number": l1[5:14].strip("<"), "nationality": l2[15:18].strip("<"), "date_of_birth": _mrz_date(l2[0:6], False), "sex": l2[7], "expiry_date": _mrz_date(l2[8:14], True), "checks": checks, "composite": composite})
    for zone in zones:
        failed = [name for name, (field, digit) in zone["checks"].items() if not (digit == "<" and set(field) <= {"<"}) and (not digit.isdigit() or check_digit(field) != int(digit))]
        field, digit = zone["composite"]
        if not digit.isdigit() or check_digit(field) != int(digit):
            failed.append("composite")
        zone["failed_checks"] = failed
        del zone["checks"], zone["composite"]
    return zones


def _finding(kind: str, fact_ids: list[str], description: str, details: dict) -> dict:
    ids = sorted(set(fact_ids))
    return {"finding_id": stable_id("finding", kind, ids, details.get("document_number")), "type": kind, "status": "INFORMATIONAL", "category": "identity_fields", "fact_ids": ids, "description": description, "details": details}


def identity_checks(document_id: str, facts: list[dict], segments: list[dict], material: dict[str, bool], settings: dict | None = None, today: date | None = None) -> tuple[list[dict], list[dict]]:
    """MRZ, expiry and proof-of-address checks for one document. Returns (findings, issues)."""
    today = today or datetime.now(timezone.utc).date()
    max_age = int((settings or {}).get("proof_of_address_max_age_days", 90))
    severity = "high" if material.get("identity_fields") else "medium"
    findings, issues = [], []
    by_predicate: dict[str, list[dict]] = {}
    for f in facts:
        by_predicate.setdefault(f["predicate"], []).append(f)
    documents = by_predicate.get("holds_identity_document", [])

    for f in by_predicate.get("identity_document_expires_on", []):
        if f.get("effective_date") and len(f["effective_date"]) == 10 and date.fromisoformat(f["effective_date"]) < today:
            issues.append(make_issue("IDENTITY_DOCUMENT_EXPIRED", "The identity document's expiry date is before the date of this run; it cannot evidence identity now.", document_id=document_id, categories=["identity_fields"], severity=severity, fact_ids=[f["fact_id"]], details={"name": f["subject"], "document_number": f.get("object"), "expired": f["effective_date"]}))

    text = "\n".join(s["text"] for s in segments)
    for zone in parse_mrz(text):
        # The visual fields this zone should agree with: the document with the same number, else the only one.
        doc = next((d for d in documents if re.sub(r"\s", "", str(d.get("value") or "")).upper() == zone["document_number"]), documents[0] if len(documents) == 1 else None)
        holder = doc["subject"] if doc else None
        related = [f for f in facts if f["subject"] == holder] if holder else []
        visual = {
            "document_number": re.sub(r"\s", "", str(doc.get("value") or "")).upper() if doc else None,
            "date_of_birth": next((f["effective_date"] for f in related if f["predicate"] == "has_date_of_birth"), None),
            "expiry_date": next((f["effective_date"] for f in related if f["predicate"] == "identity_document_expires_on"), None),
            "name": holder,
        }
        mismatches = []
        for key in ("document_number", "date_of_birth", "expiry_date"):
            if visual[key] and zone[key] and visual[key] != zone[key]:
                mismatches.append({"field": key, "visual": visual[key], "mrz": zone[key]})
        surname = person_key(zone["surname"]).split()
        if holder and surname:
            # The printed name must end with the MRZ surname; the MRZ may truncate given names,
            # so each of its given names need only start one of the printed ones.
            printed = person_key(holder).split()
            given = person_key(zone["given_names"]).split()
            if printed[-len(surname):] != surname or not all(any(p.startswith(g) for p in printed[:-len(surname)]) for g in given):
                mismatches.append({"field": "name", "visual": holder, "mrz": f"{zone['given_names']} {zone['surname']}".strip()})
        ids = [f["fact_id"] for f in related] if related else []
        details = {"format": zone["format"], "mrz_lines": zone["lines"], "document_number": zone["document_number"], "issuing_state": zone["issuing_state"], "nationality": zone["nationality"]}
        if zone["failed_checks"]:
            issues.append(make_issue("MRZ_CHECK_DIGIT_INVALID", "A check digit in the document's machine-readable zone does not match its field; the zone was misread, mistyped or altered.", document_id=document_id, categories=["identity_fields"], severity=severity, fact_ids=ids, details={**details, "failed_checks": zone["failed_checks"]}))
        if mismatches:
            issues.append(make_issue("MRZ_VISUAL_MISMATCH", "The machine-readable zone and the printed fields of the document disagree; the document may have been altered or misread.", document_id=document_id, categories=["identity_fields"], severity=severity, fact_ids=ids, details={**details, "mismatches": mismatches}))
        if not zone["failed_checks"] and not mismatches:
            compared = [k for k in ("document_number", "date_of_birth", "expiry_date", "name") if visual[k]]
            findings.append(_finding("MRZ_CONSISTENT", ids, "The machine-readable zone's check digits are valid and it agrees with the printed fields compared. This does not show the document is genuine.", {**details, "fields_compared": compared}))

    for f in by_predicate.get("has_residential_address", []):
        if f.get("qualifier") == "proof of address" and f.get("effective_date") and len(f["effective_date"]) == 10:
            age = (today - date.fromisoformat(f["effective_date"])).days
            if age > max_age:
                issues.append(make_issue("PROOF_OF_ADDRESS_OUT_OF_DATE", f"The proof of address is dated more than {max_age} days before this run.", document_id=document_id, categories=["address"], severity="medium", fact_ids=[f["fact_id"]], details={"name": f["subject"], "document_date": f["effective_date"], "age_days": age, "max_age_days": max_age}))
    return findings, issues
