"""Registry and licence checks.

Two kinds of check, both raising issues for review rather than deciding anything:

* Local checks on extracted facts: LEI check digits (ISO 17442), an adverse
  company status (dissolved, in liquidation, ...), an adverse licence status
  (suspended, revoked, ...) and a licence expiry date that has passed.
* An opt-in lookup of each LEI in the public GLEIF register, compared with
  what the documents say: legal name, previous names, jurisdiction,
  registration number, incorporation date, and the entity's and LEI's status.
  Only the LEI is sent, but a lookup still tells GLEIF which entity is of
  interest, so it runs only when explicitly requested.

GLEIF data is itself self-reported by the entity and validated by its LEI
issuer to varying degrees (``corroboration_level``); a match is not proof of
identity and a mismatch may be a stale record.
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import httpx

from .entities import legal_key
from .evidence import make_issue
from .util import identity_key, lei_check_digits_valid

GLEIF_API = "https://api.gleif.org/api/v1/lei-records/"
GLEIF_RECORD_PAGE = "https://search.gleif.org/#/record/"

_ADVERSE_COMPANY = re.compile(r"\b(?:dissolved|struck\s+off|strike[\s-]+off|liquidat\w*|winding[\s-]+up|wound\s+up|administration|receivership|insolven\w*|inactive|ceased|removed|revoked|cancell?ed|forfeited|suspended|dormant|bankrupt\w*)\b", re.I)
_ADVERSE_LICENCE = re.compile(r"\b(?:suspended|revoked|cancell?ed|withdrawn|lapsed|expired|surrendered|terminated|refused|restricted|varied|inactive|no\s+longer\s+authori[sz]ed|unauthori[sz]ed)\b", re.I)
_LEI_NOT_ISSUED = {"LAPSED", "RETIRED", "ANNULLED", "DUPLICATE", "MERGED", "TRANSFERRED", "PENDING_ARCHIVAL", "CANCELLED"}

_US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puerto rico": "PR",
}
# Common incorporation jurisdictions; anything else is reported as not compared.
_COUNTRIES = {
    "united states": "US", "united states of america": "US", "usa": "US", "united kingdom": "GB", "england and wales": "GB",
    "england": "GB", "wales": "GB", "scotland": "GB", "northern ireland": "GB", "ireland": "IE", "luxembourg": "LU",
    "netherlands": "NL", "the netherlands": "NL", "germany": "DE", "france": "FR", "belgium": "BE", "switzerland": "CH",
    "austria": "AT", "italy": "IT", "spain": "ES", "portugal": "PT", "sweden": "SE", "norway": "NO", "denmark": "DK",
    "finland": "FI", "cyprus": "CY", "malta": "MT", "jersey": "JE", "guernsey": "GG", "isle of man": "IM", "gibraltar": "GI",
    "cayman islands": "KY", "british virgin islands": "VG", "bermuda": "BM", "bahamas": "BS", "panama": "PA", "canada": "CA",
    "australia": "AU", "new zealand": "NZ", "singapore": "SG", "hong kong": "HK", "japan": "JP", "india": "IN", "china": "CN",
    "united arab emirates": "AE", "south africa": "ZA", "brazil": "BR", "mexico": "MX",
}


class RegistryLookupError(RuntimeError):
    """Sanitized lookup failure: never includes response bodies."""


def jurisdiction_code(text: str | None) -> str | None:
    """'Delaware' -> 'US-DE', 'England and Wales' -> 'GB'; an ISO code is kept; None if unknown."""
    if not text:
        return None
    raw = text.strip()
    if re.fullmatch(r"[A-Z]{2}(?:-[A-Z0-9]{1,3})?", raw):
        return raw
    key = identity_key(re.sub(r"^(?:the\s+)?state\s+of\s+", "", raw, flags=re.I))
    if key in _US_STATES:
        return "US-" + _US_STATES[key]
    return _COUNTRIES.get(key)


def _registry_number(value: str | None, *, drop_prefix: bool = False) -> str:
    number = re.sub(r"[^A-Z0-9]", "", (value or "").upper())
    return (re.sub(r"^[A-Z]+", "", number) if drop_prefix else number).lstrip("0")


def status_issues(document_id: str, facts: list[dict], material: dict[str, bool], today: date | None = None) -> list[dict]:
    """Issues that need only the extracted facts: LEI check digits, adverse statuses, a passed expiry date."""
    today = today or datetime.now(timezone.utc).date()
    issues = []
    for f in facts:
        predicate, obj = f["predicate"], (f.get("object") or "").strip()
        severity = "high" if material.get(f["category"]) else "medium"
        if predicate == "has_lei" and obj and not lei_check_digits_valid(obj.replace(" ", "").upper()):
            issues.append(make_issue("LEI_CHECK_DIGITS_INVALID", "The LEI fails its ISO 17442 check digits; it may be misprinted, misread or not a real LEI, and was not used to identify the entity.", document_id=document_id, categories=["legal_identity"], severity=severity, fact_ids=[f["fact_id"]], details={"lei": obj}))
        elif predicate == "has_legal_status" and _ADVERSE_COMPANY.search(obj):
            issues.append(make_issue("REGISTRY_STATUS_ADVERSE", "The registry status suggests the entity is not active or in good standing; confirm its current status with the registry.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=[f["fact_id"]], details={"status": obj, "as_of": f.get("effective_date")}))
        elif predicate == "holds_license" and isinstance(f.get("value"), str) and _ADVERSE_LICENCE.search(f["value"]):
            issues.append(make_issue("LICENSE_STATUS_ADVERSE", "The licence status suggests it is not in full force; confirm its current status and scope with the regulator.", document_id=document_id, categories=["license_status"], severity="medium", fact_ids=[f["fact_id"]], details={"license": obj, "status": f["value"], "as_of": f.get("effective_date")}))
        elif predicate == "license_expires_on" and f.get("effective_date"):
            try:
                expires = date.fromisoformat(f["effective_date"]) if len(f["effective_date"]) == 10 else None
            except ValueError:
                expires = None
            if expires and expires < today:
                issues.append(make_issue("LICENSE_EXPIRY_PASSED", "The licence's stated expiry date is before the date of this run; confirm whether it was renewed.", document_id=document_id, categories=["license_status"], severity="medium", fact_ids=[f["fact_id"]], details={"license": obj or None, "expires": f["effective_date"]}))
    return issues


def summarize_gleif(raw: dict) -> dict:
    """The fields compared or shown to a reviewer, from a GLEIF lei-records response."""
    data = raw["data"]
    attributes = data["attributes"]
    entity, registration = attributes["entity"], attributes["registration"]
    address = entity.get("legalAddress") or {}
    return {
        "lei": data["id"],
        "legal_name": (entity.get("legalName") or {}).get("name"),
        "other_names": [{"name": n.get("name"), "type": n.get("type")} for n in entity.get("otherNames") or [] if n.get("name")],
        "jurisdiction": entity.get("jurisdiction"),
        "registered_at": (entity.get("registeredAt") or {}).get("id"),
        "registered_as": entity.get("registeredAs"),
        "legal_address": {"lines": address.get("addressLines") or [], "city": address.get("city"), "region": address.get("region"), "country": address.get("country"), "postal_code": address.get("postalCode")},
        "entity_status": entity.get("status"),
        "creation_date": (entity.get("creationDate") or "")[:10] or None,
        "registration_status": registration.get("status"),
        "corroboration_level": registration.get("corroborationLevel"),
        "last_update": registration.get("lastUpdateDate"),
        "next_renewal": registration.get("nextRenewalDate"),
    }


class GleifClient:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 20.0, max_retries: int = 2):
        self.client = client or httpx.Client(timeout=timeout, headers={"Accept": "application/vnd.api+json"})
        self.max_retries = max_retries

    def lookup(self, lei: str) -> dict | None:
        """The summarized record, or None if GLEIF has no record for this LEI."""
        if not re.fullmatch(r"[A-Z0-9]{20}", lei):
            raise RegistryLookupError("Not a well-formed LEI")
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(GLEIF_API + lei)
            except httpx.HTTPError:
                if attempt < self.max_retries:
                    time.sleep(1.0 * 2 ** attempt)
                    continue
                raise RegistryLookupError("GLEIF request failed") from None
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                time.sleep(1.0 * 2 ** attempt)
                continue
            if not response.is_success:
                raise RegistryLookupError(f"GLEIF returned HTTP {response.status_code}")
            try:
                record = summarize_gleif(response.json())
            except (ValueError, KeyError, TypeError, AttributeError):
                raise RegistryLookupError("GLEIF returned an unexpected response") from None
            if record["lei"] != lei:
                raise RegistryLookupError("GLEIF returned a different LEI")
            return record
        raise RegistryLookupError("GLEIF request failed")


def _compare(field: str, document: list[str], registry: Any, result: str) -> dict:
    return {"field": field, "document": document, "registry": registry, "result": result}


def check_gleif(document_id: str, case_entities: list[dict], facts_by_entity: dict[str, list[dict]], entity_ids: set[str], lookup: Callable[[str], tuple[dict | None, str, bool]]) -> tuple[list[dict], list[dict]]:
    """Compare each LEI-identified case entity in ``entity_ids`` with its GLEIF record.

    ``lookup(lei)`` returns (record or None, fetched_at, served_from_cache) and
    may raise RegistryLookupError. Returns (checks, issues).
    """
    checks, issues = [], []
    for entity in case_entities:
        if entity["entity_id"] not in entity_ids:
            continue
        facts = facts_by_entity.get(entity["entity_id"], [])
        for lei in sorted({i["value"] for i in entity["identifiers"] if i["scheme"] == "LEI"}):
            lei_facts = [f["fact_id"] for f in facts if f["predicate"] == "has_lei" and (f.get("object") or "").replace(" ", "").upper() == lei]
            if not lei_check_digits_valid(lei):
                continue  # already raised as LEI_CHECK_DIGITS_INVALID
            check = {"lei": lei, "entity_id": entity["entity_id"], "entity_name": entity["name"], "source": "GLEIF", "record_url": GLEIF_RECORD_PAGE + lei}
            try:
                record, fetched_at, cached = lookup(lei)
            except RegistryLookupError as exc:
                checks.append({**check, "status": "LOOKUP_FAILED", "error": str(exc)})
                issues.append(make_issue("REGISTRY_LOOKUP_FAILED", "The GLEIF lookup failed, so this LEI was not checked; retry the run or check it manually.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=lei_facts, details={"lei": lei, "error": str(exc)}))
                continue
            check.update({"fetched_at": fetched_at, "served_from_cache": cached})
            if record is None:
                checks.append({**check, "status": "NOT_FOUND"})
                issues.append(make_issue("REGISTRY_RECORD_NOT_FOUND", "GLEIF has no record for this LEI although its check digits are valid; it may be misprinted or invented.", document_id=document_id, categories=["legal_identity"], fact_ids=lei_facts, details={"lei": lei}))
                continue

            def ids(*predicates: str) -> list[str]:
                return [f["fact_id"] for f in facts if f["predicate"] in predicates]

            comparisons = []
            # Legal name: the documents' legal or full names against GLEIF's legal and other names.
            names = sorted({n["name"] for n in entity["names"] if n["kind"] in {"legal_name", "full_name"}}) or ([entity["name"]] if entity["name"] else [])
            keys = {legal_key(n) for n in names}
            others = {legal_key(n["name"]): n["type"] for n in record["other_names"]}
            if legal_key(record["legal_name"] or "") in keys:
                comparisons.append(_compare("legal_name", names, record["legal_name"], "MATCH"))
            elif keys & set(others):
                comparisons.append(_compare("legal_name", names, record["legal_name"], "MATCHES_OTHER_NAME"))
                issues.append(make_issue("REGISTRY_NAME_IS_OTHER_NAME", "The documents use a name GLEIF lists as a previous or other name, not the current legal name; confirm the rename.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_legal_name"), details={"lei": lei, "document_names": names, "registry_legal_name": record["legal_name"], "other_name_types": sorted({others[k] for k in keys & set(others)})}))
            elif names:
                comparisons.append(_compare("legal_name", names, record["legal_name"], "MISMATCH"))
                issues.append(make_issue("REGISTRY_NAME_MISMATCH", "The documents' legal name differs from the GLEIF record for the LEI they print; the LEI may belong to another entity.", document_id=document_id, categories=["legal_identity"], fact_ids=ids("has_legal_name", "has_lei"), details={"lei": lei, "document_names": names, "registry_legal_name": record["legal_name"]}))
            # Former names stated in the documents should appear among GLEIF's other names.
            former = sorted({f["object"] for f in facts if f["predicate"] == "has_former_name" and f.get("object")})
            if former:
                known = {legal_key(n) for n in former} & set(others)
                comparisons.append(_compare("former_names", former, [n["name"] for n in record["other_names"]], "MATCH" if len(known) == len(former) else "NOT_IN_REGISTRY"))
            # Jurisdiction, mapped to ISO 3166 codes where the documents' wording is known.
            stated = entity.get("jurisdictions") or []
            codes = {jurisdiction_code(j) for j in stated}
            registry_code = record["jurisdiction"]
            if stated and registry_code:
                if None in codes and len(codes) == 1:
                    result = "NOT_COMPARED"
                elif registry_code in codes or any(c and "-" not in c and registry_code.split("-")[0] == c for c in codes):
                    result = "MATCH"
                else:
                    result = "MISMATCH"
                    issues.append(make_issue("REGISTRY_JURISDICTION_MISMATCH", "The documents' jurisdiction differs from the GLEIF record for this LEI.", document_id=document_id, categories=["legal_identity"], fact_ids=ids("has_jurisdiction"), details={"lei": lei, "document": stated, "registry": registry_code}))
                comparisons.append(_compare("jurisdiction", stated, registry_code, result))
            # Registration number, ignoring punctuation, case and leading zeros.
            numbers = sorted({i["value"] for i in entity["identifiers"] if i["scheme"] == "REGISTRATION_NUMBER"})
            if numbers and record["registered_as"]:
                # Registers differ on letter prefixes ("C0806592" and "806592" in California).
                if _registry_number(record["registered_as"]) in {_registry_number(n) for n in numbers}:
                    result = "MATCH"
                elif _registry_number(record["registered_as"], drop_prefix=True) in {_registry_number(n, drop_prefix=True) for n in numbers}:
                    result = "MATCH_IGNORING_PREFIX"
                else:
                    result = "MISMATCH"
                comparisons.append(_compare("registration_number", numbers, record["registered_as"], result))
                if result == "MISMATCH":
                    issues.append(make_issue("REGISTRY_NUMBER_MISMATCH", "The documents' registration number differs from the one GLEIF holds; GLEIF may use another register or format.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_registration_number"), details={"lei": lei, "document": numbers, "registry": record["registered_as"], "registered_at": record["registered_at"]}))
            # Incorporation date.
            dates = sorted({f["effective_date"] for f in facts if f["predicate"] == "has_incorporation_date" and f.get("effective_date")})
            if dates and record["creation_date"]:
                match = any(record["creation_date"].startswith(d) for d in dates)
                comparisons.append(_compare("incorporation_date", dates, record["creation_date"], "MATCH" if match else "MISMATCH"))
                if not match:
                    issues.append(make_issue("REGISTRY_INCORPORATION_DATE_MISMATCH", "The documents' incorporation date differs from GLEIF's entity creation date.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_incorporation_date"), details={"lei": lei, "document": dates, "registry": record["creation_date"]}))
            if record["entity_status"] and record["entity_status"] != "ACTIVE":
                issues.append(make_issue("REGISTRY_ENTITY_INACTIVE", "GLEIF records this entity as not active.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=lei_facts, details={"lei": lei, "entity_status": record["entity_status"]}))
            if record["registration_status"] in _LEI_NOT_ISSUED:
                issues.append(make_issue("LEI_NOT_CURRENT", "The LEI's registration is not current (for example lapsed or retired); its reference data may be out of date.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=lei_facts, details={"lei": lei, "registration_status": record["registration_status"]}))
            checks.append({**check, "status": "CHECKED", "record": record, "comparisons": comparisons})
    return checks, issues


def cached_lookup(store: Any, client: GleifClient, max_age: timedelta = timedelta(hours=24)) -> Callable[[str], tuple[dict | None, str, bool]]:
    """A lookup that reuses a stored GLEIF record younger than ``max_age``."""
    def lookup(lei: str) -> tuple[dict | None, str, bool]:
        cached = store.get_registry_record("GLEIF", lei)
        if cached and datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"]) < max_age:
            return cached["record"], cached["fetched_at"], True
        record = client.lookup(lei)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.put_registry_record("GLEIF", lei, record, fetched_at)
        return record, fetched_at, False
    return lookup
