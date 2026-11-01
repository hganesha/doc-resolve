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
* An opt-in lookup of each UK company number in the Companies House register,
  compared with the documents: name, previous names, jurisdiction, status,
  legal form, incorporation date, registered office, officers, and the people
  with significant control against reported stakes. Only the company number
  is sent.

GLEIF data is itself self-reported by the entity and validated by its LEI
issuer to varying degrees (``corroboration_level``); a match is not proof of
identity and a mismatch may be a stale record. Companies House records what
companies file, largely without verifying it.
"""
from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import httpx

from .credentials import get_secret
from .entities import legal_key, same_person
from .evidence import make_issue
from .util import identity_key, lei_check_digits_valid

GLEIF_API = "https://api.gleif.org/api/v1/lei-records/"
GLEIF_RECORD_PAGE = "https://search.gleif.org/#/record/"
COMPANIES_HOUSE_API = "https://api.company-information.service.gov.uk"
COMPANIES_HOUSE_PAGE = "https://find-and-update.company-information.service.gov.uk/company/"

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


def _get_json(client: httpx.Client, url: str, max_retries: int, source: str, params: dict | None = None) -> dict | None:
    """The decoded JSON body, or None on 404; retries rate limits, 5xx and network errors."""
    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, params=params)
        except httpx.HTTPError:
            if attempt < max_retries:
                time.sleep(1.0 * 2 ** attempt)
                continue
            raise RegistryLookupError(f"{source} request failed") from None
        if response.status_code == 404:
            return None
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            time.sleep(1.0 * 2 ** attempt)
            continue
        if not response.is_success:
            raise RegistryLookupError(f"{source} returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError:
            raise RegistryLookupError(f"{source} returned an unexpected response") from None
    raise RegistryLookupError(f"{source} request failed")


class GleifClient:
    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 20.0, max_retries: int = 2):
        self.client = client or httpx.Client(timeout=timeout, headers={"Accept": "application/vnd.api+json"})
        self.max_retries = max_retries

    def lookup(self, lei: str) -> dict | None:
        """The summarized record, or None if GLEIF has no record for this LEI."""
        if not re.fullmatch(r"[A-Z0-9]{20}", lei):
            raise RegistryLookupError("Not a well-formed LEI")
        raw = _get_json(self.client, GLEIF_API + lei, self.max_retries, "GLEIF")
        if raw is None:
            return None
        try:
            record = summarize_gleif(raw)
        except (KeyError, TypeError, AttributeError):
            raise RegistryLookupError("GLEIF returned an unexpected response") from None
        if record["lei"] != lei:
            raise RegistryLookupError("GLEIF returned a different LEI")
        return record


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


def cached_lookup(store: Any, client: Any, source: str = "GLEIF", max_age: timedelta = timedelta(hours=24)) -> Callable[[str], tuple[dict | None, str, bool]]:
    """A lookup that reuses a stored record from ``source`` younger than ``max_age``."""
    def lookup(identifier: str) -> tuple[dict | None, str, bool]:
        cached = store.get_registry_record(source, identifier)
        if cached and datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"]) < max_age:
            return cached["record"], cached["fetched_at"], True
        record = client.lookup(identifier)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.put_registry_record(source, identifier, record, fetched_at)
        return record, fetched_at, False
    return lookup


# Companies House --------------------------------------------------------------

_CH_DIRECTOR_ROLES = {"director", "corporate-director", "nominee-director", "corporate-nominee-director"}
_CH_CLOSED = {"dissolved", "removed", "closed", "converted-closed"}
_CH_ACTIVE = {"active", "open", "registered"}
# Companies House company types and the wording documents commonly use for them.
_CH_FORMS = {
    "ltd": {"private limited company", "private company limited by shares", "limited company"},
    "plc": {"public limited company", "public company limited by shares"},
    "llp": {"limited liability partnership"},
    "private-limited-guarant-nsc": {"private company limited by guarantee", "company limited by guarantee"},
    "private-limited-guarant-nsc-limited-exemption": {"private company limited by guarantee", "company limited by guarantee"},
    "private-unlimited": {"private unlimited company", "unlimited company"},
    "limited-partnership": {"limited partnership"},
    "scottish-partnership": {"scottish partnership"},
    "oversea-company": {"overseas company", "oversea company"},
}
_CH_JURISDICTIONS = {"england and wales": {"england-wales", "wales"}, "england": {"england-wales"}, "wales": {"england-wales", "wales"}, "scotland": {"scotland"}, "northern ireland": {"northern-ireland"}}
_CH_FLAGS = {"has_insolvency_history": "insolvency history", "registered_office_is_in_dispute": "registered office in dispute", "undeliverable_registered_office_address": "undeliverable registered office", "accounts_overdue": "accounts overdue", "confirmation_statement_overdue": "confirmation statement overdue"}
_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I)
_BAND = re.compile(r"(ownership-of-shares|voting-rights)-(\d+)-to-(\d+)-percent")


def companies_house_number(value: str | None) -> str | None:
    """'9876543' -> '09876543', 'SC 123456' -> 'SC123456'; None if not in Companies House format."""
    raw = re.sub(r"[\s-]", "", (value or "").upper())
    if re.fullmatch(r"\d{1,8}", raw):
        return raw.zfill(8)
    m = re.fullmatch(r"([A-Z]{2})(\d{1,6})", raw)
    return m.group(1) + m.group(2).zfill(6) if m else None


def _officer_name(name: str) -> str:
    """Companies House lists individuals as 'SMITH, Jane Elizabeth'."""
    surname, _, forenames = name.partition(",")
    return f"{forenames.strip()} {surname.strip()}".strip() if forenames else name.strip()


def summarize_companies_house(company: dict, officers: list[dict], pscs: list[dict]) -> dict:
    """The fields compared or shown to a reviewer, from the company, officers and PSC responses."""
    address = company.get("registered_office_address") or {}
    bands = lambda natures: {kind: [int(low), int(high)] for kind, low, high in (m.groups() for m in map(_BAND.match, natures) if m)}
    return {
        "company_number": company["company_number"],
        "company_name": company.get("company_name"),
        "company_status": company.get("company_status"),
        "company_status_detail": company.get("company_status_detail"),
        "type": company.get("type"),
        "jurisdiction": company.get("jurisdiction"),
        "date_of_creation": company.get("date_of_creation"),
        "date_of_cessation": company.get("date_of_cessation"),
        "registered_office_address": {k: address.get(k) for k in ("premises", "address_line_1", "address_line_2", "locality", "region", "postal_code", "country")},
        "previous_company_names": [{"name": n.get("name"), "effective_from": n.get("effective_from"), "ceased_on": n.get("ceased_on")} for n in company.get("previous_company_names") or [] if n.get("name")],
        "flags": sorted(label for key, label in _CH_FLAGS.items() if company.get(key) or (key == "accounts_overdue" and (company.get("accounts") or {}).get("overdue")) or (key == "confirmation_statement_overdue" and (company.get("confirmation_statement") or {}).get("overdue"))),
        "officers": [{"name": _officer_name(o.get("name") or ""), "officer_role": o.get("officer_role"), "appointed_on": o.get("appointed_on"), "resigned_on": o.get("resigned_on")} for o in officers if o.get("name")],
        "persons_with_significant_control": [{"name": p.get("name"), "kind": p.get("kind"), "natures_of_control": p.get("natures_of_control") or [], "bands": bands(p.get("natures_of_control") or []), "notified_on": p.get("notified_on"), "ceased_on": p.get("ceased_on")} for p in pscs if p.get("name")],
    }


class CompaniesHouseClient:
    """The Companies House public data API; the key is sent as the basic-auth user name."""

    def __init__(self, api_key: str | None = None, *, client: httpx.Client | None = None, timeout: float = 20.0, max_retries: int = 2, max_pages: int = 5):
        if client is None:
            api_key = api_key or get_secret("COMPANIES_HOUSE_API_KEY")
            if not api_key:
                raise RegistryLookupError("COMPANIES_HOUSE_API_KEY is not set")
            client = httpx.Client(base_url=COMPANIES_HOUSE_API, timeout=timeout, auth=httpx.BasicAuth(api_key, ""), headers={"Accept": "application/json"})
        self.client = client
        self.max_retries = max_retries
        self.max_pages = max_pages

    def _items(self, path: str) -> list[dict]:
        items: list[dict] = []
        for _ in range(self.max_pages):
            page = _get_json(self.client, path, self.max_retries, "Companies House", {"items_per_page": 100, "start_index": len(items)})
            if page is None:
                break  # no PSC register (some listed companies are exempt)
            batch = page.get("items") or []
            items += batch
            if not batch or len(items) >= (page.get("total_results") or 0):
                break
        return items

    def lookup(self, number: str) -> dict | None:
        """The summarized record, or None if Companies House has no company with this number."""
        if companies_house_number(number) != number:
            raise RegistryLookupError("Not a well-formed company number")
        company = _get_json(self.client, f"/company/{number}", self.max_retries, "Companies House")
        if company is None:
            return None
        try:
            if company["company_number"] != number:
                raise RegistryLookupError("Companies House returned a different company")
            return summarize_companies_house(company, self._items(f"/company/{number}/officers"), self._items(f"/company/{number}/persons-with-significant-control"))
        except (KeyError, TypeError, AttributeError):
            raise RegistryLookupError("Companies House returned an unexpected response") from None


def _is_uk(entity: dict) -> bool | None:
    """True for a UK jurisdiction, False for another one, None if the documents state none."""
    codes = {jurisdiction_code(j) for j in entity.get("jurisdictions") or []} | {jurisdiction_code(i.get("jurisdiction")) for i in entity["identifiers"] if i.get("jurisdiction")}
    codes.discard(None)
    if not codes and not entity.get("jurisdictions"):
        return None
    return "GB" in codes


def _postcode(text: str | None) -> str | None:
    m = _UK_POSTCODE.search(text or "")
    return re.sub(r"\s", "", m.group(0)).upper() if m else None


def check_companies_house(document_id: str, case_entities: list[dict], facts_by_entity: dict[str, list[dict]], held_by_entity: dict[str, list[dict]], all_facts: list[dict], entity_ids: set[str], lookup: Callable[[str], tuple[dict | None, str, bool]]) -> tuple[list[dict], list[dict]]:
    """Compare each UK case entity in ``entity_ids`` that has a company number with its Companies House record.

    ``facts_by_entity`` holds facts about each entity; ``held_by_entity`` the
    ownership facts in which it is the held party. ``lookup(number)`` returns
    (record or None, fetched_at, served_from_cache) and may raise
    RegistryLookupError. Returns (checks, issues).
    """
    checks, issues = [], []
    for entity in case_entities:
        if entity["entity_id"] not in entity_ids:
            continue
        stated = sorted({i["value"] for i in entity["identifiers"] if i["scheme"] == "REGISTRATION_NUMBER"})
        uk = _is_uk(entity)
        if not stated or uk is False:
            continue
        facts = facts_by_entity.get(entity["entity_id"], [])
        for value in stated:
            number = companies_house_number(value)
            number_facts = [f["fact_id"] for f in facts if f["predicate"] == "has_registration_number" and (f.get("object") or "").strip() == value]
            check = {"company_number": number or value, "entity_id": entity["entity_id"], "entity_name": entity["name"], "source": "COMPANIES_HOUSE"}
            if uk is None or number is None:
                # Never sent: the number may belong to another country's register.
                checks.append({**check, "status": "NOT_LOOKED_UP", "reason": "no UK jurisdiction stated" if uk is None else "not in Companies House number format"})
                continue
            check["record_url"] = COMPANIES_HOUSE_PAGE + number
            try:
                record, fetched_at, cached = lookup(number)
            except RegistryLookupError as exc:
                checks.append({**check, "status": "LOOKUP_FAILED", "error": str(exc)})
                issues.append(make_issue("REGISTRY_LOOKUP_FAILED", "The Companies House lookup failed, so this company was not checked; retry the run or check it manually.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=number_facts, details={"source": "COMPANIES_HOUSE", "company_number": number, "error": str(exc)}))
                continue
            check.update({"fetched_at": fetched_at, "served_from_cache": cached})
            ref = {"source": "COMPANIES_HOUSE", "company_number": number}
            if record is None:
                checks.append({**check, "status": "NOT_FOUND"})
                issues.append(make_issue("REGISTRY_RECORD_NOT_FOUND", "Companies House has no company with this number; it may be misprinted, invented, or registered elsewhere.", document_id=document_id, categories=["legal_identity"], fact_ids=number_facts, details=ref))
                continue

            def ids(*predicates: str) -> list[str]:
                return [f["fact_id"] for f in facts if f["predicate"] in predicates]

            comparisons = []
            # Name, against the current and previous names.
            names = sorted({n["name"] for n in entity["names"] if n["kind"] in {"legal_name", "full_name"}}) or ([entity["name"]] if entity["name"] else [])
            keys = {legal_key(n) for n in names}
            previous = {legal_key(n["name"]) for n in record["previous_company_names"]}
            if legal_key(record["company_name"] or "") in keys:
                comparisons.append(_compare("legal_name", names, record["company_name"], "MATCH"))
            elif keys & previous:
                comparisons.append(_compare("legal_name", names, record["company_name"], "MATCHES_OTHER_NAME"))
                issues.append(make_issue("REGISTRY_NAME_IS_OTHER_NAME", "The documents use a name Companies House lists as a previous name; confirm the rename.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_legal_name"), details={**ref, "document_names": names, "registry_legal_name": record["company_name"]}))
            elif names:
                comparisons.append(_compare("legal_name", names, record["company_name"], "MISMATCH"))
                issues.append(make_issue("REGISTRY_NAME_MISMATCH", "The documents' name differs from the Companies House record for the number they print; the number may belong to another company.", document_id=document_id, categories=["legal_identity"], fact_ids=ids("has_legal_name", "has_registration_number"), details={**ref, "document_names": names, "registry_legal_name": record["company_name"]}))
            former = sorted({f["object"] for f in facts if f["predicate"] == "has_former_name" and f.get("object")})
            if former:
                comparisons.append(_compare("former_names", former, [n["name"] for n in record["previous_company_names"]], "MATCH" if {legal_key(n) for n in former} <= previous else "NOT_IN_REGISTRY"))
            # Jurisdiction within the UK, where the documents name a part of it.
            wording = entity.get("jurisdictions") or []
            expected = set().union(*[_CH_JURISDICTIONS.get(identity_key(j), set()) for j in wording]) if wording else set()
            if wording and record["jurisdiction"]:
                result = "NOT_COMPARED" if not expected else "MATCH" if record["jurisdiction"] in expected else "MISMATCH"
                comparisons.append(_compare("jurisdiction", wording, record["jurisdiction"], result))
                if result == "MISMATCH":
                    issues.append(make_issue("REGISTRY_JURISDICTION_MISMATCH", "The documents' jurisdiction differs from the Companies House record.", document_id=document_id, categories=["legal_identity"], fact_ids=ids("has_jurisdiction"), details={**ref, "document": wording, "registry": record["jurisdiction"]}))
            # Status: the documents' wording is shown; an inactive company is raised whatever the documents say.
            status = record["company_status"]
            doc_status = sorted({f["object"] for f in facts if f["predicate"] == "has_legal_status" and f.get("object")})
            if doc_status and status:
                agree = all((not _ADVERSE_COMPANY.search(s)) == (status in _CH_ACTIVE) for s in doc_status)
                comparisons.append(_compare("legal_status", doc_status, status, "MATCH" if agree else "MISMATCH"))
            if status and status not in _CH_ACTIVE:
                issues.append(make_issue("REGISTRY_ENTITY_INACTIVE", "Companies House records this company as not active.", document_id=document_id, categories=["legal_identity"], severity="high" if status in _CH_CLOSED else "medium", fact_ids=ids("has_legal_status") + number_facts, details={**ref, "company_status": status, "company_status_detail": record["company_status_detail"], "date_of_cessation": record["date_of_cessation"]}))
            # Legal form.
            forms = sorted({f["object"] for f in facts if f["predicate"] == "has_legal_form" and f.get("object")})
            if forms and record["type"]:
                known = _CH_FORMS.get(record["type"])
                result = "NOT_COMPARED" if known is None else "MATCH" if all(identity_key(f) in known for f in forms) else "MISMATCH"
                comparisons.append(_compare("legal_form", forms, record["type"], result))
                if result == "MISMATCH":
                    issues.append(make_issue("REGISTRY_LEGAL_FORM_MISMATCH", "The documents' legal form differs from the company type Companies House holds.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_legal_form"), details={**ref, "document": forms, "registry": record["type"]}))
            # Incorporation date.
            dates = sorted({f["effective_date"] for f in facts if f["predicate"] == "has_incorporation_date" and f.get("effective_date")})
            if dates and record["date_of_creation"]:
                match = record["date_of_creation"] in dates
                comparisons.append(_compare("incorporation_date", dates, record["date_of_creation"], "MATCH" if match else "MISMATCH"))
                if not match:
                    issues.append(make_issue("REGISTRY_INCORPORATION_DATE_MISMATCH", "The documents' incorporation date differs from the Companies House record.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_incorporation_date"), details={**ref, "document": dates, "registry": record["date_of_creation"]}))
            # Registered office, by postcode: the office may have moved since the document's date.
            addresses = [f for f in facts if f["predicate"] == "has_registered_address" and f.get("object")]
            registry_postcode = _postcode(record["registered_office_address"].get("postal_code"))
            if addresses and registry_postcode:
                stated_codes = sorted({c for c in (_postcode(f["object"]) for f in addresses) if c})
                result = "NOT_COMPARED" if not stated_codes else "MATCH" if registry_postcode in stated_codes else "MISMATCH"
                comparisons.append(_compare("registered_address", [f["object"] for f in addresses], record["registered_office_address"], result))
                if result == "MISMATCH":
                    issues.append(make_issue("REGISTRY_ADDRESS_MISMATCH", "The documents' registered office postcode differs from the Companies House record; the office may have moved since the document was issued.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=ids("has_registered_address"), details={**ref, "document": stated_codes, "registry": registry_postcode, "as_of": sorted({f["effective_date"] for f in addresses if f.get("effective_date")})}))
            if record["flags"]:
                issues.append(make_issue("REGISTRY_RECORD_FLAGGED", "The Companies House record carries flags a reviewer should see.", document_id=document_id, categories=["legal_identity"], severity="medium", fact_ids=number_facts, details={**ref, "flags": record["flags"]}))
            officers = _compare_officers(document_id, ref, keys, all_facts, record, issues)
            pscs = _compare_pscs(document_id, ref, held_by_entity.get(entity["entity_id"], []), record, issues)
            checks.append({**check, "status": "CHECKED", "record": record, "comparisons": comparisons, "officers": officers, "persons_with_significant_control": pscs})
    return checks, issues


def _compare_officers(document_id: str, ref: dict, company_keys: set[str], all_facts: list[dict], record: dict, issues: list[dict]) -> dict:
    """Each director or officer the documents name for this company, against the register's appointments."""
    stated = [f for f in all_facts if f["predicate"] in {"is_director_of", "is_officer_of"} and legal_key(f.get("object") or "") in company_keys]
    results, matched = [], set()
    for f in stated:
        roles = _CH_DIRECTOR_ROLES if f["predicate"] == "is_director_of" else None
        found = [(n, o) for n, o in enumerate(record["officers"]) if (roles is None or o["officer_role"] in roles) and same_person(f["subject"], o["name"])]
        matched |= {n for n, _ in found}
        as_of = f.get("effective_date")
        serving = [o for _, o in found if not o["resigned_on"] or (as_of and o["resigned_on"] >= as_of)]
        result = "NOT_FOUND" if not found else "CURRENT" if any(not o["resigned_on"] for o in serving) else "RESIGNED_SINCE" if serving else "RESIGNED"
        results.append({"name": f["subject"], "role": f["predicate"], "as_of": as_of, "result": result, "registry": [o for _, o in found], "fact_id": f["fact_id"]})
        if result == "NOT_FOUND":
            issues.append(make_issue("REGISTRY_OFFICER_NOT_FOUND", "Companies House lists no matching appointment for this officer; the name may be spelt differently, or the appointment never filed.", document_id=document_id, categories=["legal_identity", "related_parties"], severity="medium", fact_ids=[f["fact_id"]], details={**ref, "name": f["subject"], "role": f["predicate"]}))
        elif result == "RESIGNED":
            issues.append(make_issue("REGISTRY_OFFICER_RESIGNED", "Companies House records this officer as resigned before the date the documents name them.", document_id=document_id, categories=["legal_identity", "related_parties"], severity="medium", fact_ids=[f["fact_id"]], details={**ref, "name": f["subject"], "as_of": as_of, "resigned_on": sorted(o["resigned_on"] for _, o in found)}))
    # Current appointments the documents don't mention are listed, not raised: a document may name only some officers.
    registry_only = [o for n, o in enumerate(record["officers"]) if n not in matched and not o["resigned_on"]]
    return {"stated": results, "registry_only": registry_only}


def _compare_pscs(document_id: str, ref: dict, held: list[dict], record: dict, issues: list[dict]) -> dict:
    """Reported stakes in this company, against the bands of its people with significant control."""
    current = [p for p in record["persons_with_significant_control"] if not p["ceased_on"]]
    results = []
    for f in held:
        if f["predicate"] != "owns_equity_in" or not isinstance(f.get("value"), (int, float)):
            continue
        holder = f["subject"]
        psc = next((p for p in current if legal_key(p["name"]) == legal_key(holder) or same_person(holder, p["name"])), None)
        band = (psc or {}).get("bands", {}).get("ownership-of-shares")
        if psc is None:
            result = "NOT_A_PSC" if f["value"] <= 25 else "MISSING_FROM_REGISTER"
        else:
            result = "NO_SHARE_BAND" if band is None else "WITHIN_BAND" if band[0] < f["value"] <= band[1] or (band[1] == 100 and f["value"] == 100) else "OUTSIDE_BAND"
        results.append({"holder": holder, "stake": f["value"], "as_of": f.get("effective_date"), "result": result, "psc": psc, "fact_id": f["fact_id"]})
        if result == "MISSING_FROM_REGISTER":
            issues.append(make_issue("REGISTRY_PSC_MISSING", "The documents report a stake above 25%, but the holder is not a current person with significant control at Companies House. The stake may be held indirectly, may have changed, or the register may be incomplete.", document_id=document_id, categories=["ownership", "control"], severity="medium", fact_ids=[f["fact_id"]], details={**ref, "holder": holder, "stake": f["value"], "as_of": f.get("effective_date")}))
        elif result == "OUTSIDE_BAND":
            issues.append(make_issue("REGISTRY_PSC_BAND_MISMATCH", "The reported stake falls outside the shareholding band the company filed for this person with significant control.", document_id=document_id, categories=["ownership", "control"], severity="medium", fact_ids=[f["fact_id"]], details={**ref, "holder": holder, "stake": f["value"], "registry_band": band, "as_of": f.get("effective_date")}))
    return {"stated": results, "registry": current}
