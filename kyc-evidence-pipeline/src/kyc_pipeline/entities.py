"""Entity layer: which legal entity (or group) each fact is about.

Identity comes from three cited sources, never from model judgment:

1. ``legal_identity`` facts — registry fields and the SEC cover page: legal
   name, identifiers and jurisdiction.
2. The document's own definitions — ``Berkshire Hathaway Inc. ("Berkshire")``,
   ``... and subsidiaries (the "Company")``, ``the terms "we" ... refer to ...``.
3. Names used in facts, which only ever match the same name.

Each document is resolved on its own first. Records then merge across the
case on a shared identifier, or on the same legal name where no identifier or
jurisdiction contradicts it. Groups ("X and its subsidiaries", "X or a
subsidiary") are separate entities pointing at X and are never merged into X:
a group-level claim does not say which legal entity holds a stake.
"""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Any

from .evidence import _evidence_record, make_issue
from .util import identity_key, lei_check_digits_valid, stable_id, temporal_overlap

_FORMS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "lp": "lp", "llp": "llp", "ltd": "ltd", "limited": "ltd", "plc": "plc", "sa": "sa", "ag": "ag",
    "gmbh": "gmbh", "nv": "nv", "bv": "bv", "se": "se", "spa": "spa", "sas": "sas", "pty": "pty", "na": "na",
}
_ORG_WORDS = set(_FORMS.values()) | {"group", "holdings", "bank", "trust", "partners", "fund"}
_PRONOUNS = {"we", "us", "our", "company", "registrant", "corporation"}
_CONNECTORS = {"of", "&", "de", "du", "des", "la", "le", "the", "für", "y"}
# A name stops at a security ("Preferred Stock of X") and at "and" ("NW Merger Sub Inc. and X Corp.").
_NAME_STOPS = {"and", "stock", "shares", "notes", "preferred", "common", "units", "bonds", "debentures", "warrants", "securities", "interest", "interests"}
_ABBREVIATIONS = {"Bros.", "Intl.", "Mfg.", "Natl.", "Assn.", "St.", "Mt."}
_ALIAS_FILLER = {"the", "collectively", "together", "referred", "to", "herein", "hereinafter", "as", "or", "and", "similar", "expressions", "each", "a", "an"}
_GENERIC_PARENTS = {"parent company", "company", "registrant", "parent"}
# Identifiers that are unique on their own; a registration number needs its jurisdiction.
_IDENTIFIER_PREDICATES = {"has_sec_file_number": "SEC_FILE_NUMBER", "has_us_employer_identification_number": "US_EIN", "has_registration_number": "REGISTRATION_NUMBER", "has_lei": "LEI"}
_SCHEME_RANK = ["LEI", "SEC_FILE_NUMBER", "US_EIN", "REGISTRATION_NUMBER"]
_ENTITY_OBJECTS = {"owns_equity_in", "has_voting_rights_over", "disposed_interest_in", "is_beneficial_owner_of", "is_director_of", "is_officer_of"}
# Predicates whose subject is a natural person unless the name is an organization's.
_PERSON_PREDICATES = {"is_beneficial_owner_of", "is_director_of", "is_officer_of", "has_date_of_birth", "has_nationality", "has_place_of_birth", "holds_identity_document", "identity_document_issued_on", "identity_document_expires_on", "has_residential_address", "holds_public_function", "declared_pep_status"}
_PLACEHOLDERS = {"UNRESOLVED_REGISTRANT", "UNRESOLVED_REGISTRY_SUBJECT", "UNRESOLVED_REPORTING_ENTITY"}

_QUOTED = re.compile(r"[“\"]([^”\"]{1,60})[”\"]")
_PAREN = re.compile(r"\(([^()]{1,200})\)")
_TERMS = re.compile(r"the\s+terms?\s+(?P<terms>(?:[“\"][^”\"]{1,25}[”\"][,\s]*(?:or\s+|and\s+)?)+)(?:are\s+used\s+in\s+this\s+(?:report|document)\s+to\s+)?refers?\s+(?:collectively\s+)?to\s+(?P<ref>[^.;]{3,160})", re.I)
_GROUP_TAIL = re.compile(r"\s+and\s+(?:its\s+|the\s+|certain\s+(?:of\s+its\s+)?)?(?:consolidated\s+)?(?:subsidiaries|affiliates)\b.*$", re.I)
_GROUP_END = re.compile(r"\s+and\s+(?:its\s+|the\s+|certain\s+(?:of\s+its\s+)?)?(?:consolidated\s+)?(?:subsidiaries|affiliates)$", re.I)
_OR_SUBSIDIARY = re.compile(r"^(?P<base>.+?)\s+(?:or|and/or)\s+(?:a|any|one\s+or\s+more\s+of\s+its|its)\s+subsidiar(?:y|ies)$", re.I)


def legal_key(name: str) -> str:
    """Name key with legal forms normalized (Inc./Incorporated, Co./Company) and a leading 'The' dropped."""
    tokens = identity_key(name.replace(".", "").replace("’s", "").replace("'s", "")).split()
    if tokens[:1] == ["the"]:
        tokens = tokens[1:]
    return " ".join(_FORMS.get(t, t) for t in tokens)


_TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr", "sir", "dame", "prof"}


def person_key(name: str | None) -> str:
    """'Ms Jane Smith' and 'JANE SMITH' agree; 'J Smith' does not."""
    return " ".join(t for t in legal_key(name or "").split() if t not in _TITLES)


def same_person(document: str, registry: str) -> bool:
    """Same first name and surname, and every name part in ``document`` also in ``registry``.

    'Jane Smith' matches 'Mrs Jane Elizabeth Smith'; 'J Smith' and 'Jane Smyth' do not.
    """
    d, r = person_key(document).split(), person_key(registry).split()
    return bool(d and r) and d[0] == r[0] and d[-1] == r[-1] and set(d) <= set(r)


def alias_key(name: str) -> str:
    tokens = identity_key(name).split()
    return " ".join(tokens[1:] if tokens[:1] == ["the"] else tokens)


def is_organization(name: str) -> bool:
    """Ends in a legal form or organization word, and is more than that word ("Company")."""
    tokens = legal_key(name).split()
    return len(tokens) >= 2 and tokens[-1] in _ORG_WORDS and any(t not in _ORG_WORDS for t in tokens)


def _is_form(token: str) -> bool:
    return _FORMS.get(identity_key(token.replace(".", "")), "") != ""


def _name_before(text: str) -> tuple[str, int, bool] | None:
    """The proper name ending at ``text``'s end, its offset, and whether it names a group."""
    end = len(text.rstrip())
    pre = re.sub(r"[’']s$", "", text[:end]).rstrip(" ,")
    if pre.endswith(")"):
        return None  # "(formerly known as X) (“Y”)": Y is not X
    group = _GROUP_END.search(pre)
    if group:
        pre = pre[:group.start()]
    # "Primary Group—The Berkshire Hathaway Primary Group": a dash starts a new name.
    offset = max(pre.rfind("—"), pre.rfind("–")) + 1
    tokens = [m for m in re.finditer(r"\S+", pre) if m.start() >= offset]
    i = len(tokens)
    while i > 0:
        tok = tokens[i - 1].group()
        word = tok.rstrip(",")
        last = i == len(tokens)
        if re.fullmatch(r"[(“\"][A-Z][^()“”\"]*", word):
            i -= 1  # "(Precision Castparts Corp. (“PCC”))": the name opens the bracket
            break
        if re.search(r"[()“”\"]", tok) or word.casefold() in _NAME_STOPS:
            break
        if tok.endswith(",") and (last or not _is_form(tokens[i].group())):
            break
        if not last and word.endswith(".") and not _is_form(word) and word not in _ABBREVIATIONS and not re.fullmatch(r"(?:[A-Z]\.)+", word):
            break
        if word[:1].isdigit() and (last or not _is_form(tokens[i].group())):
            break  # "Enterprises 18 Corp." keeps its number; "November 2023 of" does not
        if word[:1].isupper() or word[:1].isdigit() or word.casefold() in _CONNECTORS:
            i -= 1
            continue
        break
    while i < len(tokens) and (tokens[i].group().casefold() in _CONNECTORS - {"the"} or tokens[i].group() == "the" or tokens[i].group()[:1].isdigit()):
        i += 1
    if i >= len(tokens):
        return None
    start = tokens[i].start() + (1 if pre[tokens[i].start()] in "(“\"" else 0)
    return pre[start:].strip(" ,"), start, bool(group)


def _definitions(document: dict, segments: list[dict]) -> list[dict]:
    found: list[dict] = []
    for seg in segments:
        if seg["kind"] not in {"paragraph", "footnote"}:
            continue
        text = seg["text"]
        for m in _PAREN.finditer(text):
            aliases = [a.strip(" ,.;") for a in _QUOTED.findall(m.group(1))]
            filler = _QUOTED.sub(" ", m.group(1)).replace(",", " ").casefold().split()
            if not aliases or any(w not in _ALIAS_FILLER for w in filler):
                continue
            named = _name_before(text[:m.start()])
            if not named:
                continue
            name, start, group = named
            keys = [alias_key(a) for a in aliases if alias_key(a)]
            if not keys or not (is_organization(name) or all(k in _PRONOUNS for k in keys)):
                continue
            found.append({"aliases": aliases, "alias_keys": keys, "target": name, "group": group, "generic": False, "evidence": _evidence_record(document, seg, start, m.end())})
        for m in _TERMS.finditer(text):
            aliases = [a.strip(" ,.;") for a in _QUOTED.findall(m.group("terms"))]
            ref = m.group("ref").strip()
            group = bool(_GROUP_TAIL.search(ref))
            base = _GROUP_TAIL.sub("", ref).strip()
            keys = [alias_key(a) for a in aliases if alias_key(a)]
            if not keys or not base:
                continue
            generic = alias_key(base) in _GENERIC_PARENTS
            found.append({"aliases": aliases, "alias_keys": keys, "target": None if generic else base, "group": group, "generic": generic, "evidence": _evidence_record(document, seg, m.start(), m.end())})
    return found


class _DocumentEntities:
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.parent: dict[str, str] = {}
        self.attrs: dict[str, dict] = {}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def ensure(self, key: str, entity_type: str) -> str:
        if key not in self.parent:
            self.parent[key] = key
            self.attrs[key] = {"entity_type": entity_type, "names": [], "identifiers": [], "jurisdictions": [], "roles": set(), "group_of": None, "resolution_evidence_ids": set(), "birth_dates": []}
        return self.find(key)

    def org(self, name: str, kind: str, evidence_ids: list[str]) -> str:
        key = self.ensure("org:" + legal_key(name), "organization" if is_organization(name) else "unknown")
        self.add_name(key, name, kind, evidence_ids)
        return key

    def group(self, parent: str) -> str:
        parent = self.find(parent)
        key = self.ensure("group:" + parent, "group")
        self.attrs[key]["group_of"] = parent
        return key

    def add_name(self, key: str, name: str, kind: str, evidence_ids: list[str]) -> None:
        names = self.attrs[self.find(key)]["names"]
        for existing in names:
            if existing["name"] == name and existing["kind"] == kind:
                existing["evidence_ids"] = sorted(set(existing["evidence_ids"]) | set(evidence_ids))
                return
        names.append({"name": name, "kind": kind, "evidence_ids": sorted(set(evidence_ids))})

    def merge(self, a: str, b: str, evidence_ids: list[str]) -> str:
        a, b = self.find(a), self.find(b)
        if a == b:
            return a
        if a == "org:@registrant":
            a, b = b, a
        self.parent[b] = a
        target, source = self.attrs[a], self.attrs.pop(b)
        for field in ("names", "identifiers", "jurisdictions", "birth_dates"):
            target[field].extend(source[field])
        target["roles"] |= source["roles"]
        target["resolution_evidence_ids"] |= source["resolution_evidence_ids"] | set(evidence_ids)
        if target["entity_type"] == "unknown":
            target["entity_type"] = source["entity_type"]
        return a

    def entity_id(self, key: str) -> str:
        return stable_id("ent", self.document_id, self.find(key))


    def preferred_name(self, key: str) -> str | None:
        names = self.attrs[self.find(key)]["names"]
        return next((n["name"] for kind in ("legal_name", "full_name", "defined_term", "mention") for n in names if n["kind"] == kind), None)

    def merge_groups(self) -> None:
        """Groups whose parents were merged are the same group."""
        by_parent: dict[str, str] = {}
        for key in [k for k in self.attrs if self.attrs[k]["entity_type"] == "group"]:
            root = self.find(self.attrs[key]["group_of"])
            self.attrs[key]["group_of"] = root
            if root in by_parent:
                self.merge(by_parent[root], key, [])
            else:
                by_parent[root] = key


def resolve_document(document: dict, segments: list[dict], facts: list[dict], issues: list[dict], requirements: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Link each fact's subject (and owned or controlled object) to a document entity.

    Returns (entities, facts, new evidence, issues). An ENTITY_IDENTITY_UNRESOLVED
    issue is dropped where the document itself resolves the subject; a subject
    resolved only to a group gets ENTITY_RESOLVED_TO_GROUP instead.
    """
    doc_id = document["document_id"]
    material = {r["category"]: r["material"] for r in requirements}
    graph = _DocumentEntities(doc_id)
    new_issues: list[dict] = []

    # 1. Identity facts: registry fields and the SEC cover page.
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for f in facts:
        if f["category"] == "legal_identity" and f["subject"] != "UNRESOLVED_REGISTRY_SUBJECT":
            by_subject[f["subject"]].append(f)
    for subject, items in by_subject.items():
        key = graph.ensure("org:@registrant", "organization") if subject == "UNRESOLVED_REGISTRANT" else graph.org(subject, "legal_name", items[0]["evidence_ids"][:1])
        attrs = graph.attrs[key]
        jurisdiction = next((f["object"] for f in items if f["predicate"] == "has_jurisdiction" and f.get("object")), None)
        for f in items:
            ev = f["evidence_ids"][:1]
            if not f.get("object"):
                continue
            if f["predicate"] == "has_legal_name":
                graph.add_name(key, f["object"], "legal_name", ev)
            elif f["predicate"] == "has_jurisdiction":
                attrs["jurisdictions"].append({"value": f["object"], "evidence_ids": ev, "fact_id": f["fact_id"]})
            elif f["predicate"] in _IDENTIFIER_PREDICATES:
                scheme = _IDENTIFIER_PREDICATES[f["predicate"]]
                if scheme == "LEI" and not lei_check_digits_valid(f["object"].replace(" ", "").upper()):
                    continue  # a malformed LEI must not identify, or merge, entities
                attrs["identifiers"].append({"scheme": scheme, "value": f["object"].strip(), "jurisdiction": jurisdiction if scheme == "REGISTRATION_NUMBER" else None, "evidence_ids": ev, "fact_id": f["fact_id"]})
            if f["predicate"] == "has_sec_file_number":
                attrs["roles"].add("sec_registrant")
    # Cover-page fields split across text blocks belong to the one registrant.
    named_registrants = [k for k, a in graph.attrs.items() if "sec_registrant" in a["roles"] and k != "org:@registrant"]
    if "org:@registrant" in graph.parent and len(named_registrants) == 1:
        graph.merge(named_registrants[0], "org:@registrant", [])

    # 2. The document's own definitions. A named definition beats a generic one.
    definitions = _definitions(document, segments)
    evidence = {d["evidence"]["evidence_id"]: d["evidence"] for d in definitions}
    named: dict[str, list[dict]] = defaultdict(list)
    generic: dict[str, list[dict]] = defaultdict(list)
    for d in definitions:
        for key in d["alias_keys"]:
            (generic if d["generic"] else named)[key].append(d)
    alias_targets: dict[str, str | None] = {}
    alias_evidence: dict[str, list[str]] = {}

    def resolve_name(name: str, depth: int) -> str | None:
        if is_organization(name):
            return graph.org(name, "full_name", [])
        key = alias_key(name)
        if key not in alias_targets and depth < 3 and key in named:
            resolve_alias(key, depth + 1)
        return alias_targets.get(key)

    def resolve_alias(key: str, depth: int = 0) -> None:
        if key in alias_targets:
            return
        alias_targets[key] = None  # guards definition cycles
        resolved: dict[str, list[str]] = defaultdict(list)
        for d in named[key]:
            base = resolve_name(d["target"], depth)
            if base is None:
                continue
            if is_organization(d["target"]):
                graph.add_name(base, d["target"], "full_name", [d["evidence"]["evidence_id"]])
            target = graph.group(base) if d["group"] else graph.find(base)
            resolved[target].append(d["evidence"]["evidence_id"])
        if len(resolved) > 1:
            # "McLane Company, Inc." and "McLane Company" defined as the same term are one entity.
            stems = {" ".join(t for t in legal_key(graph.preferred_name(graph.attrs[graph.find(k)]["group_of"] or k) or "").split() if t not in _FORMS.values()) for k in resolved}
            kinds = {graph.attrs[graph.find(k)]["entity_type"] for k in resolved}
            if len(stems) == 1 and "" not in stems and len(kinds) == 1:
                keys = list(resolved)
                ids = [e for k in keys for e in resolved[k]]
                if kinds == {"group"}:
                    parents = [graph.attrs[graph.find(k)]["group_of"] for k in keys]
                    for other in parents[1:]:
                        graph.merge(parents[0], other, ids)
                    graph.merge_groups()
                    merged = graph.find(keys[0])
                else:
                    merged = keys[0]
                    for other in keys[1:]:
                        merged = graph.merge(merged, other, ids)
                resolved = {graph.find(merged): ids}
        if len(resolved) > 1:
            new_issues.append(make_issue("DEFINED_TERM_AMBIGUOUS", "The document defines the same term for different entities; facts using it were not linked.", document_id=doc_id, categories=["legal_identity"], severity="medium", details={"term": key, "evidence_ids": sorted({e for ids in resolved.values() for e in ids})}))
        elif resolved:
            (target, ids), = resolved.items()
            for alias in {a for d in named[key] for a in d["aliases"] if alias_key(a) == key}:
                graph.add_name(target, alias, "pronoun" if key in _PRONOUNS else "defined_term", ids)
            alias_targets[key] = target
            alias_evidence[key] = ids

    for key in list(named):
        resolve_alias(key)
    if generic:
        registrant = graph.find(named_registrants[0]) if named_registrants else graph.ensure("org:@registrant", "organization")
        for key, defs in generic.items():
            target = alias_targets.get(key)
            unnamed = graph.find(registrant) == graph.find("org:@registrant") if "org:@registrant" in graph.parent else False
            if target and unnamed and any(d["group"] for d in defs) and graph.attrs[graph.find(target)]["entity_type"] == "group":
                # "the parent company and its subsidiaries" and "X and its subsidiaries"
                # define the same term, so the unnamed registrant is X.
                ids = [d["evidence"]["evidence_id"] for d in defs] + alias_evidence[key]
                registrant = graph.merge(graph.attrs[graph.find(target)]["group_of"], registrant, ids)
        for key, defs in generic.items():
            if alias_targets.get(key):
                continue
            ids = [d["evidence"]["evidence_id"] for d in defs]
            target = graph.group(registrant) if any(d["group"] for d in defs) else graph.find(registrant)
            for alias in {a for d in defs for a in d["aliases"] if alias_key(a) == key}:
                graph.add_name(target, alias, "pronoun", ids)
            alias_targets[key] = target
            alias_evidence[key] = ids

    # 3. Link facts.
    def link(name: str) -> tuple[str | None, str, list[str]]:
        name = name.strip()
        if name == "UNRESOLVED_REGISTRANT":
            return (graph.find("org:@registrant"), "REGISTRANT_COVER", []) if "org:@registrant" in graph.parent else (None, "UNRESOLVED", [])
        if name in _PLACEHOLDERS:
            return None, "UNRESOLVED", []
        group = _OR_SUBSIDIARY.match(name) or re.match(r"^(?P<base>.+?)\s+and\s+(?:its\s+)?(?:consolidated\s+)?subsidiaries$", name, re.I)
        if group:
            base, _, ids = link(group.group("base"))
            return (graph.group(base), "GROUP", ids) if base else (None, "UNRESOLVED", [])
        key = alias_key(name)
        if alias_targets.get(key):
            return graph.find(alias_targets[key]), "DEFINED_TERM", alias_evidence[key]
        if not key or key in _PRONOUNS:
            return None, "UNRESOLVED", []
        if "org:" + legal_key(name) in graph.parent:
            return graph.find("org:" + legal_key(name)), "LEGAL_NAME", []
        return graph.ensure("org:" + legal_key(name), "organization" if is_organization(name) else "unknown"), "MENTION", []

    pending: list[tuple[dict, dict]] = []
    for f in facts:
        links = {}
        # An article's headline is not a party.
        roles = [] if f.get("category") == "source_metadata" else [("subject", f["subject"])] + ([("object", f["object"])] if f["predicate"] in _ENTITY_OBJECTS and f.get("object") else [])
        for role, name in roles:
            key, basis, ids = link(name)
            if key:
                links[role] = (key, basis, ids)
                if basis in {"LEGAL_NAME", "MENTION"}:
                    graph.add_name(key, name, "mention", f["evidence_ids"][:1])
                    # A director, a declared beneficial owner, or a holder the source calls an individual.
                    attrs = graph.attrs[graph.find(key)]
                    if role == "subject" and attrs["entity_type"] == "unknown" and not is_organization(name) and (f["predicate"] in _PERSON_PREDICATES or f.get("qualifier") == "individual"):
                        attrs["entity_type"] = "person"
                if role == "subject" and f["predicate"] == "has_date_of_birth" and f.get("effective_date"):
                    graph.attrs[graph.find(key)]["birth_dates"].append({"value": f["effective_date"], "fact_id": f["fact_id"]})
        pending.append((f, links))
    graph.merge_groups()

    # 4. Final records, then facts against the final (merged) entities.
    records: dict[str, dict] = {}
    for key, attrs in graph.attrs.items():
        names = attrs["names"]
        if attrs["entity_type"] == "group":
            parent_name = graph.preferred_name(attrs["group_of"])
            name = f"{parent_name} and subsidiaries" if parent_name else None
        else:
            name = graph.preferred_name(key)
        identified = bool(attrs["identifiers"]) or any(n["kind"] == "legal_name" for n in names)
        records[key] = {
            "entity_id": graph.entity_id(key), "document_id": doc_id, "entity_type": attrs["entity_type"],
            "name": name, "names": names, "identifiers": attrs["identifiers"], "jurisdictions": attrs["jurisdictions"],
            "roles": sorted(attrs["roles"]), "group_of": graph.entity_id(attrs["group_of"]) if attrs["group_of"] else None,
            "birth_dates": sorted({b["value"] for b in attrs["birth_dates"]}),
            "strength": "IDENTIFIED" if identified else "DEFINED" if any(n["kind"] in {"full_name", "defined_term", "pronoun"} for n in names) else "MENTIONED",
            "resolution_evidence_ids": sorted(attrs["resolution_evidence_ids"]),
        }
        if "sec_registrant" in attrs["roles"] and not name:
            new_issues.append(make_issue("REGISTRANT_NAME_NOT_IN_TEXT", "The filing's cover page identifies a registrant, but its name is not in the parsed text (it may be an image) and the document's definitions do not supply it.", document_id=doc_id, categories=["legal_identity"], severity="medium", details={"entity_id": records[key]["entity_id"]}))

    resolved_facts: set[str] = set()
    out_facts: list[dict] = []
    for f, links in pending:
        fact = dict(f)
        fact["entity_links"] = {role: {"entity_id": records[graph.find(key)]["entity_id"], "basis": basis, "evidence_ids": sorted(set(ids))} for role, (key, basis, ids) in links.items()}
        subject = records[graph.find(links["subject"][0])] if "subject" in links else None
        if subject is None:
            fact["entity_resolution"] = "UNRESOLVED"
        elif subject["entity_type"] == "group":
            fact["entity_resolution"] = "RESOLVED_TO_GROUP"
            parent = records[graph.find(graph.attrs[graph.find(links["subject"][0])]["group_of"])]
            new_issues.append(make_issue("ENTITY_RESOLVED_TO_GROUP", f"The source attributes this claim to {subject['name'] or 'a corporate group'}, not to one legal entity; the direct holder or actor is not stated.", document_id=doc_id, categories=[fact["category"]], severity="high" if material.get(fact["category"]) else "medium", fact_ids=[fact["fact_id"]], details={"subject": fact["subject"], "group_entity_id": subject["entity_id"], "parent_entity_id": parent["entity_id"]}))
        elif subject["strength"] != "MENTIONED" and subject["name"]:
            fact["entity_resolution"] = "RESOLVED"
        else:
            fact["entity_resolution"] = "TEXT_MATCH_ONLY"
        if fact["entity_resolution"] in {"RESOLVED", "RESOLVED_TO_GROUP"}:
            resolved_facts.add(fact["fact_id"])
        out_facts.append(fact)

    # A placeholder subject ("UNRESOLVED_REGISTRANT") resolved from the cover page
    # is no longer an unsupported subject; other literal-support failures stay.
    placeholder_resolved = {f["fact_id"] for f in out_facts if f["subject"] in _PLACEHOLDERS and f["fact_id"] in resolved_facts}
    kept = [i for i in issues if not (
        (i["code"] == "ENTITY_IDENTITY_UNRESOLVED" and i["fact_ids"] and set(i["fact_ids"]) <= resolved_facts)
        or (i["code"] == "CLAIM_LITERAL_SUPPORT_INCOMPLETE" and i["details"].get("failed_checks") == ["subject_literal_present"] and i["fact_ids"] and set(i["fact_ids"]) <= placeholder_resolved))]
    entities = [records[k] for k in graph.attrs]
    return entities, out_facts, list(evidence.values()), kept + new_issues


def _ids(record: dict) -> set[tuple]:
    return {(i["scheme"], i["value"], i["jurisdiction"] if i["scheme"] == "REGISTRATION_NUMBER" else None) for i in record["identifiers"]}


def resolve_case(entities: list[dict], document_id: str) -> tuple[list[dict], dict[str, str], list[dict]]:
    """Merge document entities across a case.

    Returns (case entities, document entity id -> case entity id, issues).
    """
    by_id = {e["entity_id"]: e for e in entities}
    parent = {eid: eid for eid in by_id}
    members: dict[str, list[str]] = {eid: [eid] for eid in by_id}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
            members[a] += members.pop(b)

    def schemes(root: str) -> dict[tuple, set[str]]:
        found: dict[tuple, set[str]] = defaultdict(set)
        for eid in members[root]:
            for scheme, value, jurisdiction in _ids(by_id[eid]):
                found[(scheme, jurisdiction)].add(value)
        return found

    def jurisdictions(root: str) -> set[str]:
        return {identity_key(j["value"]) for eid in members[root] for j in by_id[eid]["jurisdictions"]}

    def births(root: str) -> set[str]:
        return {d for eid in members[root] for d in by_id[eid].get("birth_dates", [])}

    def born_apart(a: str, b: str) -> bool:
        """Both have dates of birth and none of them agree: different people with one name."""
        da, db = births(a), births(b)
        return bool(da and db) and not any(temporal_overlap(x, y) for x in da for y in db)

    orgs = [e for e in entities if e["entity_type"] != "group"]
    holders: dict[tuple, list[str]] = defaultdict(list)
    for e in orgs:
        for identifier in _ids(e):
            holders[identifier].append(e["entity_id"])
    for ids in holders.values():
        for other in ids[1:]:
            union(ids[0], other)
    issues: list[dict] = []
    by_name: dict[str, list[str]] = defaultdict(list)
    for e in orgs:
        # People are matched on their name without titles ("Ms Jane Smith" is "Jane Smith"), and only by name.
        name_key = person_key if e["entity_type"] == "person" else legal_key
        for key in {name_key(n["name"]) for n in e["names"] if n["kind"] in {"legal_name", "full_name", "mention"}}:
            if key:
                by_name[key].append(e["entity_id"])
    for key, ids in by_name.items():
        for other in ids[1:]:
            a, b = find(ids[0]), find(other)
            if a == b:
                continue
            sa, sb = schemes(a), schemes(b)
            clash = sorted(s[0] for s in set(sa) & set(sb) if sa[s] != sb[s])
            ja, jb = jurisdictions(a), jurisdictions(b)
            if clash or (ja and jb and not ja & jb) or born_apart(a, b):
                issues.append(make_issue("ENTITY_NAME_COLLISION", "Entities with the same name have different identifiers, jurisdictions or dates of birth and were kept apart; confirm they are different entities.", document_id=document_id, categories=["legal_identity"], severity="medium", details={"name_key": key, "entity_ids": sorted({ids[0], other}), "conflicting_schemes": clash, "jurisdictions": sorted(ja | jb), "birth_dates": sorted(births(a) | births(b))}))
                continue
            union(a, b)
    # A person named with a middle name in one document ("Jane Elizabeth Smith" on a passport) and
    # without it in another ("Jane Smith" as a director) is joined when no date of birth disagrees.
    people = [e for e in orgs if e["entity_type"] == "person"]
    for x, y in combinations(people, 2):
        a, b = find(x["entity_id"]), find(y["entity_id"])
        if a == b or born_apart(a, b):
            continue
        names_a = {n["name"] for eid in members[a] for n in by_id[eid]["names"]}
        names_b = {n["name"] for eid in members[b] for n in by_id[eid]["names"]}
        pair = next(((p, q) for p in names_a for q in names_b if same_person(p, q) or same_person(q, p)), None)
        if pair:
            union(a, b)
            issues.append(make_issue("PERSON_MATCHED_ON_NAME_PARTS", "Two names were joined as one person because one contains all parts of the other (for example a middle name); confirm they are the same person.", document_id=document_id, categories=["identity_fields", "related_parties"], severity="medium", details={"names": sorted(pair), "entity_ids": sorted({x["entity_id"], y["entity_id"]}), "birth_dates": sorted(births(find(a)))}))

    mapping: dict[str, str] = {}
    case_entities: list[dict] = []
    for root in [r for r in members if r in {find(e["entity_id"]) for e in orgs}]:
        group = [by_id[eid] for eid in sorted(members[root])]
        identifiers = sorted({i for m in group for i in _ids(m)}, key=lambda i: (_SCHEME_RANK.index(i[0]) if i[0] in _SCHEME_RANK else 99, i[1], i[2] or ""))
        names = sorted({legal_key(n["name"]) for m in group for n in m["names"] if n["kind"] in {"legal_name", "full_name", "mention"}} - {""})
        # The strongest stable key names the case entity, so ids survive new documents.
        anchor = f"{identifiers[0][0]}:{identifiers[0][1]}@{identifiers[0][2] or ''}" if identifiers else "name:" + (names[0] if names else root)
        case_id = stable_id("entity", anchor)
        for m in group:
            mapping[m["entity_id"]] = case_id
        documents = sorted({m["document_id"] for m in group})
        shared = any(len({by_id[eid]["document_id"] for eid in ids if find(eid) == root}) > 1 for ids in holders.values())
        values: dict[tuple, set[str]] = defaultdict(set)
        fact_ids: set[str] = set()
        for m in group:
            for i in m["identifiers"]:
                values[(i["scheme"], i["jurisdiction"])].add(i["value"])
                if i.get("fact_id"):
                    fact_ids.add(i["fact_id"])
        conflicts = {s + (f"@{j}" if j else ""): sorted(v) for (s, j), v in values.items() if len(v) > 1}
        if conflicts:
            issues.append(make_issue("ENTITY_IDENTIFIER_CONFLICT", "One entity has different values for the same identifier across sources; this may be a data error, or two entities joined by another shared identifier.", document_id=document_id, categories=["legal_identity"], fact_ids=sorted(fact_ids), details={"entity_id": case_id, "conflicts": conflicts}))
        case_entities.append({
            "entity_id": case_id,
            "entity_type": next((m["entity_type"] for m in group if m["entity_type"] != "unknown"), "unknown"),
            "name": next((m["name"] for strength in ("IDENTIFIED", "DEFINED", "MENTIONED") for m in group if m["strength"] == strength and m["name"]), None),
            "names": sorted(({"name": n["name"], "kind": n["kind"]} for m in group for n in m["names"]), key=lambda n: (n["kind"], n["name"])),
            "identifiers": [{"scheme": s, "value": v, "jurisdiction": j} for s, v, j in identifiers],
            "jurisdictions": sorted({j["value"] for m in group for j in m["jurisdictions"]}),
            "roles": sorted({r for m in group for r in m["roles"]}),
            "birth_dates": sorted({d for m in group for d in m.get("birth_dates", [])}),
            "documents": documents, "member_entity_ids": [m["entity_id"] for m in group],
            "match_basis": "SINGLE_DOCUMENT" if len(documents) == 1 else "IDENTIFIER" if shared else "PERSON_NAME" if any(m["entity_type"] == "person" for m in group) else "LEGAL_NAME",
            "group_of": None,
        })
    groups: dict[str, dict] = {}
    for e in entities:
        if e["entity_type"] != "group":
            continue
        parent_case = mapping.get(e["group_of"])
        case_id = stable_id("entity", "group", parent_case or e["entity_id"])
        record = groups.setdefault(case_id, {"entity_id": case_id, "entity_type": "group", "name": e["name"], "names": [], "identifiers": [], "jurisdictions": [], "roles": [], "documents": [], "member_entity_ids": [], "match_basis": "GROUP_OF_PARENT", "group_of": parent_case})
        record["names"] += [n for n in ({"name": n["name"], "kind": n["kind"]} for n in e["names"]) if n not in record["names"]]
        record["documents"] = sorted(set(record["documents"]) | {e["document_id"]})
        record["member_entity_ids"].append(e["entity_id"])
        mapping[e["entity_id"]] = case_id
    # Deduplicate names (same name and kind may come from several documents).
    for record in case_entities:
        record["names"] = [dict(t) for t in sorted({tuple(sorted(n.items())) for n in record["names"]})]
    return case_entities + list(groups.values()), mapping, issues
