from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any
from pydantic import ValidationError

from .extraction import OpenAICompatible, ProviderError
from .models import CandidateFact
from .schemas import check_fact, coerce_value
from .util import date_present_in_text, identity_key, json_text, stable_id


def make_issue(code: str, message: str, *, document_id: str, categories: list[str], severity: str = "high", fact_ids: list[str] | None = None, details: dict | None = None, allowed_actions: list[str] | None = None) -> dict:
    fact_ids = sorted(set(fact_ids or []))
    details = details or {}
    issue_id = stable_id("issue", code, document_id, fact_ids, details)
    return {"issue_id": issue_id, "code": code, "message": message, "document_id": document_id, "categories": categories, "severity": severity, "fact_ids": fact_ids, "details": details, "status": "OPEN", "allowed_actions": allowed_actions or ["defer"]}


def _evidence_record(document: dict, seg: dict, start: int, end: int) -> dict:
    text = seg["text"]
    assert 0 <= start < end <= len(text)
    return {
        "evidence_id": stable_id("ev", document["document_id"], seg["segment_id"], start, end),
        "document_id": document["document_id"], "document_sha256": document["sha256"],
        "segment_id": seg["segment_id"], "normalized_segment_sha256": seg["text_sha256"],
        "quote": text[start:end], "start": start, "end": end,
        "offset_basis": "Unicode code-point offsets in the stored normalized segment, not raw file bytes",
        "locator": seg["locator"], "page": seg["page"], "bbox": seg["bbox"],
        "section_path": seg["section_path"], "representation": seg["representation"],
    }


def _contains_phrase(haystack: str, needle: str) -> bool:
    n = identity_key(needle)
    return bool(n and f" {n} " in f" {identity_key(haystack)} ")


def _contains_name_parts(haystack: str, name: str) -> bool:
    words = set(identity_key(haystack).split())
    parts = identity_key(name).split()
    return len(parts) >= 2 and all(p in words for p in parts)


def _numeric_in(value: float, text: str) -> bool:
    if not math.isfinite(value):
        return False
    for token in re.findall(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w.])", text):
        try:
            if math.isclose(float(token.replace(",", "")), value, rel_tol=1e-10, abs_tol=1e-10):
                return True
        except ValueError:
            continue
    return False


def verify_candidates(document: dict, segments: list[dict], chunks: list[dict], extraction_log: list[dict], requirements: list[dict], provider: OpenAICompatible | None, verify_material: bool) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Validate exact source pointers and flag, rather than certify, candidate claims."""
    segment_map = {s["segment_id"]: s for s in segments}
    chunk_map = {c["chunk_id"]: c for c in chunks}
    material = {r["category"]: r["material"] for r in requirements}
    allowed = {r["category"] for r in requirements if r["applicable"]}
    typed = {r["category"]: r.get("predicates", {}) for r in requirements}
    evidences: dict[str, dict] = {}
    facts: dict[str, dict] = {}
    issues: list[dict] = []
    updated_log: list[dict] = []
    rejected: dict[str, list[str]] = defaultdict(list)
    off_schema: dict[str, list[dict]] = defaultdict(list)
    for original in extraction_log:
        log = dict(original)
        candidates = log.pop("candidates", [])
        chunk = chunk_map[log["chunk_id"]]
        seg = segment_map[chunk["segment_id"]]
        if log["status"] != "OK":
            updated_log.append(log)
            continue
        for raw in candidates:
            try:
                c = CandidateFact.model_validate(raw)
                if c.category not in allowed:
                    raise ValueError("category outside coverage contract")
                if isinstance(c.value, float) and not math.isfinite(c.value):
                    raise ValueError("non-finite number")
            except (ValidationError, ValueError):
                log["status"] = "ERROR"
                log["error"] = "Candidate did not pass schema validation"
                continue
            schema = typed.get(c.category, {})
            spec = schema.get(c.predicate)
            if schema and spec is None:
                # A typed category accepts only its declared predicates; anything
                # else would be invisible to the checks that reason about them.
                log.setdefault("rejected_candidates", []).append({"category": c.category, "predicate": c.predicate, "reason": "Predicate is not declared in the category's typed extraction schema"})
                off_schema[c.category].append({"chunk_id": log["chunk_id"], "predicate": c.predicate})
                continue
            c.value = coerce_value(spec, c.value)
            quote_offset = chunk["text"].find(c.source_quote)
            supporting_offsets = [seg["text"].find(q) for q in c.supporting_quotes]
            allowed_context = {x["segment_id"] for x in chunk.get("context_refs", []) + chunk.get("section_context", [])}
            ref_segments = [segment_map.get(ref.segment_id) for ref in c.supporting_refs]
            bad_ref = any(ref.segment_id not in allowed_context or ref_seg is None or ref_seg["text"].find(ref.quote) < 0 for ref, ref_seg in zip(c.supporting_refs, ref_segments))
            if quote_offset < 0 or any(x < 0 for x in supporting_offsets) or bad_ref:
                # The chunk was still processed: reject only this claim, and taint
                # only its category, rather than failing every category in the run.
                log.setdefault("rejected_candidates", []).append({"category": c.category, "predicate": c.predicate, "reason": "Candidate citation did not match the normalized source"})
                rejected[c.category].append(log["chunk_id"])
                continue
            start = chunk["start"] + quote_offset
            spans = [(start, start + len(c.source_quote))] + [(x, x + len(q)) for x, q in zip(supporting_offsets, c.supporting_quotes)]
            evs = [_evidence_record(document, seg, a, b) for a, b in spans]
            evidence_roles = [{"evidence_id": evs[0]["evidence_id"], "role": "primary_quote"}]
            evidence_roles.extend({"evidence_id": ev["evidence_id"], "role": "supporting_quote"} for ev in evs[1:])
            for ref, ref_seg in zip(c.supporting_refs, ref_segments):
                offset = ref_seg["text"].find(ref.quote)
                ref_ev = _evidence_record(document, ref_seg, offset, offset + len(ref.quote))
                if all(existing["evidence_id"] != ref_ev["evidence_id"] for existing in evs):
                    evs.append(ref_ev)
                    evidence_roles.append({"evidence_id": ref_ev["evidence_id"], "role": "supporting_reference"})
            # A visible footnote marker is an explicit structural link, not an inference
            # that the qualification has been semantically incorporated into the claim.
            for note_ref in chunk.get("footnote_context", []):
                note_seg = segment_map.get(note_ref["segment_id"])
                if note_seg and note_seg["text"]:
                    note_ev = _evidence_record(document, note_seg, 0, len(note_seg["text"]))
                    if all(existing["evidence_id"] != note_ev["evidence_id"] for existing in evs):
                        evs.append(note_ev)
                        evidence_roles.append({"evidence_id": note_ev["evidence_id"], "role": "referenced_footnote"})
            for ev in evs:
                evidences[ev["evidence_id"]] = ev
            fields = c.model_dump(exclude={"source_quote", "supporting_quotes", "supporting_refs"})
            violations = check_fact(spec, fields) if spec else []
            fact_id = stable_id("fact", document["document_id"], fields, [e["evidence_id"] for e in evs])
            if fact_id in facts:
                continue
            combined_quote = "\n".join(e["quote"] for e in evs)
            checks: dict[str, bool | None] = {
                "exact_quote_match": True,
                # An identity document prints the surname and given names on separate lines.
                "subject_literal_present": (_contains_phrase(combined_quote, c.subject) or bool(spec and spec.get("subject_name_parts") and _contains_name_parts(combined_quote, c.subject))) if not c.subject.startswith("UNRESOLVED_") else False,
                "object_literal_present": _contains_phrase(combined_quote, c.object) if c.object else None,
                # A magnitude (a debit of 250) may be printed signed: -250.00.
                "value_literal_present": (_numeric_in(c.value, combined_quote) or bool(spec and spec.get("magnitude") and _numeric_in(-c.value, combined_quote))) if isinstance(c.value, (int, float)) else (_contains_phrase(combined_quote, c.value) if isinstance(c.value, str) else None),
                "date_literal_present": date_present_in_text(c.effective_date, combined_quote) if c.effective_date else None,
            }
            subject_key = identity_key(c.subject)
            ambiguous_entity = bool(re.search(r"\b(?:or|and/or|and\s+its\s+subsidiaries)\b", c.subject, re.I) or subject_key in {"company", "the company", "we", "our", "the registrant", "registrant", "unresolved registry subject", "unresolved reporting entity", "unresolved registrant"})
            fact = {
                "fact_id": fact_id, "document_id": document["document_id"], **fields,
                "evidence_ids": [e["evidence_id"] for e in evs], "evidence_roles": evidence_roles,
                "extractor": log["method"],
                "verification_status": "PENDING_HUMAN",
                "source_support": {"quote_match": "EXACT_NORMALIZED_TEXT_MATCH", "literal_checks": checks, "semantic_check": "NOT_REQUESTED"},
                "entity_resolution": "UNRESOLVED" if ambiguous_entity else "TEXT_MATCH_ONLY",
                "schema_check": {"status": "UNTYPED" if spec is None else "VIOLATIONS" if violations else "CONFORMS", "violations": violations},
            }
            if ambiguous_entity:
                issues.append(make_issue("ENTITY_IDENTITY_UNRESOLVED", "The source uses a generic, pronoun, or disjunctive subject; its precise legal entity is not established.", document_id=document["document_id"], categories=[c.category], severity="high" if material.get(c.category) else "medium", fact_ids=[fact_id], details={"subject": c.subject}))
            # Literal overlap alone does not verify the relationship. These narrow
            # guards identify some common failures, not semantic proof.
            severity = "high" if material.get(c.category) else "medium"
            if spec and spec.get("evidence_terms") and not re.search(spec["evidence_terms"], combined_quote, re.I):
                issues.append(make_issue("PREDICATE_NOT_LITERAL_IN_EVIDENCE", f"The cited spans lack the wording {c.predicate} needs; the claim may misread the source (for example a tax payment as revenue, or control as voting rights).", document_id=document["document_id"], categories=[c.category], severity=severity, fact_ids=[fact_id], details={"predicate": c.predicate}))
            if violations:
                issues.append(make_issue("FACT_SCHEMA_VIOLATION", f"The claim does not fit the typed schema for {c.predicate}; review it before relying on it.", document_id=document["document_id"], categories=[c.category], severity=severity, fact_ids=[fact_id], details={"predicate": c.predicate, "violations": violations}))
            false_checks = [k for k, v in checks.items() if v is False]
            # Exact words alone can still misstate the source: flag narrowed
            # disjunctions, dropped quantitative qualifiers and joined entities.
            if c.subject and not re.search(r"\bor\b", c.subject, re.I) and re.search(re.escape(c.subject) + r"\s+or\s+(?:a|its|any)\s+subsidiar", combined_quote, re.I):
                issues.append(make_issue("SUBJECT_SCOPE_NARROWED", "The claim names one subject while the cited source says that subject OR a subsidiary; do not resolve the owner without further evidence.", document_id=document["document_id"], categories=[c.category], severity="high" if material.get(c.category) else "medium", fact_ids=[fact_id]))
            if c.object and re.search(r"\band\b", c.object, re.I) and c.predicate == "owns_equity_in":
                issues.append(make_issue("OBJECT_SCOPE_AMBIGUOUS", "The extracted object joins names with 'and'; review which entity the stated ownership percentage actually qualifies.", document_id=document["document_id"], categories=[c.category], severity="high" if material.get(c.category) else "medium", fact_ids=[fact_id]))
            number = None
            if isinstance(c.value, (int, float)):
                number = float(c.value)
            elif isinstance(c.value, str) and re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", c.value.strip()):
                number = float(c.value.replace(",", ""))
            numeric = spec.get("value") in {"percent", "amount", "count"} if spec else c.predicate in {"owns_equity_in", "reported_revenue"}
            if number is not None and not c.qualifier and numeric:
                literal = re.escape(str(int(number)) if number.is_integer() else str(number))
                if re.search(r"\b(?:approximately|about|roughly|generally|at\s+least|up\s+to)\b.{0,25}(?<!\d)" + literal + r"(?!\d)", combined_quote, re.I):
                    issues.append(make_issue("QUANTITATIVE_QUALIFIER_DROPPED", "The cited source qualifies the numerical claim but the extracted fact does not preserve the qualification.", document_id=document["document_id"], categories=[c.category], severity="high" if material.get(c.category) else "medium", fact_ids=[fact_id]))
            if false_checks:
                issues.append(make_issue("CLAIM_LITERAL_SUPPORT_INCOMPLETE", "A claim contains a subject, object, value, or date not literally present in its cited quotes; context or human review is required.", document_id=document["document_id"], categories=[c.category], severity="high" if material.get(c.category) else "medium", fact_ids=[fact_id], details={"failed_checks": false_checks}))
            if verify_material and material.get(c.category):
                if provider is None:
                    raise ValueError("Semantic verification requires the OpenAI-compatible extractor")
                try:
                    result = provider.verify_support(fact, evs)
                    fact["source_support"]["semantic_check"] = result["support"]
                    fact["source_support"]["semantic_explanation"] = result["explanation"]
                    if result["support"] != "SUPPORTED":
                        issues.append(make_issue("MATERIAL_SEMANTIC_SUPPORT_UNCERTAIN", "The model source-support check did not establish support; this does not determine external truth.", document_id=document["document_id"], categories=[c.category], fact_ids=[fact_id], details={"verdict": result["support"]}))
                except ProviderError:
                    fact["source_support"]["semantic_check"] = "VERIFICATION_FAILED"
                    issues.append(make_issue("MATERIAL_SEMANTIC_VERIFICATION_FAILED", "Material source-support verification failed and requires review.", document_id=document["document_id"], categories=[c.category], fact_ids=[fact_id]))
            facts[fact_id] = fact
        updated_log.append(log)
    for category, chunk_ids in sorted(rejected.items()):
        issues.append(make_issue("CANDIDATE_CITATION_REJECTED", "One or more candidate claims cited text that is not in the source and were discarded; a real claim may have been lost. Retry or review these chunks.", document_id=document["document_id"], categories=[category], severity="high" if material.get(category) else "medium", details={"rejected_count": len(chunk_ids), "chunk_ids": sorted(set(chunk_ids))}))
    for category, items in sorted(off_schema.items()):
        issues.append(make_issue("CANDIDATE_SCHEMA_REJECTED", "One or more candidate claims used a predicate outside this category's typed schema and were discarded; a real claim may have been lost. Retry, or extend the schema if the predicate is needed.", document_id=document["document_id"], categories=[category], severity="high" if material.get(category) else "medium", details={"rejected_count": len(items), "predicates": sorted({x["predicate"] for x in items}), "chunk_ids": sorted({x["chunk_id"] for x in items})}))
    return list(evidences.values()), list(facts.values()), issues, updated_log
