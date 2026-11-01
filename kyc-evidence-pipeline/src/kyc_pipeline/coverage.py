from __future__ import annotations

from typing import Any

# Failures of extraction or of a claim in this category. Parse gaps are tracked
# separately: they limit what absence can mean but do not void what was extracted.
FAILURE_CODES = {"EXTRACTION_FAILED", "CANDIDATE_CITATION_REJECTED", "CANDIDATE_SCHEMA_REJECTED", "FACT_SCHEMA_VIOLATION", "CLAIM_LITERAL_SUPPORT_INCOMPLETE", "SUBJECT_SCOPE_NARROWED", "OBJECT_SCOPE_AMBIGUOUS", "QUANTITATIVE_QUALIFIER_DROPPED", "MATERIAL_SEMANTIC_SUPPORT_UNCERTAIN", "MATERIAL_SEMANTIC_VERIFICATION_FAILED"}


def _note(status: str, extraction_method: str, parse_gaps: list[dict], pages: list[int]) -> str:
    if parse_gaps and status in {"FOUND", "NOT_REVIEWED", "CONFLICTED"}:
        where = f" on pages {', '.join(map(str, pages))}" if pages else ""
        return f"Parts of the source{where} were not fully parsed; this category's evidence there is unreviewed and absence cannot be assessed."
    if status == "NOT_FOUND_AFTER_REVIEW":
        return "Not found after automated review is not proof of absence."
    if extraction_method == "rules":
        return "Rules are illustrative and do not provide exhaustive semantic review."
    return "Facts remain provisional until appropriately verified."


def compute_coverage(requirements: list[dict], chunks: list[dict], extraction_log: list[dict], facts: list[dict], issues: list[dict], extraction_method: str) -> list[dict]:
    total = len(chunks)
    processed = sum(log["status"] == "OK" for log in extraction_log)
    failed = sum(log["status"] == "ERROR" for log in extraction_log)
    skipped = sum(log["status"] == "NOT_REVIEWED" for log in extraction_log)
    open_issues = [i for i in issues if i["status"] == "OPEN"]
    records = []
    for req in requirements:
        category = req["category"]
        category_facts = [f for f in facts if f["category"] == category]
        relevant = [i for i in open_issues if "*" in i["categories"] or category in i["categories"]]
        failures = [i for i in relevant if i["code"] in FAILURE_CODES]
        parse_gaps = [i for i in relevant if i["code"] == "PARSE_OR_STRUCTURE_INCOMPLETE"]
        gap_pages = sorted({p for i in parse_gaps for p in i.get("details", {}).get("pages", [])})
        conflicts = [i for i in relevant if i["code"] in {"POSSIBLE_OWNERSHIP_CONFLICT", "POSSIBLE_ATTRIBUTE_CONFLICT"}]
        if not req["applicable"]:
            status = "NOT_APPLICABLE"
        elif failed or failures or (parse_gaps and total == 0):
            status = "EXTRACTION_FAILED"
        elif conflicts:
            status = "CONFLICTED"
        elif category_facts:
            status = "FOUND"
        elif skipped:
            status = "NOT_REVIEWED"
        elif extraction_method == "openai" and total > 0 and processed == total and not parse_gaps:
            status = "NOT_FOUND_AFTER_REVIEW"
        else:
            status = "NOT_REVIEWED"
        records.append({
            "category": category, "description": req["description"], "material": req["material"],
            "status": status, "applicability_reason": req.get("reason"),
            "fact_count": len(category_facts), "chunks_total": total, "chunks_processed": processed,
            "chunks_failed": failed, "chunks_not_reviewed": skipped, "open_issue_ids": [i["issue_id"] for i in relevant],
            "review_method": ("partial_automated_model_pass" if skipped else "automated_model_pass") if extraction_method == "openai" else "limited_pattern_rules",
            "all_parsed_chunks_processed": bool(total and processed == total and failed == 0 and skipped == 0),
            "source_fully_parsed": not parse_gaps,
            "parse_gap_issue_ids": [i["issue_id"] for i in parse_gaps],
            "parse_gap_pages": gap_pages,
            "completeness_asserted": False,
            "note": _note(status, extraction_method, parse_gaps, gap_pages),
        })
    return records


def complete_section_inventory(sections: list[dict], chunks: list[dict], logs: list[dict]) -> list[dict]:
    by_chunk = {log["chunk_id"]: log for log in logs}
    result = []
    for section in sections:
        ids = [c["chunk_id"] for c in chunks if c["section_id"] == section["section_id"]]
        result.append({**section, "chunks_processed": sum(by_chunk.get(i, {}).get("status") == "OK" for i in ids), "chunks_failed": sum(by_chunk.get(i, {}).get("status") == "ERROR" for i in ids), "chunks_not_reviewed": sum(by_chunk.get(i, {}).get("status") == "NOT_REVIEWED" for i in ids)})
    return result
