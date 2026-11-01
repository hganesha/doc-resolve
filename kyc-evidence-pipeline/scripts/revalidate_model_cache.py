#!/usr/bin/env python3
"""Re-run the current source validator on an existing persisted run's candidate cache.

This is a retrospective diagnostic, not a new model run, canonical package,
independent human adjudication, or a substitute for graph reconciliation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json

from kyc_pipeline.config import load_contract, requirements_for
from kyc_pipeline.evidence import verify_candidates
from kyc_pipeline.store import EvidenceStore


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--profile", default="financial_annual_report")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    store = EvidenceStore(args.db)
    package = store.export_case(args.case_id)
    if len(package["documents"]) != 1 or len(package["runs"]) != 1:
        raise ValueError("This diagnostic requires one current document and run")
    doc, run = package["documents"][0], package["runs"][0]["run_id"]
    with store._connect() as conn:
        segments = [json.loads(x[0]) for x in conn.execute("SELECT payload_json FROM segments WHERE run_id=? ORDER BY ordinal", (run,))]
    chunks = store.current_chunks(args.case_id)
    units = store.get_extraction_units(run)
    logs = [dict(units[c["chunk_id"]]) for c in chunks if c["chunk_id"] in units]
    # A mixed candidate batch may have contained both accepted claims and a bad
    # citation. Recheck the entire cached list; do not turn provider failures into OK.
    for log in logs:
        if log.get("candidates") and log.get("error") in {"Candidate citation did not match the normalized source", "Candidate did not pass schema validation"}:
            log["status"] = "OK"
    contract = load_contract()
    requirements = requirements_for(contract, contract["profiles"][args.profile])
    evidence, facts, issues, checked = verify_candidates(doc, segments, chunks, logs, requirements, None, False)
    flags_by_fact = {f["fact_id"]: sorted({i["code"] for i in issues if f["fact_id"] in i["fact_ids"]}) for f in facts}
    rows = [{"fact_id": f["fact_id"], "subject": f["subject"], "predicate": f["predicate"], "object": f["object"], "value": f["value"], "unit": f["unit"], "current_validator_flags": flags_by_fact[f["fact_id"]]} for f in facts]
    result = {"provenance": "Retrospective local validation of cached model candidates; no new model calls, graph run, human review or canonical mutation", "original_run_id": run, "candidate_claims_after_revalidation": len(facts), "source_span_count": len(evidence), "issue_counts": dict(Counter(i["code"] for i in issues)), "claims_with_one_or_more_validator_flags": sum(bool(x["current_validator_flags"]) for x in rows), "claims": rows, "disclaimer": "Validator warnings are heuristics; a warning is not proof of error and no warning is not proof of support."}
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "claims"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
