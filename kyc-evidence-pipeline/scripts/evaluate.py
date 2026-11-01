#!/usr/bin/env python3
"""Tiny, labeled, synthetic smoke evaluation; not a production recall estimate."""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("category", "subject", "predicate", "object", "value", "unit", "effective_date")
SCENARIOS = [
    ("annual", "examples/synthetic_annual_report.txt", "financial_annual_report", "examples/synthetic_annual_gold.json", "STALE_OWNERSHIP_AFTER_DISPOSAL"),
    ("conflict", "examples/synthetic_ownership_conflict.txt", "ownership_control", "examples/synthetic_conflict_gold.json", "POSSIBLE_OWNERSHIP_CONFLICT"),
]


def call(*args: str) -> dict:
    completed = subprocess.run([sys.executable, "-m", "kyc_pipeline", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)


def signature(item: dict) -> tuple:
    return tuple(item.get(k) for k in FIELDS)


def main() -> None:
    total_gold = total_predicted = total_matched = total_unsupported = review_hits = 0
    details = []
    with tempfile.TemporaryDirectory(prefix="kyc-eval-") as tmp:
        db, cp = str(Path(tmp) / "evidence.sqlite"), str(Path(tmp) / "checkpoints.sqlite")
        for name, doc, profile, gold_path, expected_issue in SCENARIOS:
            gold = json.loads((ROOT / gold_path).read_text())["claims"]
            start = call("ingest", doc, "--case-id", f"eval-{name}", "--profile", profile, "--db", db, "--checkpoint-db", cp)
            issue_codes = [i["code"] for i in start.get("review_request", {}).get("issues", [])]
            review_hits += int(expected_issue in issue_codes)
            if start["status"] == "PAUSED_FOR_REVIEW":
                decision = {"reviewer_id": "SYNTHETIC_EVALUATION_ONLY", "decisions": [{"issue_id": i["issue_id"], "action": "defer", "reason": "Synthetic evaluation: do not resolve the fictional issue."} for i in start["review_request"]["issues"]]}
                file = Path(tmp) / f"{name}-decision.json"
                file.write_text(json.dumps(decision))
                call("resume", "--run-id", start["run_id"], "--decision", str(file), "--db", db, "--checkpoint-db", cp)
            package = call("report", "--case-id", f"eval-{name}", "--db", db)
            predicted = package["facts"]
            gold_set = {signature(g) for g in gold}
            predicted_set = {signature(p) for p in predicted}
            matched = len(gold_set & predicted_set)
            unsupported = sum(signature(p) not in gold_set for p in predicted)
            total_gold += len(gold_set)
            total_predicted += len(predicted)
            total_matched += matched
            total_unsupported += unsupported
            table_links = None
            if name == "annual":
                table = next((f for f in predicted if f["predicate"] == "reported_revenue"), None)
                table_links = bool(table and {"primary_quote", "supporting_quote", "referenced_footnote"} <= {r["role"] for r in table["evidence_roles"]})
            details.append({"scenario": name, "gold_facts": len(gold_set), "predicted_facts": len(predicted), "matched_facts": matched, "expected_review_issue_detected": expected_issue in issue_codes, "table_header_and_footnote_linked": table_links})
    print(json.dumps({"dataset": "two tiny fictional documents with exhaustive labels for this demonstration", "expected_facts": total_gold, "predicted_facts": total_predicted, "matched_facts": total_matched, "fact_recall": total_matched / total_gold if total_gold else None, "unsupported_claim_rate_vs_synthetic_gold": total_unsupported / total_predicted if total_predicted else 0.0, "expected_review_issues_detected": f"{review_hits}/{len(SCENARIOS)}", "scenarios": details, "note": "Plumbing check only, not a performance estimate for real documents. Add independently annotated diverse sources, complex tables, OCR, entity ambiguity and temporal cases."}, indent=2))


if __name__ == "__main__":
    main()
