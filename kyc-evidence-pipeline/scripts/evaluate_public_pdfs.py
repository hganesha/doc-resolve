#!/usr/bin/env python3
"""Re-run the real-PDF offline benchmark. Requires the packaged public reports.

This intentionally defers all graph interrupts without representing that action
as human review. It does not use a model or claim exhaustive semantic coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kyc_pipeline.retrieval import search_chunks
from kyc_pipeline.local_embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL, LocalEmbeddingProvider
from kyc_pipeline.store import EvidenceStore

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "berkshire": "berkshire_2024_annual_report.pdf",
    "disney": "disney_2024_annual_report.pdf",
}
COMPARE_FIELDS = ("category", "predicate", "object", "value", "unit", "effective_date", "qualifier")


def run(*args: str) -> dict:
    p = subprocess.run([sys.executable, "-m", "kyc_pipeline", *map(str, args)], cwd=ROOT, text=True, capture_output=True, timeout=600)
    if p.returncode:
        raise RuntimeError(f"pipeline returned {p.returncode}: {p.stderr[-1000:]}")
    return json.loads(p.stdout)


def normalized(s: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(s).casefold(), re.UNICODE))


def evaluate_gold(package: dict, gold: list[dict]) -> list[dict]:
    evidence = {e["evidence_id"]: e for e in package["evidence"]}
    result = []
    for target in gold:
        exact, partial = [], []
        for f in package["facts"]:
            primary = evidence.get(f["evidence_ids"][0], {})
            if primary.get("page") != target["page"]:
                continue
            if any(f.get(field) != target[field] for field in COMPARE_FIELDS if field in target):
                continue
            if normalized(f["subject"]) == normalized(target["subject"]):
                exact.append(f["fact_id"])
            else:
                partial.append(f["fact_id"])
        result.append({"id": target["id"], "page": target["page"], "status": "MATCHED_SOURCE_CLAIM" if exact else "PARTIAL_SUBJECT_UNRESOLVED_OR_DIFFERENT" if partial else "MISSED", "matching_fact_ids": exact, "partial_fact_ids": partial, "caveat": target["explanation"]})
    return result


def retrieval_eval(chunks: list[dict], gold: list[dict], provider=None) -> list[dict]:
    output = []
    for target in gold:
        result = search_chunks(chunks, target["query"], top_k=5, provider=provider)
        strict_child_rank = strict_parent_rank = relevant_rank = None
        allowed = [{"page": target["page"], "terms": target["passage_terms"]}] + target.get("relevant_alternatives", [])
        for i, hit in enumerate(result["results"], 1):
            child = hit["child"]
            units = hit.get("parent_context_units", [])
            def matches(unit, passage):
                body = normalized(unit["text"])
                return unit.get("page") == passage["page"] and all(normalized(term) in body for term in passage["terms"])
            if strict_child_rank is None and matches(child, allowed[0]):
                strict_child_rank = i
            if strict_parent_rank is None and any(matches(unit, allowed[0]) for unit in units):
                strict_parent_rank = i
            if relevant_rank is None and any(matches(unit, passage) for unit in units for passage in allowed):
                relevant_rank = i
        output.append({"id": target["id"], "query": target["query"], "annotated_page": target["page"], "annotated_child_rank_at_5": strict_child_rank, "annotated_parent_rank_at_5": strict_parent_rank, "relevant_parent_rank_at_5_including_labeled_alternatives": relevant_rank, "returned_pages": [r["child"]["page"] for r in result["results"]]})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="real_evaluation/reproducible_run")
    parser.add_argument("--embeddings", choices=["lexical", "local"], default="lexical", help="Local dense retrieval requires the local-embeddings extra and model weights")
    parser.add_argument("--embedding-model", default=DEFAULT_LOCAL_EMBEDDING_MODEL)
    args = parser.parse_args()
    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gold = json.loads((ROOT / "real_evaluation/gold_sample.json").read_text())["facts"]
    sources = {x["filename"]: x for x in json.loads((ROOT / "examples/public_reports/sources.json").read_text())}
    all_claims, all_retrieval, all_counts, review_rows = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="kyc-real-pdf-") as temp:
        for name, filename in REPORTS.items():
            source = ROOT / "examples/public_reports" / filename
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != sources[filename]["sha256"]:
                raise RuntimeError(f"Missing or altered source: {source}; check sources.json before rerunning")
            db, checkpoint = Path(temp)/f"{name}-evidence.sqlite", Path(temp)/f"{name}-checkpoint.sqlite"
            started = time.monotonic()
            result = run("ingest", source, "--case-id", name, "--profile", "financial_annual_report", "--embeddings", args.embeddings, "--embedding-model", args.embedding_model, "--db", db, "--checkpoint-db", checkpoint)
            if result["status"] == "PAUSED_FOR_REVIEW":
                decision = {"reviewer_id": "AUTOMATED_BENCHMARK_DEFER_ONLY_NOT_HUMAN", "decisions": [{"issue_id": issue["issue_id"], "action": "defer", "reason": "Automated benchmark deferral. No analyst reviewed or resolved this issue."} for issue in result["review_request"]["issues"]]}
                decision_file = Path(temp) / f"{name}-decision.json"
                decision_file.write_text(json.dumps(decision), encoding="utf-8")
                run("resume", "--run-id", result["run_id"], "--decision", decision_file, "--db", db, "--checkpoint-db", checkpoint)
            package = run("report", "--case-id", name, "--db", db)
            (out / f"{name}_evidence.json").write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            store = EvidenceStore(db)
            chunks = store.current_chunks(name)
            targets = [x for x in gold if x["document"] == name]
            claims = evaluate_gold(package, targets)
            retrieval_provider = LocalEmbeddingProvider(args.embedding_model) if args.embeddings == "local" else None
            retrieval = retrieval_eval(chunks, targets, retrieval_provider)
            if retrieval_provider is not None:
                del retrieval_provider
                import gc
                gc.collect()
            all_claims.extend(claims); all_retrieval.extend(retrieval)
            evidence = {e["evidence_id"]: e for e in package["evidence"]}
            quote_ok = 0
            for e in evidence.values():
                segment = store.get_segment(name, e["segment_id"])
                if segment and segment["text"][e["start"]:e["end"]] == e["quote"] and hashlib.sha256(segment["text"].encode()).hexdigest() == e["normalized_segment_sha256"]:
                    quote_ok += 1
            for f in package["facts"]:
                primary = evidence.get(f["evidence_ids"][0], {})
                related = [i["code"] for i in package["issues"] if f["fact_id"] in i["fact_ids"]]
                supporting = []
                for role in f.get("evidence_roles", []):
                    span = evidence.get(role["evidence_id"], {})
                    supporting.append(f"{role['role']} / physical PDF p.{span.get('page')} / {span.get('representation')}: {span.get('quote')}")
                review_rows.append({"document": name, "source_url": sources[filename]["source_url"], "source_sha256": sources[filename]["sha256"], "fact_id": f["fact_id"], "page": primary.get("page"), "subject": f["subject"], "predicate": f["predicate"], "object": f.get("object"), "value": f.get("value"), "unit": f.get("unit"), "effective_date": f.get("effective_date"), "qualifier": f.get("qualifier"), "quote": primary.get("quote"), "quote_representation": primary.get("representation"), "all_evidence_quotes": "\n".join(supporting), "machine_flags": "; ".join(related), "source_support_label": "", "reviewer": "", "notes": ""})
            c = package["coverage"][0] if package["coverage"] else {}
            all_counts.append({"document": name, "pdf_pages": sources[filename]["pdf_pages"], "chunks_total": len(chunks), "chunks_processed_by_rules": c.get("chunks_processed"), "chunks_failed": c.get("chunks_failed"), "chunks_not_reviewed": c.get("chunks_not_reviewed"), "facts": len(package["facts"]), "evidence_spans": len(evidence), "normalized_quote_offsets_valid": quote_ok, "status": package["status"], "seconds": round(time.monotonic() - started, 1)})
            print(f"{name}: {len(chunks)} chunks, {len(package['facts'])} provisional facts; case {package['status']}", file=sys.stderr)
    with (out / "fact_review_template.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0]) if review_rows else ["fact_id", "source_support_label", "reviewer", "notes"])
        writer.writeheader(); writer.writerows(review_rows)
    summary = {"dataset": "two public, 2024 primary-source annual reports; 8 source-checked examples, not exhaustive or independently human-adjudicated", "mode": "offline_rules_no_extraction_model", "retrieval_embeddings": args.embeddings, "embedding_model": args.embedding_model if args.embeddings == "local" else None, "scope_note": "All parsed chunks were processed by narrow rules; this is NOT exhaustive semantic review.", "documents": all_counts, "fact_targets": all_claims, "annotated_passage_retrieval": all_retrieval, "matched_source_claims": sum(c["status"] == "MATCHED_SOURCE_CLAIM" for c in all_claims), "partial_subject_claims": sum(c["status"] == "PARTIAL_SUBJECT_UNRESOLVED_OR_DIFFERENT" for c in all_claims), "target_count": len(all_claims), "annotated_child_hits_at_5": sum(x["annotated_child_rank_at_5"] is not None for x in all_retrieval), "annotated_parent_hits_at_5": sum(x["annotated_parent_rank_at_5"] is not None for x in all_retrieval), "relevant_parent_hits_at_5_including_labeled_alternatives": sum(x["relevant_parent_rank_at_5_including_labeled_alternatives"] is not None for x in all_retrieval), "unsupported_claim_rate": None, "unsupported_rate_note": "Not measured: source-support labels must be independently adjudicated. The blank CSV template is provided; exact quote/offset matching is not a semantic accuracy metric.", "review_note": "All graph interrupts in this benchmark were automatically deferred, never resolved or represented as human review."}
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
