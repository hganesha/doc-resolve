from __future__ import annotations

import json
import os
from contextlib import contextmanager
import sqlite3
from pathlib import Path
from typing import Any

from .entities import resolve_case
from .ownership import build_ownership_graph, graph_settings
from .util import digest, json_text, stable_id, utc_now


class EvidenceStore:
    """Transactional local SQLite proof-of-concept store; NOT encrypted at rest."""

    def __init__(self, path: str | Path):
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, document_id TEXT, profile_id TEXT,
              policy_id TEXT, status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
              summary_json TEXT, error TEXT
            );
            CREATE INDEX IF NOT EXISTS runs_case ON runs(case_id, started_at);
            CREATE TABLE IF NOT EXISTS documents (
              case_id TEXT NOT NULL, document_id TEXT NOT NULL, current_run_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (case_id, document_id)
            );
            CREATE TABLE IF NOT EXISTS segments (
              run_id TEXT NOT NULL, segment_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              section_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload_json TEXT NOT NULL,
              PRIMARY KEY (run_id, segment_id)
            );
            CREATE TABLE IF NOT EXISTS chunks (
              run_id TEXT NOT NULL, chunk_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              section_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload_json TEXT NOT NULL,
              PRIMARY KEY (run_id, chunk_id)
            );
            -- Pre-commit vectors are keyed by run and invisible until canonical chunks are committed.
            -- Keep them outside LangGraph checkpoint state to bound memory on long PDFs.
            CREATE TABLE IF NOT EXISTS chunk_vectors (
              run_id TEXT NOT NULL, chunk_id TEXT NOT NULL, model TEXT NOT NULL, vector_json TEXT NOT NULL,
              PRIMARY KEY (run_id, chunk_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
              run_id TEXT NOT NULL, evidence_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS facts (
              run_id TEXT NOT NULL, fact_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              category TEXT NOT NULL, payload_json TEXT NOT NULL, PRIMARY KEY (run_id, fact_id)
            );
            CREATE TABLE IF NOT EXISTS entities (
              run_id TEXT NOT NULL, entity_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, entity_id)
            );
            CREATE TABLE IF NOT EXISTS findings (
              run_id TEXT NOT NULL, finding_id TEXT NOT NULL, case_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, finding_id)
            );
            CREATE TABLE IF NOT EXISTS issues (
              run_id TEXT NOT NULL, issue_id TEXT NOT NULL, case_id TEXT NOT NULL, status TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, issue_id)
            );
            CREATE TABLE IF NOT EXISTS coverage (
              run_id TEXT NOT NULL, category TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, category)
            );
            CREATE TABLE IF NOT EXISTS sections (
              run_id TEXT NOT NULL, section_id TEXT NOT NULL, case_id TEXT NOT NULL, document_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, section_id)
            );
            CREATE TABLE IF NOT EXISTS reviews (
              run_id TEXT NOT NULL, issue_id TEXT NOT NULL, case_id TEXT NOT NULL,
              payload_json TEXT NOT NULL, PRIMARY KEY (run_id, issue_id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
              event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, case_id TEXT NOT NULL,
              happened_at TEXT NOT NULL, action TEXT NOT NULL, details_json TEXT NOT NULL
            );
            -- Raw layout-service output, so a retry of identical bytes reuses the same
            -- segments (and extraction cache) instead of a fresh, possibly different, analysis.
            CREATE TABLE IF NOT EXISTS layout_results (
              document_sha256 TEXT NOT NULL, fingerprint TEXT NOT NULL, payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL, PRIMARY KEY (document_sha256, fingerprint)
            );
            -- Public-register lookups (GLEIF), reused for a while so retries see the same record.
            CREATE TABLE IF NOT EXISTS registry_records (
              source TEXT NOT NULL, identifier TEXT NOT NULL, payload_json TEXT NOT NULL,
              fetched_at TEXT NOT NULL, PRIMARY KEY (source, identifier)
            );
            CREATE TABLE IF NOT EXISTS extraction_units (
              run_id TEXT NOT NULL, chunk_id TEXT NOT NULL, status TEXT NOT NULL,
              payload_json TEXT NOT NULL, updated_at TEXT NOT NULL,
              PRIMARY KEY (run_id, chunk_id)
            );
            """)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def put_chunk_vectors(self, run_id: str, model: str, chunks: list[dict], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count does not match source chunks")
        with self._connect() as conn:
            conn.executemany("INSERT INTO chunk_vectors VALUES(?,?,?,?) ON CONFLICT(run_id,chunk_id) DO UPDATE SET model=excluded.model,vector_json=excluded.vector_json", [(run_id, chunk["chunk_id"], model, json_text(vector)) for chunk, vector in zip(chunks, vectors)])

    def get_layout_result(self, document_sha256: str, fingerprint: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM layout_results WHERE document_sha256=? AND fingerprint=?", (document_sha256, fingerprint)).fetchone()
        return json.loads(row[0]) if row else None

    def put_layout_result(self, document_sha256: str, fingerprint: str, result: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO layout_results VALUES(?,?,?,?)", (document_sha256, fingerprint, json_text(result), utc_now()))

    def get_registry_record(self, source: str, identifier: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json, fetched_at FROM registry_records WHERE source=? AND identifier=?", (source, identifier)).fetchone()
        return {"record": json.loads(row[0]), "fetched_at": row[1]} if row else None

    def put_registry_record(self, source: str, identifier: str, record: dict | None, fetched_at: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO registry_records VALUES(?,?,?,?)", (source, identifier, json_text(record), fetched_at))

    def put_extraction_unit(self, run_id: str, unit: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO extraction_units VALUES(?,?,?,?,?) ON CONFLICT(run_id,chunk_id) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json,updated_at=excluded.updated_at", (run_id, unit["chunk_id"], unit["status"], json_text(unit), utc_now()))

    def get_extraction_units(self, run_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id,payload_json FROM extraction_units WHERE run_id=?", (run_id,)).fetchall()
        return {r["chunk_id"]: json.loads(r["payload_json"]) for r in rows}

    def mark_unit_rejections(self, run_id: str, chunk_id: str, rejected: list[dict]) -> None:
        """Record discarded candidates so a retry re-extracts the chunk instead of reusing it."""
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM extraction_units WHERE run_id=? AND chunk_id=?", (run_id, chunk_id)).fetchone()
            if row:
                unit = json.loads(row[0])
                unit["rejected_candidates"] = rejected
                conn.execute("UPDATE extraction_units SET payload_json=?,updated_at=? WHERE run_id=? AND chunk_id=?", (json_text(unit), utc_now(), run_id, chunk_id))

    def mark_unit_error(self, run_id: str, chunk_id: str, error: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM extraction_units WHERE run_id=? AND chunk_id=?", (run_id, chunk_id)).fetchone()
            if row:
                unit = json.loads(row[0])
                unit["status"] = "ERROR"
                unit["error"] = error
                conn.execute("UPDATE extraction_units SET status=?,payload_json=?,updated_at=? WHERE run_id=? AND chunk_id=?", ("ERROR", json_text(unit), utc_now(), run_id, chunk_id))

    def _audit(self, conn: sqlite3.Connection, run_id: str, case_id: str, action: str, details: dict, *, event_id: str | None = None) -> None:
        happened = utc_now()
        eid = event_id or stable_id("audit", run_id, action, happened)
        conn.execute("INSERT OR IGNORE INTO audit_events VALUES (?,?,?,?,?,?)", (eid, run_id, case_id, happened, action, json_text(details)))

    def start_run(self, run_id: str, case_id: str, started_at: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO runs(run_id,case_id,status,started_at) VALUES(?,?,?,?)", (run_id, case_id, "PROCESSING", started_at))
            self._audit(conn, run_id, case_id, "RUN_STARTED", {})

    def set_run_status(self, run_id: str, status: str, *, error: str | None = None) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT case_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Run not found")
            conn.execute("UPDATE runs SET status=?,error=? WHERE run_id=?", (status, error, run_id))
            self._audit(conn, run_id, row["case_id"], "RUN_" + status, {"error": error} if error else {})

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def _case_revision(conn: sqlite3.Connection, case_id: str) -> str:
        rows = conn.execute("SELECT document_id,current_run_id FROM documents WHERE case_id=? ORDER BY document_id", (case_id,)).fetchall()
        return digest(json_text([[r["document_id"], r["current_run_id"]] for r in rows]))

    def get_case_snapshot(self, case_id: str, exclude_document_id: str | None = None) -> tuple[str, list[dict]]:
        # Revision and claims must come from the same SQLite read snapshot.
        with self._connect() as conn:
            conn.execute("BEGIN")
            revision = self._case_revision(conn, case_id)
            rows = conn.execute("""SELECT f.payload_json FROM facts f JOIN documents d
                ON d.current_run_id=f.run_id AND d.case_id=f.case_id AND d.document_id=f.document_id
                WHERE f.case_id=? AND (? IS NULL OR f.document_id<>?)""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
            facts = [json.loads(r[0]) for r in rows]
        return revision, facts

    def get_case_entities(self, case_id: str, exclude_document_id: str | None = None) -> list[dict]:
        """Document-level entity records from each document's current run."""
        with self._connect() as conn:
            rows = conn.execute("""SELECT e.payload_json FROM entities e JOIN documents d
                ON d.current_run_id=e.run_id AND d.case_id=e.case_id AND d.document_id=e.document_id
                WHERE e.case_id=? AND (? IS NULL OR e.document_id<>?) ORDER BY e.document_id, e.entity_id""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def get_case_registry_checks(self, case_id: str, exclude_document_id: str | None = None) -> list[dict]:
        """Register checks recorded by each document's current run."""
        with self._connect() as conn:
            rows = conn.execute("""SELECT r.summary_json FROM runs r JOIN documents d ON d.current_run_id=r.run_id AND d.case_id=r.case_id
                WHERE r.case_id=? AND (? IS NULL OR d.document_id<>?)""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
        return [c for (summary,) in rows if summary for c in json.loads(summary).get("registry_checks", [])]

    def get_current_facts(self, case_id: str, exclude_document_id: str | None = None) -> list[dict]:
        return self.get_case_snapshot(case_id, exclude_document_id)[1]

    def persist_run(self, state: dict[str, Any]) -> None:
        run_id, case_id = state["run_id"], state["case_id"]
        doc = state["document"]
        document_id = doc["document_id"]
        issues = state.get("issues", [])
        coverage = state.get("coverage", [])
        requires_review = any(i["status"] == "OPEN" for i in issues) or any(c["material"] and (c["status"] in {"NOT_REVIEWED", "NOT_FOUND_AFTER_REVIEW", "EXTRACTION_FAILED", "CONFLICTED"} or c.get("chunks_not_reviewed", 0) > 0 or not c.get("source_fully_parsed", True)) for c in coverage)
        status = "REVIEW_REQUIRED" if requires_review else "PROVISIONAL"
        summary = {
            "run_id": run_id, "case_id": case_id, "document_id": document_id,
            "profile_id": state["profile_id"], "policy_id": state["contract"]["policy"]["policy_id"],
            "profiles_sha256": state["contract"]["profiles_sha256"], "policy_sha256": state["contract"]["policy_sha256"], "schemas_sha256": state["contract"].get("schemas_sha256"),
            "extractor": state["extractor"], "extraction_model": state.get("model") if state["extractor"] == "openai" else None,
            "embeddings": state["embeddings"], "embedding_model": state.get("embedding_model") if state["embeddings"] in {"openai", "local"} else None,
            "provider_endpoint_sha256": state.get("provider_endpoint_sha256"),
            "semantic_verification_requested": state.get("verify_material", False),
            "extraction_pages": state.get("extraction_pages"), "model_batch_size": state.get("model_batch_size") if state["extractor"] == "openai" else None,
            "max_model_requests": state.get("max_model_requests") if state["extractor"] == "openai" else None,
            "model_max_tokens": state.get("model_max_tokens"), "model_reasoning_effort": state.get("model_reasoning_effort"),
            "model_provider": state.get("model_provider", "openai") if state["extractor"] == "openai" or state["embeddings"] == "openai" else None,
            "model_output_mode": state.get("model_output_mode", "strict") if state["extractor"] == "openai" else None,
            "model_usage": state.get("model_usage"),
            "registry_check": state.get("registry_check"), "registry_checks": state.get("registry_checks", []),
            "ownership_graph_settings": graph_settings(state["contract"]["policy"]),
            "status": status, "fact_count": len(state.get("facts", [])), "evidence_count": len(state.get("evidence", [])),
            "open_issue_count": sum(i["status"] == "OPEN" for i in issues),
            "started_at": state["started_at"], "finished_at": utc_now(),
            "disclaimer": "Provisional evidence preparation only; not a KYC decision or a guarantee of completeness.",
        }
        doc_payload = {**doc, "profile_id": state["profile_id"], "family": state["profile"]["family"], "current_run_id": run_id}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT current_run_id FROM documents WHERE case_id=? AND document_id=?", (case_id, document_id)).fetchone()
            # A commit may have succeeded just before a graph checkpoint was written.
            if current and current["current_run_id"] == run_id:
                return
            if self._case_revision(conn, case_id) != state.get("case_revision"):
                raise ValueError("Case evidence changed during processing or human review; retry this run to re-reconcile against the latest case.")
            for table in ("segments", "chunks", "evidence", "facts", "entities", "findings", "issues", "coverage", "sections", "reviews"):
                conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            conn.execute("INSERT INTO documents VALUES(?,?,?,?) ON CONFLICT(case_id,document_id) DO UPDATE SET current_run_id=excluded.current_run_id,payload_json=excluded.payload_json", (case_id, document_id, run_id, json_text(doc_payload)))
            conn.executemany("INSERT INTO segments VALUES(?,?,?,?,?,?,?)", [(run_id, s["segment_id"], case_id, document_id, s["section_id"], s["ordinal"], json_text(s)) for s in state["segments"]])
            conn.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", [(run_id, c["chunk_id"], case_id, document_id, c["section_id"], i, json_text(c)) for i, c in enumerate(state["chunks"])])
            conn.executemany("INSERT INTO evidence VALUES(?,?,?,?,?)", [(run_id, e["evidence_id"], case_id, document_id, json_text(e)) for e in state["evidence"]])
            conn.executemany("INSERT INTO facts VALUES(?,?,?,?,?,?)", [(run_id, f["fact_id"], case_id, document_id, f["category"], json_text(f)) for f in state["facts"]])
            conn.executemany("INSERT INTO entities VALUES(?,?,?,?,?)", [(run_id, e["entity_id"], case_id, document_id, json_text(e)) for e in state.get("entities", [])])
            conn.executemany("INSERT INTO findings VALUES(?,?,?,?)", [(run_id, f["finding_id"], case_id, json_text(f)) for f in state["findings"]])
            conn.executemany("INSERT INTO issues VALUES(?,?,?,?,?)", [(run_id, i["issue_id"], case_id, i["status"], json_text(i)) for i in issues])
            conn.executemany("INSERT INTO coverage VALUES(?,?,?,?,?)", [(run_id, c["category"], case_id, document_id, json_text(c)) for c in coverage])
            conn.executemany("INSERT INTO sections VALUES(?,?,?,?,?)", [(run_id, s["section_id"], case_id, document_id, json_text(s)) for s in state["sections"]])
            conn.executemany("INSERT INTO reviews VALUES(?,?,?,?)", [(run_id, r["issue_id"], case_id, json_text(r)) for r in state.get("review_decisions", [])])
            for review in state.get("review_decisions", []):
                self._audit(conn, run_id, case_id, "REVIEW_RECORDED", {"issue_id": review["issue_id"], "action": review["action"], "reviewer_id": review["reviewer_id"], "reviewer_identity_authenticated": False}, event_id=stable_id("audit", run_id, "review", review["issue_id"]))
            conn.execute("UPDATE runs SET document_id=?,profile_id=?,policy_id=?,status=?,finished_at=?,summary_json=?,error=NULL WHERE run_id=?", (document_id, state["profile_id"], summary["policy_id"], status, summary["finished_at"], json_text(summary), run_id))
            self._audit(conn, run_id, case_id, "CANONICAL_RECORDS_PERSISTED", {"status": status, "fact_count": summary["fact_count"]}, event_id=stable_id("audit", run_id, "persist"))

    def current_chunks(self, case_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT c.payload_json, v.model AS vector_model, v.vector_json FROM chunks c JOIN documents d
                ON d.current_run_id=c.run_id AND d.case_id=c.case_id AND d.document_id=c.document_id
                LEFT JOIN chunk_vectors v ON v.run_id=c.run_id AND v.chunk_id=c.chunk_id
                WHERE c.case_id=? ORDER BY c.document_id,c.ordinal""", (case_id,)).fetchall()
        result = []
        for row in rows:
            chunk = json.loads(row["payload_json"])
            if row["vector_json"] is not None:
                chunk["embedding"] = json.loads(row["vector_json"])
                chunk["embedding_model"] = row["vector_model"]
            result.append(chunk)
        return result

    def get_segment(self, case_id: str, segment_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("""SELECT s.payload_json FROM segments s JOIN documents d
                ON d.current_run_id=s.run_id AND d.case_id=s.case_id AND d.document_id=s.document_id
                WHERE s.case_id=? AND s.segment_id=?""", (case_id, segment_id)).fetchone()
        return json.loads(row[0]) if row else None

    def export_case(self, case_id: str) -> dict:
        with self._connect() as conn:
            docs = [json.loads(r["payload_json"]) for r in conn.execute("SELECT payload_json FROM documents WHERE case_id=? ORDER BY document_id", (case_id,))]
            current_runs = [d["current_run_id"] for d in docs]
            runs = [dict(r) for r in conn.execute("SELECT run_id,document_id,profile_id,policy_id,status,started_at,finished_at,error,summary_json FROM runs WHERE case_id=? ORDER BY started_at", (case_id,))]
            current_set = set(current_runs)
            active_runs = [r for r in runs if r["run_id"] in current_set]
            for r in active_runs:
                r["summary"] = json.loads(r.pop("summary_json")) if r.get("summary_json") else None
            pending = [{k: r[k] for k in ("run_id", "document_id", "status", "started_at", "error")} for r in runs if r["run_id"] not in current_set and r["status"] in {"PROCESSING", "REVIEW_REQUIRED", "FAILED"} and not r["finished_at"]]
            def rows(table: str) -> list[dict]:
                if not current_runs:
                    return []
                marks = ",".join("?" for _ in current_runs)
                rs = conn.execute(f"SELECT run_id,payload_json FROM {table} WHERE case_id=? AND run_id IN ({marks})", (case_id, *current_runs)).fetchall()
                return [{**json.loads(r["payload_json"]), "run_id": r["run_id"]} for r in rs]
            facts, evidence, cov, sections, raw_findings, raw_issues, reviews, document_entities = (rows(t) for t in ("facts", "evidence", "coverage", "sections", "findings", "issues", "reviews", "entities"))
            audit = [dict(r) for r in conn.execute("SELECT event_id,run_id,happened_at,action,details_json FROM audit_events WHERE case_id=? ORDER BY happened_at", (case_id,))]
        # The newest current run wins if the same issue or finding is rediscovered.
        run_order = {r["run_id"]: r["started_at"] for r in runs}
        def latest(items: list[dict], key: str) -> list[dict]:
            output: dict[str, dict] = {}
            for item in sorted(items, key=lambda x: run_order.get(x["run_id"], "")):
                output[item[key]] = item
            return list(output.values())
        issues = latest(raw_issues, "issue_id")
        findings = latest(raw_findings, "finding_id")
        for row in audit:
            row["details"] = json.loads(row.pop("details_json"))
        open_issues = [i for i in issues if i["status"] == "OPEN"]
        # Case entities are recomputed from current documents, so they reflect every merge.
        entities, entity_map, _ = resolve_case([{k: v for k, v in e.items() if k != "run_id"} for e in document_entities], "case")
        for fact in facts:
            fact["case_entity_ids"] = {role: entity_map.get(link["entity_id"]) for role, link in fact.get("entity_links", {}).items()}
        # The newest run's policy settings apply; the graph is rebuilt so it reflects every review.
        settings = next((r["summary"]["ownership_graph_settings"] for r in sorted(active_runs, key=lambda r: r["started_at"], reverse=True) if (r.get("summary") or {}).get("ownership_graph_settings")), None)
        registry_checks = [c for r in active_runs for c in (r.get("summary") or {}).get("registry_checks", [])]
        ownership_graph, _ = build_ownership_graph(facts, entities, entity_map, issues, settings=settings, registry_checks=registry_checks)
        open_requirements = [c for c in cov if c["material"] and (c["status"] in {"NOT_REVIEWED", "NOT_FOUND_AFTER_REVIEW", "EXTRACTION_FAILED", "CONFLICTED"} or c.get("chunks_not_reviewed", 0) > 0 or not c.get("source_fully_parsed", True))]
        status = "REVIEW_REQUIRED" if pending or open_issues or open_requirements or not docs else "PROVISIONAL"
        return {"case_id": case_id, "status": status, "disclaimer": "Provisional evidence package. No completeness, identity, source independence, sanctions clearance, or KYC decision is guaranteed.", "documents": docs, "runs": active_runs, "pending_runs": pending, "evidence": evidence, "facts": facts, "entities": entities, "document_entities": document_entities, "ownership_graph": ownership_graph, "registry_checks": registry_checks, "findings": findings, "coverage": cov, "section_inventory": sections, "issues": issues, "unresolved": {"open_issue_ids": [i["issue_id"] for i in open_issues], "material_requirement_gaps": [{"run_id": c["run_id"], "category": c["category"], "status": c["status"], "chunks_not_reviewed": c.get("chunks_not_reviewed", 0), "parse_gap_pages": c.get("parse_gap_pages", [])} for c in open_requirements], "verification_note": "Source-match checks are not external verification; all facts remain pending human verification."}, "review_decisions": reviews, "audit_events": audit}
