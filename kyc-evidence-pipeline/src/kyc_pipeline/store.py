from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .crypto import PREFIX, PayloadCipher
from .entities import resolve_case
from .ownership import build_ownership_graph, graph_settings
from .util import digest, json_text, stable_id, utc_now

GENESIS = "0" * 64
# Tables a run persists, with each row's key within the run. Their content is hashed into
# the run's CANONICAL_RECORDS_PERSISTED audit event, so later edits are detectable.
RUN_TABLES = {"segments": "segment_id", "chunks": "chunk_id", "evidence": "evidence_id", "facts": "fact_id", "entities": "entity_id", "findings": "finding_id", "issues": "issue_id", "coverage": "category", "sections": "section_id", "reviews": "issue_id"}
UNFINISHED = ("PROCESSING", "REVIEW_REQUIRED", "FAILED", "SUPERSEDED")

TABLES_DDL = """
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
-- Append-only (enforced by triggers). Each case's events form a hash chain: event_hash covers
-- the event and prev_hash, so an edited, removed or reordered event breaks the chain.
-- Details never hold document content, so erasure keeps the events.
CREATE TABLE IF NOT EXISTS audit_events (
  event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, case_id TEXT NOT NULL,
  happened_at TEXT NOT NULL, action TEXT NOT NULL, details_json TEXT NOT NULL,
  seq INTEGER, prev_hash TEXT, event_hash TEXT
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
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
CHAIN_INDEX = "CREATE UNIQUE INDEX IF NOT EXISTS audit_chain ON audit_events(case_id, seq);"
SQLITE_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END;
"""
POSTGRES_TRIGGERS = """
CREATE OR REPLACE FUNCTION kyc_audit_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'audit events are append-only'; END $$;
CREATE OR REPLACE TRIGGER audit_events_append_only BEFORE UPDATE OR DELETE ON audit_events FOR EACH ROW EXECUTE FUNCTION kyc_audit_append_only();
CREATE OR REPLACE TRIGGER audit_events_no_truncate BEFORE TRUNCATE ON audit_events FOR EACH STATEMENT EXECUTE FUNCTION kyc_audit_append_only();
"""


def is_postgres(target: str | Path) -> bool:
    return str(target).startswith(("postgresql://", "postgres://"))


def check_postgres_tls(url: str) -> None:
    """A Postgres server other than this machine must be reached over verified TLS."""
    from psycopg.conninfo import conninfo_to_dict
    info = conninfo_to_dict(url)
    hosts = [h for h in str(info.get("host") or "").split(",")] + [h for h in str(info.get("hostaddr") or "").split(",") if h]
    local = all(h in ("", "localhost", "127.0.0.1", "::1") or h.startswith("/") for h in hosts)
    if not local and info.get("sslmode") not in ("verify-full", "verify-ca"):
        raise ValueError("A remote Postgres server needs sslmode=verify-full (or verify-ca) in the connection URL")


def _event_hash(record: dict) -> str:
    return digest(json_text(record))


class _Row(dict):
    """A Postgres row readable by column name or position, like sqlite3.Row."""

    def __init__(self, names: list[str], values: tuple):
        super().__init__(zip(names, values))
        self._values = tuple(values)

    def __getitem__(self, key):
        return self._values[key] if isinstance(key, int) else dict.__getitem__(self, key)


def _pg_rows(cursor):
    names = [c.name for c in cursor.description or []]
    return lambda values: _Row(names, values)


class _Conn:
    """One transaction on SQLite or Postgres, written in one dialect: ? placeholders, ON CONFLICT upserts."""

    def __init__(self, raw: Any, postgres: bool):
        self.raw, self.postgres = raw, postgres

    def execute(self, sql: str, params: tuple | list = ()):
        if self.postgres:
            cursor = self.raw.cursor()
            cursor.execute(sql.replace("?", "%s"), params)
            return cursor
        return self.raw.execute(sql, params)

    def executemany(self, sql: str, rows: list) -> None:
        if not rows:
            return
        if self.postgres:
            self.raw.cursor().executemany(sql.replace("?", "%s"), rows)
        else:
            self.raw.executemany(sql, rows)

    def lock_case(self, case_id: str) -> None:
        """Serialize writers of one case (and its audit chain) until this transaction ends."""
        if self.postgres:
            self.execute("SELECT pg_advisory_xact_lock(hashtext(?))", ("kyc-case:" + case_id,))
        else:
            self.raw.execute("BEGIN IMMEDIATE")

    def snapshot(self) -> None:
        """Read one consistent view for the rest of the transaction; call first."""
        if self.postgres:
            self.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        else:
            self.raw.execute("BEGIN")


class EvidenceStore:
    """Evidence store on SQLite (a local file) or Postgres (a postgresql:// URL).

    With the secret KYC_DATA_KEY set, every payload is encrypted (see crypto.py).
    SQLite files are not otherwise protected; Postgres needs its own encryption at rest.
    """

    def __init__(self, target: str | Path):
        self.postgres = is_postgres(target)
        self.cipher = PayloadCipher.from_secret()
        if self.postgres:
            from psycopg_pool import ConnectionPool
            check_postgres_tls(str(target))
            self.path = str(target)
            self._pool = ConnectionPool(self.path, min_size=1, max_size=4, kwargs={"row_factory": _pg_rows}, open=True)
        else:
            self.path = str(Path(target).expanduser().resolve())
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            if self.postgres:
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('kyc-schema'))")
                conn.raw.execute(TABLES_DDL)
                conn.raw.execute(CHAIN_INDEX + POSTGRES_TRIGGERS)
            else:
                conn.raw.execute("PRAGMA journal_mode=WAL")
                conn.raw.executescript(TABLES_DDL)
                columns = {r[1] for r in conn.raw.execute("PRAGMA table_info(audit_events)")}
                for column in ("seq INTEGER", "prev_hash TEXT", "event_hash TEXT"):
                    if column.split()[0] not in columns:  # stores created before the audit chain
                        conn.raw.execute(f"ALTER TABLE audit_events ADD COLUMN {column}")
                conn.raw.executescript(CHAIN_INDEX + SQLITE_TRIGGERS)
            row = conn.execute("SELECT value FROM meta WHERE name='payload_key_id'").fetchone()
            if row and not self.cipher:
                raise ValueError("This evidence store is encrypted: set the KYC_DATA_KEY secret")
            if row and row[0] != self.cipher.key_id:
                raise ValueError("KYC_DATA_KEY is not the key this evidence store was encrypted with")
            if self.cipher and not row:
                if conn.execute("SELECT 1 FROM runs LIMIT 1").fetchone():
                    raise ValueError("This evidence store holds unencrypted data; KYC_DATA_KEY can only be set for a new store")
                conn.execute("INSERT INTO meta VALUES('payload_key_id', ?)", (self.cipher.key_id,))
        if not self.postgres:
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def close(self) -> None:
        if self.postgres:
            self._pool.close()

    @contextmanager
    def _connect(self):
        if self.postgres:
            with self._pool.connection() as raw:  # committed on success, rolled back on error
                yield _Conn(raw, True)
            return
        raw = sqlite3.connect(self.path, timeout=30)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("PRAGMA busy_timeout=30000")
        raw.execute("PRAGMA secure_delete=ON")  # deleted content is overwritten, not left in free pages
        try:
            with raw:
                yield _Conn(raw, False)
        finally:
            raw.close()

    def _flush_deleted(self) -> None:
        """Move deleted SQLite pages out of the write-ahead log, so erased content does not linger there."""
        if not self.postgres:
            with closing(sqlite3.connect(self.path, timeout=30)) as raw:
                raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _dump(self, table: str, value: Any) -> str:
        text = json_text(value)
        return self.cipher.seal(text, table) if self.cipher else text

    def _load(self, table: str, value: str | None) -> Any:
        if value is None:
            return None
        if value.startswith(PREFIX):
            if not self.cipher:
                raise ValueError("This evidence store is encrypted: set the KYC_DATA_KEY secret")
            value = self.cipher.open(value, table)
        return json.loads(value)

    def put_chunk_vectors(self, run_id: str, model: str, chunks: list[dict], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count does not match source chunks")
        with self._connect() as conn:
            conn.executemany("INSERT INTO chunk_vectors VALUES(?,?,?,?) ON CONFLICT(run_id,chunk_id) DO UPDATE SET model=excluded.model,vector_json=excluded.vector_json", [(run_id, chunk["chunk_id"], model, self._dump("chunk_vectors", vector)) for chunk, vector in zip(chunks, vectors)])

    def get_layout_result(self, document_sha256: str, fingerprint: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM layout_results WHERE document_sha256=? AND fingerprint=?", (document_sha256, fingerprint)).fetchone()
        return self._load("layout_results", row[0]) if row else None

    def put_layout_result(self, document_sha256: str, fingerprint: str, result: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO layout_results VALUES(?,?,?,?) ON CONFLICT DO NOTHING", (document_sha256, fingerprint, self._dump("layout_results", result), utc_now()))

    def get_registry_record(self, source: str, identifier: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json, fetched_at FROM registry_records WHERE source=? AND identifier=?", (source, identifier)).fetchone()
        return {"record": self._load("registry_records", row[0]), "fetched_at": row[1]} if row else None

    def put_registry_record(self, source: str, identifier: str, record: dict | None, fetched_at: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO registry_records VALUES(?,?,?,?) ON CONFLICT(source,identifier) DO UPDATE SET payload_json=excluded.payload_json,fetched_at=excluded.fetched_at", (source, identifier, self._dump("registry_records", record), fetched_at))

    def put_extraction_unit(self, run_id: str, unit: dict) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO extraction_units VALUES(?,?,?,?,?) ON CONFLICT(run_id,chunk_id) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json,updated_at=excluded.updated_at", (run_id, unit["chunk_id"], unit["status"], self._dump("extraction_units", unit), utc_now()))

    def get_extraction_units(self, run_id: str) -> dict[str, dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT chunk_id,payload_json FROM extraction_units WHERE run_id=?", (run_id,)).fetchall()
        return {r["chunk_id"]: self._load("extraction_units", r["payload_json"]) for r in rows}

    def mark_unit_rejections(self, run_id: str, chunk_id: str, rejected: list[dict]) -> None:
        """Record discarded candidates so a retry re-extracts the chunk instead of reusing it."""
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM extraction_units WHERE run_id=? AND chunk_id=?", (run_id, chunk_id)).fetchone()
            if row:
                unit = self._load("extraction_units", row[0])
                unit["rejected_candidates"] = rejected
                conn.execute("UPDATE extraction_units SET payload_json=?,updated_at=? WHERE run_id=? AND chunk_id=?", (self._dump("extraction_units", unit), utc_now(), run_id, chunk_id))

    def mark_unit_error(self, run_id: str, chunk_id: str, error: str) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload_json FROM extraction_units WHERE run_id=? AND chunk_id=?", (run_id, chunk_id)).fetchone()
            if row:
                unit = self._load("extraction_units", row[0])
                unit["status"] = "ERROR"
                unit["error"] = error
                conn.execute("UPDATE extraction_units SET status=?,payload_json=?,updated_at=? WHERE run_id=? AND chunk_id=?", ("ERROR", self._dump("extraction_units", unit), utc_now(), run_id, chunk_id))

    def _audit(self, conn: _Conn, run_id: str, case_id: str, action: str, details: dict, *, event_id: str | None = None) -> None:
        """Append an event to the case's hash chain. The caller holds the case lock."""
        happened = utc_now()
        eid = event_id or stable_id("audit", run_id, action, happened)
        if conn.execute("SELECT 1 FROM audit_events WHERE event_id=?", (eid,)).fetchone():
            return
        last = conn.execute("SELECT seq, event_hash FROM audit_events WHERE case_id=? AND seq IS NOT NULL ORDER BY seq DESC LIMIT 1", (case_id,)).fetchone()
        seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS)
        record = {"event_id": eid, "run_id": run_id, "case_id": case_id, "happened_at": happened, "action": action, "details": details, "seq": seq, "prev_hash": prev}
        conn.execute("INSERT INTO audit_events(event_id,run_id,case_id,happened_at,action,details_json,seq,prev_hash,event_hash) VALUES (?,?,?,?,?,?,?,?,?)", (eid, run_id, case_id, happened, action, json_text(details), seq, prev, _event_hash(record)))

    def _run_records_digest(self, conn: _Conn, run_id: str) -> str:
        """Digest of everything a run persisted: its rows in RUN_TABLES and its summary."""
        parts: list = []
        for table, key in RUN_TABLES.items():
            rows = conn.execute(f"SELECT {key}, payload_json FROM {table} WHERE run_id=?", (run_id,)).fetchall()
            parts.append([table, sorted([r[0], self._load(table, r[1])] for r in rows)])
        summary = conn.execute("SELECT summary_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
        parts.append(["runs", self._load("runs", summary[0]) if summary else None])
        return digest(json_text(parts))

    def start_run(self, run_id: str, case_id: str, started_at: str) -> None:
        with self._connect() as conn:
            conn.lock_case(case_id)
            conn.execute("INSERT INTO runs(run_id,case_id,status,started_at) VALUES(?,?,?,?)", (run_id, case_id, "PROCESSING", started_at))
            self._audit(conn, run_id, case_id, "RUN_STARTED", {})

    def set_run_status(self, run_id: str, status: str, *, error: str | None = None) -> None:
        with self._connect() as conn:
            row = conn.execute("SELECT case_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise ValueError("Run not found")
            conn.lock_case(row[0])
            conn.execute("UPDATE runs SET status=?,error=? WHERE run_id=?", (status, error, run_id))
            self._audit(conn, run_id, row[0], "RUN_" + status, {"error": error} if error else {})

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        if run.get("summary_json"):
            run["summary_json"] = json_text(self._load("runs", run["summary_json"]))
        return run

    @staticmethod
    def _case_revision(conn: _Conn, case_id: str) -> str:
        rows = conn.execute("SELECT document_id,current_run_id FROM documents WHERE case_id=? ORDER BY document_id", (case_id,)).fetchall()
        return digest(json_text([[r["document_id"], r["current_run_id"]] for r in rows]))

    def get_case_snapshot(self, case_id: str, exclude_document_id: str | None = None) -> tuple[str, list[dict]]:
        # Revision and claims must come from the same read snapshot.
        with self._connect() as conn:
            conn.snapshot()
            revision = self._case_revision(conn, case_id)
            rows = conn.execute("""SELECT f.payload_json FROM facts f JOIN documents d
                ON d.current_run_id=f.run_id AND d.case_id=f.case_id AND d.document_id=f.document_id
                WHERE f.case_id=? AND (CAST(? AS TEXT) IS NULL OR f.document_id<>?)""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
            facts = [self._load("facts", r[0]) for r in rows]
        return revision, facts

    def get_case_entities(self, case_id: str, exclude_document_id: str | None = None) -> list[dict]:
        """Document-level entity records from each document's current run."""
        with self._connect() as conn:
            rows = conn.execute("""SELECT e.payload_json FROM entities e JOIN documents d
                ON d.current_run_id=e.run_id AND d.case_id=e.case_id AND d.document_id=e.document_id
                WHERE e.case_id=? AND (CAST(? AS TEXT) IS NULL OR e.document_id<>?) ORDER BY e.document_id, e.entity_id""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
        return [self._load("entities", r[0]) for r in rows]

    def get_case_registry_checks(self, case_id: str, exclude_document_id: str | None = None) -> list[dict]:
        """Register checks recorded by each document's current run."""
        with self._connect() as conn:
            rows = conn.execute("""SELECT r.summary_json FROM runs r JOIN documents d ON d.current_run_id=r.run_id AND d.case_id=r.case_id
                WHERE r.case_id=? AND (CAST(? AS TEXT) IS NULL OR d.document_id<>?)""", (case_id, exclude_document_id, exclude_document_id)).fetchall()
        return [c for r in rows if r[0] for c in self._load("runs", r[0]).get("registry_checks", [])]

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
            "screening_lists": state.get("screening_lists"), "screening_checks": state.get("screening_checks", []),
            "ownership_graph_settings": graph_settings(state["contract"]["policy"]),
            "status": status, "fact_count": len(state.get("facts", [])), "evidence_count": len(state.get("evidence", [])),
            "open_issue_count": sum(i["status"] == "OPEN" for i in issues),
            "started_at": state["started_at"], "finished_at": utc_now(),
            "disclaimer": "Provisional evidence preparation only; not a KYC decision or a guarantee of completeness.",
        }
        doc_payload = {**doc, "profile_id": state["profile_id"], "family": state["profile"]["family"], "current_run_id": run_id}
        with self._connect() as conn:
            conn.lock_case(case_id)
            current = conn.execute("SELECT current_run_id FROM documents WHERE case_id=? AND document_id=?", (case_id, document_id)).fetchone()
            # A commit may have succeeded just before a graph checkpoint was written.
            if current and current["current_run_id"] == run_id:
                return
            if self._case_revision(conn, case_id) != state.get("case_revision"):
                raise ValueError("Case evidence changed during processing or human review; retry this run to re-reconcile against the latest case.")
            for table in RUN_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            conn.execute("INSERT INTO documents VALUES(?,?,?,?) ON CONFLICT(case_id,document_id) DO UPDATE SET current_run_id=excluded.current_run_id,payload_json=excluded.payload_json", (case_id, document_id, run_id, self._dump("documents", doc_payload)))
            conn.executemany("INSERT INTO segments VALUES(?,?,?,?,?,?,?)", [(run_id, s["segment_id"], case_id, document_id, s["section_id"], s["ordinal"], self._dump("segments", s)) for s in state["segments"]])
            conn.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)", [(run_id, c["chunk_id"], case_id, document_id, c["section_id"], i, self._dump("chunks", c)) for i, c in enumerate(state["chunks"])])
            conn.executemany("INSERT INTO evidence VALUES(?,?,?,?,?)", [(run_id, e["evidence_id"], case_id, document_id, self._dump("evidence", e)) for e in state["evidence"]])
            conn.executemany("INSERT INTO facts VALUES(?,?,?,?,?,?)", [(run_id, f["fact_id"], case_id, document_id, f["category"], self._dump("facts", f)) for f in state["facts"]])
            conn.executemany("INSERT INTO entities VALUES(?,?,?,?,?)", [(run_id, e["entity_id"], case_id, document_id, self._dump("entities", e)) for e in state.get("entities", [])])
            conn.executemany("INSERT INTO findings VALUES(?,?,?,?)", [(run_id, f["finding_id"], case_id, self._dump("findings", f)) for f in state["findings"]])
            conn.executemany("INSERT INTO issues VALUES(?,?,?,?,?)", [(run_id, i["issue_id"], case_id, i["status"], self._dump("issues", i)) for i in issues])
            conn.executemany("INSERT INTO coverage VALUES(?,?,?,?,?)", [(run_id, c["category"], case_id, document_id, self._dump("coverage", c)) for c in coverage])
            conn.executemany("INSERT INTO sections VALUES(?,?,?,?,?)", [(run_id, s["section_id"], case_id, document_id, self._dump("sections", s)) for s in state["sections"]])
            conn.executemany("INSERT INTO reviews VALUES(?,?,?,?)", [(run_id, r["issue_id"], case_id, self._dump("reviews", r)) for r in state.get("review_decisions", [])])
            for review in state.get("review_decisions", []):
                self._audit(conn, run_id, case_id, "REVIEW_RECORDED", {"issue_id": review["issue_id"], "action": review["action"], "reviewer_id": review["reviewer_id"], "reviewer_identity_authenticated": False}, event_id=stable_id("audit", run_id, "review", review["issue_id"]))
            conn.execute("UPDATE runs SET document_id=?,profile_id=?,policy_id=?,status=?,finished_at=?,summary_json=?,error=NULL WHERE run_id=?", (document_id, state["profile_id"], summary["policy_id"], status, summary["finished_at"], self._dump("runs", summary), run_id))
            self._audit(conn, run_id, case_id, "CANONICAL_RECORDS_PERSISTED", {"status": status, "fact_count": summary["fact_count"], "records_sha256": self._run_records_digest(conn, run_id)}, event_id=stable_id("audit", run_id, "persist"))

    def current_chunks(self, case_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT c.payload_json, v.model AS vector_model, v.vector_json FROM chunks c JOIN documents d
                ON d.current_run_id=c.run_id AND d.case_id=c.case_id AND d.document_id=c.document_id
                LEFT JOIN chunk_vectors v ON v.run_id=c.run_id AND v.chunk_id=c.chunk_id
                WHERE c.case_id=? ORDER BY c.document_id,c.ordinal""", (case_id,)).fetchall()
        result = []
        for row in rows:
            chunk = self._load("chunks", row["payload_json"])
            if row["vector_json"] is not None:
                chunk["embedding"] = self._load("chunk_vectors", row["vector_json"])
                chunk["embedding_model"] = row["vector_model"]
            result.append(chunk)
        return result

    def get_segment(self, case_id: str, segment_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("""SELECT s.payload_json FROM segments s JOIN documents d
                ON d.current_run_id=s.run_id AND d.case_id=s.case_id AND d.document_id=s.document_id
                WHERE s.case_id=? AND s.segment_id=?""", (case_id, segment_id)).fetchone()
        return self._load("segments", row[0]) if row else None

    def export_case(self, case_id: str) -> dict:
        with self._connect() as conn:
            conn.snapshot()
            docs = [self._load("documents", r["payload_json"]) for r in conn.execute("SELECT payload_json FROM documents WHERE case_id=? ORDER BY document_id", (case_id,)).fetchall()]
            current_runs = [d["current_run_id"] for d in docs]
            runs = [dict(r) for r in conn.execute("SELECT run_id,document_id,profile_id,policy_id,status,started_at,finished_at,error,summary_json FROM runs WHERE case_id=? ORDER BY started_at", (case_id,)).fetchall()]
            current_set = set(current_runs)
            active_runs = [r for r in runs if r["run_id"] in current_set]
            for r in active_runs:
                r["summary"] = self._load("runs", r.pop("summary_json"))
            pending = [{k: r[k] for k in ("run_id", "document_id", "status", "started_at", "error")} for r in runs if r["run_id"] not in current_set and r["status"] in {"PROCESSING", "REVIEW_REQUIRED", "FAILED"} and not r["finished_at"]]
            def rows(table: str) -> list[dict]:
                if not current_runs:
                    return []
                marks = ",".join("?" for _ in current_runs)
                rs = conn.execute(f"SELECT run_id,payload_json FROM {table} WHERE case_id=? AND run_id IN ({marks})", (case_id, *current_runs)).fetchall()
                return [{**self._load(table, r["payload_json"]), "run_id": r["run_id"]} for r in rs]
            facts, evidence, cov, sections, raw_findings, raw_issues, reviews, document_entities = (rows(t) for t in ("facts", "evidence", "coverage", "sections", "findings", "issues", "reviews", "entities"))
            audit = [dict(r) for r in conn.execute("SELECT event_id,run_id,happened_at,action,details_json,seq,event_hash FROM audit_events WHERE case_id=? ORDER BY happened_at, seq", (case_id,)).fetchall()]
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
        screening_checks = [c for r in active_runs for c in (r.get("summary") or {}).get("screening_checks", [])]
        open_requirements = [c for c in cov if c["material"] and (c["status"] in {"NOT_REVIEWED", "NOT_FOUND_AFTER_REVIEW", "EXTRACTION_FAILED", "CONFLICTED"} or c.get("chunks_not_reviewed", 0) > 0 or not c.get("source_fully_parsed", True))]
        status = "REVIEW_REQUIRED" if pending or open_issues or open_requirements or not docs else "PROVISIONAL"
        return {"case_id": case_id, "status": status, "disclaimer": "Provisional evidence package. No completeness, identity, source independence, sanctions clearance, or KYC decision is guaranteed.", "documents": docs, "runs": active_runs, "pending_runs": pending, "evidence": evidence, "facts": facts, "entities": entities, "document_entities": document_entities, "ownership_graph": ownership_graph, "registry_checks": registry_checks, "screening_checks": screening_checks, "findings": findings, "coverage": cov, "section_inventory": sections, "issues": issues, "unresolved": {"open_issue_ids": [i["issue_id"] for i in open_issues], "material_requirement_gaps": [{"run_id": c["run_id"], "category": c["category"], "status": c["status"], "chunks_not_reviewed": c.get("chunks_not_reviewed", 0), "parse_gap_pages": c.get("parse_gap_pages", [])} for c in open_requirements], "verification_note": "Source-match checks are not external verification; all facts remain pending human verification."}, "review_decisions": reviews, "audit_events": audit}

    def verify_audit(self, case_id: str, anchor: dict | None = None) -> dict:
        """Check the case's audit chain, and each persisted run's records against its recorded digest."""
        problems: list[str] = []
        with self._connect() as conn:
            conn.snapshot()
            rows = conn.execute("SELECT * FROM audit_events WHERE case_id=? ORDER BY seq, happened_at", (case_id,)).fetchall()
            legacy = [r for r in rows if r["seq"] is None]
            chained = sorted((r for r in rows if r["seq"] is not None), key=lambda r: r["seq"])
            if not rows:
                raise ValueError("Case not found")
            prev = GENESIS
            for expected, r in enumerate(chained, 1):
                if r["seq"] != expected:
                    problems.append(f"Event {expected} is missing (found {r['seq']} next)")
                if r["prev_hash"] != prev:
                    problems.append(f"Event {r['seq']} ({r['action']}) does not follow the event before it")
                record = {"event_id": r["event_id"], "run_id": r["run_id"], "case_id": r["case_id"], "happened_at": r["happened_at"], "action": r["action"], "details": json.loads(r["details_json"]), "seq": r["seq"], "prev_hash": r["prev_hash"]}
                if _event_hash(record) != r["event_hash"]:
                    problems.append(f"Event {r['seq']} ({r['action']}) was altered: it no longer matches its hash")
                prev = r["event_hash"]
            if anchor:
                at = next((r for r in chained if r["seq"] == anchor.get("seq")), None)
                if not at or at["event_hash"] != anchor.get("event_hash"):
                    problems.append(f"The chain no longer contains the anchored event {anchor.get('seq')}: history was rewritten or truncated")
            events = [(r, json.loads(r["details_json"])) for r in chained]
            erased_at = max((r["seq"] for r, _ in events if r["action"] == "CASE_ERASED"), default=0)
            persisted = {r["run_id"]: d.get("records_sha256") for r, d in events if r["action"] == "CANONICAL_RECORDS_PERSISTED" and r["seq"] > erased_at}
            for run_id, expected_digest in persisted.items():
                if expected_digest and self._run_records_digest(conn, run_id) != expected_digest:
                    problems.append(f"Run {run_id}: its stored records differ from those recorded when it was persisted")
            stored = {r[0] for table in RUN_TABLES for r in conn.execute(f"SELECT DISTINCT run_id FROM {table} WHERE case_id=?", (case_id,)).fetchall()}
            for run_id in sorted(stored - set(persisted) - {r["run_id"] for r in legacy}):
                problems.append(f"Run {run_id}: records are stored with no audit event recording their persistence")
            # Each run's status must be the one its last audit event set.
            expected_status: dict[str, str] = {}
            for r, d in events:
                if r["action"] == "RUN_STARTED":
                    expected_status[r["run_id"]] = "PROCESSING"
                elif r["action"] == "CANONICAL_RECORDS_PERSISTED":
                    expected_status[r["run_id"]] = d.get("status")
                elif r["action"] == "CASE_ERASED":
                    expected_status.update({run: "ERASED" for run in d.get("runs", [])})
                elif r["action"] == "RUN_EXPIRED":
                    expected_status[r["run_id"]] = "EXPIRED"
                elif r["action"].startswith("RUN_"):
                    expected_status[r["run_id"]] = r["action"][4:]
            for r in conn.execute("SELECT run_id,status FROM runs WHERE case_id=?", (case_id,)).fetchall():
                if r["run_id"] in expected_status and expected_status[r["run_id"]] != r["status"]:
                    problems.append(f"Run {r['run_id']}: status {r['status']} differs from the audited {expected_status[r['run_id']]}")
        head = {"seq": chained[-1]["seq"], "event_hash": chained[-1]["event_hash"]} if chained else None
        return {"case_id": case_id, "status": "FAILED" if problems else "OK", "chained_events": len(chained), "unchained_events_before_chaining": len(legacy), "runs_checked": len(persisted), "head": head, "problems": problems,
                "note": "Detects edits, removals and insertions made without recomputing the chain. Someone who can rewrite the whole chain can only be caught against an anchor kept elsewhere: store the head from each verification outside this database."}

    def case_run_ids(self, case_id: str) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE case_id=? ORDER BY started_at", (case_id,)).fetchall()]

    def erase_case(self, case_id: str, reason: str, requested_by: str) -> dict:
        """Delete the case's content: documents, runs' records, caches of its documents. Audit events remain."""
        with self._connect() as conn:
            conn.lock_case(case_id)
            runs = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE case_id=? ORDER BY started_at", (case_id,)).fetchall()]
            if not runs:
                raise ValueError("Case not found")
            if conn.execute("SELECT 1 FROM runs WHERE case_id=? AND status<>'ERASED'", (case_id,)).fetchone() is None:
                raise ValueError("Case is already erased")
            shas = {self._load("documents", r[0])["sha256"] for r in conn.execute("SELECT payload_json FROM documents WHERE case_id=?", (case_id,)).fetchall()}
            deleted = {table: conn.execute(f"DELETE FROM {table} WHERE case_id=?", (case_id,)).rowcount for table in [*RUN_TABLES, "documents"]}
            marks = ",".join("?" for _ in runs)
            for table in ("chunk_vectors", "extraction_units"):
                deleted[table] = conn.execute(f"DELETE FROM {table} WHERE run_id IN ({marks})", runs).rowcount
            # Layout results hold the documents' full text; keep those another case still uses.
            still_used = {self._load("documents", r[0])["sha256"] for r in conn.execute("SELECT payload_json FROM documents").fetchall()}
            deleted["layout_results"] = sum(conn.execute("DELETE FROM layout_results WHERE document_sha256=?", (sha,)).rowcount for sha in sorted(shas - still_used))
            conn.execute("UPDATE runs SET summary_json=NULL, error=NULL, status='ERASED' WHERE case_id=?", (case_id,))
            self._audit(conn, "", case_id, "CASE_ERASED", {"reason": reason, "requested_by": requested_by, "requested_by_authenticated": False, "runs": runs, "deleted_rows": deleted})
        self._flush_deleted()
        return {"case_id": case_id, "runs": runs, "deleted_rows": deleted}

    def retention_due(self, retention: dict, now: datetime | None = None) -> dict:
        """What the retention periods make due for deletion; nothing is deleted."""
        now = now or datetime.now(timezone.utc)
        def cutoff(days: int) -> str:
            return (now - timedelta(days=days)).isoformat(timespec="microseconds")
        with self._connect() as conn:
            conn.snapshot()
            erased = {r[0] for r in conn.execute("SELECT DISTINCT case_id FROM audit_events WHERE action='CASE_ERASED'").fetchall()}
            last = conn.execute("SELECT case_id, MAX(happened_at) FROM audit_events GROUP BY case_id").fetchall()
            cases = [{"case_id": r[0], "last_activity": r[1]} for r in last if r[1] < cutoff(retention["case_days_after_last_activity"]) and r[0] not in erased]
            marks = ",".join("?" for _ in UNFINISHED)
            runs = [dict(r) for r in conn.execute(f"""SELECT run_id, case_id, status, started_at FROM runs WHERE status IN ({marks}) AND finished_at IS NULL AND started_at < ?
                AND run_id NOT IN (SELECT current_run_id FROM documents) ORDER BY started_at""", (*UNFINISHED, cutoff(retention["unfinished_run_days"]))).fetchall()]
            runs = [r for r in runs if r["case_id"] not in {c["case_id"] for c in cases}]
            caches = {"layout_results": conn.execute("SELECT COUNT(*) FROM layout_results WHERE created_at < ?", (cutoff(retention["cache_days"]),)).fetchone()[0],
                      "registry_records": conn.execute("SELECT COUNT(*) FROM registry_records WHERE fetched_at < ?", (cutoff(retention["cache_days"]),)).fetchone()[0]}
        return {"as_of": now.isoformat(timespec="seconds"), "retention": retention, "cases_to_erase": cases, "runs_to_expire": runs, "cache_rows_to_delete": caches}

    def expire_run(self, run_id: str, reason: str) -> dict:
        """Delete an unfinished run's working records (extraction units, vectors) and mark it EXPIRED."""
        with self._connect() as conn:
            row = conn.execute("SELECT case_id FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row:
                conn.lock_case(row["case_id"])
                row = conn.execute("SELECT case_id, status, finished_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row or row["status"] not in UNFINISHED or row["finished_at"]:
                raise ValueError(f"Run {run_id} is not an unfinished run")
            deleted = {table: conn.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,)).rowcount for table in ("chunk_vectors", "extraction_units", *RUN_TABLES)}
            conn.execute("UPDATE runs SET summary_json=NULL, error=NULL, status='EXPIRED' WHERE run_id=?", (run_id,))
            self._audit(conn, run_id, row["case_id"], "RUN_EXPIRED", {"reason": reason, "previous_status": row["status"], "deleted_rows": deleted})
        self._flush_deleted()
        return {"run_id": run_id, "case_id": row["case_id"], "deleted_rows": deleted}

    def purge_caches(self, older_than_days: int, now: datetime | None = None) -> dict:
        limit = ((now or datetime.now(timezone.utc)) - timedelta(days=older_than_days)).isoformat(timespec="microseconds")
        with self._connect() as conn:
            deleted = {"layout_results": conn.execute("DELETE FROM layout_results WHERE created_at < ?", (limit,)).rowcount,
                       "registry_records": conn.execute("DELETE FROM registry_records WHERE fetched_at < ?", (limit,)).rowcount}
        self._flush_deleted()
        return deleted
