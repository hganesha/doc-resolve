"""Case evidence packages as Delta tables: one row per record, typed columns plus the full JSON.

Each table is keyed by case_id and its record IDs. Writing a case is a MERGE per
table that updates and inserts the case's rows and deletes any of the case's
rows no longer in the package, so a table always holds the latest package of
each case. Tables are not written in one transaction: every row carries the
package's export_id, and kyc_cases is written last, so a reader that joins on
kyc_cases.export_id sees one consistent package per case.

Two writers share the table definitions and MERGE conditions:
* LocalDeltaSink: Delta tables in a local directory (deltalake package).
* SparkSink: Unity Catalog tables on Databricks, through spark.sql.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .util import digest, json_text, stable_id

SCHEMA_VERSION = "1"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,254}")
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")

# Column types: string, boolean, int, bigint, double, timestamp, array<string>.
# Every table also has export_id (the package digest) and, except kyc_cases, payload (the record's JSON).
TABLES: dict[str, dict[str, Any]] = {
    "kyc_cases": {
        "comment": "One row per case: the latest exported evidence package and its status.",
        "keys": ["case_id"],
        "columns": [("case_id", "string"), ("status", "string"), ("exported_at", "timestamp"), ("schema_version", "string"), ("document_count", "int"), ("fact_count", "int"), ("open_issue_count", "int"), ("material_gap_count", "int"), ("pending_run_count", "int"), ("disclaimer", "string")],
    },
    "kyc_documents": {
        "comment": "Documents in the case and the run whose records are current.",
        "keys": ["case_id", "document_id"],
        "columns": [("case_id", "string"), ("document_id", "string"), ("current_run_id", "string"), ("profile_id", "string"), ("family", "string"), ("filename", "string"), ("format", "string"), ("sha256", "string"), ("bytes", "bigint"), ("page_count", "int"), ("parser", "string")],
    },
    "kyc_runs": {
        "comment": "Current runs, and runs paused for review or failed that are not yet current.",
        "keys": ["case_id", "run_id"],
        "columns": [("case_id", "string"), ("run_id", "string"), ("document_id", "string"), ("is_current", "boolean"), ("status", "string"), ("profile_id", "string"), ("policy_id", "string"), ("extractor", "string"), ("started_at", "timestamp"), ("finished_at", "timestamp"), ("fact_count", "int"), ("open_issue_count", "int"), ("error", "string")],
    },
    "kyc_facts": {
        "comment": "Extracted, source-matched claims; all pending human verification. Sensitive.",
        "keys": ["case_id", "run_id", "fact_id"],
        "columns": [("case_id", "string"), ("run_id", "string"), ("fact_id", "string"), ("document_id", "string"), ("category", "string"), ("predicate", "string"), ("subject", "string"), ("object", "string"), ("value", "string"), ("value_number", "double"), ("unit", "string"), ("effective_date", "string"), ("date_role", "string"), ("claim_type", "string"), ("qualifier", "string"), ("subject_entity_id", "string"), ("object_entity_id", "string"), ("extractor", "string"), ("verification_status", "string"), ("evidence_ids", "array<string>")],
    },
    "kyc_evidence": {
        "comment": "Exact quotes and their locations in the source documents. Sensitive.",
        "keys": ["case_id", "run_id", "evidence_id"],
        "columns": [("case_id", "string"), ("run_id", "string"), ("evidence_id", "string"), ("document_id", "string"), ("segment_id", "string"), ("page", "int"), ("locator", "string"), ("section_path", "array<string>"), ("quote", "string"), ("start_offset", "int"), ("end_offset", "int")],
    },
    "kyc_entities": {
        "comment": "Case entities (companies, groups, people) resolved across the case's documents. Sensitive.",
        "keys": ["case_id", "entity_id"],
        "columns": [("case_id", "string"), ("entity_id", "string"), ("entity_type", "string"), ("name", "string"), ("names", "array<string>"), ("birth_dates", "array<string>"), ("jurisdictions", "array<string>"), ("identifiers", "array<string>"), ("match_basis", "string"), ("document_ids", "array<string>")],
    },
    "kyc_issues": {
        "comment": "Issues raised for review, with their review status.",
        "keys": ["case_id", "issue_id"],
        "columns": [("case_id", "string"), ("issue_id", "string"), ("run_id", "string"), ("document_id", "string"), ("code", "string"), ("severity", "string"), ("status", "string"), ("message", "string"), ("categories", "array<string>"), ("fact_ids", "array<string>")],
    },
    "kyc_findings": {
        "comment": "Informational findings and review outcomes.",
        "keys": ["case_id", "finding_id"],
        "columns": [("case_id", "string"), ("finding_id", "string"), ("run_id", "string"), ("type", "string"), ("status", "string"), ("category", "string"), ("issue_id", "string"), ("description", "string"), ("fact_ids", "array<string>")],
    },
    "kyc_coverage": {
        "comment": "Per run, whether each policy category was found, reviewed or left open.",
        "keys": ["case_id", "run_id", "category"],
        "columns": [("case_id", "string"), ("run_id", "string"), ("category", "string"), ("document_id", "string"), ("status", "string"), ("material", "boolean"), ("fact_count", "int"), ("chunks_total", "int"), ("chunks_processed", "int"), ("chunks_not_reviewed", "int"), ("source_fully_parsed", "boolean"), ("review_method", "string")],
    },
    "kyc_ownership_edges": {
        "comment": "Ownership and control edges between case entities, with stake ranges.",
        "keys": ["case_id", "edge_id"],
        "columns": [("case_id", "string"), ("edge_id", "string"), ("holder_entity_id", "string"), ("held_entity_id", "string"), ("type", "string"), ("status", "string"), ("percent_low", "double"), ("percent_high", "double"), ("as_of", "string"), ("flags", "array<string>"), ("fact_ids", "array<string>")],
    },
    "kyc_beneficial_owners": {
        "comment": "Each company's reported beneficial owners and whether they meet the reporting threshold. Sensitive.",
        "keys": ["case_id", "company_entity_id", "owner_entity_id"],
        "columns": [("case_id", "string"), ("company_entity_id", "string"), ("company_name", "string"), ("company_status", "string"), ("owner_entity_id", "string"), ("owner_name", "string"), ("meets_threshold", "string"), ("interest_low", "double"), ("interest_high", "double"), ("bases", "array<string>"), ("flags", "array<string>")],
    },
    "kyc_screening_checks": {
        "comment": "Sanctions-list screening of case entities against local list files.",
        "keys": ["case_id", "check_id"],
        "columns": [("case_id", "string"), ("check_id", "string"), ("entity_id", "string"), ("entity_type", "string"), ("name", "string"), ("result", "string"), ("match_count", "int"), ("list_sha256", "array<string>")],
    },
    "kyc_review_decisions": {
        "comment": "Human review decisions. Reviewer IDs are self-reported, not authenticated.",
        "keys": ["case_id", "run_id", "issue_id"],
        "columns": [("case_id", "string"), ("run_id", "string"), ("issue_id", "string"), ("action", "string"), ("reason", "string"), ("reviewer_id", "string"), ("reviewer_identity_authenticated", "boolean"), ("reviewed_at", "timestamp"), ("selected_fact_ids", "array<string>")],
    },
    "kyc_audit_events": {
        "comment": "The evidence store's audit events for the case. A copy, not a tamper-evident log.",
        "keys": ["case_id", "event_id"],
        "columns": [("case_id", "string"), ("event_id", "string"), ("run_id", "string"), ("action", "string"), ("happened_at", "timestamp")],
    },
}
# kyc_cases last: its export_id marks the package as fully written.
WRITE_ORDER = [t for t in TABLES if t != "kyc_cases"] + ["kyc_cases"]


def columns(table: str) -> list[tuple[str, str]]:
    extra = [("export_id", "string")] + ([("payload", "string")] if table != "kyc_cases" else [])
    return TABLES[table]["columns"] + extra


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _str(value: Any) -> str | None:
    return None if value is None else value if isinstance(value, str) else json_text(value)


def _strs(values: Iterable[Any] | None) -> list[str]:
    return [_str(v) for v in values or [] if v is not None]


def export_id(package: dict) -> str:
    """Digest of the package: identical packages have the same export_id."""
    return "export_" + digest(json_text(package))[:32]


def case_tables(package: dict, exported_at: str) -> dict[str, list[dict]]:
    """Rows for every table from one case's evidence package (EvidenceStore.export_case)."""
    case_id = package["case_id"]
    eid = export_id(package)
    run_docs = {r["run_id"]: r["document_id"] for r in package["runs"]}
    graph = package.get("ownership_graph") or {}

    def row(table: str, record: dict | None, **values: Any) -> dict:
        out = {"case_id": case_id, **values, "export_id": eid}
        if table != "kyc_cases":
            out["payload"] = json_text(record)
        return out

    current = [(r, True) for r in package["runs"]] + [(r, False) for r in package.get("pending_runs", [])]
    tables: dict[str, list[dict]] = {
        "kyc_documents": [row("kyc_documents", d, document_id=d["document_id"], current_run_id=d["current_run_id"], profile_id=d.get("profile_id"), family=d.get("family"), filename=d.get("filename"), format=d.get("format"), sha256=d.get("sha256"), bytes=d.get("bytes"), page_count=d.get("page_count"), parser=(d.get("parser") or {}).get("name")) for d in package["documents"]],
        "kyc_runs": [row("kyc_runs", r, run_id=r["run_id"], document_id=r.get("document_id"), is_current=is_current, status=r.get("status"), profile_id=r.get("profile_id"), policy_id=r.get("policy_id"), extractor=(r.get("summary") or {}).get("extractor"), started_at=_ts(r.get("started_at")), finished_at=_ts(r.get("finished_at")), fact_count=(r.get("summary") or {}).get("fact_count"), open_issue_count=(r.get("summary") or {}).get("open_issue_count"), error=r.get("error")) for r, is_current in current],
        "kyc_facts": [row("kyc_facts", f, run_id=f["run_id"], fact_id=f["fact_id"], document_id=f.get("document_id"), category=f.get("category"), predicate=f.get("predicate"), subject=f.get("subject"), object=f.get("object"), value=_str(f.get("value")), value_number=_num(f.get("value")), unit=f.get("unit"), effective_date=f.get("effective_date"), date_role=f.get("date_role"), claim_type=f.get("claim_type"), qualifier=f.get("qualifier"), subject_entity_id=(f.get("case_entity_ids") or {}).get("subject"), object_entity_id=(f.get("case_entity_ids") or {}).get("object"), extractor=f.get("extractor"), verification_status=f.get("verification_status"), evidence_ids=_strs(f.get("evidence_ids"))) for f in package["facts"]],
        "kyc_evidence": [row("kyc_evidence", e, run_id=e["run_id"], evidence_id=e["evidence_id"], document_id=e.get("document_id"), segment_id=e.get("segment_id"), page=e.get("page"), locator=e.get("locator"), section_path=_strs(e.get("section_path")), quote=e.get("quote"), start_offset=e.get("start"), end_offset=e.get("end")) for e in package["evidence"]],
        "kyc_entities": [row("kyc_entities", e, entity_id=e["entity_id"], entity_type=e.get("entity_type"), name=e.get("name"), names=sorted({n["name"] if isinstance(n, dict) else n for n in e.get("names") or []}), birth_dates=_strs(e.get("birth_dates")), jurisdictions=_strs(e.get("jurisdictions")), identifiers=_strs(e.get("identifiers")), match_basis=e.get("match_basis"), document_ids=_strs(e.get("documents"))) for e in package["entities"]],
        "kyc_issues": [row("kyc_issues", i, issue_id=i["issue_id"], run_id=i.get("run_id"), document_id=i.get("document_id") or run_docs.get(i.get("run_id")), code=i.get("code"), severity=i.get("severity"), status=i.get("status"), message=i.get("message"), categories=_strs(i.get("categories")), fact_ids=_strs(i.get("fact_ids"))) for i in package["issues"]],
        "kyc_findings": [row("kyc_findings", f, finding_id=f["finding_id"], run_id=f.get("run_id"), type=f.get("type"), status=f.get("status"), category=f.get("category"), issue_id=f.get("issue_id"), description=f.get("description"), fact_ids=_strs(f.get("fact_ids"))) for f in package["findings"]],
        "kyc_coverage": [row("kyc_coverage", c, run_id=c["run_id"], category=c["category"], document_id=run_docs.get(c["run_id"]), status=c.get("status"), material=c.get("material"), fact_count=c.get("fact_count"), chunks_total=c.get("chunks_total"), chunks_processed=c.get("chunks_processed"), chunks_not_reviewed=c.get("chunks_not_reviewed"), source_fully_parsed=c.get("source_fully_parsed"), review_method=c.get("review_method")) for c in package["coverage"]],
        "kyc_ownership_edges": [row("kyc_ownership_edges", e, edge_id=e["edge_id"], holder_entity_id=e.get("holder"), held_entity_id=e.get("held"), type=e.get("type"), status=e.get("status"), percent_low=_num((e.get("percent") or {}).get("low")), percent_high=_num((e.get("percent") or {}).get("high")), as_of=e.get("as_of"), flags=_strs(e.get("flags")), fact_ids=_strs(e.get("fact_ids"))) for e in graph.get("edges", [])],
        "kyc_beneficial_owners": [row("kyc_beneficial_owners", o, company_entity_id=b["company"], company_name=b.get("name"), company_status=b.get("status"), owner_entity_id=o["person"], owner_name=o.get("name"), meets_threshold=o.get("meets_threshold"), interest_low=_num((o.get("interest") or {}).get("low")), interest_high=_num((o.get("interest") or {}).get("high")), bases=_strs(o.get("bases")), flags=_strs(o.get("flags"))) for b in graph.get("beneficial_owners", []) for o in b.get("owners", [])],
        # Each run screens the whole case, so identical checks repeat across runs; keep one.
        "kyc_screening_checks": [row("kyc_screening_checks", c, check_id=stable_id("check", c), entity_id=c.get("entity_id"), entity_type=c.get("entity_type"), name=c.get("name"), result=c.get("result"), match_count=len(c.get("matches") or []), list_sha256=_strs(sorted({l.get("sha256") for l in c.get("lists") or [] if l.get("sha256")}))) for c in {stable_id("check", c): c for c in package.get("screening_checks", [])}.values()],
        "kyc_review_decisions": [row("kyc_review_decisions", r, run_id=r["run_id"], issue_id=r["issue_id"], action=r.get("action"), reason=r.get("reason"), reviewer_id=r.get("reviewer_id"), reviewer_identity_authenticated=r.get("reviewer_identity_authenticated"), reviewed_at=_ts(r.get("reviewed_at")), selected_fact_ids=_strs(r.get("selected_fact_ids"))) for r in package.get("review_decisions", [])],
        "kyc_audit_events": [row("kyc_audit_events", a, event_id=a["event_id"], run_id=a.get("run_id"), action=a.get("action"), happened_at=_ts(a.get("happened_at"))) for a in package.get("audit_events", [])],
    }
    unresolved = package.get("unresolved") or {}
    tables["kyc_cases"] = [row("kyc_cases", None, status=package["status"], exported_at=_ts(exported_at), schema_version=SCHEMA_VERSION, document_count=len(package["documents"]), fact_count=len(package["facts"]), open_issue_count=len(unresolved.get("open_issue_ids", [])), material_gap_count=len(unresolved.get("material_requirement_gaps", [])), pending_run_count=len(package.get("pending_runs", [])), disclaimer=package.get("disclaimer"))]
    for table, rows in tables.items():
        names = [c for c, _ in columns(table)]
        keys = TABLES[table]["keys"]
        seen: set[tuple] = set()
        for r in rows:
            if list(r) != names:
                raise ValueError(f"Row for {table} does not match its columns")
            key = tuple(r[k] for k in keys)
            if None in key or key in seen:
                raise ValueError(f"Missing or duplicate key {key} in {table}")
            seen.add(key)
    return tables


def _check_identifier(value: str, what: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{what} must be letters, digits and underscores, starting with a letter or underscore")
    return value


def _case_literal(case_id: str) -> str:
    # Case IDs are restricted to letters, digits, dots, underscores and hyphens, so quoting is safe.
    if not CASE_ID.fullmatch(case_id):
        raise ValueError("Invalid case ID")
    return f"'{case_id}'"


def merge_condition(table: str) -> str:
    return " AND ".join(f"t.{k} = s.{k}" for k in TABLES[table]["keys"])


def delete_condition(case_id: str) -> str:
    return f"t.case_id = {_case_literal(case_id)}"


def table_ddl(catalog: str, schema: str) -> list[str]:
    """CREATE TABLE IF NOT EXISTS statements for Unity Catalog."""
    prefix = f"`{_check_identifier(catalog, 'Catalog')}`.`{_check_identifier(schema, 'Schema')}`"
    statements = []
    for table, spec in TABLES.items():
        cols = ",\n  ".join(f"{name} {kind.upper()}" + (" NOT NULL" if name in spec["keys"] else "") for name, kind in columns(table))
        comment = spec["comment"].replace("'", "")
        statements.append(f"CREATE TABLE IF NOT EXISTS {prefix}.`{table}` (\n  {cols}\n) USING DELTA\nCOMMENT '{comment}'\nTBLPROPERTIES ('kyc.schema_version' = '{SCHEMA_VERSION}', 'delta.enableChangeDataFeed' = 'true')")
    return statements


def merge_sql(catalog: str, schema: str, table: str, source_view: str, case_id: str) -> str:
    target = f"`{_check_identifier(catalog, 'Catalog')}`.`{_check_identifier(schema, 'Schema')}`.`{table}`"
    return (f"MERGE INTO {target} AS t\nUSING `{_check_identifier(source_view, 'View')}` AS s\nON {merge_condition(table)}\n"
            f"WHEN MATCHED THEN UPDATE SET *\nWHEN NOT MATCHED THEN INSERT *\nWHEN NOT MATCHED BY SOURCE AND {delete_condition(case_id)} THEN DELETE")


def _arrow_schema(table: str):
    import pyarrow as pa
    kinds = {"string": pa.string(), "boolean": pa.bool_(), "int": pa.int32(), "bigint": pa.int64(), "double": pa.float64(), "timestamp": pa.timestamp("us", tz="UTC"), "array<string>": pa.list_(pa.string())}
    keys = set(TABLES[table]["keys"])
    return pa.schema([pa.field(name, kinds[kind], nullable=name not in keys) for name, kind in columns(table)])


class LocalDeltaSink:
    """Delta tables under a local directory, one folder per table."""

    def __init__(self, root: str | Path):
        try:
            import deltalake  # noqa: F401
        except ImportError:
            raise ValueError("Local Delta output needs the deltalake package: pip install -e '.[delta]'") from None
        self.root = Path(root).expanduser().resolve()

    def _path(self, table: str) -> str:
        return str(self.root / table)

    def setup(self) -> None:
        from deltalake import DeltaTable
        self.root.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            DeltaTable.create(self._path(table), schema=_arrow_schema(table), mode="ignore", name=table, description=TABLES[table]["comment"], configuration={"delta.enableChangeDataFeed": "true"})
            existing = [f.name for f in DeltaTable(self._path(table)).schema().fields]
            if existing != [c for c, _ in columns(table)]:
                raise ValueError(f"Table {table} has different columns from schema version {SCHEMA_VERSION}; migrate or recreate it")

    def _query(self, table: str, sql: str) -> list[dict]:
        # DataFusion, not to_pyarrow_table: pyarrow dataset reads of Delta tables can hang the interpreter at exit.
        import pyarrow as pa
        from deltalake import DeltaTable, QueryBuilder
        return pa.table(QueryBuilder().register("t", DeltaTable(self._path(table))).execute(sql).read_all()).to_pylist()

    def current_export_id(self, case_id: str) -> str | None:
        rows = self._query("kyc_cases", f"SELECT export_id FROM t WHERE case_id = {_case_literal(case_id)}")
        return rows[0]["export_id"] if rows else None

    def write(self, table: str, rows: list[dict], case_id: str) -> None:
        import pyarrow as pa
        from deltalake import DeltaTable
        source = pa.Table.from_pylist(rows, schema=_arrow_schema(table))
        (DeltaTable(self._path(table)).merge(source, predicate=merge_condition(table), source_alias="s", target_alias="t")
            .when_matched_update_all().when_not_matched_insert_all().when_not_matched_by_source_delete(predicate=delete_condition(case_id)).execute())

    def read(self, table: str) -> list[dict]:
        return self._query(table, "SELECT * FROM t")


class SparkSink:
    """Unity Catalog tables through a Spark session (on Databricks). The schema must already exist."""

    def __init__(self, spark: Any, catalog: str, schema: str):
        self.spark, self.catalog, self.schema = spark, _check_identifier(catalog, "Catalog"), _check_identifier(schema, "Schema")

    def _name(self, table: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{table}`"

    def setup(self) -> None:
        for statement in table_ddl(self.catalog, self.schema):
            self.spark.sql(statement)
        for table in TABLES:
            existing = list(self.spark.table(self._name(table)).columns)
            if existing != [c for c, _ in columns(table)]:
                raise ValueError(f"Table {table} has different columns from schema version {SCHEMA_VERSION}; migrate or recreate it")

    def current_export_id(self, case_id: str) -> str | None:
        rows = self.spark.sql(f"SELECT export_id FROM {self._name('kyc_cases')} WHERE case_id = :case_id", args={"case_id": case_id}).collect()
        return rows[0]["export_id"] if rows else None

    def write(self, table: str, rows: list[dict], case_id: str) -> None:
        cols = columns(table)
        ddl = ", ".join(f"{name} {kind.upper()}" for name, kind in cols)
        frame = self.spark.createDataFrame([tuple(r[name] for name, _ in cols) for r in rows], schema=ddl)
        view = f"kyc_source_{table}"
        frame.createOrReplaceTempView(view)
        self.spark.sql(merge_sql(self.catalog, self.schema, table, view, case_id))


def write_case(sink: Any, package: dict, exported_at: str, *, force: bool = False) -> dict:
    """Write one case's package to every table. Skipped when the tables already hold this package."""
    tables = case_tables(package, exported_at)
    eid = tables["kyc_cases"][0]["export_id"]
    if not force and sink.current_export_id(package["case_id"]) == eid:
        return {"case_id": package["case_id"], "export_id": eid, "written": False, "rows": {}}
    for table in WRITE_ORDER:
        sink.write(table, tables[table], package["case_id"])
    return {"case_id": package["case_id"], "export_id": eid, "written": True, "rows": {t: len(tables[t]) for t in WRITE_ORDER}}


def delete_case(sink: Any, case_id: str) -> None:
    """Remove every row of the case: kyc_cases first, so readers stop seeing it at once."""
    for table in ["kyc_cases"] + [t for t in WRITE_ORDER if t != "kyc_cases"]:
        sink.write(table, [], case_id)


def ddl_script(catalog: str, schema: str) -> str:
    return ";\n\n".join(table_ddl(catalog, schema)) + ";\n"


def payload(row: dict) -> dict | None:
    return json.loads(row["payload"]) if row.get("payload") else None
