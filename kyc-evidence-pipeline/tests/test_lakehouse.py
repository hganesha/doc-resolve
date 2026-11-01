from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from conftest import ROOT, defer_all
from kyc_pipeline import cli as cli_module
from kyc_pipeline.databricks_job import _copy_in, _copy_out, main as job_main, run_job
from kyc_pipeline.lakehouse import TABLES, WRITE_ORDER, LocalDeltaSink, SparkSink, case_tables, columns, ddl_script, merge_sql, write_case
from kyc_pipeline.store import EvidenceStore

pytest.importorskip("deltalake")

EXAMPLES = ROOT / "examples"
LIST = str(EXAMPLES / "synthetic_screening_list.csv")


def _case(cli, paths, *names: str) -> dict:
    """Ingest and defer-resume each document, then return the case package."""
    for name in names:
        result = cli("ingest", EXAMPLES / name, "--case-id", "birch", "--screening-list", LIST, "--db", paths["db"], "--checkpoint-db", paths["cp"])
        while result["status"] == "PAUSED_FOR_REVIEW":
            result = cli("resume", "--run-id", result["run_id"], "--decision", defer_all(result, paths["tmp"]), "--db", paths["db"], "--checkpoint-db", paths["cp"])
    return EvidenceStore(paths["db"]).export_case("birch")


def test_case_tables_flatten_the_package_with_typed_columns(cli, paths):
    package = _case(cli, paths, "synthetic_registry_extract.txt", "synthetic_beneficial_ownership_declaration.txt")
    tables = case_tables(package, "2026-11-01T10:00:00+00:00")
    assert set(tables) == set(TABLES) and WRITE_ORDER[-1] == "kyc_cases"
    for table, rows in tables.items():
        assert all(list(r) == [c for c, _ in columns(table)] for r in rows)
    export = tables["kyc_cases"][0]["export_id"]
    assert {r["export_id"] for rows in tables.values() for r in rows} == {export}
    assert len(tables["kyc_facts"]) == len(package["facts"]) and len(tables["kyc_evidence"]) == len(package["evidence"])
    # Facts carry case entity IDs; numeric values also get a number column; the payload is the full record.
    owner = next(r for r in tables["kyc_facts"] if r["predicate"] == "is_beneficial_owner_of")
    assert owner["subject_entity_id"].startswith("entity_") and owner["object_entity_id"].startswith("entity_")
    assert owner["value_number"] == float(owner["value"]) and json.loads(owner["payload"])["fact_id"] == owner["fact_id"]
    owners = {r["meets_threshold"]: r for r in tables["kyc_beneficial_owners"]}
    assert "smith" in owners["YES"]["owner_name"].lower() and owners["YES"]["interest_low"] == 40.0
    # Each run screens the whole case, so identical checks are kept once.
    assert len({r["check_id"] for r in tables["kyc_screening_checks"]}) == len(tables["kyc_screening_checks"])
    assert len(tables["kyc_screening_checks"]) < len(package["screening_checks"])
    assert tables["kyc_cases"][0]["document_count"] == 2
    # The same package always gets the same export_id.
    assert case_tables(package, "2026-11-02T10:00:00+00:00")["kyc_cases"][0]["export_id"] == export


def test_local_delta_tables_hold_the_latest_package_of_each_case(cli, paths, tmp_path):
    package = _case(cli, paths, "synthetic_registry_extract.txt", "synthetic_beneficial_ownership_declaration.txt")
    sink = LocalDeltaSink(tmp_path / "delta")
    sink.setup()
    sink.setup()  # idempotent
    first = write_case(sink, package, "2026-11-01T10:00:00+00:00")
    assert first["written"] and first["rows"]["kyc_facts"] == len(package["facts"])
    assert not write_case(sink, package, "2026-11-01T11:00:00+00:00")["written"]  # unchanged package
    other = {**package, "case_id": "other"}
    assert write_case(sink, other, "2026-11-01T10:00:00+00:00")["written"]
    # A case that shrinks loses its old rows; the other case is untouched.
    smaller = {**package, "facts": package["facts"][:3]}
    write_case(sink, smaller, "2026-11-01T12:00:00+00:00")
    facts = sink.read("kyc_facts")
    assert sum(r["case_id"] == "birch" for r in facts) == 3 and sum(r["case_id"] == "other" for r in facts) == len(package["facts"])
    cases = {r["case_id"]: r for r in sink.read("kyc_cases")}
    assert set(cases) == {"birch", "other"}
    assert {r["export_id"] for r in facts if r["case_id"] == "birch"} == {cases["birch"]["export_id"]}
    assert sink.current_export_id("birch") == cases["birch"]["export_id"] and sink.current_export_id("missing") is None


def test_local_delta_refuses_tables_with_other_columns(tmp_path):
    import pyarrow as pa
    from deltalake import write_deltalake
    write_deltalake(str(tmp_path / "kyc_cases"), pa.table({"case_id": ["x"]}))
    with pytest.raises(ValueError, match="different columns"):
        LocalDeltaSink(tmp_path).setup()


def test_unity_catalog_sql_is_quoted_and_merges_one_case():
    sql = merge_sql("main", "kyc", "kyc_facts", "kyc_source_kyc_facts", "case-1.a")
    assert "MERGE INTO `main`.`kyc`.`kyc_facts` AS t" in sql
    assert "ON t.case_id = s.case_id AND t.run_id = s.run_id AND t.fact_id = s.fact_id" in sql
    assert "WHEN NOT MATCHED BY SOURCE AND t.case_id = 'case-1.a' THEN DELETE" in sql
    ddl = ddl_script("main", "kyc")
    assert ddl.count("CREATE TABLE IF NOT EXISTS") == len(TABLES) and "fact_id STRING NOT NULL" in ddl and "evidence_ids ARRAY<STRING>" in ddl
    for bad in ("main; DROP TABLE x", "a.b", "`x`"):
        with pytest.raises(ValueError):
            merge_sql(bad, "kyc", "kyc_facts", "v", "c")
    with pytest.raises(ValueError):
        merge_sql("main", "kyc", "kyc_facts", "v", "x' OR '1'='1")


class _FakeSpark:
    """Records statements; tables report the expected columns and kyc_cases is empty."""

    def __init__(self):
        self.statements, self.frames = [], []

    def sql(self, statement, args=None):
        self.statements.append((statement, args))
        return self

    def collect(self):
        return []

    def table(self, name):
        table = name.split(".")[-1].strip("`")
        return type("T", (), {"columns": [c for c, _ in columns(table)]})()

    def createDataFrame(self, rows, schema):
        self.frames.append((rows, schema))
        return type("F", (), {"createOrReplaceTempView": lambda _self, view: None})()


def test_spark_sink_creates_tables_and_merges_rows_in_column_order(cli, paths):
    package = _case(cli, paths, "synthetic_registry_extract.txt")
    spark = _FakeSpark()
    sink = SparkSink(spark, "main", "kyc")
    sink.setup()
    assert sum("CREATE TABLE IF NOT EXISTS" in s for s, _ in spark.statements) == len(TABLES)
    result = write_case(sink, package, "2026-11-01T10:00:00+00:00")
    assert result["written"]
    lookup = next((s, a) for s, a in spark.statements if s.startswith("SELECT export_id"))
    assert lookup[1] == {"case_id": "birch"}  # a parameter, not interpolated
    merges = [s for s, _ in spark.statements if s.startswith("MERGE INTO")]
    assert [m.split("`")[5] for m in merges] == WRITE_ORDER
    rows, schema = spark.frames[[m.split("`")[5] for m in merges].index("kyc_facts")]
    assert schema.startswith("case_id STRING, run_id STRING, fact_id STRING") and "evidence_ids ARRAY<STRING>" in schema
    assert all(isinstance(r, tuple) and len(r) == len(columns("kyc_facts")) for r in rows)


def _decide(job: dict) -> None:
    for case in job["cases"]:
        for waiting in case["waiting_for_review"]:
            decision = {"reviewer_id": "TEST_ANALYST", "decisions": [{"issue_id": i["issue_id"], "action": "defer", "reason": "Fictional test case."} for i in waiting["issues"]]}
            Path(waiting["decision_file"]).write_text(json.dumps(decision))


def test_job_ingests_in_order_waits_for_review_and_writes_tables(tmp_path):
    inbox, state = tmp_path / "inbox", tmp_path / "state"
    folder = inbox / "cases" / "birch"
    folder.mkdir(parents=True)
    (inbox / "decisions").mkdir()
    shutil.copy(EXAMPLES / "synthetic_registry_extract.txt", folder / "01_registry.txt")
    shutil.copy(EXAMPLES / "synthetic_beneficial_ownership_declaration.txt", folder / "02_declaration.txt")
    (folder / "notes.docx").write_bytes(b"not supported natively")
    (folder / ".DS_Store").write_bytes(b"")
    sink = LocalDeltaSink(tmp_path / "delta")

    first = run_job(inbox, state, sink, screening_lists=[LIST])
    case = first["cases"][0]
    # The first document pauses, so the second waits for the review.
    assert first["status"] == "WAITING_FOR_REVIEW" and [i["file"] for i in case["ingested"]] == ["01_registry.txt"]
    assert case["skipped"] == [{"file": "notes.docx", "reason": "unsupported file type .docx"}]
    waiting = case["waiting_for_review"][0]
    assert waiting["issues"] and waiting["decision_file"].endswith(waiting["run_id"] + ".json")
    assert (state / "evidence.sqlite").is_file() and len(list((state / "reports").iterdir())) == 1
    assert {(r["is_current"], r["status"]) for r in sink.read("kyc_runs")} == {(False, "REVIEW_REQUIRED")}

    assert run_job(inbox, state, sink, screening_lists=[LIST])["status"] == "WAITING_FOR_REVIEW"  # no decision yet
    _decide(first)
    second = run_job(inbox, state, sink, screening_lists=[LIST])
    assert second["cases"][0]["resumed"][0]["status"] == "PERSISTED"
    assert [i["file"] for i in second["cases"][0]["ingested"]] == ["02_declaration.txt"]
    _decide(second)
    third = run_job(inbox, state, sink, screening_lists=[LIST])
    assert third["status"] == "OK" and third["tables"][0]["written"]
    assert {r["is_current"] for r in sink.read("kyc_runs")} == {True} and sink.read("kyc_cases")[0]["document_count"] == 2
    # Nothing new: no ingest and no table write.
    fourth = run_job(inbox, state, sink, screening_lists=[LIST])
    assert not fourth["cases"][0]["ingested"] and not fourth["tables"][0]["written"]


def test_job_refuses_to_overwrite_state_changed_by_another_writer(tmp_path):
    state, work = tmp_path / "state", tmp_path / "work"
    state.mkdir()
    work.mkdir()
    EvidenceStore(state / "evidence.sqlite")
    digests = _copy_in(state, work)
    (state / "evidence.sqlite").write_bytes(b"changed elsewhere")
    with pytest.raises(ValueError, match="changed while this job ran"):
        _copy_out(work, state, digests)
    assert (state / "evidence.sqlite").read_bytes() == b"changed elsewhere"


def test_job_command_line_reports_errors(tmp_path, capsys):
    (tmp_path / "inbox" / "cases" / "bad name").mkdir(parents=True)
    assert job_main(["--inbox", str(tmp_path / "inbox"), "--state", str(tmp_path / "state"), "--local-delta", str(tmp_path / "delta")]) == 1
    assert "valid case IDs" in json.loads(capsys.readouterr().err)["message"]
    assert job_main(["--inbox", str(tmp_path / "inbox"), "--state", str(tmp_path / "state"), "--catalog", "main"]) == 1
    assert "--schema" in json.loads(capsys.readouterr().err)["message"]


def test_cli_lakehouse_commands(cli, paths, tmp_path, capsys):
    _case(cli, paths, "synthetic_registry_extract.txt")
    result = cli("lakehouse", "--case-id", "birch", "--delta-dir", tmp_path / "delta", "--db", paths["db"])
    assert result["written"] and result["rows"]["kyc_cases"] == 1
    assert not cli("lakehouse", "--case-id", "birch", "--delta-dir", tmp_path / "delta", "--db", paths["db"])["written"]
    assert cli("lakehouse", "--case-id", "birch", "--delta-dir", tmp_path / "delta", "--db", paths["db"], "--force")["written"]
    assert cli("lakehouse", "--case-id", "nobody", "--delta-dir", tmp_path / "delta", "--db", paths["db"], expected=1)["status"] == "ERROR"
    assert cli_module.main(["lakehouse-ddl", "--catalog", "main", "--schema", "kyc"]) == 0
    assert capsys.readouterr().out.count("CREATE TABLE IF NOT EXISTS `main`.`kyc`.") == len(TABLES)
