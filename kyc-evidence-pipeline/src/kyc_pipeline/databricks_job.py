"""Batch job: process the case folders in an inbox and write each case's package to Delta tables.

Folder layout (Unity Catalog volumes on Databricks, or any local directories):

    <inbox>/cases/<case_id>/<document files>   documents to ingest, oldest name first
    <inbox>/decisions/<run_id>.json            review decisions for paused runs (resume format)
    <state>/evidence.sqlite, checkpoints.sqlite the evidence store and graph checkpoints
    <state>/reports/<job time>.json            what each job run did

Per case: paused runs that have a decision file are resumed first. While a run in
the case is still paused, no new documents are ingested, so cross-document checks
see the reviewed document. Otherwise new documents (by SHA-256) are ingested one
at a time, stopping at the first that pauses. Then the case's package is written
to the tables if it changed.

SQLite needs random writes and file locks, which volumes do not provide, so the
databases are copied to local disk, processed there, and copied back. Run one
job at a time (the bundle sets max_concurrent_runs: 1); the copy back also fails
if the stored databases changed while the job ran.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

from . import cli
from .lakehouse import CASE_ID, LocalDeltaSink, SparkSink, write_case
from .store import EvidenceStore
from .util import digest, json_text, utc_now

STATE_FILES = ("evidence.sqlite", "checkpoints.sqlite")
NATIVE_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".pdf"}


def _file_digest(path: Path) -> str | None:
    return digest(path.read_bytes()) if path.is_file() else None


def _copy_in(state: Path, work: Path) -> dict[str, str | None]:
    """Copy the stored databases to local disk; returns their digests for the copy-back check."""
    digests = {}
    for name in STATE_FILES:
        source = state / name
        digests[name] = _file_digest(source)
        if digests[name]:
            shutil.copyfile(source, work / name)
    return digests


def _copy_out(work: Path, state: Path, digests: dict[str, str | None]) -> None:
    """Write consistent copies back: SQLite backup to local disk, then a sequential copy and rename."""
    for name in STATE_FILES:
        if _file_digest(state / name) != digests[name]:
            raise ValueError(f"{name} in the state folder changed while this job ran; another job may be writing it. Local results were not copied back.")
    state.mkdir(parents=True, exist_ok=True)
    for name in STATE_FILES:
        if not (work / name).is_file():
            continue
        snapshot = work / (name + ".snapshot")
        with closing(sqlite3.connect(work / name)) as src, closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst)
        partial = state / (name + ".partial")
        shutil.copyfile(snapshot, partial)
        partial.replace(state / name)


def _run_cli(argv: list[str]) -> dict:
    return cli.execute(cli.make_parser().parse_args(argv))


def _paused_runs(store: EvidenceStore, case_id: str) -> list[dict]:
    return [r for r in store.export_case(case_id)["pending_runs"] if r["status"] == "REVIEW_REQUIRED"]


def process_case(case_id: str, folder: Path, decisions: Path, db: Path, checkpoints: Path, screening_lists: list[str]) -> dict:
    """Resume decided runs, then ingest new documents. Returns what happened to each."""
    store = EvidenceStore(db)
    paths = ["--db", str(db), "--checkpoint-db", str(checkpoints)]
    report: dict[str, Any] = {"case_id": case_id, "resumed": [], "ingested": [], "waiting_for_review": [], "failed": [], "skipped": []}
    for run in _paused_runs(store, case_id):
        decision = decisions / f"{run['run_id']}.json"
        if not decision.is_file():
            continue
        try:
            result = _run_cli(["resume", "--run-id", run["run_id"], "--decision", str(decision), *paths])
            report["resumed"].append({"run_id": run["run_id"], "status": result["status"]})
        except ValueError as exc:
            report["failed"].append({"run_id": run["run_id"], "step": "resume", "error": str(exc)})
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    report["skipped"] = [{"file": p.name, "reason": f"unsupported file type {p.suffix or '(none)'}"} for p in files if p.suffix.lower() not in NATIVE_SUFFIXES]
    blocked = _paused_runs(store, case_id)
    if not blocked:
        known = {d["sha256"] for d in store.export_case(case_id)["documents"]}
        for path in (p for p in files if p.suffix.lower() in NATIVE_SUFFIXES):
            sha = digest(path.read_bytes())
            if sha in known:
                continue
            known.add(sha)
            try:
                result = _run_cli(["ingest", str(path), "--case-id", case_id, *[a for p in screening_lists for a in ("--screening-list", p)], *paths])
            except ValueError as exc:
                # Later documents wait too, so documents enter the case in name order.
                report["failed"].append({"file": path.name, "step": "ingest", "error": str(exc)})
                break
            report["ingested"].append({"file": path.name, "run_id": result["run_id"], "status": result["status"]})
            if result["status"] == "PAUSED_FOR_REVIEW":
                break
        blocked = _paused_runs(store, case_id)
    for run in blocked:
        request = _run_cli(["status", "--run-id", run["run_id"], *paths]).get("review_request") or {}
        issues = [{k: i.get(k) for k in ("issue_id", "code", "severity", "message", "allowed_actions")} for i in request.get("issues", [])]
        report["waiting_for_review"].append({"run_id": run["run_id"], "document_id": request.get("document_id"), "issues": issues, "decision_file": str(decisions / f"{run['run_id']}.json")})
    return report


def run_job(inbox: str | Path, state: str | Path, sink: Any, *, case_ids: list[str] | None = None, screening_lists: list[str] | None = None) -> dict:
    inbox, state = Path(inbox), Path(state)
    cases_root = inbox / "cases"
    folders = {p.name: p for p in sorted(cases_root.iterdir()) if p.is_dir() and not p.name.startswith(".")} if cases_root.is_dir() else {}
    for name in case_ids or []:
        if name not in folders:
            raise ValueError(f"No inbox folder for case {name!r}")
    invalid = [name for name in folders if not CASE_ID.fullmatch(name)]
    if invalid:
        raise ValueError(f"Case folder names must be valid case IDs (letters, digits, dots, underscores, hyphens): {', '.join(invalid)}")
    selected = case_ids or sorted(folders)
    started = utc_now()
    job: dict[str, Any] = {"started_at": started, "inbox": str(inbox), "cases": [], "tables": []}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        digests = _copy_in(state, work)
        try:
            sink.setup()
            for case_id in selected:
                job["cases"].append(process_case(case_id, folders[case_id], inbox / "decisions", work / "evidence.sqlite", work / "checkpoints.sqlite", screening_lists or []))
            store = EvidenceStore(work / "evidence.sqlite")
            for case_id in selected:
                package = store.export_case(case_id)
                if package["documents"] or package["pending_runs"]:
                    job["tables"].append(write_case(sink, package, utc_now()))
        finally:
            # The store is the record of what ran; keep it even if a table write failed.
            if (work / "evidence.sqlite").is_file():
                _copy_out(work, state, digests)
    job["finished_at"] = utc_now()
    job["status"] = "FAILED" if any(c["failed"] for c in job["cases"]) else "WAITING_FOR_REVIEW" if any(c["waiting_for_review"] for c in job["cases"]) else "OK"
    reports = state / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / (started.replace(":", "").replace("+", "_") + ".json")).write_text(json_text(job, pretty=True) + "\n", encoding="utf-8")
    return job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kyc-databricks-job", description="Process inbox case folders and write case packages to Delta tables")
    parser.add_argument("--inbox", required=True, help="Folder with cases/<case_id>/ documents and decisions/<run_id>.json")
    parser.add_argument("--state", required=True, help="Folder for the evidence and checkpoint databases and job reports")
    parser.add_argument("--case-id", action="append", type=cli._case, help="Process only this case (repeatable); default: every case folder")
    parser.add_argument("--screening-list", action="append", default=[], metavar="PATH", help="Local sanctions list file passed to every ingest (repeatable)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--catalog", help="Unity Catalog catalog (with --schema): write through the job's Spark session")
    target.add_argument("--local-delta", metavar="DIR", help="Write Delta tables to this local folder instead (deltalake package)")
    parser.add_argument("--schema", help="Unity Catalog schema; it must already exist")
    args = parser.parse_args(argv)
    try:
        if args.catalog:
            if not args.schema:
                raise ValueError("--catalog needs --schema")
            from pyspark.sql import SparkSession
            sink: Any = SparkSink(SparkSession.builder.getOrCreate(), args.catalog, args.schema)
        else:
            sink = LocalDeltaSink(args.local_delta)
        job = run_job(args.inbox, args.state, sink, case_ids=args.case_id, screening_lists=args.screening_list)
    except (ValueError, OSError) as exc:
        sys.stderr.write(json_text({"status": "ERROR", "message": str(exc)}, pretty=True) + "\n")
        return 1
    sys.stdout.write(json_text(job, pretty=True) + "\n")
    return 1 if job["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
