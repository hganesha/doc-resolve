from __future__ import annotations
import base64
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from conftest import ROOT
from kyc_pipeline.credentials import get_secret
from kyc_pipeline.store import EvidenceStore, check_postgres_tls, is_postgres

REGISTRY = ROOT / "examples/synthetic_registry.txt"
SECRET_TEXT = "TX-123456"  # the registration number in the registry extract


def _ingest(cli, paths, case_id="reg", doc=REGISTRY, profile="corporate_registry"):
    return cli("ingest", doc, "--case-id", case_id, "--profile", profile, "--db", paths["db"], "--checkpoint-db", paths["cp"])


def _sql(paths, statement: str) -> None:
    """Run one statement directly against the evidence store, bypassing the pipeline."""
    if is_postgres(paths["db"]):
        import psycopg
        with psycopg.connect(paths["db"], autocommit=True) as conn:
            conn.execute(statement)
    else:
        with closing(sqlite3.connect(paths["db"])) as conn, conn:
            conn.execute(statement)


def _unguarded(paths, statement: str) -> None:
    """Someone with the database owner's rights removes the append-only guard (the store restores it on open)."""
    if is_postgres(paths["db"]):
        _sql(paths, "ALTER TABLE audit_events DISABLE TRIGGER USER")
    else:
        _sql(paths, "DROP TRIGGER audit_events_no_update")
        _sql(paths, "DROP TRIGGER audit_events_no_delete")
    _sql(paths, statement)


def _dump(target: str) -> str:
    """Every row of every table, as text."""
    if is_postgres(target):
        import psycopg
        with psycopg.connect(target) as conn:
            tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()")]
            return "\n".join(str(row) for t in tables for row in conn.execute(f'SELECT * FROM "{t}"'))
    with closing(sqlite3.connect(target)) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return "\n".join(str(row) for t in tables for row in conn.execute(f'SELECT * FROM "{t}"'))


def _raw(paths) -> str:
    """The stores' rows and, on SQLite, the database files' bytes (free pages and write-ahead logs included)."""
    text = _dump(paths["db"]) + _dump(paths["cp"])
    if not is_postgres(paths["db"]):
        for name in (paths["db"], paths["cp"]):
            for f in (Path(name), Path(name + "-wal")):
                if f.is_file():
                    text += f.read_bytes().decode("latin-1")
    return text


def _verify(paths, case_id="reg", *extra, expected=0) -> dict:
    """verify-audit exits 1 with its report on stdout when the check fails."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([sys.executable, "-m", "kyc_pipeline", "verify-audit", "--case-id", case_id, "--db", paths["db"], *map(str, extra)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
    assert proc.returncode == expected, proc.stdout + proc.stderr
    return json.loads(proc.stdout)


def test_audit_chain_verifies_and_detects_edits(cli, paths):
    _ingest(cli, paths)
    ok = _verify(paths)
    assert ok["status"] == "OK" and ok["chained_events"] >= 2 and ok["runs_checked"] == 1
    assert ok["head"]["seq"] == ok["chained_events"]

    # A stored fact rewritten after persistence no longer matches the run's recorded digest.
    _sql(paths, "UPDATE facts SET payload_json = REPLACE(payload_json, 'Texas', 'Delaware')")
    failed = _verify(paths, expected=1)
    assert failed["status"] == "FAILED"
    assert any("differ from those recorded" in p for p in failed["problems"])


def test_audit_events_are_append_only(cli, paths):
    _ingest(cli, paths)
    for statement in ("UPDATE audit_events SET action='X'", "DELETE FROM audit_events"):
        with pytest.raises(Exception, match="append-only"):
            _sql(paths, statement)


def test_rewritten_history_is_caught_by_the_chain_or_an_anchor(cli, paths):
    _ingest(cli, paths)
    anchor = paths["tmp"] / "anchor.json"
    anchor.write_text(json.dumps(_verify(paths)))

    # Removing the newest event leaves a valid chain, but the run's records are now unaudited;
    # an anchor kept elsewhere also shows the chain was cut.
    head = json.loads(anchor.read_text())["head"]["seq"]
    _unguarded(paths, f"DELETE FROM audit_events WHERE case_id = 'reg' AND seq = {head}")
    truncated = _verify(paths, expected=1)
    assert not any("follow" in p or "altered" in p for p in truncated["problems"])
    assert any("no audit event recording their persistence" in p for p in truncated["problems"])
    anchored = _verify(paths, "reg", "--anchor", anchor, expected=1)
    assert any("anchored event" in p for p in anchored["problems"])

    # Editing an earlier event breaks the chain.
    _unguarded(paths, "UPDATE audit_events SET happened_at = '2000-01-01T00:00:00+00:00' WHERE case_id = 'reg' AND seq = 1")
    edited = _verify(paths, expected=1)
    assert any("Event 1" in p and "was altered" in p for p in edited["problems"])


def test_erase_case_removes_content_and_keeps_a_verifiable_log(cli, paths):
    first = _ingest(cli, paths)
    other = _ingest(cli, paths, case_id="other", doc=ROOT / "examples/synthetic_annual_report.txt", profile="financial_annual_report")
    assert SECRET_TEXT in _raw(paths)
    with pytest.raises(AssertionError):
        cli("erase-case", "--case-id", "reg", "--confirm", "nope", "--reason", "test", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    erased = cli("erase-case", "--case-id", "reg", "--confirm", "reg", "--reason", "Erasure request (synthetic test)", "--requested-by", "TEST_ANALYST", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert erased["runs"] == [first["run_id"]]
    assert erased["deleted_rows"]["documents"] == 1
    assert SECRET_TEXT not in _raw(paths)

    package = cli("report", "--case-id", "reg", "--db", paths["db"])
    assert package["documents"] == [] and package["facts"] == []
    assert package["audit_events"][-1]["action"] == "CASE_ERASED"
    assert package["audit_events"][-1]["details"]["requested_by"] == "TEST_ANALYST"
    assert _verify(paths)["status"] == "OK"
    # The other case's paused run and its checkpoint are untouched.
    assert cli("status", "--run-id", other["run_id"], "--db", paths["db"], "--checkpoint-db", paths["cp"])["review_request"]
    assert _verify(paths, "other")["status"] == "OK"
    again = cli("erase-case", "--case-id", "reg", "--confirm", "reg", "--reason", "again", "--db", paths["db"], "--checkpoint-db", paths["cp"], expected=1)
    assert "already erased" in again["message"]


def test_retention_lists_then_applies(cli, paths):
    paused = _ingest(cli, paths, case_id="annual", doc=ROOT / "examples/synthetic_annual_report.txt", profile="financial_annual_report")
    assert paused["status"] == "PAUSED_FOR_REVIEW"
    _ingest(cli, paths)
    common = ("--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert cli("retention", *common)["runs_to_expire"] == []

    # After the unfinished-run period (90 days) the paused run expires; the cases stay.
    later = cli("retention", "--as-of", "2100-01-01", *common)
    assert {c["case_id"] for c in later["cases_to_erase"]} == {"annual", "reg"}
    soon = cli("retention", "--policy-file", _policy(paths, case_days=36500), "--as-of", "2100-01-01", *common)
    assert soon["cases_to_erase"] == [] and [r["run_id"] for r in soon["runs_to_expire"]] == [paused["run_id"]]
    assert soon["applied"] is False
    applied = cli("retention", "--policy-file", _policy(paths, case_days=36500), "--as-of", "2100-01-01", "--apply", *common)
    assert applied["applied"] is True
    status = cli("status", "--run-id", paused["run_id"], *common)
    assert status["run"]["status"] == "EXPIRED" and status["review_request"] is None
    assert _verify(paths, "annual")["status"] == "OK"
    assert cli("report", "--case-id", "reg", "--db", paths["db"])["documents"]

    # After the case period both cases are erased.
    cli("retention", "--as-of", "2100-01-01", "--apply", *common)
    assert cli("report", "--case-id", "reg", "--db", paths["db"])["documents"] == []
    assert _verify(paths)["status"] == "OK"
    assert cli("retention", "--as-of", "2100-01-01", *common)["cases_to_erase"] == []


def _policy(paths, case_days: int) -> Path:
    import yaml
    policy = yaml.safe_load((ROOT / "src/kyc_pipeline/configs/kyc_policy.yaml").read_text())
    policy["retention"]["case_days_after_last_activity"] = case_days
    path = paths["tmp"] / f"policy_{case_days}.yaml"
    path.write_text(yaml.safe_dump(policy))
    return path


def _key() -> str:
    return base64.b64encode(os.urandom(32)).decode()


def test_encrypted_store_hides_content_and_needs_its_key(cli, paths, monkeypatch):
    monkeypatch.setenv("KYC_DATA_KEY", _key())
    paused = _ingest(cli, paths, case_id="annual", doc=ROOT / "examples/synthetic_annual_report.txt", profile="financial_annual_report")
    _ingest(cli, paths)
    raw = _raw(paths)
    assert SECRET_TEXT not in raw and "Aster Holdings" not in raw
    assert "enc1:" in raw
    package = cli("report", "--case-id", "reg", "--db", paths["db"])
    assert any(f["object"] == SECRET_TEXT or SECRET_TEXT in json.dumps(f) for f in package["facts"])
    # The paused run's checkpoint decrypts with the same key.
    assert cli("status", "--run-id", paused["run_id"], "--db", paths["db"], "--checkpoint-db", paths["cp"])["review_request"]
    assert _verify(paths)["status"] == "OK"

    monkeypatch.setenv("KYC_DATA_KEY", _key())
    wrong = cli("report", "--case-id", "reg", "--db", paths["db"], expected=1)
    assert "not the key" in wrong["message"]
    monkeypatch.delenv("KYC_DATA_KEY")
    missing = cli("report", "--case-id", "reg", "--db", paths["db"], expected=1)
    assert "encrypted" in missing["message"]


def test_key_cannot_be_added_to_an_unencrypted_store(cli, paths, monkeypatch):
    _ingest(cli, paths)
    monkeypatch.setenv("KYC_DATA_KEY", _key())
    refused = cli("report", "--case-id", "reg", "--db", paths["db"], expected=1)
    assert "unencrypted data" in refused["message"]


def test_secrets_lookup_order_and_file_permissions(tmp_path, monkeypatch):
    for name in ("DEMO_SECRET", "DEMO_SECRET_FILE", "KYC_SECRETS_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert get_secret("DEMO_SECRET", "fallback") == "fallback"
    monkeypatch.setenv("DEMO_SECRET", "from-env")
    assert get_secret("DEMO_SECRET") == "from-env"
    folder = tmp_path / "secrets"
    folder.mkdir()
    (folder / "DEMO_SECRET").write_text("from-folder\n")
    (folder / "DEMO_SECRET").chmod(0o600)
    monkeypatch.setenv("KYC_SECRETS_DIR", str(folder))
    assert get_secret("DEMO_SECRET") == "from-folder"
    single = tmp_path / "one"
    single.write_text("from-file")
    single.chmod(0o600)
    monkeypatch.setenv("DEMO_SECRET_FILE", str(single))
    assert get_secret("DEMO_SECRET") == "from-file"
    single.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        get_secret("DEMO_SECRET")


def test_remote_postgres_needs_verified_tls():
    check_postgres_tls("postgresql://kyc@127.0.0.1:5432/kyc")
    check_postgres_tls("postgresql://kyc@localhost/kyc")
    check_postgres_tls("postgresql://kyc@db.example.com/kyc?sslmode=verify-full")
    for url in ("postgresql://kyc@db.example.com/kyc", "postgresql://kyc@db.example.com/kyc?sslmode=require", "postgresql://kyc@localhost,db.example.com/kyc"):
        with pytest.raises(ValueError, match="verify-full"):
            check_postgres_tls(url)
