from __future__ import annotations
from conftest import ROOT


def test_adverse_media_claim_remains_attributed_as_an_allegation(cli, paths):
    run = cli("ingest", ROOT / "examples/synthetic_adverse_media.txt", "--case-id", "media", "--profile", "adverse_media", "--db", paths["db"], "--checkpoint-db", paths["cp"])
    assert run["status"] == "PERSISTED"
    package = cli("report", "--case-id", "media", "--db", paths["db"])
    claim = next(f for f in package["facts"] if f["predicate"] == "accused_of")
    assert claim["claim_type"] == "allegation"
    assert claim["verification_status"] == "PENDING_HUMAN"
    assert next(c for c in package["coverage"] if c["category"] == "source_metadata")["status"] == "NOT_REVIEWED"
