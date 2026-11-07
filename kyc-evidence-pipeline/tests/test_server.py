from __future__ import annotations

import base64
import http.client
import json
import threading
from pathlib import Path

import pytest

from conftest import ROOT
from kyc_pipeline.server import DEMO_TOKEN, EvidenceServer, EvidenceService, ServerConfig

TOKEN = "test-token-0123456789abcdef"


@pytest.fixture
def server(tmp_path):
    config = ServerConfig(token=TOKEN, mode="live", data_dir=tmp_path / "data", max_body_bytes=64 * 1024, host="127.0.0.1", port=0)
    srv = EvidenceServer(config)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def call(srv, method: str, path: str, body=None, *, token: str | None = TOKEN, headers: dict | None = None, raw: bytes | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
    sent = {"Content-Type": "application/json"} if body is not None or raw is not None else {}
    if token is not None:
        sent["Authorization"] = f"Bearer {token}"
    sent.update(headers or {})
    payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    conn.request(method, path, body=payload, headers=sent)
    response = conn.getresponse()
    data = json.loads(response.read() or b"null")
    conn.close()
    return response.status, data


def document(name: str, profile: str | None = None) -> dict:
    body = {"filename": name, "content_base64": base64.b64encode((ROOT / "examples" / name).read_bytes()).decode()}
    if profile:
        body["profile"] = profile
    return body


def test_health_is_public_and_everything_else_needs_the_token(server):
    assert call(server, "GET", "/healthz", token=None) == (200, {"status": "ok", "mode": "live", "note": "Evidence preparation only; not a KYC decision engine."})
    status, body = call(server, "GET", "/v1/cases/acme/package", token=None)
    assert status == 401 and body["code"] == "UNAUTHORIZED"
    status, _ = call(server, "GET", "/v1/cases/acme/package", token="wrong-token")
    assert status == 401
    status, _ = call(server, "POST", "/v1/cases/acme/documents", document("synthetic_registry.txt"), token="wrong-token")
    assert status == 401
    assert call(server, "GET", "/v1/unknown")[0] == 404
    assert call(server, "DELETE", "/v1/cases/acme/package")[0] == 405


def test_ingest_review_resume_package_and_ownership(server):
    status, paused = call(server, "POST", "/v1/cases/case-annual/documents", document("synthetic_annual_report.txt", "financial_annual_report"))
    assert status == 200, paused
    assert paused["status"] == "PAUSED_FOR_REVIEW"
    assert {i["code"] for i in paused["review_request"]["issues"]} == {"STALE_OWNERSHIP_AFTER_DISPOSAL"}
    # The raw upload is not retained once the run has parsed it.
    assert not list((server.service.config.data_dir / "cases" / "case-annual" / "documents").iterdir())

    status, run = call(server, "GET", f"/v1/runs/{paused['run_id']}")
    assert status == 200 and run["run"]["case_id"] == "case-annual"
    assert run["review_request"]["issues"][0]["issue_id"] == paused["review_request"]["issues"][0]["issue_id"]

    decision = {"reviewer_id": "TEST_ANALYST", "decisions": [{"issue_id": i["issue_id"], "action": "defer", "reason": "Fictional test case needs further evidence."} for i in paused["review_request"]["issues"]]}
    status, done = call(server, "POST", f"/v1/runs/{paused['run_id']}/resume", decision)
    assert status == 200, done
    assert done["status"] == "PERSISTED"
    status, again = call(server, "POST", f"/v1/runs/{paused['run_id']}/resume", decision)
    assert status == 422 and "not paused" in again["message"]

    status, package = call(server, "GET", "/v1/cases/case-annual/package")
    assert status == 200
    assert package["case_id"] == "case-annual"
    assert len(package["facts"]) == 5
    assert all(f["verification_status"] == "PENDING_HUMAN" for f in package["facts"])
    assert any(x["action"] == "REVIEW_RECORDED" for x in package["audit_events"])

    status, ownership = call(server, "GET", "/v1/cases/case-annual/ownership")
    assert status == 200 and ownership["case_id"] == "case-annual" and "ownership_graph" in ownership


def test_invalid_requests_are_rejected_before_the_pipeline(server):
    assert call(server, "POST", "/v1/cases/bad%20id/documents", document("synthetic_registry.txt"))[0] == 400
    assert call(server, "POST", "/v1/cases/..a/documents", document("synthetic_registry.txt"))[0] == 400
    assert call(server, "GET", "/v1/cases/" + "a" * 81 + "/package")[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", {"filename": "x.txt", "content_base64": "not base64!"})[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", {"filename": "x.txt", "content_base64": "eA==", "extractor": "openai"})[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", {"filename": "x.txt", "content_base64": "eA==", "profile": "../x"})[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", {"filename": "...", "content_base64": "eA=="})[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", [1, 2])[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", raw=b"{not json")[0] == 400
    assert call(server, "POST", "/v1/cases/acme/documents", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 415
    status, body = call(server, "POST", "/v1/cases/acme/documents", raw=b"x" * (64 * 1024 + 1))
    assert status == 413 and body["code"] == "BODY_TOO_LARGE"
    assert call(server, "GET", "/v1/runs/not-a-run")[0] == 400
    assert call(server, "GET", "/v1/runs/run_" + "0" * 32)[0] == 404
    assert call(server, "POST", "/v1/runs/run_" + "0" * 32 + "/resume", {"reviewer_id": "x", "decisions": []})[0] == 404
    assert call(server, "GET", "/v1/cases/never-seen/package")[0] == 404
    assert call(server, "GET", "/v1/cases/never-seen/ownership")[0] == 404


def test_filename_cannot_escape_the_case_folder(server):
    status, result = call(server, "POST", "/v1/cases/reg/documents", {**document("synthetic_registry.txt", "corporate_registry"), "filename": "../../etc/registry.txt"})
    assert status == 200, result
    assert result["status"] == "PERSISTED"
    assert not (server.service.config.data_dir / "etc").exists()


def test_concurrent_ingests_for_one_case_serialize(server):
    results: list[tuple[int, dict]] = []

    def ingest(name: str, profile: str) -> None:
        results.append(call(server, "POST", "/v1/cases/shared/documents", document(name, profile)))

    threads = [threading.Thread(target=ingest, args=args) for args in [("synthetic_registry.txt", "corporate_registry"), ("synthetic_annual_report.txt", "financial_annual_report")]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert sorted(status for status, _ in results) == [200, 200], results
    assert {body["status"] for _, body in results} == {"PERSISTED", "PAUSED_FOR_REVIEW"}
    status, package = call(server, "GET", "/v1/cases/shared/package")
    assert status == 200 and len(package["documents"]) == 1 and len(package["pending_runs"]) == 1
    service: EvidenceService = server.service
    assert service.case_lock("shared") is service.case_lock("shared")
    assert service.case_lock("shared") is not service.case_lock("other")


def test_configuration_refuses_to_start_without_a_token(tmp_path):
    with pytest.raises(ValueError, match="EVIDENCE_PIPELINE_TOKEN is required"):
        ServerConfig.from_env({"EVIDENCE_PIPELINE_DATA_DIR": str(tmp_path)})
    with pytest.raises(ValueError, match="at least"):
        ServerConfig.from_env({"EVIDENCE_PIPELINE_TOKEN": "short"})
    with pytest.raises(ValueError, match="demo or live"):
        ServerConfig.from_env({"EVIDENCE_PIPELINE_MODE": "open", "EVIDENCE_PIPELINE_TOKEN": TOKEN})
    demo = ServerConfig.from_env({"EVIDENCE_PIPELINE_MODE": "demo", "EVIDENCE_PIPELINE_DATA_DIR": str(tmp_path)})
    assert demo.token == DEMO_TOKEN and demo.data_dir == Path(tmp_path).resolve()
    live = ServerConfig.from_env({"EVIDENCE_PIPELINE_TOKEN": TOKEN, "EVIDENCE_PIPELINE_MAX_BODY_BYTES": "2048", "EVIDENCE_PIPELINE_PORT": "9000"})
    assert (live.mode, live.max_body_bytes, live.port, live.host) == ("live", 2048, 9000, "127.0.0.1")


def test_main_exits_without_a_token(monkeypatch, capsys):
    from kyc_pipeline import server as module
    monkeypatch.delenv("EVIDENCE_PIPELINE_TOKEN", raising=False)
    monkeypatch.delenv("EVIDENCE_PIPELINE_TOKEN_FILE", raising=False)
    monkeypatch.delenv("EVIDENCE_PIPELINE_MODE", raising=False)
    assert module.main([]) == 2
    assert "refusing to start" in capsys.readouterr().err
