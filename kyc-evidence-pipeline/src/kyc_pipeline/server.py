"""A thin, authenticated HTTP front end for the evidence pipeline.

The server adds no pipeline behaviour of its own: every endpoint builds the argument vector
the ``kyc-pipeline`` command would receive and calls :func:`kyc_pipeline.cli.execute` in
process. It uses only the standard library, so the image carries no web framework.

It always runs the offline pipeline (``--parser native --extractor rules --embeddings
lexical``): nothing a caller sends can make it call a model, OCR or registry endpoint.
Like the CLI, it prepares evidence for a human reviewer and never decides a KYC case.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import os
import re
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import cli
from .credentials import get_secret
from .extraction import ProviderError
from .store import is_postgres
from .util import json_text

DEMO_TOKEN = "evidence-demo-token-local-only"
DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024
MIN_TOKEN_LENGTH = 16
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
RUN_ID = re.compile(r"run_[0-9a-f]{32}")
PROFILE_ID = re.compile(r"[a-z0-9_]{1,80}")
FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
NOT_FOUND_MESSAGES = {"Case not found", "Run not found", "Case has no persisted documents"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ServerConfig:
    token: str
    mode: str
    data_dir: Path
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ServerConfig":
        """Read EVIDENCE_PIPELINE_* settings; refuse a live server without a bearer token."""
        source = os.environ if env is None else env
        mode = (source.get("EVIDENCE_PIPELINE_MODE") or "live").strip().lower()
        if mode not in {"demo", "live"}:
            raise ValueError("EVIDENCE_PIPELINE_MODE must be demo or live")
        # get_secret also honours EVIDENCE_PIPELINE_TOKEN_FILE and KYC_SECRETS_DIR.
        token = (get_secret("EVIDENCE_PIPELINE_TOKEN") if env is None else source.get("EVIDENCE_PIPELINE_TOKEN", "")).strip()
        if not token:
            if mode != "demo":
                raise ValueError("EVIDENCE_PIPELINE_TOKEN is required (set EVIDENCE_PIPELINE_MODE=demo only for local synthetic data)")
            token = DEMO_TOKEN
        elif mode != "demo" and len(token) < MIN_TOKEN_LENGTH:
            raise ValueError(f"EVIDENCE_PIPELINE_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")
        try:
            max_body = int(source.get("EVIDENCE_PIPELINE_MAX_BODY_BYTES") or DEFAULT_MAX_BODY_BYTES)
            port = int(source.get("EVIDENCE_PIPELINE_PORT") or 8080)
        except ValueError:
            raise ValueError("EVIDENCE_PIPELINE_MAX_BODY_BYTES and EVIDENCE_PIPELINE_PORT must be integers") from None
        if not 1024 <= max_body <= 256 * 1024 * 1024:
            raise ValueError("EVIDENCE_PIPELINE_MAX_BODY_BYTES must be between 1 KiB and 256 MiB")
        data_dir = Path(source.get("EVIDENCE_PIPELINE_DATA_DIR") or "kyc_data/server").expanduser().resolve()
        return cls(token=token, mode=mode, data_dir=data_dir, max_body_bytes=max_body, host=source.get("EVIDENCE_PIPELINE_HOST") or "127.0.0.1", port=port)


class EvidenceService:
    """Maps API operations onto CLI commands, serializing graph runs per case."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        # SQLite admits one writer at a time, so graph runs for different cases also take a
        # process-wide lock there instead of racing for the database lock. Postgres needs
        # only the per-case lock.
        self._sqlite_writer = threading.Lock()
        _private_dir(config.data_dir)
        self.db = get_secret("KYC_DATABASE_URL") or str(config.data_dir / "evidence.sqlite")
        self.checkpoint_db = get_secret("KYC_CHECKPOINT_DATABASE_URL") or (self.db if is_postgres(self.db) else str(config.data_dir / "checkpoints.sqlite"))

    def case_lock(self, case_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(case_id, threading.Lock())

    @contextmanager
    def writing(self, case_id: str) -> Iterator[None]:
        """Concurrent ingests and resumes for one case run one at a time."""
        with self.case_lock(case_id):
            if is_postgres(self.db) and is_postgres(self.checkpoint_db):
                yield
            else:
                with self._sqlite_writer:
                    yield

    def _run(self, argv: list[str]) -> Any:
        try:
            args = cli.make_parser().parse_args(argv)
        except SystemExit:
            raise ApiError(400, "INVALID_ARGUMENTS", "The request could not be mapped to a pipeline command") from None
        try:
            return cli.execute(args)
        except ProviderError as exc:
            raise ApiError(502, "PROVIDER_ERROR", str(exc)) from None
        except ValueError as exc:
            message = str(exc)
            if message in NOT_FOUND_MESSAGES:
                raise ApiError(404, "NOT_FOUND", message) from None
            raise ApiError(422, "PIPELINE_REJECTED", message) from None

    def _stores(self, checkpoint: bool = False) -> list[str]:
        return ["--db", self.db, *(["--checkpoint-db", self.checkpoint_db] if checkpoint else [])]

    def ingest(self, case_id: str, body: Any) -> dict:
        case_id = _case_id(case_id)
        if not isinstance(body, dict):
            raise ApiError(400, "INVALID_BODY", "Body must be a JSON object")
        unknown = set(body) - {"filename", "content_base64", "profile"}
        if unknown:
            raise ApiError(400, "INVALID_BODY", "Unknown fields: " + ", ".join(sorted(unknown)))
        filename = _safe_filename(body.get("filename"))
        content = body.get("content_base64")
        if not isinstance(content, str) or not content:
            raise ApiError(400, "INVALID_BODY", "content_base64 must be a nonempty base64 string")
        try:
            data = base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError):
            raise ApiError(400, "INVALID_BODY", "content_base64 is not valid base64") from None
        if not data:
            raise ApiError(400, "INVALID_BODY", "The document is empty")
        profile = body.get("profile")
        if profile is not None and (not isinstance(profile, str) or not PROFILE_ID.fullmatch(profile)):
            raise ApiError(400, "INVALID_BODY", "profile must be a profile ID such as financial_annual_report")
        folder = _private_dir(self.config.data_dir / "cases" / case_id / "documents")
        path = folder / f"{uuid.uuid4().hex[:12]}-{filename}"
        with self.writing(case_id):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            try:
                argv = ["ingest", str(path), "--case-id", case_id, "--parser", "native", "--extractor", "rules", "--embeddings", "lexical", *self._stores(checkpoint=True)]
                if profile:
                    argv += ["--profile", profile]
                return self._run(argv)
            finally:
                # The evidence and checkpoint stores keep what the run parsed (and the
                # document hash); the server does not retain the raw upload.
                path.unlink(missing_ok=True)

    def run_status(self, run_id: str) -> dict:
        return self._run(["status", "--run-id", _run_id(run_id), *self._stores(checkpoint=True)])

    def resume(self, run_id: str, body: Any) -> dict:
        run_id = _run_id(run_id)
        if not isinstance(body, dict):
            raise ApiError(400, "INVALID_BODY", "The review decision must be a JSON object")
        case_id = self.run_status(run_id)["run"]["case_id"]
        folder = _private_dir(self.config.data_dir / "cases" / case_id / "decisions")
        path = folder / f"{run_id}-{uuid.uuid4().hex[:8]}.json"
        with self.writing(case_id):
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(body, handle)
            try:
                return self._run(["resume", "--run-id", run_id, "--decision", str(path), *self._stores(checkpoint=True)])
            finally:
                path.unlink(missing_ok=True)

    def package(self, case_id: str) -> dict:
        return self._run(["report", "--case-id", _case_id(case_id), *self._stores()])

    def ownership(self, case_id: str) -> dict:
        return self._run(["ownership", "--case-id", _case_id(case_id), "--format", "json", *self._stores()])


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def _case_id(value: str) -> str:
    if not CASE_ID.fullmatch(value) or ".." in value:
        raise ApiError(400, "INVALID_CASE_ID", "case ID must be 1-80 letters, digits, dots, underscores, or hyphens")
    return value


def _run_id(value: str) -> str:
    if not RUN_ID.fullmatch(value):
        raise ApiError(400, "INVALID_RUN_ID", "run ID must be run_ followed by 32 lowercase hex digits")
    return value


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, "INVALID_BODY", "filename must be a nonempty string")
    name = FILENAME_UNSAFE.sub("_", Path(value.replace("\\", "/")).name).lstrip(".")
    if not name or len(name) > 120:
        raise ApiError(400, "INVALID_BODY", "filename must keep 1-120 safe characters")
    return name


ROUTES: list[tuple[str, re.Pattern[str], str]] = [
    ("GET", re.compile(r"/healthz"), "health"),
    ("POST", re.compile(r"/v1/cases/([^/]+)/documents"), "ingest"),
    ("POST", re.compile(r"/v1/runs/([^/]+)/resume"), "resume"),
    ("GET", re.compile(r"/v1/runs/([^/]+)"), "run"),
    ("GET", re.compile(r"/v1/cases/([^/]+)/package"), "package"),
    ("GET", re.compile(r"/v1/cases/([^/]+)/ownership"), "ownership"),
]


class Handler(BaseHTTPRequestHandler):
    server_version = "kyc-evidence-pipeline"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 60  # seconds a client may take to send its request

    @property
    def service(self) -> EvidenceService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler API
        # Request line and status only; never bodies or headers (which carry the token).
        sys.stderr.write("%s %s\n" % (self.address_string(), format % args))

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def _not_allowed(self) -> None:
        self._send(405, {"status": "ERROR", "code": "METHOD_NOT_ALLOWED", "message": "Method not allowed"})

    do_PUT = do_PATCH = do_DELETE = _not_allowed

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        try:
            for route_method, pattern, name in ROUTES:
                match = pattern.fullmatch(path)
                if not match or route_method != method:
                    continue
                if name == "health":
                    self._send(200, {"status": "ok", "mode": self.service.config.mode, "note": "Evidence preparation only; not a KYC decision engine."})
                    return
                self._authenticate()
                body = self._body() if method == "POST" else None
                arg = match.group(1)
                operations = {
                    "ingest": lambda: self.service.ingest(arg, body),
                    "resume": lambda: self.service.resume(arg, body),
                    "run": lambda: self.service.run_status(arg),
                    "package": lambda: self.service.package(arg),
                    "ownership": lambda: self.service.ownership(arg),
                }
                self._send(200, operations[name]())
                return
            if any(pattern.fullmatch(path) for _, pattern, _ in ROUTES):
                raise ApiError(405, "METHOD_NOT_ALLOWED", "Method not allowed")
            raise ApiError(404, "NOT_FOUND", "Unknown endpoint")
        except ApiError as exc:
            # A rejected request may leave an unread body on the socket; never parse it as
            # the next request.
            self.close_connection = True
            self._send(exc.status, {"status": "ERROR", "code": exc.code, "message": exc.message})
        except Exception as exc:  # never leak internals to the caller
            sys.stderr.write(f"internal error: {type(exc).__name__}\n")
            self.close_connection = True
            self._send(500, {"status": "ERROR", "code": "INTERNAL", "message": "Internal error; see server diagnostics."})

    def _authenticate(self) -> None:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            raise ApiError(401, "UNAUTHORIZED", "Bearer token required")
        if not hmac.compare_digest(header[7:].encode("utf-8"), self.service.config.token.encode("utf-8")):
            raise ApiError(401, "UNAUTHORIZED", "Invalid token")

    def _body(self) -> Any:
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise ApiError(411, "LENGTH_REQUIRED", "Send a Content-Length; chunked bodies are not accepted")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            raise ApiError(411, "LENGTH_REQUIRED", "Content-Length required") from None
        if length < 0 or length > self.service.config.max_body_bytes:
            self.close_connection = True  # the unread body must not be parsed as the next request
            raise ApiError(413, "BODY_TOO_LARGE", f"Body exceeds {self.service.config.max_body_bytes} bytes")
        raw = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json")
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "INVALID_JSON", "Body is not valid JSON") from None

    def _send(self, status: int, payload: Any) -> None:
        data = (payload if isinstance(payload, str) else json_text(payload)).encode("utf-8")
        self.send_response(status, HTTPStatus(status).phrase)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)


class EvidenceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, config: ServerConfig, service: EvidenceService | None = None):
        self.service = service or EvidenceService(config)
        super().__init__((config.host, config.port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kyc-evidence-server", description="Authenticated HTTP API over the offline evidence pipeline, configured from EVIDENCE_PIPELINE_* environment variables")
    parser.add_argument("--host", help="Overrides EVIDENCE_PIPELINE_HOST (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Overrides EVIDENCE_PIPELINE_PORT (default 8080)")
    args = parser.parse_args(argv)
    try:
        config = ServerConfig.from_env()
    except ValueError as exc:
        sys.stderr.write(f"refusing to start: {exc}\n")
        return 2
    if args.host is not None:
        config = replace(config, host=args.host)
    if args.port is not None:
        config = replace(config, port=args.port)
    if config.mode == "demo":
        sys.stderr.write("DEMO MODE: synthetic data only" + (" (published demo token in use)" if config.token == DEMO_TOKEN else "") + "\n")
    server = EvidenceServer(config)
    sys.stderr.write(f"kyc-evidence-server listening on {config.host}:{server.server_address[1]}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
