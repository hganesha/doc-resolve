from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .azure_di import AzureLayoutClient, endpoint_from_env
from .config import load_contract, requirements_for
from .extraction import OpenAICompatible, ProviderError, provider_endpoint
from .graph import build_graph, validate_review_response
from .local_embeddings import DEFAULT_LOCAL_EMBEDDING_MODEL, LocalEmbeddingProvider
from .ownership import to_mermaid
from .screening import load_list
from .retrieval import search_chunks
from .store import EvidenceStore
from .util import digest, json_text, utc_now

DEFAULT_DB = "kyc_data/evidence.sqlite"
DEFAULT_CHECKPOINT = "kyc_data/checkpoints.sqlite"


def _case(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise argparse.ArgumentTypeError("case ID must be 1–80 letters, digits, dots, underscores, or hyphens")
    return value


def _pages(value: str) -> list[int]:
    result: set[int] = set()
    try:
        for part in value.split(","):
            if "-" in part:
                first, last = (int(x) for x in part.split("-", 1))
                if first < 1 or last < first or last - first > 1000:
                    raise ValueError()
                result.update(range(first, last + 1))
            else:
                number = int(part)
                if number < 1:
                    raise ValueError()
                result.add(number)
        if not result or len(result) > 1000:
            raise ValueError()
        return sorted(result)
    except ValueError:
        raise argparse.ArgumentTypeError("Use positive PDF page numbers/ranges, e.g. 13,38,45-47") from None


def _positive_int(value: str) -> int:
    try:
        number = int(value)
        if number < 1 or number > 200000:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("Value must be an integer from 1 to 200000") from None


def _retries(value: str) -> int:
    try:
        number = int(value)
        if not 0 <= number <= 10:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("Retries must be an integer from 0 to 10") from None


def _timeout(value: str) -> float:
    try:
        number = float(value)
        if not 5 <= number <= 900:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("Timeout must be between 5 and 900 seconds") from None


def _provider_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-provider", choices=["openai", "azure-openai"], default="openai", help="openai: OPENAI_BASE_URL (OpenAI or a compatible server). azure-openai: AZURE_OPENAI_ENDPOINT; --model and --embedding-model are deployment names")


def _interval(value: str) -> float:
    try:
        number = float(value)
        if not 0 <= number <= 300:
            raise ValueError()
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("Interval must be between 0 and 300 seconds") from None


def _db_options(parser: argparse.ArgumentParser, *, checkpoint: bool = False) -> None:
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Evidence SQLite file (default: {DEFAULT_DB})")
    if checkpoint:
        parser.add_argument("--checkpoint-db", default=DEFAULT_CHECKPOINT, help=f"LangGraph SQLite checkpoints (default: {DEFAULT_CHECKPOINT})")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kyc-pipeline", description="Auditable KYC evidence preparation proof of concept; not a KYC decision engine")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("profiles", help="List declarative processing profiles and illustrative coverage categories")
    p.add_argument("--profiles-file")
    p.add_argument("--policy-file")
    p.add_argument("--schemas-file", help="Typed extraction schemas (defaults to the bundled illustrative set)")
    p = sub.add_parser("ingest", help="Parse and prepare one document")
    p.add_argument("file")
    p.add_argument("--case-id", type=_case, required=True)
    p.add_argument("--profile", help="Explicit document profile; automatic selection fails closed if ambiguous")
    p.add_argument("--parser", choices=["native", "azure-di"], default="native", help="native: local text/HTML/CSV/text-layer PDF parsers. azure-di: Azure Document Intelligence prebuilt-layout (OCR, scans and images; sends the file to Azure)")
    p.add_argument("--extractor", choices=["rules", "openai"], default="rules")
    p.add_argument("--embeddings", choices=["lexical", "openai", "local"], default="lexical")
    p.add_argument("--verify-material", action="store_true", help="Run an additional model source-support check for material facts (extra requests)")
    p.add_argument("--extract-pages", type=_pages, help="Optional physical page scope, e.g. 13,38,45-47. Other pages remain NOT_REVIEWED.")
    p.add_argument("--model-batch-size", type=_positive_int, default=12, help="Maximum source chunks per model extraction request (default: 12)")
    p.add_argument("--model-batch-max-chars", type=_positive_int, default=18000, help="Maximum approximate source characters per batch (default: 18000)")
    p.add_argument("--max-model-requests", type=_positive_int, default=20, help="Extraction request budget; remaining chunks are NOT_REVIEWED (default: 20)")
    p.add_argument("--model-request-interval", type=_interval, default=0.0, help="Minimum seconds between model extraction requests")
    p.add_argument("--model-max-tokens", type=_positive_int, help="Optional output token limit for providers that support max_tokens")
    p.add_argument("--model-reasoning-effort", choices=["low", "medium", "high"], help="Optional provider-specific reasoning_effort (temperature is then left at the model's default)")
    p.add_argument("--registry-check", choices=["none", "gleif", "companies-house"], action="append", default=[], help="gleif: look up each LEI the document names in the public GLEIF register (sends only the LEI). companies-house: look up each UK company number at Companies House (sends only the number; needs COMPANIES_HOUSE_API_KEY). Repeat to use both.")
    p.add_argument("--screening-list", action="append", default=[], metavar="PATH", help="Screen the people and organizations this document names against a local list file: a CSV with a 'name' column (optional type, list, id, aliases separated by ';', date_of_birth, program) or OFAC's SDN CSV (alt.csv beside it adds aliases). Nothing is downloaded. Repeat for several lists.")
    _provider_option(p)
    p.add_argument("--model-output", choices=["strict", "json-object"], default="strict", help="strict: JSON-schema structured output (default). json-object: for compatible servers without json_schema support")
    p.add_argument("--model-timeout", type=_timeout, default=120.0, help="Seconds to wait for one model response (default: 120)")
    p.add_argument("--model-max-retries", type=_retries, default=4, help="Retries with exponential backoff for rate limits, timeouts and 5xx errors (default: 4)")
    p.add_argument("--allow-remote-sensitive-data", action="store_true", help="Explicitly permit sending document content to the configured API endpoint")
    p.add_argument("--model", help="Chat model, or the Azure deployment name (defaults: OPENAI_MODEL or gpt-4.1-mini; AZURE_OPENAI_DEPLOYMENT)")
    p.add_argument("--embedding-model", help="Dense embedding model; defaults depend on --embeddings")
    p.add_argument("--profiles-file")
    p.add_argument("--policy-file")
    p.add_argument("--schemas-file", help="Typed extraction schemas (defaults to the bundled illustrative set)")
    _db_options(p, checkpoint=True)
    p = sub.add_parser("resume", help="Resume an interrupted run using a human review JSON file")
    p.add_argument("--run-id", required=True)
    p.add_argument("--decision", required=True)
    _db_options(p, checkpoint=True)
    p = sub.add_parser("retry", help="Create a new run, reusing successful extraction units and retrying failed units")
    p.add_argument("--run-id", required=True, help="Previous run ID")
    p.add_argument("--file", help="Optional replacement path to identical source bytes")
    p.add_argument("--allow-remote-sensitive-data", action="store_true", help="Re-confirm permission for any model calls")
    _db_options(p, checkpoint=True)
    p = sub.add_parser("status", help="Inspect run status and any human-review request")
    p.add_argument("--run-id", required=True)
    _db_options(p, checkpoint=True)
    p = sub.add_parser("report", help="Export the canonical case evidence package")
    p.add_argument("--case-id", type=_case, required=True)
    p.add_argument("--output", help="Write JSON to this file instead of stdout")
    _db_options(p)
    p = sub.add_parser("ownership", help="Export the case ownership and control graph")
    p.add_argument("--case-id", type=_case, required=True)
    p.add_argument("--format", choices=["json", "mermaid"], default="json")
    p.add_argument("--output", help="Write to this file instead of stdout")
    _db_options(p)
    p = sub.add_parser("query", help="Retrieve relevant child chunks with parent-section context")
    p.add_argument("--case-id", type=_case, required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--embeddings", choices=["lexical", "openai", "local"], default="lexical")
    p.add_argument("--embedding-model", help="Dense embedding model; defaults depend on --embeddings")
    _provider_option(p)
    p.add_argument("--allow-remote-sensitive-data", action="store_true")
    _db_options(p)
    p = sub.add_parser("segment", help="Inspect the normalized source segment behind an evidence offset")
    p.add_argument("--case-id", type=_case, required=True)
    p.add_argument("--segment-id", required=True)
    _db_options(p)
    return parser


def _checkpoint_path(path: str) -> str:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return str(p)


def _new_run_id() -> str:
    return "run_" + uuid.uuid4().hex


def _parser_endpoint_hash(parser: str) -> str | None:
    return digest(endpoint_from_env()) if parser == "azure_di" else None


def _default_model(provider: str) -> str:
    return os.environ.get("AZURE_OPENAI_DEPLOYMENT", "") if provider == "azure_openai" else os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def _default_embedding_model(embeddings: str, provider: str) -> str:
    if embeddings == "local":
        return os.environ.get("LOCAL_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL)
    if provider == "azure_openai":
        return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _endpoint_hash(extractor: str, embeddings: str, provider: str) -> str | None:
    return digest(provider_endpoint(provider)) if extractor == "openai" or embeddings == "openai" else None


def _preflight_remote(extractor: str, embeddings: str, verify_material: bool, allow_remote: bool, model: str, embedding_model: str, parser: str = "native", provider: str = "openai") -> None:
    if parser == "azure_di":
        AzureLayoutClient(allow_remote=allow_remote).client.close()
    if verify_material and extractor != "openai":
        raise ValueError("--verify-material requires --extractor openai")
    if extractor == "openai" or embeddings == "openai":
        client = OpenAICompatible(allow_remote=allow_remote, model=model, embedding_model=embedding_model, provider=provider)
        client.client.close()
        if provider == "azure_openai" and extractor == "openai" and not model:
            raise ValueError("Azure OpenAI needs the chat deployment name: pass --model or set AZURE_OPENAI_DEPLOYMENT")
        if provider == "azure_openai" and embeddings == "openai" and not embedding_model:
            raise ValueError("Azure OpenAI embeddings need a deployment name: pass --embedding-model or set AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def _interrupt_value(result: dict) -> dict | None:
    interrupts = result.get("__interrupt__", ())
    return interrupts[0].value if interrupts else None


def _snapshot_request(snapshot: Any) -> dict | None:
    for task in snapshot.tasks:
        for item in task.interrupts:
            return item.value
    return None


def _drive(graph: Any, store: EvidenceStore, run_id: str, case_id: str, input_value: Any) -> dict:
    config = {"configurable": {"thread_id": run_id}}
    try:
        result = graph.invoke(input_value, config=config)
    except Exception as exc:
        safe = str(exc) if isinstance(exc, ValueError) else f"Pipeline failed ({type(exc).__name__}); see local diagnostics."
        store.set_run_status(run_id, "FAILED", error=safe)
        raise ValueError(safe) from exc
    request = _interrupt_value(result)
    if request:
        store.set_run_status(run_id, "REVIEW_REQUIRED")
        return {"status": "PAUSED_FOR_REVIEW", "run_id": run_id, "case_id": case_id, "review_request": request, "next_step": "Submit a review decision with the resume command, or retry failed extraction units."}
    row = store.get_run(run_id)
    return {"status": "PERSISTED", "run_id": run_id, "case_id": case_id, "run_status": row["status"], "summary": json.loads(row["summary_json"]) if row and row["summary_json"] else None, "next_step": "Export the case with the report command; all facts remain provisional."}


def _write_output(data: dict | str, output: str | None = None) -> None:
    text = data if isinstance(data, str) else json_text(data, pretty=True) + "\n"
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    else:
        sys.stdout.write(text)


def execute(args: argparse.Namespace) -> dict | str:
    if args.command == "profiles":
        contract = load_contract(args.profiles_file, args.policy_file, args.schemas_file)
        return {"policy_id": contract["policy"]["policy_id"], "profiles": [{"profile_id": pid, **p, "requirements": requirements_for(contract, p)} for pid, p in contract["profiles"].items()], "note": "Profiles and policy are illustrative, not bank-approved policy."}
    store = EvidenceStore(args.db)
    if args.command == "report":
        package = store.export_case(args.case_id)
        if not package["documents"] and not package["pending_runs"] and not package["audit_events"]:
            raise ValueError("Case not found")
        return package
    if args.command == "ownership":
        package = store.export_case(args.case_id)
        if not package["documents"]:
            raise ValueError("Case has no persisted documents")
        graph = package["ownership_graph"]
        return to_mermaid(graph) if args.format == "mermaid" else {"case_id": args.case_id, "status": package["status"], "ownership_graph": graph}
    if args.command == "segment":
        segment = store.get_segment(args.case_id, args.segment_id)
        if not segment:
            raise ValueError("Segment not found in the current case evidence")
        return segment
    if args.command == "query":
        provider = None
        model_provider = args.model_provider.replace("-", "_")
        if args.embeddings == "openai":
            provider = OpenAICompatible(allow_remote=args.allow_remote_sensitive_data, embedding_model=args.embedding_model or _default_embedding_model("openai", model_provider), provider=model_provider)
        elif args.embeddings == "local":
            provider = LocalEmbeddingProvider(args.embedding_model or _default_embedding_model("local", model_provider))
        try:
            chunks = store.current_chunks(args.case_id)
            if not chunks:
                raise ValueError("Case has no persisted searchable chunks")
            return search_chunks(chunks, args.query, args.top_k, provider=provider)
        finally:
            if provider and hasattr(provider, "client"):
                provider.client.close()
    checkpoint_path = _checkpoint_path(args.checkpoint_db)
    with SqliteSaver.from_conn_string(checkpoint_path) as saver:
        graph = build_graph(store, saver)
        if args.command == "ingest":
            contract = load_contract(args.profiles_file, args.policy_file, args.schemas_file)
            model_provider = args.model_provider.replace("-", "_")
            args.model = args.model or _default_model(model_provider)
            args.embedding_model = args.embedding_model or _default_embedding_model(args.embeddings, model_provider)
            parser = args.parser.replace("-", "_")
            _preflight_remote(args.extractor, args.embeddings, args.verify_material, args.allow_remote_sensitive_data, args.model, args.embedding_model, parser, model_provider)
            if "companies-house" in args.registry_check and not os.environ.get("COMPANIES_HOUSE_API_KEY"):
                raise ValueError("--registry-check companies-house needs COMPANIES_HOUSE_API_KEY (a free key from the Companies House developer hub)")
            for path in args.screening_list:
                load_list(path)  # a missing or unreadable list fails before the run starts
            run_id = _new_run_id()
            state = {"run_id": run_id, "case_id": args.case_id, "started_at": utc_now(), "doc_path": str(Path(args.file).expanduser().resolve()), "requested_profile": args.profile, "contract": contract, "parser": parser, "parser_endpoint_sha256": _parser_endpoint_hash(parser), "extractor": args.extractor, "embeddings": args.embeddings, "allow_remote": args.allow_remote_sensitive_data, "verify_material": args.verify_material, "model": args.model, "embedding_model": args.embedding_model, "extraction_pages": args.extract_pages, "model_batch_size": args.model_batch_size, "model_batch_max_chars": args.model_batch_max_chars, "max_model_requests": args.max_model_requests, "model_request_interval": args.model_request_interval, "model_max_tokens": args.model_max_tokens, "model_reasoning_effort": args.model_reasoning_effort, "model_provider": model_provider, "model_output_mode": args.model_output.replace("-", "_"), "model_timeout": args.model_timeout, "model_max_retries": args.model_max_retries, "registry_check": sorted(set(args.registry_check) - {"none"}) or None, "screening_lists": [str(Path(p).expanduser().resolve()) for p in args.screening_list] or None, "provider_endpoint_sha256": _endpoint_hash(args.extractor, args.embeddings, model_provider), "retry_from_run": None, "expected_document_sha256": None}
            store.start_run(run_id, args.case_id, state["started_at"])
            return _drive(graph, store, run_id, args.case_id, state)
        row = store.get_run(args.run_id)
        if not row:
            raise ValueError("Run not found")
        config = {"configurable": {"thread_id": args.run_id}}
        snapshot = graph.get_state(config)
        if args.command == "status":
            row["summary"] = json.loads(row.pop("summary_json")) if row.get("summary_json") else None
            return {"run": row, "review_request": _snapshot_request(snapshot)}
        if args.command == "resume":
            if row["status"] != "REVIEW_REQUIRED" or row["finished_at"] is not None or not _snapshot_request(snapshot):
                raise ValueError("Run is not paused at a human-review interrupt")
            try:
                response = json.loads(Path(args.decision).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise ValueError("Decision file must be readable JSON") from None
            validate_review_response(response, snapshot.values["issues"])
            return _drive(graph, store, args.run_id, row["case_id"], Command(resume=response))
        if args.command == "retry":
            prior = snapshot.values
            if not prior or not prior.get("document") or not store.get_extraction_units(args.run_id):
                raise ValueError("Run has no checkpointed document and extraction units to retry")
            path = Path(args.file or prior["doc_path"]).expanduser().resolve()
            if not path.is_file() or digest(path.read_bytes()) != prior["document"]["sha256"]:
                raise ValueError("Retry requires the identical original source bytes; use ingest for changed documents")
            model_provider = prior.get("model_provider", "openai")
            _preflight_remote(prior["extractor"], prior["embeddings"], prior.get("verify_material", False), args.allow_remote_sensitive_data, prior["model"], prior["embedding_model"], provider=model_provider)
            endpoint_hash = _endpoint_hash(prior["extractor"], prior["embeddings"], model_provider)
            if endpoint_hash != prior.get("provider_endpoint_sha256"):
                raise ValueError("Retry requires the same provider endpoint; use a fresh ingestion if it changed")
            parser = prior.get("parser", "native")
            if _parser_endpoint_hash(parser) != prior.get("parser_endpoint_sha256"):
                raise ValueError("Retry requires the same Azure Document Intelligence endpoint; use a fresh ingestion if it changed")
            new_id = _new_run_id()
            state = {"run_id": new_id, "case_id": prior["case_id"], "started_at": utc_now(), "doc_path": str(path), "requested_profile": prior["profile_id"], "contract": prior["contract"], "parser": parser, "parser_endpoint_sha256": prior.get("parser_endpoint_sha256"), "extractor": prior["extractor"], "embeddings": prior["embeddings"], "allow_remote": args.allow_remote_sensitive_data, "verify_material": prior.get("verify_material", False), "model": prior["model"], "embedding_model": prior["embedding_model"], "extraction_pages": prior.get("extraction_pages"), "model_batch_size": prior.get("model_batch_size", 12), "model_batch_max_chars": prior.get("model_batch_max_chars", 18000), "max_model_requests": prior.get("max_model_requests", 20), "model_request_interval": prior.get("model_request_interval", 0.0), "model_max_tokens": prior.get("model_max_tokens"), "model_reasoning_effort": prior.get("model_reasoning_effort"), "model_provider": model_provider, "model_output_mode": prior.get("model_output_mode", "strict"), "model_timeout": prior.get("model_timeout", 120.0), "model_max_retries": prior.get("model_max_retries", 4), "registry_check": prior.get("registry_check"), "screening_lists": prior.get("screening_lists"), "provider_endpoint_sha256": endpoint_hash, "retry_from_run": args.run_id, "expected_document_sha256": prior["document"]["sha256"]}
            store.start_run(new_id, state["case_id"], state["started_at"])
            if row["finished_at"] is None and row["status"] in {"REVIEW_REQUIRED", "FAILED", "PROCESSING"}:
                store.set_run_status(args.run_id, "SUPERSEDED")
            return _drive(graph, store, new_id, state["case_id"], state)
    raise ValueError("Unsupported command")


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = execute(args)
        _write_output(result, getattr(args, "output", None))
        return 0
    except (ValueError, OSError, ProviderError) as exc:
        sys.stderr.write(json_text({"status": "ERROR", "message": str(exc)}, pretty=True) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
