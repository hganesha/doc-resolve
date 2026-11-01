from __future__ import annotations

from collections import deque
from typing import Any, Callable, TypedDict
import time

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from .chunking import build_chunks
from .config import requirements_for, select_profile
from .coverage import complete_section_inventory, compute_coverage
from .azure_di import AzureLayoutClient, layout_fingerprint, parser_metadata
from .entities import resolve_case, resolve_document
from .evidence import make_issue, verify_candidates
from .extraction import OpenAICompatible, ProviderError, rules_extract
from .local_embeddings import LocalEmbeddingProvider
from .parsing import parse_file
from .ownership import build_ownership_graph, graph_settings
from .reconciliation import issue_finding, reconcile
from .funds import inflow_checks, statement_checks
from .identity import identity_checks
from .media import media_checks
from .prior import prior_checks
from .screening import load_list, report_checks, screen_entities
from .registry import CompaniesHouseClient, GleifClient, RegistryLookupError, cached_lookup, check_companies_house, check_gleif, status_issues
from .store import EvidenceStore
from .util import digest, json_text, utc_now

# Increment whenever extraction prompts, rules, or output schema change.
EXTRACTION_CACHE_VERSION = {"rules": "11", "openai": "5"}
# A batch that fails for one of these reasons is split in half and retried, so
# one long or filtered chunk does not cost the whole batch.
_SPLIT_ON = {"output_truncated", "content_filter"}


class PipelineState(TypedDict, total=False):
    run_id: str
    case_id: str
    started_at: str
    doc_path: str
    requested_profile: str | None
    contract: dict
    parser: str
    parser_endpoint_sha256: str | None
    extractor: str
    embeddings: str
    allow_remote: bool
    verify_material: bool
    model: str
    embedding_model: str
    extraction_pages: list[int] | None
    model_batch_size: int
    model_batch_max_chars: int
    max_model_requests: int
    model_request_interval: float
    model_max_tokens: int | None
    model_reasoning_effort: str | None
    model_provider: str
    model_output_mode: str
    model_timeout: float
    model_max_retries: int
    model_usage: dict
    registry_check: list[str] | str | None
    registry_checks: list[dict]
    screening_lists: list[str] | None
    screening_checks: list[dict]
    provider_endpoint_sha256: str | None
    retry_from_run: str | None
    expected_document_sha256: str | None
    document: dict
    profile_id: str
    profile: dict
    requirements: list[dict]
    segments: list[dict]
    chunks: list[dict]
    sections: list[dict]
    parse_issues: list[dict]
    index_issues: list[dict]
    extraction_log: list[dict]
    evidence: list[dict]
    facts: list[dict]
    entities: list[dict]
    case_entities: list[dict]
    issues: list[dict]
    findings: list[dict]
    case_revision: str
    coverage: list[dict]
    review_decisions: list[dict]
    persisted: bool


def _provider(state: PipelineState) -> OpenAICompatible:
    return OpenAICompatible(allow_remote=state["allow_remote"], model=state.get("model"), embedding_model=state.get("embedding_model"), max_output_tokens=state.get("model_max_tokens"), reasoning_effort=state.get("model_reasoning_effort"), provider=state.get("model_provider", "openai"), output_mode=state.get("model_output_mode", "strict"), timeout=state.get("model_timeout", 120.0), max_retries=state.get("model_max_retries", 4))


def _layout_client(state: PipelineState) -> AzureLayoutClient:
    return AzureLayoutClient(allow_remote=state["allow_remote"])


def _registry_client(state: PipelineState, source: str) -> Any:
    return GleifClient() if source == "gleif" else CompaniesHouseClient()


def registry_sources(value: list[str] | str | None) -> list[str]:
    """Runs before Companies House support stored a single source name."""
    return [value] if isinstance(value, str) else list(value or [])


def validate_review_response(response: Any, issues: list[dict]) -> list[dict]:
    if not isinstance(response, dict):
        raise ValueError("Review response must be a JSON object")
    reviewer = response.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 150:
        raise ValueError("Review requires a nonempty reviewer_id (self-reported, not authenticated)")
    decisions = response.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Review requires a decisions array")
    reviewable = {i["issue_id"]: i for i in issues if i["status"] == "OPEN" and i["severity"] == "high"}
    if len(decisions) != len(reviewable):
        raise ValueError("Provide exactly one decision for each review item")
    seen: set[str] = set()
    normalized = []
    for d in decisions:
        if not isinstance(d, dict) or d.get("issue_id") not in reviewable or d.get("issue_id") in seen:
            raise ValueError("Review contains an unknown or duplicate issue_id")
        issue = reviewable[d["issue_id"]]
        action = d.get("action")
        if action not in issue["allowed_actions"]:
            raise ValueError(f"Action is not permitted for issue {issue['issue_id']}")
        reason = d.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("Every decision requires a nonempty reason, at most 2000 characters")
        selected = d.get("selected_fact_ids", [])
        if not isinstance(selected, list) or any(not isinstance(x, str) for x in selected):
            raise ValueError("selected_fact_ids must be an array of fact IDs")
        if action == "resolve" and (len(selected) != 1 or selected[0] not in issue["fact_ids"]):
            raise ValueError("Resolving a conflict requires selecting exactly one fact from that issue")
        if action == "defer" and selected:
            raise ValueError("Do not select a fact when deferring")
        seen.add(issue["issue_id"])
        normalized.append({"issue_id": issue["issue_id"], "action": action, "reason": reason.strip(), "selected_fact_ids": selected, "reviewer_id": reviewer.strip(), "reviewed_at": utc_now(), "reviewer_identity_authenticated": False})
    return normalized


def build_graph(store: EvidenceStore, checkpointer: Any, provider_factory: Callable[[PipelineState], Any] = _provider, layout_factory: Callable[[PipelineState], Any] = _layout_client, registry_factory: Callable[[PipelineState], Any] = _registry_client):
    def parse_node(state: PipelineState) -> dict:
        layout = None
        if state.get("parser", "native") == "azure_di":
            def layout(raw: bytes, sha256: str) -> tuple[dict, dict]:
                fingerprint = digest(json_text(layout_fingerprint(state.get("parser_endpoint_sha256"))))
                result = store.get_layout_result(sha256, fingerprint)
                if result is None:
                    client = layout_factory(state)
                    try:
                        result = client.analyze(raw)
                    finally:
                        if hasattr(client, "client"):
                            client.client.close()
                    store.put_layout_result(sha256, fingerprint, result)
                return result, parser_metadata(result, state.get("parser_endpoint_sha256"))
        document, segments, parse_issues = parse_file(state["doc_path"], layout)
        expected = state.get("expected_document_sha256")
        scope_pages = state.get("extraction_pages")
        if scope_pages and document.get("page_count") is None:
            raise ValueError("Page-limited extraction requires a PDF or a text file with form-feed page anchors")
        if scope_pages and any(page > document["page_count"] for page in scope_pages):
            raise ValueError("The selected extraction pages exceed the document's page count")
        if expected and expected != document["sha256"]:
            raise ValueError("The retry input differs from the original document; start a fresh ingestion instead")
        sample = "\n".join(s["text"] for s in segments[:50])[:20000]
        profile_id, profile = select_profile(state["contract"], state.get("requested_profile"), document["filename"], sample)
        requirements = requirements_for(state["contract"], profile)
        return {"document": document, "segments": segments, "parse_issues": parse_issues, "profile_id": profile_id, "profile": profile, "requirements": requirements}

    def chunk_node(state: PipelineState) -> dict:
        chunks, sections, warnings = build_chunks(state["segments"], state["profile"])
        return {"chunks": chunks, "sections": sections, "parse_issues": state["parse_issues"] + warnings}

    def index_node(state: PipelineState) -> dict:
        chunks = [dict(c) for c in state["chunks"]]
        failures = 0
        if state["embeddings"] in {"openai", "local"}:
            provider = provider_factory(state) if state["embeddings"] == "openai" else LocalEmbeddingProvider(state["embedding_model"])
            for start in range(0, len(chunks), 32):
                group = chunks[start:start + 32]
                try:
                    vectors = provider.embed([((c.get("table_header") or "") + "\n" + c["text"] + "\n" + " ".join(ref.get("text", "") for ref in c.get("context_refs", []))).strip() for c in group])
                    if len(vectors) != len(group):
                        raise ProviderError("Embedding count does not match source chunks")
                    store.put_chunk_vectors(state["run_id"], state["embedding_model"], group, vectors)
                except ProviderError:
                    failures += len(group)
            if hasattr(provider, "client"):
                provider.client.close()
        issues = []
        if failures:
            issues.append(make_issue("EMBEDDING_INDEX_INCOMPLETE", "Some dense embeddings could not be created; lexical retrieval remains available.", document_id=state["document"]["document_id"], categories=["*"], severity="medium", details={"failed_chunks": failures}))
        return {"chunks": chunks, "index_issues": issues}

    def extract_node(state: PipelineState) -> dict:
        provider = provider_factory(state) if state["extractor"] == "openai" else None
        current_units = store.get_extraction_units(state["run_id"])
        old_units = store.get_extraction_units(state["retry_from_run"]) if state.get("retry_from_run") else {}
        logs_by_id: dict[str, dict] = {}
        pending: list[tuple[dict, str]] = []
        scope_pages = set(state.get("extraction_pages") or [])
        for chunk in state["chunks"]:
            fingerprint = digest(json_text({"version": EXTRACTION_CACHE_VERSION[state["extractor"]], "chunk_id": chunk["chunk_id"], "profile_id": state["profile_id"], "profiles_sha256": state["contract"]["profiles_sha256"], "policy_sha256": state["contract"]["policy_sha256"], "schemas_sha256": state["contract"].get("schemas_sha256"), "extractor": state["extractor"], "model": state.get("model") if state["extractor"] == "openai" else None, "provider_endpoint_sha256": state.get("provider_endpoint_sha256") if state["extractor"] == "openai" else None, "model_max_tokens": state.get("model_max_tokens"), "model_reasoning_effort": state.get("model_reasoning_effort"), "model_output_mode": state.get("model_output_mode", "strict") if state["extractor"] == "openai" else None}))
            if scope_pages and chunk.get("page") not in scope_pages:
                log = {"chunk_id": chunk["chunk_id"], "status": "NOT_REVIEWED", "method": state["extractor"], "candidate_count": 0, "candidates": [], "reason": "outside explicitly selected extraction pages", "cache_fingerprint": fingerprint}
                store.put_extraction_unit(state["run_id"], log)
                logs_by_id[chunk["chunk_id"]] = log
                continue
            cached = current_units.get(chunk["chunk_id"]) or old_units.get(chunk["chunk_id"])
            if cached and cached.get("status") == "OK" and not cached.get("rejected_candidates") and cached.get("cache_fingerprint") == fingerprint:
                log = {**cached, "reused_from_run": state.get("retry_from_run") or state["run_id"]}
                store.put_extraction_unit(state["run_id"], log)
                logs_by_id[chunk["chunk_id"]] = log
            else:
                pending.append((chunk, fingerprint))
        if provider is None:
            for chunk, fingerprint in pending:
                try:
                    candidates = rules_extract(chunk, state["requirements"])
                    log = {"chunk_id": chunk["chunk_id"], "status": "OK", "method": "rules", "candidate_count": len(candidates), "candidates": candidates, "cache_fingerprint": fingerprint}
                except (ValidationError, ValueError):
                    log = {"chunk_id": chunk["chunk_id"], "status": "ERROR", "method": "rules", "candidate_count": 0, "candidates": [], "error": "Candidate extraction or validation failed", "cache_fingerprint": fingerprint}
                store.put_extraction_unit(state["run_id"], log)
                logs_by_id[chunk["chunk_id"]] = log
        else:
            batch_size = state.get("model_batch_size", 12)
            max_chars = state.get("model_batch_max_chars", 18000)
            max_requests = state.get("max_model_requests", 20)
            interval = state.get("model_request_interval", 0.0)
            batches: list[list[tuple[dict, str]]] = []
            group: list[tuple[dict, str]] = []
            chars = 0
            for chunk, fingerprint in pending:
                cost = len(chunk["text"]) + len(chunk.get("table_header") or "") + sum(len(ref["text"]) for ref in chunk.get("section_context", []))
                if group and (len(group) >= batch_size or chars + cost > max_chars):
                    batches.append(group)
                    group = []
                    chars = 0
                group.append((chunk, fingerprint))
                chars += cost
            if group:
                batches.append(group)
            last_started: float | None = None
            queue = deque(batches)
            requests = 0
            usage = {"requests": 0, "http_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0, "split_batches": 0}

            def record(batch: list[tuple[dict, str]], **fields: Any) -> None:
                """Log every chunk of a batch that produced no candidates."""
                for chunk, fingerprint in batch:
                    log = {"chunk_id": chunk["chunk_id"], "method": "openai", "candidate_count": 0, "candidates": [], **fields, "cache_fingerprint": fingerprint}
                    store.put_extraction_unit(state["run_id"], log)
                    logs_by_id[chunk["chunk_id"]] = log

            while queue:
                batch = queue.popleft()
                if requests >= max_requests:
                    record(batch, status="NOT_REVIEWED", reason="model request budget reached")
                    continue
                if last_started is not None:
                    wait = interval - (time.monotonic() - last_started)
                    if wait > 0:
                        time.sleep(wait)
                last_started = time.monotonic()
                requests += 1
                try:
                    mapped = provider.extract_batch([c for c, _ in batch], state["requirements"])
                    error = None
                except (ProviderError, ValidationError, ValueError) as exc:
                    error = exc
                call = getattr(provider, "last_call", None)
                usage["requests"] += 1
                if call:
                    usage["http_attempts"] += call.get("attempts") or 0
                    usage["prompt_tokens"] += call.get("prompt_tokens") or 0
                    usage["completion_tokens"] += call.get("completion_tokens") or 0
                if error is None:
                    for chunk, fingerprint in batch:
                        candidates = mapped.get(chunk["chunk_id"], [])
                        log = {"chunk_id": chunk["chunk_id"], "status": "OK", "method": "openai", "candidate_count": len(candidates), "candidates": candidates, "batch_number": requests, "model_call": call, "cache_fingerprint": fingerprint}
                        store.put_extraction_unit(state["run_id"], log)
                        logs_by_id[chunk["chunk_id"]] = log
                    continue
                code = getattr(error, "code", None)
                if code in _SPLIT_ON and len(batch) > 1:
                    half = len(batch) // 2
                    queue.extendleft([batch[half:], batch[:half]])
                    usage["split_batches"] += 1
                    continue
                safe = str(error) if isinstance(error, ProviderError) else "Batch extraction or validation failed"
                record(batch, status="ERROR", error=safe, error_code=code, provider_request_id=getattr(error, "request_id", None), batch_number=requests)
            if hasattr(provider, "client"):
                provider.client.close()
        result = {"extraction_log": [logs_by_id[c["chunk_id"]] for c in state["chunks"]]}
        if provider is not None:
            result["model_usage"] = usage
        return result

    def verify_node(state: PipelineState) -> dict:
        provider = provider_factory(state) if state.get("verify_material") else None
        evidence, facts, issues, logs = verify_candidates(state["document"], state["segments"], state["chunks"], state["extraction_log"], state["requirements"], provider, state.get("verify_material", False))
        if provider is not None and hasattr(provider, "client"):
            provider.client.close()
        for log in logs:
            if log["status"] == "ERROR":
                store.mark_unit_error(state["run_id"], log["chunk_id"], log.get("error", "Extraction failed"))
            elif log.get("rejected_candidates"):
                store.mark_unit_rejections(state["run_id"], log["chunk_id"], log["rejected_candidates"])
        grouped: dict[tuple[str, str], list[dict]] = {}
        for item in state["parse_issues"]:
            grouped.setdefault((item["code"], item["message"]), []).append(item)
        parse_issues = [make_issue("PARSE_OR_STRUCTURE_INCOMPLETE", message, document_id=state["document"]["document_id"], categories=["*"], details={"source_code": code, "occurrences": len(items), "pages": sorted({x["page"] for x in items if x.get("page") is not None}), "items": [{k: v for k, v in x.items() if k not in {"code", "message"}} for x in items]}) for (code, message), items in grouped.items()]
        failed = [x for x in logs if x["status"] == "ERROR"]
        extraction_issues = []
        if failed:
            extraction_issues.append(make_issue("EXTRACTION_FAILED", "One or more chunks failed extraction or schema validation; retry or review them before relying on coverage.", document_id=state["document"]["document_id"], categories=["*"], details={"failed_count": len(failed), "chunk_ids": [x["chunk_id"] for x in failed], "error_types": sorted({x.get("error", "Extraction failed") for x in failed})}))
        entities, facts, definition_evidence, issues = resolve_document(state["document"], state["segments"], facts, issues, state["requirements"])
        return {"evidence": evidence + definition_evidence, "facts": facts, "entities": entities, "issues": issues + parse_issues + extraction_issues + state["index_issues"], "extraction_log": logs}

    def reconcile_node(state: PipelineState) -> dict:
        revision, previous = store.get_case_snapshot(state["case_id"], exclude_document_id=state["document"]["document_id"])
        previous_entities = store.get_case_entities(state["case_id"], exclude_document_id=state["document"]["document_id"])
        case_entities, entity_map, entity_issues = resolve_case(previous_entities + state["entities"], state["document"]["document_id"])
        # Issues about other documents' entities alone were raised when those were ingested.
        current = {e["entity_id"] for e in state["entities"]}
        current_case = {entity_map[eid] for eid in current if eid in entity_map}
        entity_issues = [i for i in entity_issues if set(i["details"].get("entity_ids", [])) & current or i["details"].get("entity_id") in current_case]
        findings, issues = reconcile(state["document"], state["facts"], previous, state["issues"] + entity_issues, entity_map, state["contract"].get("schemas"))
        known = {i["issue_id"] for i in issues}
        material = {r["category"]: r["material"] for r in state["requirements"]}
        extra = status_issues(state["document"]["document_id"], state["facts"], material)
        statement_findings, statement_issues = statement_checks(state["document"]["document_id"], state["facts"], state["evidence"], state["segments"], material)
        inflow_findings, inflow_issues = inflow_checks(state["document"]["document_id"], state["facts"], previous)
        identity_findings, identity_issues = identity_checks(state["document"]["document_id"], state["facts"], state["segments"], material, state["contract"]["policy"].get("identity_checks"))
        screening_findings, screening_issues = report_checks(state["document"]["document_id"], state["facts"], previous, material, state["contract"]["policy"].get("screening"), entity_map=entity_map)
        prior_findings, prior_issues = prior_checks(state["document"]["document_id"], state["facts"], previous, entity_map)
        media_findings, media_issues = media_checks(state["document"]["document_id"], state["facts"], previous, entity_map)
        extra += statement_issues + inflow_issues + identity_issues + screening_issues + prior_issues + media_issues
        check_findings = statement_findings + inflow_findings + identity_findings + screening_findings + prior_findings + media_findings
        screening: list[dict] = []
        if state.get("screening_lists"):
            # Only the entities this document names are screened; the others were screened with their own documents.
            screening, list_issues = screen_entities(state["document"]["document_id"], case_entities, current_case, [load_list(p) for p in state["screening_lists"]], state["contract"]["policy"].get("screening"))
            extra += list_issues
        checks: list[dict] = []
        sources = registry_sources(state.get("registry_check"))
        if sources:
            # Only the entities this document is about are looked up.
            facts_by_entity: dict[str, list[dict]] = {}
            held_by_entity: dict[str, list[dict]] = {}
            for f in previous + state["facts"]:
                for role, index in (("subject", facts_by_entity), ("object", held_by_entity)):
                    link = f.get("entity_links", {}).get(role)
                    if link and link["entity_id"] in entity_map:
                        index.setdefault(entity_map[link["entity_id"]], []).append(f)
            for source in sources:
                try:
                    client = registry_factory(state, source)
                    lookup = cached_lookup(store, client, "GLEIF" if source == "gleif" else "COMPANIES_HOUSE")
                except RegistryLookupError as exc:
                    # A missing API key fails every lookup visibly rather than skipping the check.
                    client, error = None, exc

                    def lookup(_: str, error: RegistryLookupError = error) -> tuple[dict | None, str, bool]:
                        raise error
                try:
                    if source == "gleif":
                        source_checks, source_issues = check_gleif(state["document"]["document_id"], case_entities, facts_by_entity, current_case, lookup)
                    else:
                        source_checks, source_issues = check_companies_house(state["document"]["document_id"], case_entities, facts_by_entity, held_by_entity, previous + state["facts"], current_case, lookup)
                finally:
                    if hasattr(client, "client"):
                        client.client.close()
                checks += source_checks
                extra += source_issues
        # The graph itself is rebuilt from the stored case on export; here it only raises issues for review.
        # It runs after the register checks so that this run's PSCs are seen.
        previous_checks = store.get_case_registry_checks(state["case_id"], exclude_document_id=state["document"]["document_id"])
        _, graph_issues = build_ownership_graph(previous + state["facts"], case_entities, entity_map, issues, document_id=state["document"]["document_id"], current_fact_ids={f["fact_id"] for f in state["facts"]}, settings=graph_settings(state["contract"]["policy"]), registry_checks=previous_checks + checks)
        graph_issues = [i for i in graph_issues if i["issue_id"] not in known]
        known |= {i["issue_id"] for i in graph_issues}
        extra = [i for i in {i["issue_id"]: i for i in extra}.values() if i["issue_id"] not in known]
        new = graph_issues + extra
        return {"findings": findings + check_findings + [issue_finding(i) for i in new], "issues": issues + new, "case_revision": revision, "case_entities": case_entities, "registry_checks": checks, "screening_checks": screening}

    def review_node(state: PipelineState) -> dict:
        items = [i for i in state["issues"] if i["status"] == "OPEN" and i["severity"] == "high"]
        if not items:
            return {"review_decisions": []}
        fact_map = {f["fact_id"]: f for f in state["facts"]}
        # Cross-document conflict facts are already persisted and must also be shown.
        fact_map.update({f["fact_id"]: f for f in store.get_current_facts(state["case_id"], exclude_document_id=state["document"]["document_id"])})
        related = sorted({fid for item in items for fid in item["fact_ids"]})
        related_facts = [fact_map[fid] for fid in related if fid in fact_map]
        ev_map = {e["evidence_id"]: e for e in store.export_case(state["case_id"])["evidence"]}
        ev_map.update({e["evidence_id"]: e for e in state["evidence"]})
        related_evidence_ids = sorted({eid for fact in related_facts for eid in fact["evidence_ids"]})
        request = {"run_id": state["run_id"], "case_id": state["case_id"], "document_id": state["document"]["document_id"], "issues": items, "facts": related_facts, "evidence": [ev_map[eid] for eid in related_evidence_ids if eid in ev_map], "instructions": "Review every issue against the source; resolve only where permitted, or defer. A selection is a self-reported human judgment, not an authenticated KYC decision."}
        response = interrupt(request)
        decisions = validate_review_response(response, state["issues"])
        by_issue = {d["issue_id"]: d for d in decisions}
        issues = []
        for item in state["issues"]:
            issue = dict(item)
            decision = by_issue.get(issue["issue_id"])
            if decision and decision["action"] == "resolve":
                issue["status"] = "RESOLVED_BY_HUMAN"
                issue["selected_fact_ids"] = decision["selected_fact_ids"]
            issues.append(issue)
        issue_map = {i["issue_id"]: i for i in issues}
        findings = [{**f, "status": issue_map[f["issue_id"]]["status"]} if f.get("issue_id") in issue_map else f for f in state["findings"]]
        return {"issues": issues, "findings": findings, "review_decisions": decisions}

    def coverage_node(state: PipelineState) -> dict:
        coverage = compute_coverage(state["requirements"], state["chunks"], state["extraction_log"], state["facts"], state["issues"], state["extractor"])
        sections = complete_section_inventory(state["sections"], state["chunks"], state["extraction_log"])
        return {"coverage": coverage, "sections": sections}

    def persist_node(state: PipelineState) -> dict:
        store.persist_run(state)
        return {"persisted": True}

    graph = StateGraph(PipelineState)
    graph.add_node("parse_and_classify", parse_node)
    graph.add_node("structure_and_chunk", chunk_node)
    graph.add_node("index", index_node)
    graph.add_node("extract_document_wide", extract_node)
    graph.add_node("validate_source", verify_node)
    graph.add_node("reconcile", reconcile_node)
    graph.add_node("human_review", review_node)
    graph.add_node("coverage_ledger", coverage_node)
    graph.add_node("persist", persist_node)
    graph.add_edge(START, "parse_and_classify")
    graph.add_edge("parse_and_classify", "structure_and_chunk")
    graph.add_edge("structure_and_chunk", "index")
    graph.add_edge("index", "extract_document_wide")
    graph.add_edge("extract_document_wide", "validate_source")
    graph.add_edge("validate_source", "reconcile")
    graph.add_edge("reconcile", "human_review")
    graph.add_edge("human_review", "coverage_ledger")
    graph.add_edge("coverage_ledger", "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer)
