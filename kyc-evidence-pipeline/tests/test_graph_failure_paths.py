from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from kyc_pipeline.config import load_contract
from kyc_pipeline.extraction import ProviderError, rules_extract
from kyc_pipeline.graph import build_graph
from kyc_pipeline.store import EvidenceStore
from kyc_pipeline.util import utc_now


class FakeProvider:
    def __init__(self):
        self.mode = "empty"
        self.calls = []
        self.fail = False
        self.support = "SUPPORTED"

    def extract(self, chunk, requirements):
        self.calls.append(chunk["chunk_id"])
        if self.fail and "65%" in chunk["text"]:
            raise ProviderError("Provider returned HTTP 503")
        if self.mode == "empty":
            return []
        if self.mode == "wrong_quote" and "65%" in chunk["text"]:
            return [{"category": "ownership", "subject": "Aster Holdings Ltd", "predicate": "owns_equity_in", "object": "Birch Ltd", "value": 99, "unit": "percent", "source_quote": "Aster owns 99% of Birch"}]
        return rules_extract(chunk, requirements)

    def extract_batch(self, chunks, requirements):
        return {chunk["chunk_id"]: self.extract(chunk, requirements) for chunk in chunks}

    def verify_support(self, fact, evidence):
        return {"support": self.support, "explanation": "Synthetic checker response"}


def make_state(tmp_path, run_id="run_test", **overrides):
    path = tmp_path / "source.txt"
    path.write_text("# Ownership\nAster Holdings Ltd owns 65% of Birch Ltd.\n")
    state = {"run_id": run_id, "case_id": "test", "started_at": utc_now(), "doc_path": str(path), "requested_profile": "ownership_control", "contract": load_contract(), "extractor": "openai", "embeddings": "lexical", "allow_remote": True, "verify_material": False, "model": "fake-model", "embedding_model": "fake-embedding", "retry_from_run": None, "expected_document_sha256": None}
    state.update(overrides)
    return state


def run_graph(tmp_path, fake, state):
    store = EvidenceStore(tmp_path / "db.sqlite")
    store.start_run(state["run_id"], state["case_id"], state["started_at"])
    graph = build_graph(store, InMemorySaver(), provider_factory=lambda _: fake)
    config = {"configurable": {"thread_id": state["run_id"]}}
    result = graph.invoke(state, config)
    return store, graph, config, result


def test_model_empty_pass_marks_automated_not_found_not_proof(tmp_path):
    fake = FakeProvider()
    state = make_state(tmp_path)
    store, graph, config, result = run_graph(tmp_path, fake, state)
    assert "__interrupt__" not in result
    package = store.export_case("test")
    assert all(c["status"] == "NOT_FOUND_AFTER_REVIEW" for c in package["coverage"])
    assert all(c["review_method"] == "automated_model_pass" and not c["completeness_asserted"] for c in package["coverage"])
    assert package["status"] == "REVIEW_REQUIRED"


def test_hallucinated_citation_escalates_and_fails_only_its_category(tmp_path):
    fake = FakeProvider(); fake.mode = "wrong_quote"
    state = make_state(tmp_path)
    store, graph, config, result = run_graph(tmp_path, fake, state)
    request = result["__interrupt__"][0].value
    rejected = next(i for i in request["issues"] if i["code"] == "CANDIDATE_CITATION_REJECTED")
    assert rejected["categories"] == ["ownership"] and rejected["details"]["rejected_count"] == 1
    assert not any(i["code"] == "EXTRACTION_FAILED" for i in request["issues"])
    decisions = [{"issue_id": i["issue_id"], "action": "defer", "reason": "Needs another extraction."} for i in request["issues"]]
    graph.invoke(Command(resume={"reviewer_id": "TEST", "decisions": decisions}), config)
    package = store.export_case("test")
    assert package["facts"] == []
    coverage = {c["category"]: c["status"] for c in package["coverage"]}
    assert coverage.pop("ownership") == "EXTRACTION_FAILED"
    assert set(coverage.values()) == {"NOT_FOUND_AFTER_REVIEW"}
    assert package["status"] == "REVIEW_REQUIRED"


def test_retry_reextracts_chunks_with_rejected_citations(tmp_path):
    fake = FakeProvider(); fake.mode = "wrong_quote"
    initial = make_state(tmp_path, run_id="run_first", model_batch_size=1)
    store, graph, config, result = run_graph(tmp_path, fake, initial)
    units = store.get_extraction_units("run_first")
    rejected = {key for key, u in units.items() if u.get("rejected_candidates")}
    assert len(rejected) == 1 and all(u["status"] == "OK" for u in units.values())
    prior = graph.get_state(config).values
    fake.mode = "rules"
    fake.calls.clear()
    retry = make_state(tmp_path, run_id="run_retry", retry_from_run="run_first", expected_document_sha256=prior["document"]["sha256"], model_batch_size=1)
    store.start_run("run_retry", "test", retry["started_at"])
    graph.invoke(retry, {"configurable": {"thread_id": "run_retry"}})
    assert set(fake.calls) == rejected


def test_model_source_support_check_is_not_external_verification(tmp_path):
    fake = FakeProvider(); fake.mode = "rules"; fake.support = "UNSUPPORTED"
    state = make_state(tmp_path, verify_material=True)
    store, graph, config, result = run_graph(tmp_path, fake, state)
    request = result["__interrupt__"][0].value
    assert any(i["code"] == "MATERIAL_SEMANTIC_SUPPORT_UNCERTAIN" for i in request["issues"])
    assert request["facts"][0]["source_support"]["semantic_check"] == "UNSUPPORTED"
    assert request["facts"][0]["verification_status"] == "PENDING_HUMAN"


def test_retry_reuses_successful_chunks_and_retries_only_failed(tmp_path):
    fake = FakeProvider(); fake.mode = "rules"; fake.fail = True
    initial = make_state(tmp_path, run_id="run_first", model_batch_size=1)
    store, graph, config, result = run_graph(tmp_path, fake, initial)
    assert "__interrupt__" in result
    first_units = store.get_extraction_units("run_first")
    failed = {key for key, u in first_units.items() if u["status"] == "ERROR"}
    assert len(failed) == 1
    prior = graph.get_state(config).values
    fake.fail = False
    fake.calls.clear()
    retry = make_state(tmp_path, run_id="run_retry", retry_from_run="run_first", expected_document_sha256=prior["document"]["sha256"], model_batch_size=1)
    store.start_run("run_retry", "test", retry["started_at"])
    result2 = graph.invoke(retry, {"configurable": {"thread_id": "run_retry"}})
    assert "__interrupt__" not in result2
    assert set(fake.calls) == failed
    units = store.get_extraction_units("run_retry")
    assert all(u["status"] == "OK" for u in units.values())
    assert any(u.get("reused_from_run") == "run_first" for u in units.values())


def test_model_request_budget_leaves_explicit_unreviewed_chunks(tmp_path):
    fake = FakeProvider()
    state = make_state(tmp_path, model_batch_size=1, max_model_requests=1)
    store, graph, config, result = run_graph(tmp_path, fake, state)
    assert "__interrupt__" not in result
    package = store.export_case("test")
    assert all(c["status"] == "NOT_REVIEWED" and c["chunks_not_reviewed"] > 0 for c in package["coverage"])
    assert all(not c["all_parsed_chunks_processed"] for c in package["coverage"])
    assert package["status"] == "REVIEW_REQUIRED"


def test_local_vector_index_and_query_stay_in_the_local_provider(tmp_path, monkeypatch):
    import kyc_pipeline.graph as graph_module
    from kyc_pipeline.retrieval import search_chunks
    class FakeLocal:
        def __init__(self, model): self.embedding_model = model
        def embed(self, texts): return [[1.0, 0.0] for _ in texts]
        def embed_query(self, text): return [1.0, 0.0]
    monkeypatch.setattr(graph_module, "LocalEmbeddingProvider", FakeLocal)
    fake = FakeProvider()
    state = make_state(tmp_path, extractor="rules", embeddings="local", embedding_model="local-test", allow_remote=False)
    store, graph, config, result = run_graph(tmp_path, fake, state)
    assert "__interrupt__" not in result
    chunks = store.current_chunks("test")
    assert all(c["embedding_model"] == "local-test" and c["embedding"] == [1.0, 0.0] for c in chunks)
    query = search_chunks(chunks, "Who owns Birch?", provider=FakeLocal("local-test"))
    assert query["method"] == "hybrid_bm25_embedding" and not query["warnings"]
    assert query["results"][0]["parent_context_units"][0]["segment_id"]
    assert not fake.calls
