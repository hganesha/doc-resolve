from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from kyc_pipeline.extraction import OpenAICompatible, ProviderError, provider_endpoint
from kyc_pipeline.models import BatchExtractionResponse, strict_schema
from test_graph_failure_paths import FakeProvider, make_state, run_graph

REQS = [{"category": "ownership", "description": "Ownership", "applicable": True}, {"category": "control", "description": "Control", "applicable": True}]
CHUNKS = [{"chunk_id": "c1", "section_path": ["Ownership"], "text": "Acme owns 65% of Beta.", "table_header": None}, {"chunk_id": "c2", "section_path": ["Ownership"], "text": "Beta owns 10% of Gamma.", "table_header": None}]


def _ok(content, **headers):
    return httpx.Response(200, headers=headers, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(content)}}], "usage": {"prompt_tokens": 120, "completion_tokens": 7}})


def _provider(monkeypatch, handle, *, base_url="https://example.invalid/v1", **kwargs):
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return OpenAICompatible(allow_remote=True, model="fake", client=httpx.Client(transport=httpx.MockTransport(handle)), **kwargs)


@pytest.fixture
def sleeps(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr("time.sleep", waits.append)
    return waits


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def test_strict_schema_closes_every_object_and_keeps_property_names():
    schema = strict_schema(BatchExtractionResponse)
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False and obj["required"] == list(obj["properties"])
    text = json.dumps(schema)
    assert all(f'"{k}"' not in text for k in ("maxLength", "minLength", "pattern", "default", "maxItems"))
    fact = schema["$defs"]["CandidateFact"]["properties"]
    assert fact["object"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    class Odd(BaseModel):
        pattern: str
        format: int | None = None
    assert strict_schema(Odd)["required"] == ["pattern", "format"]


def test_strict_request_restricts_chunks_and_categories(monkeypatch, sleeps):
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return _ok({"claims": []})
    provider = _provider(monkeypatch, handle)
    assert provider.extract_batch(CHUNKS, REQS) == {"c1": [], "c2": []}
    body = seen[0]
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["name"] == "BatchExtractionResponse" and fmt["json_schema"]["strict"] is True
    defs = fmt["json_schema"]["schema"]["$defs"]
    assert defs["BatchClaim"]["properties"]["chunk_id"]["enum"] == ["c1", "c2"]
    assert defs["CandidateFact"]["properties"]["category"]["enum"] == ["control", "ownership"]
    # The schema travels in response_format, not in the prompt.
    assert "schema" not in json.loads(body["messages"][1]["content"])
    assert body["temperature"] == 0 and "reasoning_effort" not in body
    assert provider.last_call == {"request_id": None, "attempts": 1, "prompt_tokens": 120, "completion_tokens": 7}


def test_json_object_mode_reasoning_and_token_parameter(monkeypatch, sleeps):
    seen = []
    def handle(request):
        seen.append(json.loads(request.content))
        return _ok({"facts": []})
    provider = _provider(monkeypatch, handle, output_mode="json_object", reasoning_effort="low", max_output_tokens=500)
    provider.extract(CHUNKS[0], REQS)
    body = seen[0]
    assert body["response_format"] == {"type": "json_object"} and "schema" in json.loads(body["messages"][1]["content"])
    # Reasoning models accept only their default temperature; a custom server gets max_tokens.
    assert body["reasoning_effort"] == "low" and "temperature" not in body and body["max_tokens"] == 500
    official = _provider(monkeypatch, handle, base_url="https://api.openai.com/v1", max_output_tokens=500)
    official.extract(CHUNKS[0], REQS)
    assert seen[1]["max_completion_tokens"] == 500 and "max_tokens" not in seen[1]
    with pytest.raises(ValueError, match="output mode"):
        _provider(monkeypatch, handle, output_mode="text")


def test_rate_limits_back_off_and_honour_retry_after(monkeypatch, sleeps):
    replies = iter([
        httpx.Response(429, headers={"retry-after-ms": "1500"}, json={"error": {"code": "rate_limit_exceeded"}}),
        httpx.Response(503, headers={"retry-after": "2"}),
        httpx.Response(500),
        _ok({"facts": []}, **{"x-request-id": "req_abc123"}),
    ])
    provider = _provider(monkeypatch, lambda request: next(replies))
    assert provider.extract(CHUNKS[0], REQS) == []
    # Without Retry-After, the third wait is 2s with jitter of up to half.
    assert sleeps[:2] == [1.5, 2.0] and 1.0 <= sleeps[2] <= 2.0
    assert provider.last_call["attempts"] == 4 and provider.last_call["request_id"] == "req_abc123"


def test_exhausted_quota_and_bad_requests_fail_at_once_without_leaking_bodies(monkeypatch, sleeps):
    calls = []
    def handle(request):
        calls.append(1)
        return httpx.Response(429, json={"error": {"code": "insufficient_quota", "message": "secret account detail"}})
    with pytest.raises(ProviderError) as info:
        _provider(monkeypatch, handle).extract(CHUNKS[0], REQS)
    assert len(calls) == 1 and info.value.code == "insufficient_quota" and "secret" not in str(info.value)
    bad = httpx.Response(400, headers={"x-request-id": "req_9"}, json={"error": {"code": "unsupported_parameter", "param": "temperature", "message": "Acme owns 65% of Beta"}})
    with pytest.raises(ProviderError) as info:
        _provider(monkeypatch, lambda request: bad).extract(CHUNKS[0], REQS)
    assert str(info.value) == "Provider returned HTTP 400 (unsupported_parameter: temperature)"
    assert (info.value.status, info.value.request_id) == (400, "req_9") and not sleeps
    # Only short identifier-like codes are echoed.
    odd = httpx.Response(400, json={"error": {"code": "Acme owns 65% of Beta", "param": None}})
    with pytest.raises(ProviderError, match=r"^Provider returned HTTP 400$"):
        _provider(monkeypatch, lambda request: odd).extract(CHUNKS[0], REQS)


def test_timeouts_retry_then_fail_with_a_code(monkeypatch, sleeps):
    def handle(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(ProviderError, match="timed out") as info:
        _provider(monkeypatch, handle, max_retries=2).extract(CHUNKS[0], REQS)
    assert info.value.code == "timeout" and len(sleeps) == 2
    sleeps.clear()
    with pytest.raises(ProviderError):
        _provider(monkeypatch, handle, max_retries=0).extract(CHUNKS[0], REQS)
    assert not sleeps


@pytest.mark.parametrize("choice, code", [
    ({"finish_reason": "length", "message": {"content": "{\"facts\": ["}}, "output_truncated"),
    ({"finish_reason": "content_filter", "message": {"content": None}}, "content_filter"),
    ({"finish_reason": "stop", "message": {"content": None, "refusal": "I can't help with that."}}, "refusal"),
    ({"finish_reason": "stop", "message": {"content": "{\"facts\": [{\"category\": \"ownership\"}]}"}}, "invalid_output"),
])
def test_unusable_outputs_fail_closed_with_a_reason(monkeypatch, sleeps, choice, code):
    provider = _provider(monkeypatch, lambda request: httpx.Response(200, json={"choices": [choice]}))
    with pytest.raises(ProviderError) as info:
        provider.extract(CHUNKS[0], REQS)
    assert info.value.code == code


def test_azure_openai_uses_deployments_api_version_and_api_key(monkeypatch, sleeps):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://kyc-test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test-key")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    seen = []
    def handle(request):
        seen.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})
        return _ok({"facts": []}, **{"apim-request-id": "0f9a-11"})
    provider = OpenAICompatible(allow_remote=True, provider="azure_openai", model="kyc-gpt", embedding_model="kyc-embed", client=httpx.Client(transport=httpx.MockTransport(handle)), max_output_tokens=900)
    provider.extract(CHUNKS[0], REQS)
    provider.embed(["text"])
    chat, embed = seen
    assert str(chat.url) == "https://kyc-test.openai.azure.com/openai/deployments/kyc-gpt/chat/completions?api-version=2024-10-21"
    assert embed.url.path == "/openai/deployments/kyc-embed/embeddings"
    assert chat.headers["api-key"] == "azure-test-key" and "authorization" not in chat.headers
    assert json.loads(chat.content)["max_completion_tokens"] == 900
    assert provider.last_call["request_id"] == "0f9a-11"
    # A prompt blocked by Azure's content filter is reported as such, not as a generic 400.
    blocked = httpx.Response(400, json={"error": {"code": "content_filter", "message": "The prompt was filtered", "innererror": {"code": "ResponsibleAIPolicyViolation"}}})
    provider.client = httpx.Client(transport=httpx.MockTransport(lambda request: blocked))
    with pytest.raises(ProviderError, match="content filter blocked") as info:
        provider.extract(CHUNKS[0], REQS)
    assert info.value.code == "content_filter"


def test_azure_settings_are_checked_before_any_request(monkeypatch):
    for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_AD_TOKEN", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        OpenAICompatible(allow_remote=True, provider="azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://kyc-test.openai.azure.com")
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN"):
        OpenAICompatible(allow_remote=True, provider="azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "entra-token")
    provider = OpenAICompatible(allow_remote=True, provider="azure_openai")
    assert provider._headers() == {"Authorization": "Bearer entra-token"}
    with pytest.raises(ProviderError, match="deployment name"):
        provider.extract(CHUNKS[0], REQS)
    before = provider_endpoint("azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    assert provider_endpoint("azure_openai") != before


class TruncatingProvider(FakeProvider):
    """Truncates any batch of more than one chunk, and filters one chunk outright."""
    def __init__(self):
        super().__init__()
        self.mode = "rules"
        self.batches = []
        self.last_call = None

    def extract_batch(self, chunks, requirements):
        self.batches.append([c["chunk_id"] for c in chunks])
        self.last_call = {"request_id": f"req_{len(self.batches)}", "attempts": 1, "prompt_tokens": 100, "completion_tokens": 10}
        if len(chunks) > 1:
            raise ProviderError("Provider output was truncated; extraction cannot be treated as complete", code="output_truncated")
        if "Cedar" in chunks[0]["text"]:
            raise ProviderError("Provider content filter withheld the output", code="content_filter", request_id="req_filtered")
        return super().extract_batch(chunks, requirements)


def test_truncated_batches_are_split_and_a_filtered_chunk_fails_alone(tmp_path):
    fake = TruncatingProvider()
    state = make_state(tmp_path)
    (tmp_path / "source.txt").write_text("# Ownership\nAster Holdings Ltd owns 65% of Birch Ltd.\n\n# Control\nCedar Ltd has voting rights over Dune Ltd.\n\n# Other\nBirch Ltd owns 40% of Elm Ltd.\n")
    store, graph, config, result = run_graph(tmp_path, fake, state)
    run = store.get_extraction_units(state["run_id"])
    # The first request held every chunk; halving ends with one request per chunk.
    assert sorted(fake.batches[0]) == sorted(run) and len(run) > 2
    assert sorted(b[0] for b in fake.batches if len(b) == 1) == sorted(run)
    errors = [u for u in run.values() if u["status"] == "ERROR"]
    assert len(errors) == 1 and errors[0]["error_code"] == "content_filter" and errors[0]["provider_request_id"] == "req_filtered"
    ok = [u for u in run.values() if u["status"] == "OK"]
    assert ok and all(u["model_call"]["attempts"] == 1 for u in ok)
    usage = (result.get("__interrupt__") and graph.get_state(config).values or result)["model_usage"]
    assert usage["requests"] == len(fake.batches) and usage["split_batches"] >= 1 and usage["prompt_tokens"] == 100 * len(fake.batches)


def test_split_requests_count_against_the_budget(tmp_path):
    fake = TruncatingProvider()
    state = make_state(tmp_path, max_model_requests=1)
    (tmp_path / "source.txt").write_text("# Ownership\nAster Holdings Ltd owns 65% of Birch Ltd.\n\n# Other\nBirch Ltd owns 40% of Elm Ltd.\n")
    store, *_ = run_graph(tmp_path, fake, state)
    units = store.get_extraction_units(state["run_id"]).values()
    assert len(fake.batches) == 1 and {u["status"] for u in units} == {"NOT_REVIEWED"}
