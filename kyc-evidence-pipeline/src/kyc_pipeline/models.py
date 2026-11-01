from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .util import normalize_date


class SupportingReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str = Field(min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=1200)


class CandidateFact(BaseModel):
    """Model output is untrusted until its citations and basic fields are checked."""
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=250)
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    object: str | None = Field(default=None, max_length=250)
    value: float | str | None = None
    unit: str | None = Field(default=None, max_length=80)
    effective_date: str | None = None
    date_role: Literal["as_of", "event", "reporting_period", "unspecified"] = "unspecified"
    claim_type: Literal["assertion", "reported_statement", "allegation", "finding", "order", "disposition", "denial"] = "reported_statement"
    qualifier: str | None = Field(default=None, max_length=150)
    attribution: str | None = Field(default=None, max_length=250)
    source_quote: str = Field(min_length=1, max_length=1200)
    supporting_quotes: list[str] = Field(default_factory=list, max_length=4)
    supporting_refs: list[SupportingReference] = Field(default_factory=list, max_length=6)

    @field_validator("effective_date", mode="before")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("effective_date must be a string")
            return normalize_date(value)
        return value

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.strip().casefold() in {"%", "percent", "percentage", "per cent"}:
            return "percent"
        return value

    @field_validator("subject", "source_quote")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("supporting_quotes")
    @classmethod
    def valid_quotes(cls, value: list[str]) -> list[str]:
        if any(not s.strip() or len(s) > 1200 for s in value):
            raise ValueError("supporting quotes must contain 1–1200 characters")
        return value


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: list[CandidateFact] = Field(max_length=100)


class BatchClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(min_length=1, max_length=100)
    fact: CandidateFact


class BatchExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[BatchClaim] = Field(max_length=200)


class SupportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    support: Literal["SUPPORTED", "UNSUPPORTED", "UNCERTAIN"]
    explanation: str = Field(max_length=500)


# Keywords OpenAI strict structured output rejects or may reject. They are
# dropped from the request schema only; responses are still validated against
# the full pydantic model, so lengths and patterns are enforced locally.
_NOT_STRICT = {"default", "title", "minLength", "maxLength", "pattern", "format", "minItems", "maxItems"}


def strict_schema(model: type[BaseModel]) -> dict:
    """The model's JSON schema in the subset accepted by strict structured output.

    Every property is required and every object is closed; fields that are
    optional in the model stay nullable, and fields with a non-null default
    must be given explicitly.
    """
    def walk(node, *, names: bool = False):
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if names:
            return {k: walk(v) for k, v in node.items()}
        out = {k: walk(v, names=k in {"properties", "$defs"}) for k, v in node.items() if k not in _NOT_STRICT}
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out
    return walk(model.model_json_schema())
