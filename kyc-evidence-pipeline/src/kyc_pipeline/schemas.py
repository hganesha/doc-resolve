from __future__ import annotations

import re
from typing import Any

VALUE_TYPES = {"none", "percent", "amount", "count", "text"}
_DATE_ROLES = {"as_of", "event", "reporting_period", "unspecified"}
_CLAIM_TYPES = {"assertion", "reported_statement", "allegation", "finding", "order", "disposition", "denial"}
_NUMERIC = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def validate_schemas(data: dict[str, Any]) -> dict[str, dict]:
    predicates = data.get("predicates")
    if not isinstance(predicates, dict) or not predicates:
        raise ValueError("Extraction schemas must contain a predicates mapping")
    for name, spec in predicates.items():
        where = f"extraction schema {name!r}"
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,79}", name):
            raise ValueError(f"Invalid predicate name in {where}")
        if not isinstance(spec, dict) or not isinstance(spec.get("category"), str) or not isinstance(spec.get("description"), str) or not spec["description"].strip():
            raise ValueError(f"{where} needs a category and a description")
        unknown = set(spec) - {"category", "description", "object", "value", "value_required", "date", "date_roles", "claim_types", "evidence_terms", "single_valued", "magnitude", "subject_name_parts"}
        if unknown:
            raise ValueError(f"Unknown fields {sorted(unknown)} in {where}")
        if spec.get("object", "optional") not in {"required", "optional", "none"}:
            raise ValueError(f"{where}: object must be required, optional or none")
        if spec.get("value", "none") not in VALUE_TYPES:
            raise ValueError(f"{where}: value must be one of {sorted(VALUE_TYPES)}")
        if spec.get("date", "optional") not in {"required", "optional"}:
            raise ValueError(f"{where}: date must be required or optional")
        for key in ("value_required", "magnitude", "subject_name_parts"):
            if not isinstance(spec.get(key, False), bool):
                raise ValueError(f"{where}: {key} must be true or false")
        if spec.get("magnitude") and spec.get("value", "none") not in {"amount", "count"}:
            raise ValueError(f"{where}: magnitude needs an amount or count value")
        single = spec.get("single_valued", False)
        if not (isinstance(single, bool) or single == "per_object"):
            raise ValueError(f"{where}: single_valued must be true, false or per_object")
        if spec.get("single_valued") == "per_object" and (spec.get("object", "optional") != "required" or spec.get("value", "none") == "none"):
            raise ValueError(f"{where}: single_valued per_object needs a required object and a value type")
        if spec.get("value_required") and spec.get("value", "none") == "none":
            raise ValueError(f"{where}: value_required needs a value type")
        for key, allowed in (("date_roles", _DATE_ROLES), ("claim_types", _CLAIM_TYPES)):
            values = spec.get(key)
            if values is not None and (not isinstance(values, list) or not values or not set(values) <= allowed):
                raise ValueError(f"{where}: {key} must be a nonempty list from {sorted(allowed)}")
        if "evidence_terms" in spec:
            try:
                re.compile(spec["evidence_terms"])
            except (re.error, TypeError):
                raise ValueError(f"{where}: evidence_terms must be a valid regular expression") from None
    return predicates


def coerce_value(spec: dict | None, value: Any) -> Any:
    """'27' or '1,234.5' becomes a number where the schema expects one; anything else is left for check_fact."""
    if spec and spec.get("value") in {"percent", "amount", "count"} and isinstance(value, str) and _NUMERIC.fullmatch(value.strip()):
        return float(value.strip().replace(",", ""))
    return value


def check_fact(spec: dict, fact: dict) -> list[str]:
    """Field-level violations of a predicate's schema. An empty list is not a claim that the fact is true."""
    violations = []
    obj, value, unit = fact.get("object"), fact.get("value"), fact.get("unit")
    if spec.get("object", "optional") == "required" and not (obj or "").strip():
        violations.append("object is required")
    if spec.get("object") == "none" and obj:
        violations.append("object must be empty")
    kind = spec.get("value", "none")
    if value is None:
        if spec.get("value_required"):
            violations.append(f"a {kind} value is required")
    elif kind == "none":
        violations.append("value must be empty")
    elif kind == "text":
        if not isinstance(value, str):
            violations.append("value must be text")
    elif not isinstance(value, (int, float)):
        violations.append(f"value must be a number ({kind})")
    elif kind == "percent":
        if unit != "percent":
            violations.append("unit must be percent; a currency amount may be cost or valuation, not a stake")
        if not 0 <= value <= 100:
            violations.append("percentage must be from 0 to 100")
    elif kind == "amount" and not (unit or "").strip():
        violations.append("an amount needs a currency or unit")
    elif kind == "count" and value < 0:
        violations.append("a count cannot be negative")
    if spec.get("magnitude") and isinstance(value, (int, float)) and value < 0:
        violations.append("value must not be negative; the predicate gives the direction")
    if spec.get("date") == "required" and not fact.get("effective_date"):
        violations.append("a date or reporting period is required")
    if spec.get("date_roles") and fact.get("date_role", "unspecified") not in spec["date_roles"]:
        violations.append(f"date_role must be one of {', '.join(spec['date_roles'])}")
    if spec.get("claim_types") and fact.get("claim_type", "reported_statement") not in spec["claim_types"]:
        violations.append(f"claim_type must be one of {', '.join(spec['claim_types'])}")
    return violations


def predicates_for(schemas: dict[str, dict], category: str) -> dict[str, dict]:
    return {name: spec for name, spec in schemas.items() if spec["category"] == category}
