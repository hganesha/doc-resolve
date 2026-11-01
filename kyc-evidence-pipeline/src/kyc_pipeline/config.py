from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import predicates_for, validate_schemas
from .util import digest

CONFIG_DIR = Path(__file__).parent / "configs"
BOUNDARIES = {"whitespace", "sentence", "clause"}
CONTEXT_UNITS = {"none", "parent_section", "document_header"}


def _read_yaml(path: str | Path) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML object: {path}")
    return data, digest(raw)


def load_contract(profiles_path: str | None = None, policy_path: str | None = None, schemas_path: str | None = None) -> dict[str, Any]:
    profiles_data, profile_hash = _read_yaml(profiles_path or CONFIG_DIR / "document_profiles.yaml")
    policy, policy_hash = _read_yaml(policy_path or CONFIG_DIR / "kyc_policy.yaml")
    schemas_data, schemas_hash = _read_yaml(schemas_path or CONFIG_DIR / "extraction_schemas.yaml")
    profiles = profiles_data.get("profiles")
    categories = policy.get("categories")
    families = policy.get("families")
    if not isinstance(profiles, dict) or not isinstance(categories, dict) or not isinstance(families, dict):
        raise ValueError("Profiles and policy must contain profiles, categories, and families mappings")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ValueError("The policy must have a nonempty policy_id")
    for pid, profile in profiles.items():
        if not isinstance(profile, dict) or profile.get("family") not in families:
            raise ValueError(f"Profile {pid!r} has an unknown policy family")
        max_chars, overlap = profile.get("max_chars"), profile.get("overlap_chars")
        if not isinstance(max_chars, int) or max_chars < 256 or not isinstance(overlap, int) or not 0 <= overlap < max_chars // 2:
            raise ValueError(f"Invalid chunk sizes in profile {pid!r}")
        if not isinstance(profile.get("signals"), list) or not all(isinstance(s, str) and s.strip() for s in profile["signals"]):
            raise ValueError(f"Invalid classification signals in profile {pid!r}")
        if profile.get("boundary", "whitespace") not in BOUNDARIES or profile.get("context_unit", "none") not in CONTEXT_UNITS or not isinstance(profile.get("keep_records_whole", False), bool):
            raise ValueError(f"Invalid chunking strategy settings in profile {pid!r}: boundary must be one of {sorted(BOUNDARIES)}, context_unit one of {sorted(CONTEXT_UNITS)}, keep_records_whole true or false")
    for family, requirements in families.items():
        if not isinstance(requirements, list) or not requirements:
            raise ValueError(f"Policy family {family!r} must have requirements")
        seen: set[str] = set()
        for item in requirements:
            name = item if isinstance(item, str) else item.get("category") if isinstance(item, dict) else None
            if name not in categories or name in seen:
                raise ValueError(f"Unknown or duplicate category {name!r} in policy family {family!r}")
            seen.add(name)
            if isinstance(item, dict) and item.get("applicable", True) is False and not item.get("reason"):
                raise ValueError(f"NOT_APPLICABLE needs a documented reason: {family}.{name}")
    ownership_graph = policy.get("ownership_graph", {})
    if not isinstance(ownership_graph, dict):
        raise ValueError("ownership_graph must be a mapping")
    for key in ("reporting_threshold_percent", "control_threshold_percent"):
        value = ownership_graph.get(key, 50)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 100:
            raise ValueError(f"ownership_graph.{key} must be a percentage above 0 and at most 100")
    depth = ownership_graph.get("max_chain_depth", 8)
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 20:
        raise ValueError("ownership_graph.max_chain_depth must be an integer from 1 to 20")
    identity = policy.get("identity_checks", {})
    age = identity.get("proof_of_address_max_age_days", 90) if isinstance(identity, dict) else None
    if isinstance(age, bool) or not isinstance(age, int) or not 1 <= age <= 3650:
        raise ValueError("identity_checks.proof_of_address_max_age_days must be an integer from 1 to 3650")
    screening = policy.get("screening", {})
    if not isinstance(screening, dict):
        raise ValueError("screening must be a mapping")
    days, threshold = screening.get("max_age_days", 90), screening.get("name_match_threshold", 0.88)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 3650:
        raise ValueError("screening.max_age_days must be an integer from 1 to 3650")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.5 <= threshold <= 1:
        raise ValueError("screening.name_match_threshold must be a number from 0.5 to 1")
    retention = policy.setdefault("retention", {})
    if not isinstance(retention, dict):
        raise ValueError("retention must be a mapping")
    for key, default in (("case_days_after_last_activity", 1825), ("unfinished_run_days", 90), ("cache_days", 30)):
        value = retention.setdefault(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 36500:
            raise ValueError(f"retention.{key} must be a number of days from 1 to 36500")
    for name, category in categories.items():
        if not isinstance(category, dict) or not isinstance(category.get("description"), str) or not isinstance(category.get("material"), bool):
            raise ValueError(f"Invalid category {name!r}")
    schemas = validate_schemas(schemas_data)
    for name, spec in schemas.items():
        if spec["category"] not in categories:
            raise ValueError(f"Extraction schema {name!r} names unknown policy category {spec['category']!r}")
    return {"profiles": profiles, "policy": policy, "schemas": schemas, "profiles_sha256": profile_hash, "policy_sha256": policy_hash, "schemas_sha256": schemas_hash}


def select_profile(contract: dict[str, Any], requested: str | None, filename: str, sample: str) -> tuple[str, dict[str, Any]]:
    profiles = contract["profiles"]
    if requested:
        if requested not in profiles:
            raise ValueError(f"Unknown profile {requested!r}; available: {', '.join(sorted(profiles))}")
        return requested, profiles[requested]
    text = (filename.replace("_", " ").replace("-", " ") + "\n" + sample[:20000]).casefold()
    scores = [(sum(1 for signal in p["signals"] if signal.casefold() in text), name) for name, p in profiles.items()]
    scores.sort(reverse=True)
    if not scores or scores[0][0] == 0 or (len(scores) > 1 and scores[0][0] == scores[1][0]):
        raise ValueError("Document profile is ambiguous; select one explicitly with --profile")
    return scores[0][1], profiles[scores[0][1]]


def requirements_for(contract: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Policy requirements with each category's typed predicates; a category with none is open."""
    policy = contract["policy"]
    # Contracts stored before typed schemas existed have none: every category is open.
    schemas = contract.get("schemas") or {}
    requirements = []
    for item in policy["families"][profile["family"]]:
        entry = {"category": item} if isinstance(item, str) else dict(item)
        category = policy["categories"][entry["category"]]
        requirements.append({
            **entry,
            "description": category["description"],
            "material": category["material"],
            "applicable": entry.get("applicable", True),
            "predicates": predicates_for(schemas, entry["category"]),
        })
    return requirements
