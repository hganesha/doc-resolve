from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .entities import legal_key, same_person
from .evidence import make_issue
from .reconciliation import APPROXIMATE, party_key, rounding_consistent
from .util import date_interval, identity_key, stable_id, temporal_overlap

# Illustrative defaults; a policy may override them under `ownership_graph`.
DEFAULT_SETTINGS = {"reporting_threshold_percent": 25.0, "control_threshold_percent": 50.0, "max_chain_depth": 8}
_MAX_PATHS = 5000
# Percentage points of rounding allowed before direct stakes "exceed 100%".
_TOLERANCE = 0.5
_NOTE = ("Built from extracted, source-matched claims, all pending human verification, and from register records "
         "where a register check ran. Stakes are ranges: conflicts, rounding and later disposals widen them rather "
         "than choosing a figure. A group's holding is counted as its parent's consolidated interest, not a "
         "look-through economic interest. People are only those the sources name as directors, beneficial owners, "
         "individual holders or register PSCs; an entity with no reported owners may still have them, and a "
         "beneficial owner listed here is a lead for review, not a verified identification.")
# Companies House natures of control that give control on their own.
_PSC_CONTROL = ("right-to-appoint-and-remove-directors", "significant-influence-or-control")



def graph_settings(policy: dict | None) -> dict:
    configured = (policy or {}).get("ownership_graph") or {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in configured.items() if k in DEFAULT_SETTINGS}}


def _percent(fact: dict) -> float | None:
    value = fact.get("value")
    if fact.get("unit") != "percent" or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if 0 < value <= 100 else None


def _latest(facts: list[dict]) -> str | None:
    dated = [f["effective_date"] for f in facts if date_interval(f.get("effective_date"))]
    return max(dated, key=lambda d: date_interval(d)[0]) if dated else None


def _equity_edge(holder: str, held: str, snapshots: list[dict], sales: list[dict], conflicts: dict, open_issues: dict) -> dict:
    as_of = _latest(snapshots)
    # The latest reported period wins; undated claims may be current, so they stay in the range.
    current = [s for s in snapshots if as_of is None or temporal_overlap(s.get("effective_date"), as_of)]
    values = [_percent(s) for s in current]
    low, high = min(values), max(values)
    flags = set()
    if low != high:
        flags.add("RANGE_OF_REPORTED_FIGURES")
    if any(APPROXIMATE.search(s.get("qualifier") or "") for s in current):
        flags.add("APPROXIMATE")
    if any(not s.get("effective_date") for s in current):
        flags.add("UNDATED")
    latest = date_interval(as_of)
    stale, unordered = [], []
    for sale in sales:
        when = date_interval(sale.get("effective_date"))
        if not when or not latest or temporal_overlap(sale["effective_date"], as_of):
            unordered.append(sale)
        elif when[0] > latest[1]:
            stale.append(sale)
        # Otherwise the disposal predates the latest stake, which already reflects it.
    if stale or unordered:
        # A disposal leaves the remaining stake unknown, down to nothing.
        low = 0.0
        flags.add("STALE_AFTER_DISPOSAL" if stale else "DISPOSAL_ORDER_UNKNOWN")
    conflict_ids = sorted({i for s in current for i in conflicts.get(s["fact_id"], ())})
    status = "CONFLICTED" if conflict_ids else "STALE_AFTER_DISPOSAL" if stale else "CHRONOLOGY_UNRESOLVED" if unordered else "REPORTED"
    related = current + sales
    return {
        "edge_id": stable_id("edge", "EQUITY", holder, held), "type": "EQUITY", "holder": holder, "held": held,
        "percent": {"low": low, "high": high}, "as_of": as_of, "status": status, "flags": sorted(flags),
        "fact_ids": sorted(s["fact_id"] for s in current),
        "superseded_fact_ids": sorted(s["fact_id"] for s in snapshots if s not in current),
        "disposal_fact_ids": sorted(s["fact_id"] for s in sales),
        "conflict_issue_ids": conflict_ids,
        "open_issue_ids": sorted({i for f in related for i in open_issues.get(f["fact_id"], ())}),
        "documents": sorted({f["document_id"] for f in related if f.get("document_id")}),
    }


def _hull(edges: list[dict]) -> tuple[float, float] | None:
    return (min(e["percent"]["low"] for e in edges), max(e["percent"]["high"] for e in edges)) if edges else None


def _register_holder(psc: dict, case_entities: list[dict]) -> tuple[str, str]:
    """The case entity a register PSC names, else a register-only node; returns (node key, kind)."""
    individual = "individual" in (psc.get("kind") or "")
    for e in case_entities:
        if e["entity_type"] == "group":
            continue
        names = [n["name"] for n in e["names"]] + ([e["name"]] if e["name"] else [])
        if individual and e["entity_type"] in {"person", "unknown"} and any(same_person(n, psc["name"]) for n in names):
            return e["entity_id"], "person"
        if not individual and e["entity_type"] != "person" and legal_key(psc["name"]) in {legal_key(n) for n in names}:
            return e["entity_id"], e["entity_type"]
    return "register:" + identity_key(psc["name"]), "person" if individual else "organization"


def build_ownership_graph(facts: list[dict], case_entities: list[dict], entity_map: dict[str, str], issues: list[dict], *, document_id: str = "case", current_fact_ids: set[str] | None = None, settings: dict | None = None, registry_checks: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Case ownership and control graph with look-through interests and beneficial owners.

    ``registry_checks`` adds each checked company's current Companies House PSCs.
    Returns (graph, issues). Issues are limited to those touching current_fact_ids when given.
    """
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    threshold, control_threshold, max_depth = float(settings["reporting_threshold_percent"]), float(settings["control_threshold_percent"]), int(settings["max_chain_depth"])
    entities = {e["entity_id"]: e for e in case_entities}
    group_parent = {e["entity_id"]: e["group_of"] for e in case_entities if e["entity_type"] == "group" and e.get("group_of")}

    # A human resolution keeps the selected figure and sets the others aside.
    set_aside: set[str] = set()
    conflicts: dict[str, set[str]] = defaultdict(set)
    open_issues: dict[str, set[str]] = defaultdict(set)
    for issue in issues:
        if issue["code"] == "POSSIBLE_OWNERSHIP_CONFLICT" and issue["status"] == "RESOLVED_BY_HUMAN":
            set_aside |= set(issue["fact_ids"]) - set(issue.get("selected_fact_ids", []))
        if issue["status"] == "OPEN":
            for fid in issue["fact_ids"]:
                open_issues[fid].add(issue["issue_id"])
                if issue["code"] == "POSSIBLE_OWNERSHIP_CONFLICT":
                    conflicts[fid].add(issue["issue_id"])

    nodes: dict[str, dict] = {}

    def add_node(key: str, text: str | None = None, kind: str | None = None, source: str | None = None) -> str:
        if key not in nodes:
            e = entities.get(key)
            nodes[key] = {"node_id": key, "entity_id": key if e else None, "name": (e["name"] if e else None) or text, "kind": e["entity_type"] if e else kind or "unresolved_text", "roles": e["roles"] if e else [], "identified": bool(e and (e["identifiers"] or e["roles"]))}
            if source and not e:
                nodes[key]["source"] = source
        return key

    equity: dict[tuple, list[dict]] = defaultdict(list)
    voting: dict[tuple, list[dict]] = defaultdict(list)
    disposals: dict[tuple, list[dict]] = defaultdict(list)
    declared: dict[tuple, list[dict]] = defaultdict(list)
    directors: dict[str, list[dict]] = defaultdict(list)
    for f in facts:
        predicate = f["predicate"]
        if f["fact_id"] in set_aside or not f.get("object"):
            continue
        if predicate in {"is_beneficial_owner_of", "is_director_of"}:
            pair = (party_key(f, "subject", entity_map), party_key(f, "object", entity_map))
            if all(pair) and pair[0] != pair[1]:
                if predicate == "is_director_of":
                    directors[pair[1]].append(f)
                else:
                    declared[pair].append(f)
                    add_node(pair[0], f["subject"], "person")
                    add_node(pair[1], f["object"])
            continue
        if predicate not in {"owns_equity_in", "has_voting_rights_over", "disposed_interest_in"}:
            continue
        if predicate == "owns_equity_in" and _percent(f) is None:
            continue
        pair = (party_key(f, "subject", entity_map), party_key(f, "object", entity_map))
        if not all(pair) or pair[0] == pair[1]:
            continue
        {"owns_equity_in": equity, "has_voting_rights_over": voting, "disposed_interest_in": disposals}[predicate][pair].append(f)
        if predicate != "disposed_interest_in":
            add_node(pair[0], f["subject"])
            add_node(pair[1], f["object"])

    edges = [_equity_edge(h, d, snaps, disposals.get((h, d), []), conflicts, open_issues) for (h, d), snaps in sorted(equity.items())]
    for (holder, held), claims in sorted(voting.items()):
        figures = [v for v in (_percent(c) for c in claims) if v is not None]
        edges.append({"edge_id": stable_id("edge", "VOTING_RIGHTS", holder, held), "type": "VOTING_RIGHTS", "holder": holder, "held": held, "percent": {"low": min(figures), "high": max(figures)} if figures else None, "as_of": _latest(claims), "status": "REPORTED", "flags": ["UNDATED"] if any(not c.get("effective_date") for c in claims) else [], "fact_ids": sorted(c["fact_id"] for c in claims), "superseded_fact_ids": [], "disposal_fact_ids": [], "conflict_issue_ids": [], "open_issue_ids": sorted({i for c in claims for i in open_issues.get(c["fact_id"], ())}), "documents": sorted({c["document_id"] for c in claims if c.get("document_id")})})
    # Current PSCs from register checks; the latest check of each company wins.
    register: dict[tuple, dict] = {}
    officers: dict[str, list[dict]] = defaultdict(list)
    for check in sorted(registry_checks or [], key=lambda c: c.get("fetched_at") or ""):
        if check.get("source") != "COMPANIES_HOUSE" or check.get("status") != "CHECKED":
            continue
        held = add_node(check["entity_id"], check["entity_name"])
        officers[held] = [o for o in check["record"]["officers"] if not o["resigned_on"] and o["officer_role"] in {"director", "corporate-director", "nominee-director", "corporate-nominee-director"}]
        for key in [k for k in register if k[1] == held]:
            del register[key]
        for psc in check["record"]["persons_with_significant_control"]:
            if psc["ceased_on"]:
                continue
            key, kind = _register_holder(psc, case_entities)
            holder = add_node(key, psc["name"], kind, "COMPANIES_HOUSE")
            band = psc["bands"].get("ownership-of-shares")
            register[(holder, held)] = {"edge_id": stable_id("edge", "REGISTER_PSC", holder, held), "type": "REGISTER_PSC", "holder": holder, "held": held, "percent": {"low": float(band[0]), "high": float(band[1])} if band else None, "as_of": (check.get("fetched_at") or "")[:10] or None, "status": "REGISTERED", "flags": ["REGISTER_BAND"] if band else [], "fact_ids": [], "superseded_fact_ids": [], "disposal_fact_ids": [], "conflict_issue_ids": [], "open_issue_ids": [], "documents": [], "source": "COMPANIES_HOUSE", "record_url": check.get("record_url"), "natures_of_control": psc["natures_of_control"], "voting_band": psc["bands"].get("voting-rights")}
    edges += [register[k] for k in sorted(register)]
    for group in sorted(k for k in nodes if k in group_parent):
        parent = add_node(group_parent[group])
        edges.append({"edge_id": stable_id("edge", "GROUP_PARENT", parent, group), "type": "GROUP_PARENT", "holder": parent, "held": group, "percent": None, "as_of": None, "status": "STRUCTURAL", "flags": [], "fact_ids": [], "superseded_fact_ids": [], "disposal_fact_ids": [], "conflict_issue_ids": [], "open_issue_ids": [], "documents": entities[group]["documents"]})
    edge_by_id = {e["edge_id"]: e for e in edges}
    new_issues: list[dict] = []

    # Attribute each group's holdings to its parent. The group's stake includes the
    # parent's own, so the two are combined, never added.
    def fold(node: str) -> str:
        return group_parent.get(node, node)

    parts: dict[tuple, dict[str, list[dict]]] = defaultdict(lambda: {"direct": [], "group": []})
    for e in edges:
        if e["type"] == "EQUITY" and fold(e["holder"]) != fold(e["held"]):
            parts[(fold(e["holder"]), fold(e["held"]))]["group" if e["holder"] in group_parent else "direct"].append(e)
    holdings: dict[str, dict[str, dict]] = defaultdict(dict)
    for (holder, held), split in sorted(parts.items()):
        direct, group = _hull(split["direct"]), _hull(split["group"])
        members = split["direct"] + split["group"]
        flags = {flag for e in members for flag in e["flags"]} | {e["status"] for e in members if e["status"] != "REPORTED"}
        if direct and group:
            low, high = max(direct[0], group[0]), max(group[1], direct[0])
            approximate = "approximately" if any("APPROXIMATE" in e["flags"] for e in members) else None
            if direct[0] > group[1] and not rounding_consistent({"value": direct[0], "qualifier": approximate}, {"value": group[1], "qualifier": approximate}):
                new_issues.append(make_issue("GROUP_STAKE_BELOW_PARENT_STAKE", "The parent alone is reported to hold more than its whole group; one of the figures, or the group's scope, is wrong.", document_id=document_id, categories=["ownership"], severity="medium", fact_ids=sorted({f for e in members for f in e["fact_ids"]}), details={"parent": holder, "held": held, "parent_low": direct[0], "group_high": group[1]}))
        else:
            low, high = direct or group
        if group:
            flags.add("ATTRIBUTED_FROM_GROUP")
        if any(e["open_issue_ids"] for e in members):
            flags.add("OPEN_ISSUES")
        if nodes[holder]["kind"] == "unresolved_text" or nodes[held]["kind"] == "unresolved_text":
            flags.add("UNRESOLVED_PARTY")
        holdings[holder][held] = {"low": low, "high": high, "edge_ids": sorted(e["edge_id"] for e in members), "flags": flags, "as_of": _latest([{"effective_date": e["as_of"]} for e in members])}

    # A register band stands in only where the documents report no stake.
    for e in edges:
        if e["type"] == "REGISTER_PSC" and e["percent"] and fold(e["held"]) not in holdings.get(fold(e["holder"]), {}) and fold(e["holder"]) != fold(e["held"]):
            holdings[fold(e["holder"])][fold(e["held"])] = {"low": e["percent"]["low"], "high": e["percent"]["high"], "edge_ids": [e["edge_id"]], "flags": {"REGISTER_BAND"}, "as_of": e["as_of"]}

    # Look-through: multiply along each simple path and add across paths.
    interests: dict[tuple, dict] = {}
    cycles: dict[tuple, list[str]] = {}
    truncated = False
    for start in sorted(holdings):
        stack = [(start, (start,), 100.0, 100.0, frozenset(), frozenset())]
        count = 0
        while stack:
            current, path, low, high, flags, dates = stack.pop()
            for nxt, h in sorted(holdings.get(current, {}).items()):
                if nxt in path:
                    loop = path[path.index(nxt):]
                    turn = loop.index(min(loop))
                    loop = loop[turn:] + loop[:turn]
                    cycles.setdefault(loop, sorted({eid for a, b in zip(loop, loop[1:] + loop[:1]) for eid in holdings[a][b]["edge_ids"]}))
                    continue
                n_low, n_high = low * h["low"] / 100, high * h["high"] / 100
                n_dates = dates | ({h["as_of"]} if h["as_of"] else set())
                n_flags = flags | h["flags"] | ({"MIXED_DATES"} if len(n_dates) > 1 else set())
                n_path = path + (nxt,)
                item = interests.setdefault((start, nxt), {"owner": start, "owned": nxt, "paths": []})
                item["paths"].append({"nodes": list(n_path), "percent": {"low": round(n_low, 4), "high": round(n_high, 4)}, "flags": sorted(n_flags)})
                count += 1
                if count >= _MAX_PATHS:
                    truncated = True
                elif len(n_path) - 1 < max_depth:
                    stack.append((nxt, n_path, n_low, n_high, n_flags, n_dates))
                elif holdings.get(nxt):
                    truncated = True
    interest_list = []
    for (owner, owned), item in sorted(interests.items()):
        low = sum(p["percent"]["low"] for p in item["paths"])
        high = sum(p["percent"]["high"] for p in item["paths"])
        flags = {f for p in item["paths"] for f in p["flags"]} | ({"PATHS_EXCEED_100"} if low > 100 + _TOLERANCE else set())
        low, high = min(round(low, 4), 100.0), min(round(high, 4), 100.0)
        interest_list.append({"owner": owner, "owned": owned, "percent": {"low": low, "high": high}, "direct": any(len(p["nodes"]) == 2 for p in item["paths"]), "meets_reporting_threshold": "YES" if low >= threshold else "POSSIBLE" if high >= threshold else "NO", "flags": sorted(flags), "paths": sorted(item["paths"], key=lambda p: (len(p["nodes"]), p["nodes"]))})

    # Control: reported voting rights or a majority stake; certain links chain.
    direct_control: dict[str, dict[str, dict]] = defaultdict(dict)
    for e in edges:
        if e["type"] == "VOTING_RIGHTS" and fold(e["holder"]) != fold(e["held"]):
            # Voting rights without a figure are reported as control; with one, only above the threshold.
            if e["percent"] and e["percent"]["high"] <= control_threshold:
                continue
            flags = set(e["flags"]) | ({"OPEN_ISSUES"} if e["open_issue_ids"] else set()) | ({"ATTRIBUTED_FROM_GROUP"} if e["holder"] in group_parent else set())
            certainty = "POSSIBLE" if e["percent"] and e["percent"]["low"] <= control_threshold else "REPORTED"
            direct_control[fold(e["holder"])][fold(e["held"])] = {"basis": "VOTING_RIGHTS", "certainty": certainty, "edge_ids": [e["edge_id"]], "flags": flags}
    for e in edges:
        if e["type"] == "REGISTER_PSC" and fold(e["held"]) not in direct_control[fold(e["holder"])]:
            voting_band = e["voting_band"]
            if any(n.startswith(_PSC_CONTROL) for n in e["natures_of_control"]) or (voting_band and voting_band[0] >= control_threshold):
                direct_control[fold(e["holder"])][fold(e["held"])] = {"basis": "REGISTER_PSC", "certainty": "REPORTED", "edge_ids": [e["edge_id"]], "flags": {"REGISTER_BAND"}}
    for holder, held_map in holdings.items():
        for held, h in held_map.items():
            if held in direct_control[holder]:
                continue
            if h["low"] > control_threshold:
                direct_control[holder][held] = {"basis": "MAJORITY_STAKE", "certainty": "REPORTED", "edge_ids": h["edge_ids"], "flags": h["flags"]}
            elif h["high"] > control_threshold:
                direct_control[holder][held] = {"basis": "MAJORITY_STAKE", "certainty": "POSSIBLE", "edge_ids": h["edge_ids"], "flags": h["flags"]}
    control = []
    for controller in sorted(direct_control):
        for controlled, link in sorted(direct_control[controller].items()):
            control.append({"controller": controller, "controlled": controlled, "basis": link["basis"], "certainty": link["certainty"], "path": [controller, controlled], "edge_ids": link["edge_ids"], "flags": sorted(link["flags"])})
        seen = {controller} | set(direct_control[controller])
        queue = [(c, [controller, c], list(link["edge_ids"]), set(link["flags"])) for c, link in sorted(direct_control[controller].items()) if link["certainty"] == "REPORTED"]
        while queue:
            node, path, edge_ids, flags = queue.pop(0)
            for nxt, link in sorted(direct_control.get(node, {}).items()):
                if nxt in seen or link["certainty"] != "REPORTED":
                    continue
                seen.add(nxt)
                chain = (path + [nxt], edge_ids + link["edge_ids"], flags | link["flags"])
                control.append({"controller": controller, "controlled": nxt, "basis": "CONTROL_CHAIN", "certainty": "DERIVED", "path": chain[0], "edge_ids": chain[1], "flags": sorted(chain[2])})
                if len(path) < max_depth:
                    queue.append((nxt, *chain))

    # Distinct holders of one entity cannot together hold more than all of it.
    reaches = set(interests)
    incoming: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for holder, held_map in holdings.items():
        for held, h in held_map.items():
            if not {"STALE_AFTER_DISPOSAL", "DISPOSAL_ORDER_UNKNOWN"} & h["flags"]:
                incoming[held].append((holder, h))
    for held, stakes in sorted(incoming.items()):
        # A holder's figure may already include what an entity it holds owns.
        stakes = [(a, h) for a, h in stakes if not any((b, a) in reaches for b, _ in stakes if b != a)]
        if len(stakes) < 2 or not all(temporal_overlap(x["as_of"], y["as_of"]) for (_, x), (_, y) in combinations(stakes, 2)):
            continue
        total = sum(h["low"] for _, h in stakes)
        if total > 100 + _TOLERANCE:
            fact_ids = sorted({f for _, h in stakes for eid in h["edge_ids"] for f in edge_by_id[eid]["fact_ids"]})
            new_issues.append(make_issue("OWNERSHIP_TOTAL_EXCEEDS_100", "Distinct holders are reported to own more than 100% of one entity in the same or unknown period; at least one figure, date or holder is wrong.", document_id=document_id, categories=["ownership"], fact_ids=fact_ids, details={"held": held, "holders": sorted(a for a, _ in stakes), "total_low_percent": round(total, 4)}))
    for loop, edge_ids in sorted(cycles.items()):
        new_issues.append(make_issue("OWNERSHIP_CYCLE", "The reported holdings form a cycle. Cross-holdings exist, but look-through figures omit the circular part; review the chain.", document_id=document_id, categories=["ownership"], severity="medium", fact_ids=sorted({f for eid in edge_ids for f in edge_by_id[eid]["fact_ids"]}), details={"nodes": list(loop)}))
    if current_fact_ids is not None:
        new_issues = [i for i in new_issues if set(i["fact_ids"]) & current_fact_ids]

    owned = {i["owned"] for i in interest_list} | {fold(e["held"]) for e in edges if e["type"] in {"VOTING_RIGHTS", "REGISTER_PSC"}}
    for node in nodes.values():
        node["has_reported_owners"] = node["node_id"] in owned or fold(node["node_id"]) in owned
    beneficial_owners, ubo_issues = _beneficial_owners(nodes, edge_by_id | {e["edge_id"]: e for e in edges}, holdings, interest_list, control, declared, directors, officers, threshold, document_id)
    if current_fact_ids is not None:
        ubo_issues = [i for i in ubo_issues if set(i["fact_ids"]) & current_fact_ids]
    new_issues += ubo_issues
    graph = {
        "settings": settings, "note": _NOTE,
        "nodes": sorted(nodes.values(), key=lambda n: n["node_id"]),
        "edges": edges, "interests": interest_list,
        "control": control,
        "beneficial_owners": beneficial_owners,
        "cycles": [list(loop) for loop in sorted(cycles)],
        "identified_entities_without_reported_owners": sorted(e["entity_id"] for e in case_entities if e["entity_type"] != "group" and (e["identifiers"] or e["roles"]) and e["entity_id"] not in owned),
        "truncated": truncated,
    }
    return graph, new_issues


_CONTROL_RANK = {"REPORTED": 3, "DERIVED": 2, "POSSIBLE": 1}


def _beneficial_owners(nodes: dict, edge_by_id: dict, holdings: dict, interests: list[dict], control: list[dict], declared: dict, directors: dict, officers: dict, threshold: float, document_id: str) -> tuple[list[dict], list[dict]]:
    """For each organization with reported owners, the natural persons who own or control it.

    Three sources are set side by side, never merged into one figure: interests
    and control derived from the holdings, the beneficial owners a declaration
    names, and the register's PSCs. Where a declaration exists, a person the
    other two identify but the declaration omits, or a declared percentage the
    holdings or register band contradict, is raised.
    """
    people = {k for k, n in nodes.items() if n["kind"] == "person"}
    register = [e for e in edge_by_id.values() if e["type"] == "REGISTER_PSC"]
    targets = sorted({i["owned"] for i in interests} | {c["controlled"] for c in control} | {held for _, held in declared} | {e["held"] for e in register})
    targets = [t for t in targets if nodes.get(t, {}).get("kind") not in {"person", "group"}]
    results, issues = [], []
    for target in targets:
        candidates: dict[str, dict] = {}

        def candidate(person: str) -> dict:
            return candidates.setdefault(person, {"person": person, "name": nodes[person]["name"], "interest": None, "meets_threshold": "NO", "control": None, "declared": None, "register_psc": None, "fact_ids": set(), "flags": set()})

        for i in interests:
            if i["owned"] == target and i["owner"] in people:
                c = candidate(i["owner"])
                c.update(interest=i["percent"], meets_threshold=i["meets_reporting_threshold"])
                c["flags"] |= set(i["flags"])
                c["fact_ids"] |= {f for p in i["paths"] for a, b in zip(p["nodes"], p["nodes"][1:]) for eid in holdings.get(a, {}).get(b, {}).get("edge_ids", []) for f in edge_by_id[eid]["fact_ids"]}
        for link in control:
            if link["controlled"] == target and link["controller"] in people:
                c = candidate(link["controller"])
                if c["control"] is None or _CONTROL_RANK[link["certainty"]] > _CONTROL_RANK[c["control"]["certainty"]]:
                    c["control"] = {"basis": link["basis"], "certainty": link["certainty"], "path": link["path"]}
                c["fact_ids"] |= {f for eid in link["edge_ids"] for f in edge_by_id[eid]["fact_ids"]}
        declaration_ids = sorted({f["fact_id"] for (person, held), claims in declared.items() if held == target for f in claims})
        for (person, held), claims in declared.items():
            if held == target:
                values = sorted({f["value"] for f in claims if isinstance(f.get("value"), (int, float))})
                candidate(person)["declared"] = {"percent": values, "nature": sorted({f["qualifier"] for f in claims if f.get("qualifier")}), "fact_ids": sorted(f["fact_id"] for f in claims), "as_of": _latest(claims)}
        for e in register:
            if e["held"] == target and e["holder"] in people:
                candidate(e["holder"])["register_psc"] = {"share_band": [e["percent"]["low"], e["percent"]["high"]] if e["percent"] else None, "natures_of_control": e["natures_of_control"], "as_of": e["as_of"], "record_url": e["record_url"]}
        owners = []
        for c in candidates.values():
            bases = (["HOLDINGS"] if c["meets_threshold"] in {"YES", "POSSIBLE"} else []) + (["CONTROL"] if c["control"] else []) + (["DECLARED"] if c["declared"] else []) + (["REGISTER_PSC"] if c["register_psc"] else [])
            if not bases:
                continue  # holds a stake below the threshold, without control
            if bases == ["DECLARED"]:
                c["flags"].add("NOT_CORROBORATED")
            owners.append({**c, "bases": bases, "fact_ids": sorted(c["fact_ids"]), "flags": sorted(c["flags"])})
            if declaration_ids and not c["declared"] and (c["meets_threshold"] == "YES" or (c["control"] and c["control"]["certainty"] != "POSSIBLE") or c["register_psc"]):
                issues.append(make_issue("UBO_NOT_DECLARED", "The holdings or the register identify this person as a beneficial owner, but the beneficial-ownership declaration does not name them.", document_id=document_id, categories=["ownership", "control"], fact_ids=sorted(set(declaration_ids) | c["fact_ids"]), details={"company": target, "company_name": nodes[target]["name"], "person": c["person"], "name": c["name"], "bases": bases, "interest": c["interest"], "control": c["control"], "register_psc": c["register_psc"]}))
            stated = c["declared"]["percent"] if c["declared"] else []
            for basis, low, high in ([("HOLDINGS", c["interest"]["low"], c["interest"]["high"])] if c["interest"] and c["fact_ids"] else []) + ([("REGISTER_PSC", *c["register_psc"]["share_band"])] if c["register_psc"] and c["register_psc"]["share_band"] else []):
                outside = [v for v in stated if not (low - _TOLERANCE <= v <= high + _TOLERANCE)]
                if outside:
                    issues.append(make_issue("UBO_DECLARED_PERCENT_DIFFERS", "The declared beneficial interest differs from what the holdings give or from the register's band.", document_id=document_id, categories=["ownership"], severity="medium", fact_ids=sorted(set(c["declared"]["fact_ids"]) | c["fact_ids"]), details={"company": target, "company_name": nodes[target]["name"], "person": c["person"], "name": c["name"], "declared": outside, "basis": basis, "low": low, "high": high}))
        identified = [o for o in owners if o["meets_threshold"] == "YES" or (o["control"] and o["control"]["certainty"] != "POSSIBLE") or o["register_psc"]]
        status = "IDENTIFIED" if identified else "DECLARED_ONLY" if any(o["declared"] for o in owners) else "POSSIBLE" if owners else "NOT_IDENTIFIED"
        # Organizations at the top of a chain hide whoever owns them.
        unresolved = [{"node": i["owner"], "name": nodes[i["owner"]]["name"], "percent": i["percent"]} for i in interests if i["owned"] == target and i["owner"] not in people and nodes[i["owner"]]["kind"] != "group" and not nodes[i["owner"]]["has_reported_owners"] and i["percent"]["high"] >= threshold]
        direct = [h for holder, held_map in holdings.items() for held, h in held_map.items() if held == target and not {"STALE_AFTER_DISPOSAL", "DISPOSAL_ORDER_UNKNOWN"} & h["flags"]]
        entry = {"company": target, "name": nodes[target]["name"], "status": status, "owners": sorted(owners, key=lambda o: (-(o["interest"] or {"high": 0})["high"], o["name"] or "")),
                 "chain_ends_at": sorted(unresolved, key=lambda u: -u["percent"]["high"]),
                 "direct_holdings_percent": {"low": min(round(sum(h["low"] for h in direct), 4), 100.0), "high": min(round(sum(h["high"] for h in direct), 4), 100.0)} if direct else None}
        if status in {"NOT_IDENTIFIED", "POSSIBLE"}:
            # Where no owner is identified, KYC rules usually fall back to senior managing officials.
            entry["senior_managing_officials"] = sorted({f["subject"] for f in directors.get(target, [])} | {o["name"] for o in officers.get(target, [])})
        results.append(entry)
    return results, issues


def to_mermaid(graph: dict) -> str:
    """A Mermaid flowchart of the graph's edges, for reading rather than for analysis."""
    ids = {n["node_id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    lines = ["flowchart TD"]
    for n in graph["nodes"]:
        label = (n["name"] or n["node_id"]).replace('"', "#quot;")
        style = {"group": ":::group", "unresolved_text": ":::unresolved", "person": ":::person"}.get(n["kind"], "")
        shape = f'(["{label}"])' if n["kind"] == "person" else f'["{label}"]'
        lines.append(f'  {ids[n["node_id"]]}{shape}{style}')
    for e in graph["edges"]:
        a, b = ids[e["holder"]], ids[e["held"]]
        if e["type"] == "GROUP_PARENT":
            lines.append(f"  {a} -. heads group .-> {b}")
            continue
        if e["type"] == "REGISTER_PSC":
            band = f'{e["percent"]["low"]:g}–{e["percent"]["high"]:g}%' if e["percent"] else "control"
            lines.append(f'  {a} -. "PSC {band} (register {e["as_of"] or "undated"})" .-> {b}')
            continue
        if e["type"] == "VOTING_RIGHTS":
            label = "voting rights" if not e["percent"] else f'{e["percent"]["low"]:g}% votes' if e["percent"]["low"] == e["percent"]["high"] else f'{e["percent"]["low"]:g}–{e["percent"]["high"]:g}% votes'
        else:
            low, high = e["percent"]["low"], e["percent"]["high"]
            label = f"{low:g}%" if low == high else f"{low:g}–{high:g}%"
        notes = [e["as_of"] or "undated"] + ([e["status"].lower().replace("_", " ")] if e["status"] != "REPORTED" else []) + (["open issues"] if e["open_issue_ids"] else [])
        lines.append(f'  {a} -- "{label} ({", ".join(notes)})" --> {b}')
    lines += ["  classDef group stroke-dasharray: 4 3", "  classDef unresolved stroke-dasharray: 2 2, color:#b45309", "  classDef person stroke-width:2px"]
    return "\n".join(lines) + "\n"
