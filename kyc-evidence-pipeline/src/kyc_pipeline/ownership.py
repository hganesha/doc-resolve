from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .evidence import make_issue
from .reconciliation import APPROXIMATE, party_key, rounding_consistent
from .util import date_interval, stable_id, temporal_overlap

# Illustrative defaults; a policy may override them under `ownership_graph`.
DEFAULT_SETTINGS = {"reporting_threshold_percent": 25.0, "control_threshold_percent": 50.0, "max_chain_depth": 8}
_MAX_PATHS = 5000
# Percentage points of rounding allowed before direct stakes "exceed 100%".
_TOLERANCE = 0.5
_NOTE = ("Built only from extracted, source-matched claims, all pending human verification. Stakes are ranges: "
         "conflicts, rounding and later disposals widen them rather than choosing a figure. A group's holding is "
         "counted as its parent's consolidated interest, not a look-through economic interest. No natural persons "
         "are extracted, so the top of a chain is not a beneficial owner, and an entity with no reported owners "
         "may still have them.")


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


def build_ownership_graph(facts: list[dict], case_entities: list[dict], entity_map: dict[str, str], issues: list[dict], *, document_id: str = "case", current_fact_ids: set[str] | None = None, settings: dict | None = None) -> tuple[dict, list[dict]]:
    """Case ownership and control graph with look-through interests.

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

    def add_node(key: str, text: str | None = None) -> str:
        if key not in nodes:
            e = entities.get(key)
            nodes[key] = {"node_id": key, "entity_id": key if e else None, "name": (e["name"] if e else None) or text, "kind": e["entity_type"] if e else "unresolved_text", "roles": e["roles"] if e else [], "identified": bool(e and (e["identifiers"] or e["roles"]))}
        return key

    equity: dict[tuple, list[dict]] = defaultdict(list)
    voting: dict[tuple, list[dict]] = defaultdict(list)
    disposals: dict[tuple, list[dict]] = defaultdict(list)
    for f in facts:
        predicate = f["predicate"]
        if f["fact_id"] in set_aside or not f.get("object") or predicate not in {"owns_equity_in", "has_voting_rights_over", "disposed_interest_in"}:
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
        edges.append({"edge_id": stable_id("edge", "VOTING_RIGHTS", holder, held), "type": "VOTING_RIGHTS", "holder": holder, "held": held, "percent": None, "as_of": _latest(claims), "status": "REPORTED", "flags": ["UNDATED"] if any(not c.get("effective_date") for c in claims) else [], "fact_ids": sorted(c["fact_id"] for c in claims), "superseded_fact_ids": [], "disposal_fact_ids": [], "conflict_issue_ids": [], "open_issue_ids": sorted({i for c in claims for i in open_issues.get(c["fact_id"], ())}), "documents": sorted({c["document_id"] for c in claims if c.get("document_id")})})
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
            flags = set(e["flags"]) | ({"OPEN_ISSUES"} if e["open_issue_ids"] else set()) | ({"ATTRIBUTED_FROM_GROUP"} if e["holder"] in group_parent else set())
            direct_control[fold(e["holder"])][fold(e["held"])] = {"basis": "VOTING_RIGHTS", "certainty": "REPORTED", "edge_ids": [e["edge_id"]], "flags": flags}
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

    owned = {i["owned"] for i in interest_list} | {fold(e["held"]) for e in edges if e["type"] == "VOTING_RIGHTS"}
    for node in nodes.values():
        node["has_reported_owners"] = node["node_id"] in owned or fold(node["node_id"]) in owned
    graph = {
        "settings": settings, "note": _NOTE,
        "nodes": sorted(nodes.values(), key=lambda n: n["node_id"]),
        "edges": edges, "interests": interest_list,
        "control": control,
        "cycles": [list(loop) for loop in sorted(cycles)],
        "identified_entities_without_reported_owners": sorted(e["entity_id"] for e in case_entities if e["entity_type"] != "group" and (e["identifiers"] or e["roles"]) and e["entity_id"] not in owned),
        "truncated": truncated,
    }
    return graph, new_issues


def to_mermaid(graph: dict) -> str:
    """A Mermaid flowchart of the graph's edges, for reading rather than for analysis."""
    ids = {n["node_id"]: f"n{i}" for i, n in enumerate(graph["nodes"])}
    lines = ["flowchart TD"]
    for n in graph["nodes"]:
        label = (n["name"] or n["node_id"]).replace('"', "#quot;")
        style = {"group": ":::group", "unresolved_text": ":::unresolved"}.get(n["kind"], "")
        lines.append(f'  {ids[n["node_id"]]}["{label}"]{style}')
    for e in graph["edges"]:
        a, b = ids[e["holder"]], ids[e["held"]]
        if e["type"] == "GROUP_PARENT":
            lines.append(f"  {a} -. heads group .-> {b}")
            continue
        if e["type"] == "VOTING_RIGHTS":
            label = "voting rights"
        else:
            low, high = e["percent"]["low"], e["percent"]["high"]
            label = f"{low:g}%" if low == high else f"{low:g}–{high:g}%"
        notes = [e["as_of"] or "undated"] + ([e["status"].lower().replace("_", " ")] if e["status"] != "REPORTED" else []) + (["open issues"] if e["open_issue_ids"] else [])
        lines.append(f'  {a} -- "{label} ({", ".join(notes)})" --> {b}')
    lines += ["  classDef group stroke-dasharray: 4 3", "  classDef unresolved stroke-dasharray: 2 2, color:#b45309"]
    return "\n".join(lines) + "\n"
