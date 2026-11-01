"""Source-of-funds checks: statement arithmetic and expected inflows.

* Statement arithmetic: within one statement, each printed running balance
  should equal the previous balance plus credits minus debits, and the last
  should equal the closing balance. A break may be an extraction error, a
  missing row, or an altered statement; it is raised for review, not judged.
* Expected inflows: a declared source-of-funds amount, or the net proceeds of
  a sale, is looked for among the credits on the case's statements for the
  same person, by amount and currency. A match shows that a payment of that
  size arrived, not that it came from the declared source.
"""
from __future__ import annotations

from collections import defaultdict

from .entities import person_key
from .evidence import make_issue
from .util import stable_id

_TOLERANCE = 0.005  # half a cent
_NEAR = 0.01  # within 1% (fees, rounding) is reported, but as a near match


def _finding(kind: str, fact_ids: list[str], category: str, description: str, details: dict) -> dict:
    ids = sorted(set(fact_ids))
    return {"finding_id": stable_id("finding", kind, ids), "type": kind, "status": "INFORMATIONAL", "category": category, "fact_ids": ids, "description": description, "details": details}


def statement_checks(document_id: str, facts: list[dict], evidence: list[dict], segments: list[dict], material: dict[str, bool]) -> tuple[list[dict], list[dict]]:
    """Running-balance and closing-balance arithmetic for each statement in this document."""
    ordinal = {s["segment_id"]: i for i, s in enumerate(segments)}
    where = {e["evidence_id"]: (ordinal.get(e["segment_id"], -1), e["start"]) for e in evidence}
    severity = "high" if material.get("transactions") else "medium"
    statements: dict[tuple, list[dict]] = defaultdict(list)
    for f in facts:
        if f["predicate"] in {"received_payment", "made_payment", "reported_account_balance"} and isinstance(f.get("value"), (int, float)) and f.get("unit"):
            statements[(person_key(f["subject"]), f["unit"])].append(f)
    findings, issues = [], []
    for (holder, unit), items in statements.items():
        opening = next((f for f in items if f.get("qualifier") == "opening balance"), None)
        closing = next((f for f in items if f.get("qualifier") == "closing balance"), None)
        rows: dict[tuple, dict] = {}
        for f in items:
            if f in (opening, closing):
                continue
            primary = next((r["evidence_id"] for r in f.get("evidence_roles", []) if r["role"] == "primary_quote"), f["evidence_ids"][0])
            row = rows.setdefault(where.get(primary, (-1, 0)), {"credits": [], "debits": [], "balance": None, "date": f.get("effective_date")})
            if f["predicate"] == "received_payment":
                row["credits"].append(f)
            elif f["predicate"] == "made_payment":
                row["debits"].append(f)
            else:
                row["balance"] = f
        ordered = [rows[k] for k in sorted(rows)]
        dates = [r["date"] for r in ordered if r["date"]]
        if dates and dates[0] > dates[-1]:
            ordered.reverse()  # newest-first statement
        if not ordered or not (opening or any(r["balance"] for r in ordered)):
            continue
        running = opening["value"] if opening else None
        previous = opening
        breaks = 0
        for row in ordered:
            delta = sum(f["value"] for f in row["credits"]) - sum(f["value"] for f in row["debits"])
            balance = row["balance"]
            if balance is None:
                running = running + delta if running is not None else None
                continue
            if running is not None and abs(running + delta - balance["value"]) > _TOLERANCE:
                breaks += 1
                ids = [f["fact_id"] for f in row["credits"] + row["debits"] + [balance]] + ([previous["fact_id"]] if previous else [])
                issues.append(make_issue("STATEMENT_BALANCE_BREAK", "A printed running balance does not equal the previous balance plus this row's credits minus its debits. A row may be missing or misread, or the statement may have been altered.", document_id=document_id, categories=["transactions"], severity=severity, fact_ids=ids, details={"date": row["date"], "unit": unit, "expected": round(running + delta, 2), "printed": balance["value"]}))
            running, previous = balance["value"], balance
        if closing and running is not None and abs(running - closing["value"]) > _TOLERANCE:
            breaks += 1
            issues.append(make_issue("STATEMENT_CLOSING_BALANCE_MISMATCH", "The statement's closing balance does not equal its last balance after the listed transactions; a row may be missing or the statement altered.", document_id=document_id, categories=["transactions"], severity=severity, fact_ids=[closing["fact_id"]] + ([previous["fact_id"]] if previous else []), details={"unit": unit, "expected": round(running, 2), "printed": closing["value"]}))
        if not breaks:
            ids = [f["fact_id"] for f in items]
            findings.append(_finding("STATEMENT_BALANCES_RECONCILE", ids, "transactions", "Every printed balance on this statement follows from the previous balance and the listed transactions. This checks the arithmetic only, not that the statement is genuine.", {"holder": holder, "unit": unit, "rows": len(ordered), "opening": opening["value"] if opening else None, "closing": closing["value"] if closing else None}))
    return findings, issues


def inflow_checks(document_id: str, current: list[dict], previous: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match declared amounts and sale proceeds against credits for the same person and currency."""
    current_ids = {f["fact_id"] for f in current}
    facts = previous + current
    expected = [f for f in facts if f["predicate"] in {"declared_source_of_funds", "received_sale_proceeds"} and isinstance(f.get("value"), (int, float)) and f.get("unit")]
    credits: dict[tuple, list[dict]] = defaultdict(list)
    for f in facts:
        if f["predicate"] == "received_payment" and isinstance(f.get("value"), (int, float)) and f.get("unit"):
            credits[(person_key(f["subject"]), f["unit"])].append(f)
    findings, issues = [], []
    for e in expected:
        pool = credits.get((person_key(e["subject"]), e["unit"]), [])
        if not pool or not ({e["fact_id"]} | {p["fact_id"] for p in pool}) & current_ids:
            continue  # no statement for this person yet, or nothing new to compare
        exact = [p for p in pool if abs(p["value"] - e["value"]) <= _TOLERANCE]
        near = [p for p in pool if p not in exact and abs(p["value"] - e["value"]) <= _NEAR * abs(e["value"])]
        what = "declared amount" if e["predicate"] == "declared_source_of_funds" else "sale proceeds"
        if exact or near:
            matched = exact or near
            findings.append(_finding("EXPECTED_FUNDS_RECEIVED", [e["fact_id"]] + [p["fact_id"] for p in matched], "wealth_and_funds", f"A credit matches the {what} {'exactly' if exact else 'within 1%'}. This shows a payment of that size arrived, not that it came from the stated source.", {"expected": e["value"], "unit": e["unit"], "result": "EXACT" if exact else "WITHIN_1_PERCENT", "credits": [{"fact_id": p["fact_id"], "value": p["value"], "date": p.get("effective_date"), "description": p.get("object")} for p in matched]}))
        else:
            dates = sorted(p["effective_date"] for p in pool if p.get("effective_date"))
            issues.append(make_issue("EXPECTED_FUNDS_NOT_FOUND", f"No credit on this person's statements in the case matches the {what}; the funds may have arrived in another account or period, in parts, or not at all.", document_id=document_id, categories=["wealth_and_funds", "transactions"], severity="medium", fact_ids=[e["fact_id"]], details={"expected": e["value"], "unit": e["unit"], "statement_dates": [dates[0], dates[-1]] if dates else [], "credits_checked": len(pool)}))
    return findings, issues
