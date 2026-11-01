from __future__ import annotations

import pytest

from conftest import ROOT
from kyc_pipeline.extraction import _amount
from kyc_pipeline.funds import inflow_checks, person_key
from kyc_pipeline.schemas import check_fact, validate_schemas
from kyc_pipeline.store import EvidenceStore
from test_registry_and_licensing import _ingest

STATEMENT = (ROOT / "examples/synthetic_bank_statement.txt").read_text()


def _by(values, predicate):
    return [f for f in values["facts"] if f["predicate"] == predicate]


def test_statement_rows_become_transactions_and_balances_that_reconcile(tmp_path):
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_bank_statement.txt")
    assert values["profile_id"] == "source_of_funds" and "__interrupt__" not in result
    assert {(f["object"], f["value"], f["unit"], f["effective_date"]) for f in _by(values, "received_payment")} == {("SALARY ACME LTD", 4200.0, "GBP", "2026-03-02"), ("COMPLETION FUNDS HARBOUR LAW LLP", 438250.0, "GBP", "2026-03-09")}
    assert sorted(f["value"] for f in _by(values, "made_payment")) == [120.0, 1850.0, 7650.0]
    balances = {(f["qualifier"], f["effective_date"]): f["value"] for f in _by(values, "reported_account_balance")}
    assert balances[("opening balance", "2026-03-01")] == 12400.0 and balances[("closing balance", "2026-03-31")] == 445230.0 and len(balances) == 7
    assert _by(values, "holds_account_at")[0]["object"] == "Northbank plc"
    assert all(f["schema_check"]["status"] == "CONFORMS" and not [k for k, v in f["source_support"]["literal_checks"].items() if v is False] for f in values["facts"])
    # Each row cites its own line, the table header, and the statement's holder and currency lines.
    salary = next(f for f in _by(values, "received_payment") if f["value"] == 4200.0)
    quotes = {e["evidence_id"]: e["quote"] for e in values["evidence"]}
    assert [quotes[e] for e in salary["evidence_ids"]] == ["| 2 March 2026 | SALARY ACME LTD | 4,200.00 | | 16,600.00 |", "| Date | Description | Money in | Money out | Balance |", "Account holder: Jane Smith", "Currency: GBP"]
    reconciled = next(f for f in values["findings"] if f["type"] == "STATEMENT_BALANCES_RECONCILE")
    assert reconciled["details"]["rows"] == 5


def test_an_altered_balance_or_closing_figure_breaks_the_arithmetic(tmp_path):
    altered = STATEMENT.replace("| 438,250.00 | | 453,000.00 |", "| 438,250.00 | | 463,000.00 |")
    _, result, values = _ingest(tmp_path, altered, profile="source_of_funds")
    breaks = [i for i in result["__interrupt__"][0].value["issues"] if i["code"] == "STATEMENT_BALANCE_BREAK"]
    # The altered row breaks, and so does the next one, which follows the true balance.
    assert [(i["details"]["expected"], i["details"]["printed"]) for i in breaks] == [(453000.0, 463000.0), (462880.0, 452880.0)]
    assert breaks[0]["severity"] == "high"
    _, result, _ = _ingest(tmp_path, STATEMENT.replace("Closing balance: 445,230.00", "Closing balance: 455,230.00"), profile="source_of_funds", run_id="run_2", case="other")
    closing = next(i for i in result["__interrupt__"][0].value["issues"] if i["code"] == "STATEMENT_CLOSING_BALANCE_MISMATCH")
    assert closing["details"] == {"unit": "GBP", "expected": 445230.0, "printed": 455230.0}


def test_signed_amounts_and_newest_first_statements(tmp_path):
    text = """# Bank Statement
Account holder: Ms Jane Smith
Statement period: 1 April 2026 to 30 April 2026

| Date | Details | Amount (GBP) | Balance (GBP) |
| --- | --- | --- | --- |
| 2026-04-20 | TRANSFER TO SAVINGS | -1,000.00 | 2,500.00 |
| 2026-04-10 | RENT ELM PROPERTIES | (1,850.00) | 3,500.00 |
| 2026-04-02 | SALARY ACME LTD | 4,200.00 | 5,350.00 |
"""
    _, result, values = _ingest(tmp_path, text, profile="source_of_funds")
    debits = sorted((f["value"], f["unit"]) for f in _by(values, "made_payment"))
    assert debits == [(1000.0, "GBP"), (1850.0, "GBP")]
    # A debit printed as -1,000.00 or (1,850.00) is a made_payment of 1,000 or 1,850 and still passes the literal check.
    assert all(f["source_support"]["literal_checks"]["value_literal_present"] for f in values["facts"])
    assert any(f["type"] == "STATEMENT_BALANCES_RECONCILE" for f in values["findings"])
    assert not [i for i in values["issues"] if i["code"].startswith("STATEMENT_")]


def test_a_statement_without_an_account_holder_yields_no_rows(tmp_path):
    store, _, values = _ingest(tmp_path, STATEMENT.replace("Account holder: Jane Smith\n", ""), profile="source_of_funds")
    assert not [f for f in values["facts"] if f["category"] == "transactions"]
    assert next(c for c in values["coverage"] if c["category"] == "transactions")["status"] == "NOT_REVIEWED"


def test_declared_funds_and_sale_proceeds_are_matched_to_credits(tmp_path):
    store = EvidenceStore(tmp_path / "db.sqlite")
    for n, name in enumerate(["synthetic_source_of_funds_declaration", "synthetic_completion_statement", "synthetic_bank_statement"]):
        _, result, values = _ingest(tmp_path, ROOT / f"examples/{name}.txt", case="sof", run_id=f"run_{n}", store=store)
        assert "__interrupt__" not in result
    declared = next(f for f in values["facts"] if f["predicate"] == "received_payment" and f["value"] == 438250.0)
    matches = [f for f in store.export_case("sof")["findings"] if f["type"] == "EXPECTED_FUNDS_RECEIVED"]
    assert len(matches) == 2 and all(declared["fact_id"] in f["fact_ids"] and f["details"]["result"] == "EXACT" for f in matches)
    case = store.export_case("sof")
    assert {f["predicate"] for f in case["facts"] if f["category"] == "wealth_and_funds"} == {"declared_source_of_funds", "declared_source_of_wealth", "sold_asset", "received_sale_proceeds"}


def test_near_and_missing_inflows():
    def fact(fid, predicate, subject, value, unit="GBP"):
        return {"fact_id": fid, "predicate": predicate, "subject": subject, "value": value, "unit": unit, "effective_date": "2026-03-09", "object": "x"}
    credit = fact("c", "received_payment", "MS JANE SMITH", 438250.0)
    findings, issues = inflow_checks("d", [credit], [fact("e", "declared_source_of_funds", "Jane Smith", 440000.0)])
    assert findings[0]["details"]["result"] == "WITHIN_1_PERCENT" and not issues
    findings, issues = inflow_checks("d", [credit], [fact("e", "declared_source_of_funds", "Jane Smith", 500000.0)])
    assert not findings and issues[0]["code"] == "EXPECTED_FUNDS_NOT_FOUND" and issues[0]["details"]["credits_checked"] == 1
    # Another currency, or no statement for this person yet, is not compared.
    assert inflow_checks("d", [credit], [fact("e", "declared_source_of_funds", "Jane Smith", 438250.0, "EUR")]) == ([], [])
    assert inflow_checks("d", [fact("e", "declared_source_of_funds", "Omar Haddad", 1.0)], []) == ([], [])


def test_amount_parsing_and_person_keys():
    assert [_amount(t) for t in ("£1,250.00", "(250.00)", "-1,000.00", "GBP 438,250.00", "1.234,56")] == [(1250.0, "GBP"), (-250.0, None), (-1000.0, None), (438250.0, "GBP"), None]
    assert person_key("Ms Jane Smith") == person_key("JANE SMITH") != person_key("J Smith")


def test_magnitude_schema_rules():
    spec = {"category": "transactions", "description": "d", "object": "required", "value": "amount", "magnitude": True}
    assert check_fact(spec, {"object": "x", "value": -5.0, "unit": "GBP"}) == ["value must not be negative; the predicate gives the direction"]
    with pytest.raises(ValueError, match="magnitude"):
        validate_schemas({"predicates": {"has_stake": {**spec, "value": "percent"}}})


def test_a_row_with_an_unreadable_date_is_skipped_and_breaks_the_next_balance(tmp_path):
    _, result, values = _ingest(tmp_path, STATEMENT.replace("| 2 March 2026 |", "| 02/03/2026 |"), profile="source_of_funds")
    assert 4200.0 not in [f["value"] for f in _by(values, "received_payment")]
    breaks = [i["details"] for i in result["__interrupt__"][0].value["issues"] if i["code"] == "STATEMENT_BALANCE_BREAK"]
    assert breaks == [{"date": "2026-03-05", "unit": "GBP", "expected": 10550.0, "printed": 14750.0}]
