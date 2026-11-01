from __future__ import annotations

from datetime import date

from conftest import ROOT
from kyc_pipeline.identity import check_digit, identity_checks, parse_mrz
from kyc_pipeline.store import EvidenceStore
from test_registry_and_licensing import _ingest

PASSPORT = (ROOT / "examples/synthetic_passport.txt").read_text()
# The specimen passport in ICAO Doc 9303 Part 4.
ICAO = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C<3UTO6908061F9406236ZE184226B<<<<<14\n"


def _codes(values, prefix=("MRZ_", "IDENTITY_", "PROOF_", "PERSON_", "ENTITY_")):
    return sorted(i["code"] for i in values["issues"] if i["code"].startswith(prefix))


def test_icao_specimen_check_digits():
    assert [check_digit(f) for f in ("L898902C<", "690806", "940623", "ZE184226B<<<<<")] == [3, 1, 6, 1]
    zone, = parse_mrz(ICAO)
    assert zone["failed_checks"] == [] and (zone["surname"], zone["given_names"], zone["document_number"], zone["date_of_birth"], zone["expiry_date"]) == ("ERIKSSON", "ANNA MARIA", "L898902C", "1969-08-06", "1994-06-23")


def test_passport_fields_and_consistent_mrz(tmp_path):
    _, result, values = _ingest(tmp_path, ROOT / "examples/synthetic_passport.txt")
    assert values["profile_id"] == "identity_verification" and "__interrupt__" not in result
    facts = {f["predicate"]: f for f in values["facts"]}
    assert facts["holds_identity_document"]["subject"] == "JANE ELIZABETH SMITH"
    assert (facts["holds_identity_document"]["object"], facts["holds_identity_document"]["value"], facts["holds_identity_document"]["qualifier"]) == ("Passport", "924163857", "United Kingdom")
    assert facts["has_date_of_birth"]["effective_date"] == "1980-03-14" and facts["identity_document_expires_on"]["effective_date"] == "2031-05-02"
    # The name is cited from its separate surname and given-name lines.
    assert all(f["source_support"]["literal_checks"]["subject_literal_present"] and f["schema_check"]["status"] == "CONFORMS" for f in values["facts"])
    mrz = next(f for f in values["findings"] if f["type"] == "MRZ_CONSISTENT")
    assert mrz["details"]["fields_compared"] == ["document_number", "date_of_birth", "expiry_date", "name"]
    assert _codes(values) == []


def test_altered_mrz_or_printed_fields(tmp_path):
    # An altered birth date in the MRZ fails its check digit and disagrees with the printed date.
    _, result, values = _ingest(tmp_path, PASSPORT.replace("9241638579GBR8003144F", "9241638579GBR8103144F"))
    assert _codes(values) == ["MRZ_CHECK_DIGIT_INVALID", "MRZ_VISUAL_MISMATCH"] and "__interrupt__" in result
    issue = next(i for i in values["issues"] if i["code"] == "MRZ_CHECK_DIGIT_INVALID")
    assert issue["details"]["failed_checks"] == ["date_of_birth", "composite"] and issue["severity"] == "high"
    # A printed birth date changed alone leaves the MRZ valid but in disagreement.
    _, _, values = _ingest(tmp_path, PASSPORT.replace("Date of birth: 14 MAR 1980", "Date of birth: 14 MAR 1981"), run_id="run_2", case="other")
    mismatch = next(i for i in values["issues"] if i["code"] == "MRZ_VISUAL_MISMATCH")
    assert mismatch["details"]["mismatches"] == [{"field": "date_of_birth", "visual": "1981-03-14", "mrz": "1980-03-14"}]


def test_identity_card_mrz(tmp_path):
    _, _, values = _ingest(tmp_path, ROOT / "examples/synthetic_identity_card.txt")
    assert next(f for f in values["findings"] if f["type"] == "MRZ_CONSISTENT")["details"]["format"] == "TD1"
    assert _codes(values) == []


def test_expiry_and_proof_of_address_age(tmp_path):
    _, _, passport = _ingest(tmp_path, ROOT / "examples/synthetic_passport.txt")
    _, issues = identity_checks("d", passport["facts"], passport["segments"], {"identity_fields": True}, today=date(2031, 6, 1))
    assert [(i["code"], i["details"]["expired"], i["severity"]) for i in issues] == [("IDENTITY_DOCUMENT_EXPIRED", "2031-05-02", "high")]
    _, _, bill = _ingest(tmp_path, ROOT / "examples/synthetic_utility_bill.txt", run_id="run_2", case="bill")
    address = next(f for f in bill["facts"] if f["predicate"] == "has_residential_address")
    assert (address["subject"], address["object"], address["qualifier"], address["effective_date"]) == ("Ms Jane Smith", "14 Elm Road, Bristol BS1 5AA", "proof of address", "2026-10-12")
    assert identity_checks("d", bill["facts"], bill["segments"], {}, today=date(2026, 12, 1))[1] == []
    _, issues = identity_checks("d", bill["facts"], bill["segments"], {}, {"proof_of_address_max_age_days": 30}, today=date(2026, 12, 1))
    assert [(i["code"], i["details"]["age_days"]) for i in issues] == [("PROOF_OF_ADDRESS_OUT_OF_DATE", 50)]


def test_one_person_across_documents_unless_born_apart(tmp_path):
    store = EvidenceStore(tmp_path / "db.sqlite")
    for n, source in enumerate([ROOT / "examples/synthetic_beneficial_ownership_declaration.txt", ROOT / "examples/synthetic_passport.txt", ROOT / "examples/synthetic_utility_bill.txt"]):
        _, _, values = _ingest(tmp_path, source, case="id", run_id=f"run_{n}", store=store, defer=True)
        if n == 1:
            assert [i["details"]["names"] for i in values["issues"] if i["code"] == "PERSON_MATCHED_ON_NAME_PARTS"] == [["JANE ELIZABETH SMITH", "Jane Smith"]]
    jane, = [e for e in store.export_case("id")["entities"] if "Smith" in e["name"].title()]
    assert jane["entity_type"] == "person" and jane["birth_dates"] == ["1980-03-14"]
    assert {n["name"] for n in jane["names"]} == {"JANE ELIZABETH SMITH", "Jane Smith", "Ms Jane Smith"}
    # A second Jane Elizabeth Smith born on another day is a different person.
    other = PASSPORT.replace("14 MAR 1980", "2 JUL 1991").replace("924163857", "551234870").split("\nP<")[0]
    _, _, values = _ingest(tmp_path, other, case="id", run_id="run_9", store=store, defer=True)
    assert "ENTITY_NAME_COLLISION" in _codes(values)
    assert len([e for e in store.export_case("id")["entities"] if "Smith" in e["name"].title()]) == 2
