#!/usr/bin/env python3
"""Score independently completed fact_review_template.csv, without guessing missing labels."""
from __future__ import annotations
import argparse
import csv
import json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_file")
    args = p.parse_args()
    with open(args.csv_file, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    counts = {"SUPPORTED": 0, "UNSUPPORTED": 0, "UNCERTAIN": 0, "UNLABELED": 0}
    for row in rows:
        label = (row.get("source_support_label") or "").strip().upper()
        reviewer = (row.get("reviewer") or "").strip()
        if not label:
            counts["UNLABELED"] += 1
        elif label not in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}:
            raise ValueError(f"Unknown source_support_label {label!r}")
        elif not reviewer:
            raise ValueError("Every nonempty label requires a reviewer")
        else:
            counts[label] += 1
    judged = counts["SUPPORTED"] + counts["UNSUPPORTED"]
    print(json.dumps({"counts": counts, "adjudicated_supported_or_unsupported": judged, "unsupported_rate_among_adjudicated": counts["UNSUPPORTED"] / judged if judged else None, "caveat": "This rate is only for the labeled sample and depends on reviewer competence; UNCERTAIN and unlabeled claims are not counted in the denominator."}, indent=2))


if __name__ == "__main__":
    main()
