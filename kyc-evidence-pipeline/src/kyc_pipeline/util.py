from __future__ import annotations

import calendar
import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def json_text(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2 if pretty else None, separators=None if pretty else (",", ":"))


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return prefix + "_" + digest(json_text(parts))[:24]


def identity_key(name: str | None) -> str:
    """Conservative textual key, NOT legal-entity resolution or an alias match."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def normalize_date(value: str) -> str:
    """Normalize only explicit, unambiguous ISO or English month-name dates."""
    value = value.strip()
    if re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        date_interval(value)
        return value
    value = re.sub(r"\s+", " ", value).replace("Sept.", "Sep").replace("Sept ", "Sep ")
    for pattern in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%B %Y", "%b %Y"):
        try:
            parsed = datetime.strptime(value, pattern)
            return parsed.strftime("%Y-%m") if pattern in {"%B %Y", "%b %Y"} else parsed.date().isoformat()
        except ValueError:
            pass
    raise ValueError("effective_date must be ISO or an unambiguous English month-name date")


def lei_check_digits_valid(lei: str) -> bool:
    """ISO 17442 / ISO 7064 MOD 97-10: letters become 10-35 and the whole number mod 97 is 1."""
    if not re.fullmatch(r"[A-Z0-9]{18}[0-9]{2}", lei):
        return False
    return int("".join(str(int(ch, 36)) for ch in lei)) % 97 == 1


def date_present_in_text(value: str, text: str) -> bool:
    """Require the normalized date to be expressed in the cited text; no inferred date."""
    if re.search(r"(?<!\d)" + re.escape(value) + r"(?!\d)", text):
        return True
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember|t)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    pat = rf"\b(?:{months}\.?\s+\d{{1,2}},?\s+\d{{4}}|\d{{1,2}}\s+{months}\.?\s+\d{{4}}|{months}\.?\s+\d{{4}})\b"
    for m in re.finditer(pat, text, re.I):
        try:
            if normalize_date(m.group().replace(".", "")) == value:
                return True
        except ValueError:
            pass
    return False


def date_interval(value: str | None) -> tuple[date, date] | None:
    if not value:
        return None
    if not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        raise ValueError("effective_date must be YYYY, YYYY-MM, or YYYY-MM-DD")
    fields = [int(p) for p in value.split("-")]
    if len(fields) == 1:
        return date(fields[0], 1, 1), date(fields[0], 12, 31)
    if len(fields) == 2:
        return date(fields[0], fields[1], 1), date(fields[0], fields[1], calendar.monthrange(fields[0], fields[1])[1])
    exact = date(*fields)
    return exact, exact


def temporal_overlap(a: str | None, b: str | None) -> bool:
    """Unknown dates conservatively overlap; partial dates are date ranges."""
    ia, ib = date_interval(a), date_interval(b)
    if not ia or not ib:
        return True
    return ia[0] <= ib[1] and ib[0] <= ia[1]
