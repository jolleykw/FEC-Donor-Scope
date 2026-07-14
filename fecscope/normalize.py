"""
Normalization helpers shared by ingestion, matching, and aggregation.

Everything that turns a raw string into a comparable key lives here so the whole
application uses one consistent definition of "the same name" or "the same
committee". Keeping it in one module is what makes the matching reproducible.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

# Suffixes / prefixes that should not participate in name identity.
_NAME_NOISE = {
    "MR", "MRS", "MS", "DR", "MISS", "HON", "REV", "SIR",
    "JR", "SR", "II", "III", "IV", "V", "ESQ", "PHD", "MD", "DDS",
}

# Corporate / committee tokens normalized to a canonical short form so that
# "POLITICAL ACTION COMMITTEE" and "PAC" collapse to the same key.
_COMMITTEE_SUBSTITUTIONS = [
    (r"\bPOLITICAL ACTION COMMITTEE\b", "PAC"),
    (r"\bPOL(?:ITICAL)? ACTION COM(?:MITTEE)?\b", "PAC"),
    (r"\bPOLITICAL ACTION\b", "PAC"),
    (r"\bCORPORATION\b", "CORP"),
    (r"\bINCORPORATED\b", "INC"),
    (r"\bLIMITED LIABILITY COMPANY\b", "LLC"),
    (r"\bLIMITED LIABILITY CO(?:MPANY)?\b", "LLC"),
    (r"\bASSOCIATION\b", "ASSN"),
    (r"\bCOMPANY\b", "CO"),
    (r"\bCOMMITTEE\b", "CMTE"),
    (r"\bFEDERAL\b", "FED"),
]

_PUNCT_RE = re.compile(r"[^A-Z0-9\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(value) -> str:
    """Upper-case, strip punctuation, and collapse whitespace.

    This is the baseline used for exact-key comparisons. It is deliberately
    aggressive about punctuation ("O'BRIEN" -> "OBRIEN") because filings are
    inconsistent about apostrophes, hyphens, and periods.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).upper().strip()
    if text in {"NAN", "NA", "NONE", "NULL"}:
        return ""
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _strip_noise_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _NAME_NOISE] or tokens


def clean_person_name(value) -> str:
    """Clean a person name and drop honorifics/suffixes."""
    tokens = _strip_noise_tokens(clean_text(value).split())
    return " ".join(tokens)


def normalize_committee_name(value) -> str:
    """Canonicalize a committee/organization name for fuzzy matching."""
    text = clean_text(value)
    if not text:
        return ""
    for pattern, replacement in _COMMITTEE_SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return _WS_RE.sub(" ", text).strip()


def parse_person_name(full_name) -> Tuple[str, str]:
    """Split a single name string into (first, last).

    Handles the two dominant filing conventions: "LAST, FIRST" (receipts
    ``contributor_name``) and "FIRST LAST" (free text). Returns cleaned,
    upper-cased parts with honorifics removed.
    """
    raw = "" if full_name is None else str(full_name).strip()
    if not raw or clean_text(raw) == "":
        return "", ""
    if "," in raw:
        last, _, first = raw.partition(",")
        return clean_person_name(first), clean_person_name(last)
    tokens = clean_person_name(raw).split()
    if not tokens:
        return "", ""
    if len(tokens) == 1:
        return "", tokens[0]
    return tokens[0], tokens[-1]


def format_person_display(name) -> str:
    """Turn a raw sponsor name into a readable label.

    'PETTERSEN, BRITTANY LOUISE MS.' -> 'Brittany Louise Pettersen'.
    Falls back to a simple title-case when there is no comma.
    """
    raw = "" if name is None else str(name).strip()
    if not raw:
        return ""
    if "," in raw:
        last, _, rest = raw.partition(",")
        rest_tokens = [t for t in rest.replace(".", "").split()
                       if t.upper() not in _NAME_NOISE]
        ordered = rest_tokens + [last.strip()]
        return " ".join(w.capitalize() for w in ordered if w).strip()
    return " ".join(w.capitalize() for w in raw.split())


def zip5(value) -> str:
    """Return the first five digits of a ZIP code, or ''."""
    digits = re.sub(r"\D", "", "" if value is None else str(value))
    return digits[:5] if digits else ""


def parse_amount(value) -> float:
    """Parse a currency string into a float, tolerant of $ , and () negatives."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "").replace("$", "").replace(",", "").strip()
    try:
        amount = float(text)
    except ValueError:
        return 0.0
    return -amount if negative else amount


def parse_date(value) -> Optional[pd.Timestamp]:
    """Parse the several date encodings FEC files use into a Timestamp."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "NAN", "NONE"}:
        return None
    # FECFILE compact form: YYYYMMDD
    if re.fullmatch(r"\d{8}", text):
        try:
            return pd.Timestamp(datetime.strptime(text, "%Y%m%d"))
        except ValueError:
            return None
    try:
        return pd.to_datetime(text, errors="coerce")
    except (ValueError, TypeError):
        return None


def cycle_end_year(year: int) -> int:
    """Even year that closes the 2-year federal cycle containing ``year``."""
    return year if year % 2 == 0 else year + 1


def cycle_label_from_year(year: Optional[int]) -> Optional[str]:
    """Return a cycle label like '2025-2026' for a calendar year."""
    if year is None:
        return None
    end = cycle_end_year(int(year))
    return f"{end - 1}-{end}"


def cycle_label_from_date(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    return cycle_label_from_year(ts.year)


def cycle_sort_key(label: str) -> int:
    """Sort cycles chronologically by their ending year."""
    match = re.search(r"(\d{4})\s*$", label)
    return int(match.group(1)) if match else 0
