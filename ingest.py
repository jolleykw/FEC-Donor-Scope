"""
Ingestion layer.

Two very different on-disk formats collapse into one canonical transactions
table here, so every downstream stage (matching, aggregation, UI) can ignore
where the data came from:

* Receipts export  -- the wide, named-column CSV you download from fec.gov
                      ("Bulk / receipts"). One row per receipt.
* Quarterly filing -- the raw electronic filing (FECFILE / .fec exported to
                      CSV) with schedule rows (HDR, F3*, SA*, SB*, ...).

Memo / attribution rows
-----------------------
Both formats disclose "memo" line items (memo code X): the itemized individuals
behind a committee transfer (e.g. a joint-fundraising-committee transfer), or
the conduit behind an earmarked gift. Their dollars are already represented by
the counted line they sit under, so they must NOT be added to totals. But they
name the people behind a transfer, which is exactly what an investigator needs.

So instead of discarding them, ingestion splits every file into two frames:
    * ``transactions``       counted receipts (memo rows removed) -> totals
    * ``memo_transactions``  the memo rows, each linked to the counted line it
                             attributes to (``transfer_source`` / amount)

Canonical transaction columns produced for both frames:
    donor_type            'individual' | 'committee'
    first_name last_name  (individuals; upper-cased, cleaned)
    display_name          human-readable donor label
    committee_id          donor committee FEC id (committees)
    committee_name        donor committee/org name (committees)
    state zip5            location
    employer occupation   (individuals)
    amount                float (can be negative for refunds)
    date                  pd.Timestamp | NaT
    cycle                 '2023-2024' style label
    recipient_id recipient_name
    schedule              originating schedule code
The memo frame additionally carries:
    transfer_source            name of the counted line it attributes to
    transfer_amount            that counted line's dollar amount
    transfer_parent_schedule   that counted line's schedule code
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from . import normalize as nz

CANONICAL_COLUMNS = [
    "donor_type", "first_name", "last_name", "display_name",
    "committee_id", "committee_name", "state", "zip5",
    "employer", "occupation", "amount", "date", "cycle",
    "recipient_id", "recipient_name", "schedule",
]
MEMO_EXTRA_COLUMNS = ["transfer_source", "transfer_amount", "transfer_parent_schedule"]


@dataclass
class LoadResult:
    """Everything the UI needs to know about an ingested file."""
    transactions: pd.DataFrame
    fmt: str                       # 'receipts' | 'quarterly'
    memo_transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    recipients: List[str] = field(default_factory=list)
    rows_read: int = 0
    rows_kept: int = 0
    memo_rows_retained: int = 0
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #
def _read_raw(file_or_path) -> str:
    if hasattr(file_or_path, "read"):
        data = file_or_path.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        return data
    with open(file_or_path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def detect_format(raw_text: str) -> str:
    """Return 'quarterly' for a schedule-based filing, else 'receipts'."""
    head = raw_text[:4000].upper()
    first_line = raw_text.lstrip().splitlines()[0].upper() if raw_text.strip() else ""
    if first_line.startswith('"HDR"') or first_line.startswith("HDR,"):
        return "quarterly"
    if "CONTRIBUTION_RECEIPT_AMOUNT" in head or "CONTRIBUTOR_LAST_NAME" in head:
        return "receipts"
    # Fall back to sniffing for schedule tokens (SA11AI etc.).
    if '"SA1' in head or ",SA1" in head or "SCHEDULE" in head:
        return "quarterly"
    return "receipts"


# --------------------------------------------------------------------------- #
# Receipts (named-column) loader
# --------------------------------------------------------------------------- #
_RECEIPT_INDIVIDUAL_ENTITIES = {"IND"}


def load_receipts(raw_text: str) -> LoadResult:
    df = pd.read_csv(io.StringIO(raw_text), dtype=str, low_memory=False, na_filter=False)
    df.columns = [c.strip().lower() for c in df.columns]
    rows_read = len(df)

    def col(name: str) -> pd.Series:
        return df[name] if name in df.columns else pd.Series([""] * len(df), index=df.index)

    entity = col("entity_type").str.upper().str.strip()
    first = col("contributor_first_name")
    last = col("contributor_last_name")
    contributor_name = col("contributor_name")     # "LAST, FIRST" or org name

    # Some receipts rows only populate contributor_name; recover parts from it.
    need_split = (first.str.strip() == "") & (last.str.strip() == "") & (entity == "IND")
    if need_split.any():
        parsed = contributor_name.where(need_split, "").map(nz.parse_person_name)
        first = first.mask(need_split, parsed.map(lambda t: t[0]))
        last = last.mask(need_split, parsed.map(lambda t: t[1]))

    is_individual = entity.isin(_RECEIPT_INDIVIDUAL_ENTITIES) | (
        (entity == "") & (last.str.strip() != "")
    )

    amount = col("contribution_receipt_amount").map(nz.parse_amount)
    date = col("contribution_receipt_date").map(nz.parse_date)

    # Prefer the filing's own 2-year period; fall back to the date's cycle.
    period = col("two_year_transaction_period").str.strip()
    cycle = period.map(lambda y: nz.cycle_label_from_year(int(y)) if y.isdigit() else None)
    cycle = cycle.fillna(pd.Series(date.map(nz.cycle_label_from_date), index=df.index))

    donor_type = pd.Series("committee", index=df.index)
    donor_type = donor_type.mask(is_individual, "individual")

    first_c = first.map(nz.clean_person_name)
    last_c = last.map(nz.clean_person_name)
    committee_name = contributor_name.where(~is_individual, "")

    display = pd.Series("", index=df.index)
    display = display.mask(is_individual, (first_c + " " + last_c).str.strip())
    display = display.mask(~is_individual, committee_name.str.strip())

    out = pd.DataFrame({
        "donor_type": donor_type,
        "first_name": first_c.where(is_individual, ""),
        "last_name": last_c.where(is_individual, ""),
        "display_name": display,
        "committee_id": col("contributor_id").where(~is_individual, ""),
        "committee_name": committee_name,
        "state": col("contributor_state").str.upper().str.strip(),
        "zip5": col("contributor_zip").map(nz.zip5),
        "employer": col("contributor_employer").where(is_individual, ""),
        "occupation": col("contributor_occupation").where(is_individual, ""),
        "amount": amount,
        "date": date,
        "cycle": cycle,
        "recipient_id": col("committee_id"),
        "recipient_name": col("committee_name"),
        "schedule": "receipts",
    })

    # Memo split. In bulk receipts, memo_code == 'X' marks a memo entry whose
    # amount is already counted on another line.
    memo_code = col("memo_code").str.upper().str.strip()
    is_memo = memo_code == "X"

    txn_id = col("transaction_id").str.strip()
    back_ref = col("back_reference_transaction_id").str.strip()
    # map a counted transaction id -> (name, amount) so memo rows can name their parent
    parent_name: Dict[str, str] = {}
    parent_amt: Dict[str, float] = {}
    for tid, nm, amt, memo in zip(txn_id, out["display_name"], out["amount"], is_memo):
        if tid and not memo:
            parent_name.setdefault(tid, nm)
            parent_amt.setdefault(tid, amt)
    resolved = back_ref.map(lambda t: parent_name.get(t, ""))
    # When the parent line is not in the file, fall back to the disclosed memo
    # reason (e.g. "REATTRIBUTION TO SPOUSE", "REDESIGNATION TO GENERAL") so the
    # entry is still self-explanatory.
    reason = col("memo_text").str.strip().str.slice(0, 70)
    reason = reason.where(reason != "", col("receipt_type_desc").str.strip())
    reason = reason.where(reason.str.strip() != "", "(memo entry)")
    out["transfer_source"] = resolved.where(resolved.str.strip() != "", reason)
    out["transfer_amount"] = back_ref.map(lambda t: parent_amt.get(t, 0.0))
    out["transfer_parent_schedule"] = "receipts"

    counted = _finalize(out[~is_memo].copy())
    memo = _finalize(out[is_memo].copy(), extra=MEMO_EXTRA_COLUMNS)

    recipients = sorted({r for r in counted["recipient_name"].unique() if r})
    notes = [f"Loaded receipts export with {rows_read:,} rows."]
    if len(memo):
        notes.append(
            f"Retained {len(memo):,} memo/attribution rows for the Transfers & "
            "attribution view; they stay out of totals to avoid double-counting."
        )
    return LoadResult(
        transactions=counted, fmt="receipts", memo_transactions=memo,
        recipients=recipients, rows_read=rows_read, rows_kept=len(counted),
        memo_rows_retained=len(memo), notes=notes,
    )


# --------------------------------------------------------------------------- #
# Quarterly (schedule-based) loader
# --------------------------------------------------------------------------- #
# Standard leading field order for Schedule A rows in the electronic filing
# format. These positions are stable across form versions; trailing memo
# fields are located dynamically below.
_SA = {
    "form": 0, "filer_id": 1, "txn_id": 2, "backref": 3,
    "entity": 5, "org_name": 6, "last": 7, "first": 8,
    "city": 14, "state": 15, "zip": 16, "election": 17,
    "date": 19, "amount": 20, "aggregate": 21,
    "employer": 23, "occupation": 24,
    "donor_cmte_id": 25, "donor_cmte_name": 26,
}


def _detect_memo_code_col(sa_rows: List[List[str]]) -> Optional[int]:
    """Find the memo-code column (values are a subset of {'X'}).

    Earmark / conduit / transfer-itemization rows carry memo code 'X' and are
    excluded from totals so a contribution is not counted once as the counted
    line and again as its attribution.
    """
    if not sa_rows:
        return None
    width = max(len(r) for r in sa_rows)
    for c in range(30, width):          # memo code always trails the core fields
        seen = set()
        has_x = False
        for r in sa_rows:
            if c < len(r):
                v = r[c].strip().upper()
                if v:
                    seen.add(v)
                    if v == "X":
                        has_x = True
        if has_x and seen <= {"X"}:
            return c
    return None


def _sa_record(r: List[str], memo_col: Optional[int], filer_id: str,
               filer_name: str) -> Optional[dict]:
    """Build one canonical record from a Schedule A row (or None to skip)."""
    def g(key: str) -> str:
        idx = _SA[key]
        return r[idx].strip() if idx < len(r) else ""

    entity = g("entity").upper()
    amount = nz.parse_amount(g("amount"))
    date = nz.parse_date(g("date"))
    cycle = nz.cycle_label_from_date(date)
    state = g("state").upper()
    zc = nz.zip5(g("zip"))

    if entity == "IND":
        first_c = nz.clean_person_name(g("first"))
        last_c = nz.clean_person_name(g("last"))
        if not first_c and not last_c:
            return None
        return {
            "donor_type": "individual",
            "first_name": first_c, "last_name": last_c,
            "display_name": f"{first_c} {last_c}".strip(),
            "committee_id": "", "committee_name": "",
            "state": state, "zip5": zc,
            "employer": g("employer"), "occupation": g("occupation"),
            "amount": amount, "date": date, "cycle": cycle,
            "recipient_id": filer_id, "recipient_name": filer_name,
            "schedule": g("form").upper(),
        }
    name = g("org_name")
    if not name.strip():
        return None
    return {
        "donor_type": "committee",
        "first_name": "", "last_name": "",
        "display_name": name.strip(),
        "committee_id": g("donor_cmte_id"), "committee_name": name.strip(),
        "state": state, "zip5": zc,
        "employer": "", "occupation": "",
        "amount": amount, "date": date, "cycle": cycle,
        "recipient_id": filer_id, "recipient_name": filer_name,
        "schedule": g("form").upper(),
    }


def load_quarterly(raw_text: str) -> LoadResult:
    reader = csv.reader(io.StringIO(raw_text))
    all_rows = [r for r in reader if r]
    rows_read = len(all_rows)

    filer_name = ""
    filer_id = ""
    for r in all_rows:
        tag = r[0].strip().upper()
        if tag.startswith("F3"):            # cover record: F3, F3N, F3X ...
            filer_id = r[1].strip() if len(r) > 1 else ""
            filer_name = r[2].strip() if len(r) > 2 else ""
            break

    sa_rows = [r for r in all_rows if r[0].strip().upper().startswith("SA")]
    memo_col = _detect_memo_code_col(sa_rows)

    counted_records: List[dict] = []
    memo_records: List[dict] = []
    # Association of a memo row with the counted line it attributes to:
    #   1. by BACK_REFERENCE_TRAN_ID when the filer populates it, else
    #   2. positionally -- memo rows follow the counted line they belong to.
    parent_by_txn: Dict[str, dict] = {}
    current_parent: Optional[dict] = None

    for r in sa_rows:
        is_memo = (memo_col is not None and memo_col < len(r)
                   and r[memo_col].strip().upper() == "X")
        rec = _sa_record(r, memo_col, filer_id, filer_name)
        if rec is None:
            continue

        if is_memo:
            back_ref = r[_SA["backref"]].strip() if _SA["backref"] < len(r) else ""
            parent = parent_by_txn.get(back_ref) if back_ref else None
            if parent is None:
                parent = current_parent
            rec["transfer_source"] = parent["name"] if parent else "(unattributed memo)"
            rec["transfer_amount"] = parent["amount"] if parent else 0.0
            rec["transfer_parent_schedule"] = parent["schedule"] if parent else ""
            memo_records.append(rec)
        else:
            counted_records.append(rec)
            txn_id = r[_SA["txn_id"]].strip() if _SA["txn_id"] < len(r) else ""
            current_parent = {
                "name": rec["display_name"], "amount": rec["amount"],
                "schedule": rec["schedule"], "txn_id": txn_id,
            }
            if txn_id:
                parent_by_txn[txn_id] = current_parent

    counted = _finalize(pd.DataFrame(counted_records, columns=CANONICAL_COLUMNS))
    memo = _finalize(pd.DataFrame(memo_records, columns=CANONICAL_COLUMNS + MEMO_EXTRA_COLUMNS),
                     extra=MEMO_EXTRA_COLUMNS)

    recipients = [filer_name] if filer_name else []
    notes = [f"Parsed quarterly filing for {filer_name or 'unknown filer'}."]
    if len(memo):
        notes.append(
            f"Retained {len(memo):,} memo/attribution rows (the individuals behind "
            "transfers and conduit gifts) for the Transfers & attribution view; they "
            "stay out of committee totals to avoid double-counting."
        )
    return LoadResult(
        transactions=counted, fmt="quarterly", memo_transactions=memo,
        recipients=recipients, rows_read=rows_read, rows_kept=len(counted),
        memo_rows_retained=len(memo), notes=notes,
    )


# --------------------------------------------------------------------------- #
def _finalize(df: pd.DataFrame, extra: tuple = ()) -> pd.DataFrame:
    columns = CANONICAL_COLUMNS + list(extra)
    if df.empty:
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in df.columns:
            df[c] = 0.0 if c in ("amount", "transfer_amount") else ""
    df = df[columns].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "transfer_amount" in df.columns:
        df["transfer_amount"] = pd.to_numeric(df["transfer_amount"], errors="coerce").fillna(0.0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cycle"] = df["cycle"].where(df["cycle"].notna(), "Unknown cycle")
    string_cols = ["first_name", "last_name", "display_name", "committee_id",
                   "committee_name", "state", "zip5", "employer", "occupation",
                   "recipient_id", "recipient_name", "schedule"]
    string_cols += [c for c in extra if c != "transfer_amount"]
    for c in string_cols:
        df[c] = df[c].fillna("").astype(str)
    return df.reset_index(drop=True)


def load_fec_file(file_or_path, forced_format: Optional[str] = None) -> LoadResult:
    """Entry point: read a file, detect its format, and canonicalize it."""
    raw = _read_raw(file_or_path)
    fmt = forced_format or detect_format(raw)
    if fmt == "quarterly":
        return load_quarterly(raw)
    return load_receipts(raw)
