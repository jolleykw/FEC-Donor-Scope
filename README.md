# Donor Scope

A bright, modern Streamlit app for analyzing Federal Election Commission (FEC)
campaign-finance data. It aggregates contributions into one record per unique
donor, cross-references those donors against a set of local masterlists, and
surfaces flagged donors, industry/PAC breakdowns, and interactive summary
statistics.

The app accepts either of the two FEC file shapes people actually work with —
the named-column **receipts export** from fec.gov, or a raw **quarterly filing**
(FECFILE) — and normalizes both into a single internal representation.

---

## Quick start

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

Streamlit opens the app in your browser (default http://localhost:8501). Upload
a file in the sidebar, choose which repositories to cross-reference, set the
High Donor floor, and select **Run analysis**.

---

## Accepted input formats

**Receipts export (Option A).** The wide CSV downloaded from fec.gov, one row per
receipt, with named columns such as `contributor_last_name`,
`contribution_receipt_amount`, `contributor_employer`, and
`two_year_transaction_period`. Committee/PAC contributions are identified by
`entity_type` and the donor committee id in `contributor_id`.

**Quarterly filing (Option B).** The raw electronic filing (FECFILE) exported to
CSV, containing schedule rows (`HDR`, `F3`, `SA11AI`, `SA11C`, `SB…`, …). The
loader reads Schedule A (contributions received) and derives the election cycle
from each contribution date.

> **Memo / attribution handling.** Filings disclose "memo" line items (memo code
> `X`): the individuals itemized behind a committee transfer, the conduit behind
> an earmarked gift, or reattribution/redesignation adjustments. Their dollars are
> already represented by the counted line they sit under, so Donor Scope detects
> the memo-code column automatically and keeps those rows **out of the totals** to
> avoid double-counting. It does **not** discard them: every memo row is retained,
> linked to the counted line it attributes to, and surfaced in the
> **Transfers & attribution** view (below) so the people behind a transfer are
> visible and cross-referenced. Disbursement schedules (`SB…`) are ignored for
> donor analysis.

Format detection is automatic; you can override it from the sidebar if needed.

---

## Repositories

The app reads masterlists from the `repositories/` folder. Include any subset of
the nine supported lists; the sidebar shows which are present and how many rows
each contains. Each list keeps its own real-world schema:

| List | Matches on | Key columns used |
|------|------------|------------------|
| bad donor masterlist | individual name | `Name`, `Blurb` |
| bad employer masterlist | employer string | `NAME`, `Flag` |
| bad group masterlist | committee id / name | `Committee.ID`, `Committee.Name`, `Flag` |
| epstein persons list | individual name (fuzzy, incl. aliases) | `Name`, `Aliases`, `Category`, … |
| iowa lobbyist masterlist | individual name | `Name`, `clients`, `year` |
| industry pac masterlist | committee id / name | `Committee.ID`, `Org.type`, `Larger.Categories` |
| federal lobbyist masterlist | individual name | `lobbyist_full_name`, `code_desc` |
| leadership pac masterlist | committee id / name | `Committee_Id`, `Sponsor_Name` |
| politico young republicans list | individual name + state | `NAME`, `State`, `Politico.Quote` |

---

## Core features

1. **Unique donor master.** One row per donor (individuals keyed by name + ZIP,
   committees by FEC id). Contribution totals are pivoted into one column per
   election cycle, so toggling the cycle checkboxes recomputes every total
   instantly. Any donor at or above the High Donor floor for the selected cycles
   is flagged.
2. **Flagged lists.** Donors grouped by the exact repository that triggered each
   flag, with a "Flagged via …" provenance string, match detail, and match
   confidence. Each repository has a fixed identity color used throughout.
3. **Industry & PAC filters.**
   - *Lobbyist sectors* — filters by the LDA disclosure area using the
     natural-language `code_desc`, not the three-letter code.
   - *Corporate PAC sectors* — corporate PACs (industry list org type `C`) with
     macro-category sub-filters and a spend-by-category chart.
   - *Leadership PACs* — leadership PACs with the owning candidate pulled from
     the masterlist's `Sponsor_Name` relationship.
4. **Transfers & attribution.** Committee transfers (e.g. a joint-fundraising
   committee moving money to the filer) appear in the totals as a single
   aggregate line, which can hide the individual donors behind them. This view
   lists every memo/attribution entry in the filing — the underlying donors
   behind each transfer, the conduits behind earmarks, and reattribution
   adjustments — grouped by the line they attribute to. Those donors are run
   through the **same** masterlist matching, so a flagged individual hiding
   inside a transfer surfaces here (and is cross-linked from the Flagged tab).
   These amounts are shown for transparency only and are never added to the
   committee totals.
5. **Interactive summary.** A configurable pivot: pick a row dimension (donor
   type, flagged status, industry category, lobbyist sector, state, …) and the
   metrics to compute (total funds, unique donors, average donation, …).

---

## Matching logic

Names, employers, and committee names are normalized consistently (upper-cased,
punctuation stripped, whitespace collapsed, honorifics and common corporate
suffixes canonicalized). Matching then runs in three lanes:

- **Exact keys** for name, employer, and committee id/name lookups (O(1) dict
  lookups) — the fast common path.
- **Confidence tiers** where extra signal exists, e.g. name + state (HIGH) vs.
  name only (MEDIUM) for the young republicans list.
- **Fuzzy matching** for the Epstein list (name + aliases) and industry
  committee names, gated by the sidebar's similarity threshold. Uses
  [`rapidfuzz`](https://pypi.org/project/rapidfuzz/) when installed, and falls
  back to the standard library's `difflib` otherwise.

Matching runs once per *unique donor* rather than once per transaction, which
keeps it fast on large files.

---

## Architecture
```
app.py                 Streamlit UI only (rendering, layout, styling)
fecscope/
  normalize.py         string / name / date / cycle normalization
  ingest.py            format detection + receipts & quarterly loaders
  repositories.py      loads the 9 masterlists, builds lookup indices, matches
  aggregate.py         unique-donor master, matching orchestration,
                       cycle math, summary pivots, transfer attribution
repositories/          the cross-reference masterlists (swap in full data here)
sample_data/           example receipts and quarterly files
.streamlit/config.toml light theme
```

`app.py` imports from `fecscope` and does no data processing of its own beyond
formatting values for display.

---

## Notes and limitations

- Matches are investigative leads, not conclusions. Names collide; verify any
  total or affiliation against the underlying FEC records before relying on it.
- Individual identity uses name + ZIP (falling back to state). A donor who files
  under materially different names or locations may appear as more than one row.
