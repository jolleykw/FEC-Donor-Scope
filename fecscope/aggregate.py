"""
Aggregation + matching orchestration.

This is where transactions become the analytical objects the UI shows:

1. ``build_unique_donors``  collapses transactions to one row per donor and
   pivots contribution totals into one column per election cycle, so cycle
   selection later is a pure vectorized column-sum (instant, no recompute of
   the underlying data).
2. ``run_matching``         runs each unique donor through the repository store
   once (not once per transaction) and attaches flags, the "flagged via"
   provenance string, and the Feature-3 attribute columns.
3. helpers                  compute the selected-cycle total, the dynamic
   High Donor flag, and the interactive summary pivot.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import normalize as nz
from .repositories import REPO_DISPLAY, RepositoryStore

AMT_PREFIX = "amt::"
CNT_PREFIX = "cnt::"
_CONF_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "": 0}


# --------------------------------------------------------------------------- #
# Small aggregation helpers
# --------------------------------------------------------------------------- #
def _first_nonempty(series: pd.Series) -> str:
    for value in series:
        if str(value).strip():
            return str(value).strip()
    return ""


def _longest_nonempty(series: pd.Series) -> str:
    best = ""
    for value in series:
        text = str(value).strip()
        if len(text) > len(best):
            best = text
    return best


def _join_unique(series: pd.Series, limit: int = 220) -> str:
    seen = []
    for value in series:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    joined = "; ".join(seen)
    return joined[:limit]


def _donor_key(row) -> str:
    if row["donor_type"] == "individual":
        loc = row["zip5"].strip() or row["state"].strip()
        return f"I|{row['first_name']}|{row['last_name']}|{loc}"
    cid = row["committee_id"].strip().upper()
    if cid:
        return f"C|{cid}"
    return f"C|{nz.normalize_committee_name(row['committee_name'])}"


# --------------------------------------------------------------------------- #
# Step 1: unique donor master
# --------------------------------------------------------------------------- #
def build_unique_donors(transactions: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Collapse transactions to one row per donor with per-cycle columns."""
    if transactions.empty:
        return pd.DataFrame(), []

    df = transactions.copy()
    df["donor_key"] = df.apply(_donor_key, axis=1)

    cycles = sorted(df["cycle"].unique(), key=nz.cycle_sort_key)

    base = df.groupby("donor_key", sort=False).agg(
        donor_type=("donor_type", "first"),
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        committee_id=("committee_id", _first_nonempty),
        committee_name=("committee_name", _longest_nonempty),
        display_name=("display_name", _longest_nonempty),
        state=("state", _first_nonempty),
        zip5=("zip5", _first_nonempty),
        employer=("employer", _join_unique),
        occupation=("occupation", _first_nonempty),
        recipient_name=("recipient_name", _join_unique),
        total_all=("amount", "sum"),
        count_all=("amount", "size"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )

    amt = df.pivot_table(index="donor_key", columns="cycle",
                         values="amount", aggfunc="sum", fill_value=0.0)
    cnt = df.pivot_table(index="donor_key", columns="cycle",
                         values="amount", aggfunc="size", fill_value=0)
    amt = amt.reindex(columns=cycles)
    cnt = cnt.reindex(columns=cycles)
    amt.columns = [f"{AMT_PREFIX}{c}" for c in amt.columns]
    cnt.columns = [f"{CNT_PREFIX}{c}" for c in cnt.columns]

    master = base.join(amt).join(cnt).reset_index()

    # Cosmetic: title-case individual display names for the UI.
    ind = master["donor_type"] == "individual"
    master.loc[ind, "display_name"] = (
        master.loc[ind, "first_name"].str.title() + " "
        + master.loc[ind, "last_name"].str.title()
    ).str.strip()
    # Committees: ensure a label even if the raw display_name was blank.
    cmte = ~ind
    blank = cmte & (master["display_name"].str.strip() == "")
    master.loc[blank, "display_name"] = master.loc[blank, "committee_name"]
    master["display_name"] = master["display_name"].replace("", "(unnamed donor)")
    return master, cycles


# --------------------------------------------------------------------------- #
# Step 2: matching
# --------------------------------------------------------------------------- #
def run_matching(master: pd.DataFrame, store: RepositoryStore,
                 enabled_repos: Sequence[str]) -> pd.DataFrame:
    """Attach flags and Feature-3 attribute columns to the donor master."""
    enabled = set(enabled_repos)
    master = master.copy()

    flag_repos_col, flagged_via_col, confidence_col, details_col = [], [], [], []
    lobby_sectors_col, iowa_clients_col = [], []
    is_corp_col, industry_cat_col = [], []
    is_lpac_col, lpac_owner_col = [], []
    epstein_col = []

    for row in master.itertuples(index=False):
        hits = []
        if row.donor_type == "individual":
            hits.extend(store.match_individual(
                row.first_name, row.last_name, row.state, enabled))
            for emp in str(row.employer).split(";"):
                emp_hit = store.match_employer(emp, enabled)
                if emp_hit:
                    hits.append(emp_hit)
                    break
        else:
            hits.extend(store.match_committee(
                row.committee_id, row.committee_name, enabled))

        # de-duplicate repos while keeping registry order
        repos, seen = [], set()
        best_conf = ""
        details = []
        sectors, iowa_clients = [], ""
        is_corp, industry_cat = False, ""
        is_lpac, lpac_owner = False, ""
        epstein_meta = None

        for hit in hits:
            if hit.repo_key not in seen:
                seen.add(hit.repo_key)
                repos.append(hit.repo_key)
            if _CONF_RANK[hit.confidence] > _CONF_RANK[best_conf]:
                best_conf = hit.confidence
            details.append({
                "repo": hit.repo_key,
                "display": REPO_DISPLAY.get(hit.repo_key, hit.repo_key),
                "detail": hit.detail,
                "confidence": hit.confidence,
            })
            if hit.repo_key == "federal_lobbyist":
                sectors = hit.extra.get("sectors", [])
            elif hit.repo_key == "iowa_lobbyist":
                iowa_clients = hit.extra.get("clients", "")
            elif hit.repo_key == "industry_pac":
                is_corp = hit.extra.get("is_corporate", False)
                industry_cat = hit.extra.get("larger", "")
            elif hit.repo_key == "leadership_pac":
                is_lpac = True
                lpac_owner = hit.extra.get("owner", "")
            elif hit.repo_key == "epstein":
                epstein_meta = hit.extra.get("meta")

        flag_repos_col.append(repos)
        flagged_via_col.append(", ".join(REPO_DISPLAY.get(k, k) for k in repos))
        confidence_col.append(best_conf)
        details_col.append(details)
        lobby_sectors_col.append(sectors)
        iowa_clients_col.append(iowa_clients)
        is_corp_col.append(is_corp)
        industry_cat_col.append(industry_cat)
        is_lpac_col.append(is_lpac)
        lpac_owner_col.append(lpac_owner)
        epstein_col.append(epstein_meta)

    master["flag_repos"] = flag_repos_col
    master["flagged_via"] = flagged_via_col
    master["flag_confidence"] = confidence_col
    master["flag_details"] = details_col
    master["flagged"] = master["flag_repos"].map(bool)
    master["lobbyist_sectors"] = lobby_sectors_col
    master["iowa_clients"] = iowa_clients_col
    master["is_corporate_pac"] = is_corp_col
    master["industry_category"] = industry_cat_col
    master["is_leadership_pac"] = is_lpac_col
    master["lpac_owner"] = lpac_owner_col
    master["epstein_meta"] = epstein_col
    return master


# --------------------------------------------------------------------------- #
# Step 3: dynamic cycle + high-donor helpers
# --------------------------------------------------------------------------- #
def selected_total(master: pd.DataFrame, selected_cycles: Sequence[str]) -> pd.Series:
    """Sum of contributions strictly within the selected cycles, per donor."""
    cols = [f"{AMT_PREFIX}{c}" for c in selected_cycles if f"{AMT_PREFIX}{c}" in master.columns]
    if not cols:
        return pd.Series(0.0, index=master.index)
    return master[cols].sum(axis=1)


def selected_count(master: pd.DataFrame, selected_cycles: Sequence[str]) -> pd.Series:
    cols = [f"{CNT_PREFIX}{c}" for c in selected_cycles if f"{CNT_PREFIX}{c}" in master.columns]
    if not cols:
        return pd.Series(0, index=master.index)
    return master[cols].sum(axis=1).astype(int)


def high_donor_mask(totals: pd.Series, floor: float) -> pd.Series:
    return totals >= float(floor)


# --------------------------------------------------------------------------- #
# Step 4: interactive summary pivot
# --------------------------------------------------------------------------- #
ROW_DIMENSIONS = {
    "Donor type": "donor_type",
    "Flagged status": "_flag_status",
    "Confidence level": "flag_confidence",
    "Industry / macro category": "industry_category",
    "Lobbyist sector": "lobbyist_sectors",     # list-valued -> exploded
    "Leadership PAC owner": "lpac_owner",
    "State": "state",
}

METRIC_FUNCS = {
    "Total funds": lambda g: g["_amount"].sum(),
    "Unique donors": lambda g: g["donor_key"].nunique(),
    "Contribution count": lambda g: g["_count"].sum(),
    "Average donation": lambda g: (g["_amount"].sum() / g["_count"].sum()
                                   if g["_count"].sum() else 0.0),
    "Median donor total": lambda g: g["_amount"].median(),
    "Largest donor total": lambda g: g["_amount"].max(),
}


def build_summary_table(master: pd.DataFrame, amount: pd.Series, count: pd.Series,
                        row_dim_label: str, metrics: Sequence[str]) -> pd.DataFrame:
    """Build a pivot-style summary the user configures interactively."""
    if master.empty or not metrics:
        return pd.DataFrame()

    work = master.copy()
    work["_amount"] = amount.values
    work["_count"] = count.values
    work["_flag_status"] = work["flagged"].map({True: "Flagged", False: "Not flagged"})

    dim = ROW_DIMENSIONS.get(row_dim_label, "donor_type")

    if dim == "lobbyist_sectors":                # explode list column
        work = work.explode("lobbyist_sectors")
        work["lobbyist_sectors"] = work["lobbyist_sectors"].fillna("")
        work = work[work["lobbyist_sectors"] != ""]
        dim = "lobbyist_sectors"
    else:
        work[dim] = work[dim].replace("", "(none)").fillna("(none)")

    rows = []
    for group_value, group in work.groupby(dim, sort=False):
        record = {row_dim_label: group_value}
        for metric in metrics:
            func = METRIC_FUNCS.get(metric)
            record[metric] = round(float(func(group)), 2) if func else 0.0
        rows.append(record)

    table = pd.DataFrame(rows)
    if "Total funds" in metrics and not table.empty:
        table = table.sort_values("Total funds", ascending=False)
    elif "Unique donors" in metrics and not table.empty:
        table = table.sort_values("Unique donors", ascending=False)
    return table.reset_index(drop=True)


def dataset_overview(master: pd.DataFrame, amount: pd.Series, count: pd.Series,
                     high_mask: pd.Series) -> Dict[str, float]:
    return {
        "unique_donors": int(len(master)),
        "individuals": int((master["donor_type"] == "individual").sum()),
        "committees": int((master["donor_type"] == "committee").sum()),
        "total_amount": float(amount.sum()),
        "contribution_count": int(count.sum()),
        "flagged_donors": int(master["flagged"].sum()),
        "high_donors": int(high_mask.sum()),
        "avg_donation": float(amount.sum() / count.sum()) if count.sum() else 0.0,
    }


# --------------------------------------------------------------------------- #
# Transfers & attribution (memo rows)
# --------------------------------------------------------------------------- #
# These build the "who is behind the transfers" view. Memo dollars are NOT
# additive to committee totals (they are already represented by the counted
# transfer line), so this path is kept entirely separate from the donor master.
def build_attributed_donors(memo_transactions: pd.DataFrame, store: RepositoryStore,
                            enabled_repos: Sequence[str]) -> pd.DataFrame:
    """Unique donors disclosed inside transfers/conduits, cross-referenced.

    Same matching machinery as the main master (so a flagged individual hiding
    inside a transfer surfaces), but amounts are labeled as attributed, not as
    contributions to the filing committee.
    """
    if memo_transactions is None or memo_transactions.empty:
        return pd.DataFrame()

    df = memo_transactions.copy()
    df["donor_key"] = df.apply(_donor_key, axis=1)
    base = df.groupby("donor_key", sort=False).agg(
        donor_type=("donor_type", "first"),
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        committee_id=("committee_id", _first_nonempty),
        committee_name=("committee_name", _longest_nonempty),
        display_name=("display_name", _longest_nonempty),
        state=("state", _first_nonempty),
        zip5=("zip5", _first_nonempty),
        employer=("employer", _join_unique),
        occupation=("occupation", _first_nonempty),
        attributed_amount=("amount", "sum"),
        attributed_count=("amount", "size"),
        transfer_sources=("transfer_source", lambda s: sorted({x for x in s if x})),
    ).reset_index()

    ind = base["donor_type"] == "individual"
    base.loc[ind, "display_name"] = (
        base.loc[ind, "first_name"].str.title() + " "
        + base.loc[ind, "last_name"].str.title()
    ).str.strip()
    blank = (~ind) & (base["display_name"].str.strip() == "")
    base.loc[blank, "display_name"] = base.loc[blank, "committee_name"]
    base["display_name"] = base["display_name"].replace("", "(unnamed donor)")

    base = run_matching(base, store, enabled_repos)
    base["transfer_via"] = base["transfer_sources"].map(lambda lst: ", ".join(lst))
    return base


def summarize_transfers(memo_transactions: pd.DataFrame,
                        attributed: pd.DataFrame) -> pd.DataFrame:
    """One row per transfer/parent line: counted amount vs. who is behind it."""
    if memo_transactions is None or memo_transactions.empty:
        return pd.DataFrame()

    df = memo_transactions.copy()
    df["donor_key"] = df.apply(_donor_key, axis=1)
    flagged_keys = set()
    if attributed is not None and not attributed.empty:
        flagged_keys = set(attributed.loc[attributed["flagged"], "donor_key"])
    df["_flagged"] = df["donor_key"].isin(flagged_keys)

    rows = []
    for source, group in df.groupby("transfer_source", sort=False):
        rows.append({
            "Attributed to / memo": source,
            "Parent schedule": group["transfer_parent_schedule"].iloc[0],
            "Counted parent amount": float(group["transfer_amount"].iloc[0]),
            "Underlying donors": int(group["donor_key"].nunique()),
            "Attributed total": round(float(group["amount"].sum()), 2),
            "Flagged underlying": int(group.loc[group["_flagged"], "donor_key"].nunique()),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Counted parent amount", "Attributed total"],
                              ascending=False)
    return out.reset_index(drop=True)
