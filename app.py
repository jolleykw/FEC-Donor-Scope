"""
Donor Scope - FEC campaign-finance analysis and cross-reference tool.

This module is UI only. Every piece of data logic lives in the ``fecscope``
package (ingest / repositories / matching / aggregation). The app reads an FEC
receipts export or a quarterly filing, cross-references donors against the local
repository masterlists, and presents the four core views: the unique-donor
master, flagged lists, industry/PAC filters, and an interactive summary pivot.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from fecscope import aggregate
from fecscope import normalize as nz
from fecscope.ingest import load_fec_file
from fecscope.repositories import REPO_DISPLAY, REPO_REGISTRY, RepositoryStore

REPO_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repositories")

st.set_page_config(
    page_title="Donor Scope - FEC Analysis",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Functional color system. Color encodes meaning, not decoration:
#   - each repository has a fixed identity color used everywhere it appears
#   - confidence and the High Donor state have their own semantic colors
# --------------------------------------------------------------------------- #
REPO_COLORS = {
    "bad_donor": "#e11d48",
    "bad_employer": "#db2777",
    "bad_group": "#b91c1c",
    "epstein": "#334155",
    "iowa_lobbyist": "#047857",
    "industry_pac": "#2563eb",
    "federal_lobbyist": "#7c3aed",
    "leadership_pac": "#0d9488",
    "young_republicans": "#ea580c",
}
CONF_COLORS = {"HIGH": "#e11d48", "MEDIUM": "#d97706", "LOW": "#64748b", "": "#94a3b8"}
HIGH_DONOR_COLOR = "#b45309"
CATEGORY_PALETTE = [
    "#2563eb", "#0d9488", "#7c3aed", "#db2777", "#ea580c", "#65a30d",
    "#0891b2", "#c026d3", "#dc2626", "#4f46e5", "#059669", "#d97706",
]


def category_color(name: str) -> str:
    if not name:
        return "#94a3b8"
    return CATEGORY_PALETTE[hash(name) % len(CATEGORY_PALETTE)]


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; }
    .block-container { padding-top: 2.2rem; max-width: 1400px; }

    /* Brand header */
    .ds-brand { display:flex; align-items:baseline; gap:.6rem; }
    .ds-brand .mark { font-family:'Space Grotesk'; font-weight:700; font-size:1.9rem;
        color:#0f172a; }
    .ds-brand .mark span { color:#4338ca; }
    .ds-sub { color:#64748b; font-size:.95rem; margin:-.2rem 0 1.2rem 0; }

    /* Metric cards */
    .ds-cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:.75rem; margin:.3rem 0 1rem 0; }
    .ds-card { background:#ffffff; border:1px solid #e5e9f2; border-radius:14px;
        padding:.85rem 1rem; box-shadow:0 1px 2px rgba(15,23,42,.04); }
    .ds-card .label { color:#64748b; font-size:.72rem; text-transform:uppercase;
        letter-spacing:.06em; font-weight:600; }
    .ds-card .value { font-family:'Space Grotesk'; font-size:1.55rem; font-weight:700;
        color:#0f172a; font-variant-numeric:tabular-nums; line-height:1.2; }
    .ds-card .value.accent { color:#4338ca; }
    .ds-card .value.alert  { color:#e11d48; }
    .ds-card .value.gold   { color:#b45309; }

    /* Chips / badges */
    .chip { display:inline-block; padding:2px 9px; border-radius:999px; font-size:.72rem;
        font-weight:600; margin:2px 4px 2px 0; border:1px solid transparent;
        white-space:nowrap; }
    .chip.soft { background:var(--c-bg); color:var(--c-fg); border-color:var(--c-bd); }

    /* Section eyebrow */
    .eyebrow { color:#4338ca; font-weight:700; font-size:.74rem; letter-spacing:.08em;
        text-transform:uppercase; margin-bottom:.1rem; }

    /* Flag group header */
    .flag-head { display:flex; align-items:center; gap:.55rem; margin:.2rem 0 .1rem 0; }
    .flag-dot { width:11px; height:11px; border-radius:3px; display:inline-block; }
    .flag-title { font-weight:700; font-size:1.02rem; color:#0f172a; }
    .flag-count { color:#64748b; font-size:.85rem; font-weight:500; }

    .legend { display:flex; flex-wrap:wrap; gap:.35rem; margin:.2rem 0 .8rem 0; }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk'; }
    .stTabs [data-baseweb="tab"] { font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Small render helpers (presentation only)
# --------------------------------------------------------------------------- #
def money(value: float) -> str:
    try:
        return f"${float(value):,.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def chip(text: str, color: str) -> str:
    bg = color + "1a"           # ~10% alpha
    return (f'<span class="chip soft" style="--c-bg:{bg};--c-fg:{color};'
            f'--c-bd:{color}40;">{text}</span>')


def repo_chip(repo_key: str) -> str:
    return chip(REPO_DISPLAY.get(repo_key, repo_key), REPO_COLORS.get(repo_key, "#64748b"))


def cards(items):
    html = '<div class="ds-cards">'
    for label, value, cls in items:
        html += (f'<div class="ds-card"><div class="label">{label}</div>'
                 f'<div class="value {cls}">{value}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data plumbing
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_store(folder: str) -> RepositoryStore:
    return RepositoryStore(folder)


@st.cache_data(show_spinner=False)
def process(file_bytes: bytes, forced_format, enabled_repos, threshold: int):
    """Ingest -> unique donors -> match. Cached on inputs so cycle/filter
    interactions never re-run the heavy work.

    Also builds the separate transfers/attribution structures from the memo
    rows so donors hidden inside transfers can be surfaced and flagged without
    ever entering the committee totals.
    """
    result = load_fec_file(io.BytesIO(file_bytes), forced_format)
    store = get_store(REPO_FOLDER)
    store.fuzzy_threshold = threshold

    master, cycles = aggregate.build_unique_donors(result.transactions)
    if not master.empty:
        master = aggregate.run_matching(master, store, list(enabled_repos))

    attributed = aggregate.build_attributed_donors(
        result.memo_transactions, store, list(enabled_repos))
    transfers = aggregate.summarize_transfers(result.memo_transactions, attributed)
    return result, master, cycles, attributed, transfers


# --------------------------------------------------------------------------- #
# Sidebar - configuration phase
# --------------------------------------------------------------------------- #
store = get_store(REPO_FOLDER)

with st.sidebar:
    st.markdown('<div class="ds-brand"><span class="mark">Donor'
                '<span>Scope</span></span></div>', unsafe_allow_html=True)
    st.caption("Campaign-finance cross-reference and flagging")

    st.markdown("##### :material/upload_file: Data input")
    uploaded = st.file_uploader(
        "FEC file (CSV)", type=["csv"],
        help="A receipts export from fec.gov, or a quarterly filing (FECFILE) CSV.",
    )
    fmt_choice = st.radio(
        "File format",
        ["Auto-detect", "Receipts export", "Quarterly filing"],
        help="Auto-detect inspects the header; override it if detection is wrong.",
    )
    forced = {"Auto-detect": None, "Receipts export": "receipts",
              "Quarterly filing": "quarterly"}[fmt_choice]

    st.markdown("##### :material/tune: Analysis settings")
    high_floor = st.number_input(
        "High Donor floor ($, cumulative)", min_value=0, value=5000, step=500,
        help="Any unique donor whose total across the selected cycles meets or "
             "exceeds this amount is flagged as a High Donor.",
    )
    threshold = st.slider(
        "Fuzzy match threshold", min_value=70, max_value=100, value=88,
        help="Similarity cutoff for name-based fuzzy matching (Epstein list) and "
             "committee-name matching (industry list). Higher is stricter.",
    )

    st.markdown("##### :material/checklist: Repository checklist")
    st.caption("Cross-reference against the lists you select.")
    enabled_repos = []
    for entry in REPO_REGISTRY:
        key = entry["key"]
        avail = store.available.get(key, False)
        count = store.counts.get(key, 0)
        label = f"{entry['display']}  ({count:,})" if avail else f"{entry['display']}  (missing)"
        checked = st.checkbox(label, value=avail, disabled=not avail, key=f"repo_{key}")
        if checked and avail:
            enabled_repos.append(key)

    run = st.button("Run analysis", type="primary", use_container_width=True,
                    disabled=uploaded is None)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="ds-brand"><span class="mark">Donor'
            '<span>Scope</span></span></div>', unsafe_allow_html=True)
st.markdown('<div class="ds-sub">Aggregate FEC contributions into unique donors, '
            'cross-reference them against local masterlists, and surface who gave, '
            'how much, and why it matters.</div>', unsafe_allow_html=True)

if uploaded is None:
    st.info("Upload an FEC receipts export or a quarterly filing in the sidebar, "
            "choose your repositories, and select Run analysis to begin.",
            icon=":material/upload:")
    with st.expander("What the two accepted formats look like"):
        st.markdown(
            "- **Receipts export** - the wide, named-column CSV downloaded from "
            "fec.gov (one row per receipt, columns like `contributor_last_name`, "
            "`contribution_receipt_amount`).\n"
            "- **Quarterly filing** - the raw electronic filing (FECFILE) exported "
            "to CSV, with schedule rows (`HDR`, `F3`, `SA11AI`, ...). Earmark and "
            "conduit memo rows are excluded automatically so contributions are not "
            "double-counted."
        )
    st.stop()

if run:
    st.session_state["analyzed"] = True
if not st.session_state.get("analyzed"):
    st.info("Select Run analysis in the sidebar to process the uploaded file.",
            icon=":material/play_circle:")
    st.stop()

# --------------------------------------------------------------------------- #
# Process (cached)
# --------------------------------------------------------------------------- #
with st.spinner("Processing filing and cross-referencing repositories..."):
    result, master, cycles, attributed, transfers = process(
        uploaded.getvalue(), forced, tuple(enabled_repos), int(threshold)
    )

if master.empty:
    st.warning("No individual or committee contributions were found in this file. "
               "If this is a quarterly filing, confirm it contains Schedule A rows.")
    st.stop()

fmt_label = "Receipts export" if result.fmt == "receipts" else "Quarterly filing"
st.success(f"{fmt_label} processed - {result.rows_kept:,} contribution rows across "
           f"{len(master):,} unique donors.", icon=":material/check_circle:")
for note in result.notes:
    st.caption(note)

# --------------------------------------------------------------------------- #
# Election-cycle selector (Feature 1 driver)
# --------------------------------------------------------------------------- #
st.markdown('<div class="eyebrow">Election cycles</div>', unsafe_allow_html=True)
st.caption("Totals throughout the app reflect only the cycles selected here.")
cycle_cols = st.columns(min(len(cycles), 6) or 1)
selected_cycles = []
for i, cyc in enumerate(cycles):
    with cycle_cols[i % len(cycle_cols)]:
        if st.checkbox(cyc, value=True, key=f"cycle_{cyc}"):
            selected_cycles.append(cyc)
if not selected_cycles:
    st.warning("Select at least one election cycle to display totals.")
    st.stop()

# Vectorized recompute for the current cycle selection.
sel_total = aggregate.selected_total(master, selected_cycles)
sel_count = aggregate.selected_count(master, selected_cycles)
high_mask = aggregate.high_donor_mask(sel_total, high_floor)
overview = aggregate.dataset_overview(master, sel_total, sel_count, high_mask)

work = master.copy()
work["sel_total"] = sel_total.values
work["sel_count"] = sel_count.values
work["is_high_donor"] = high_mask.values

# --------------------------------------------------------------------------- #
# Overview cards
# --------------------------------------------------------------------------- #
cards([
    ("Unique donors", f"{overview['unique_donors']:,}", "accent"),
    ("Individuals", f"{overview['individuals']:,}", ""),
    ("Committees / PACs", f"{overview['committees']:,}", ""),
    ("Total (selected cycles)", money(overview["total_amount"]), "accent"),
    ("Flagged donors", f"{overview['flagged_donors']:,}", "alert"),
    ("High Donors", f"{overview['high_donors']:,}", "gold"),
])

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
transfer_badge = ""
if attributed is not None and not attributed.empty:
    n_underlying = len(attributed)
    transfer_badge = f" ({n_underlying})"

tab_master, tab_flags, tab_transfers, tab_filters, tab_summary = st.tabs([
    ":material/table_rows: Unique donor master",
    ":material/flag: Flagged donors",
    f":material/account_tree: Transfers & attribution{transfer_badge}",
    ":material/filter_alt: Industry & PAC filters",
    ":material/insights: Summary statistics",
])


# ------------------------------ helpers for tables ------------------------- #
def status_text(row) -> str:
    parts = []
    if row["is_high_donor"]:
        parts.append("High Donor")
    if row["flagged"]:
        parts.append(row["flagged_via"])
    return " | ".join(parts)


TOTAL_COL = st.column_config.NumberColumn("Total (selected)", format="$%.2f")
COUNT_COL = st.column_config.NumberColumn("Contributions", format="%d")


# ============================ TAB 1: MASTER ================================= #
with tab_master:
    st.markdown('<div class="eyebrow">Feature 1</div>', unsafe_allow_html=True)
    st.markdown("#### One record per unique donor")
    st.caption("Individual contributions are grouped by donor across the selected "
               "cycles. Toggle cycles above to recompute every total instantly.")

    c1, c2, c3, c4 = st.columns([2.2, 1, 1, 1.2])
    search = c1.text_input("Search name / committee", placeholder="e.g. Smith, ActBlue",
                           label_visibility="collapsed")
    dtype = c2.selectbox("Type", ["All", "Individuals", "Committees"],
                         label_visibility="collapsed")
    only_flagged = c3.toggle("Flagged only")
    only_high = c4.toggle("High Donors only")

    view = work
    if search:
        s = search.strip().lower()
        view = view[view["display_name"].str.lower().str.contains(s, na=False)
                    | view["committee_id"].str.lower().str.contains(s, na=False)]
    if dtype == "Individuals":
        view = view[view["donor_type"] == "individual"]
    elif dtype == "Committees":
        view = view[view["donor_type"] == "committee"]
    if only_flagged:
        view = view[view["flagged"]]
    if only_high:
        view = view[view["is_high_donor"]]

    view = view.sort_values("sel_total", ascending=False)
    display = pd.DataFrame({
        "Donor": view["display_name"],
        "Type": view["donor_type"].map({"individual": "Individual", "committee": "Committee"}),
        "State": view["state"],
        "Total (selected)": view["sel_total"],
        "Contributions": view["sel_count"],
        "High Donor": view["is_high_donor"],
        "Flagged via": view["flagged_via"],
    })
    st.caption(f"Showing {len(display):,} of {len(work):,} unique donors.")
    st.dataframe(
        display, use_container_width=True, hide_index=True, height=460,
        column_config={
            "Total (selected)": TOTAL_COL,
            "Contributions": COUNT_COL,
            "High Donor": st.column_config.CheckboxColumn("High Donor"),
        },
    )

    csv_bytes = view.drop(columns=["flag_details", "epstein_meta", "flag_repos",
                                   "lobbyist_sectors"], errors="ignore").to_csv(index=False)
    st.download_button("Download master (CSV)", csv_bytes,
                       file_name="donor_master.csv", mime="text/csv",
                       icon=":material/download:")


# ============================ TAB 2: FLAGGED =============================== #
with tab_flags:
    st.markdown('<div class="eyebrow">Feature 2</div>', unsafe_allow_html=True)
    st.markdown("#### Flagged donors, grouped by the list that triggered the flag")

    # legend
    legend = '<div class="legend">'
    for entry in REPO_REGISTRY:
        if entry["key"] in enabled_repos:
            legend += repo_chip(entry["key"])
    legend += chip("High Donor", HIGH_DONOR_COLOR) + "</div>"
    st.markdown(legend, unsafe_allow_html=True)

    # Cross-link: flagged people who appear only inside transfers, not as direct
    # receipts, would otherwise be invisible on this tab.
    if attributed is not None and not attributed.empty:
        hidden_flagged = int(attributed["flagged"].sum())
        if hidden_flagged:
            st.warning(
                f"{hidden_flagged} flagged donor(s) appear inside transfers rather "
                "than as direct receipts to this committee. See the "
                "Transfers & attribution tab to review who is behind each transfer.",
                icon=":material/account_tree:")

    flagged = work[work["flagged"] | work["is_high_donor"]]
    if flagged.empty:
        st.info("No donors matched the selected repositories or the High Donor floor "
                "for the chosen cycles.", icon=":material/info:")
    else:
        # High Donor section (dynamic, repository-independent)
        highs = work[work["is_high_donor"]].sort_values("sel_total", ascending=False)
        st.markdown(
            f'<div class="flag-head"><span class="flag-dot" style="background:{HIGH_DONOR_COLOR}">'
            f'</span><span class="flag-title">High Donors</span>'
            f'<span class="flag-count">- {len(highs):,} at or above {money(high_floor)}'
            f' cumulative</span></div>', unsafe_allow_html=True)
        if highs.empty:
            st.caption("None for the current floor and cycle selection.")
        else:
            hd = pd.DataFrame({
                "Donor": highs["display_name"], "State": highs["state"],
                "Total (selected)": highs["sel_total"],
                "Also flagged via": highs["flagged_via"].replace("", "-"),
            })
            st.dataframe(hd, use_container_width=True, hide_index=True,
                         column_config={"Total (selected)": TOTAL_COL})

        st.divider()

        # One section per repository, in registry order
        for entry in REPO_REGISTRY:
            key = entry["key"]
            if key not in enabled_repos:
                continue
            subset = work[work["flag_repos"].map(lambda r, k=key: k in r)]
            if subset.empty:
                continue
            color = REPO_COLORS.get(key, "#64748b")
            st.markdown(
                f'<div class="flag-head"><span class="flag-dot" style="background:{color}">'
                f'</span><span class="flag-title">{entry["display"]}</span>'
                f'<span class="flag-count">- {len(subset):,} donor(s)</span></div>',
                unsafe_allow_html=True)

            def detail_for(details, k=key):
                for d in details:
                    if d["repo"] == k:
                        return d["detail"]
                return ""

            def conf_for(details, k=key):
                for d in details:
                    if d["repo"] == k:
                        return d["confidence"]
                return ""

            sub = subset.sort_values("sel_total", ascending=False)
            tbl = pd.DataFrame({
                "Donor": sub["display_name"],
                "State": sub["state"],
                "Match detail": sub["flag_details"].map(detail_for),
                "Confidence": sub["flag_details"].map(conf_for),
                "Total (selected)": sub["sel_total"],
                "Other flags": sub["flagged_via"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True,
                         column_config={"Total (selected)": TOTAL_COL})

        flag_csv = flagged.assign(
            match_summary=flagged["flag_details"].map(
                lambda ds: " || ".join(f"{d['display']}: {d['detail']}" for d in ds))
        )[["display_name", "donor_type", "state", "sel_total", "is_high_donor",
           "flagged_via", "match_summary"]].to_csv(index=False)
        st.download_button("Download flagged donors (CSV)", flag_csv,
                           file_name="flagged_donors.csv", mime="text/csv",
                           icon=":material/download:")


# ===================== TAB 3: TRANSFERS & ATTRIBUTION ==================== #
with tab_transfers:
    st.markdown('<div class="eyebrow">Transfer transparency</div>', unsafe_allow_html=True)
    st.markdown("#### Who is behind each transfer")
    st.caption("Filings disclose the individual donors behind committee transfers, "
               "the conduits behind earmarked gifts, and reattribution/redesignation "
               "adjustments as memo entries. Their dollars are already represented by "
               "the counted line they sit under, so they are kept out of every total "
               "here to avoid double-counting - but they are listed and "
               "cross-referenced so a committee cannot use a transfer to hide who its "
               "money came from. This view reflects the whole uploaded filing, "
               "independent of the cycle selection above.")

    if attributed is None or attributed.empty:
        st.info("This filing contains no memo or attribution entries - every "
                "contribution is a direct, counted receipt.", icon=":material/info:")
    else:
        n_hidden_flagged = int(attributed["flagged"].sum())
        cards([
            ("Transfers / parent lines", f"{len(transfers):,}", "accent"),
            ("Underlying donors", f"{len(attributed):,}", ""),
            ("Attributed (not in totals)", money(attributed["attributed_amount"].sum()), ""),
            ("Flagged underlying donors", f"{n_hidden_flagged:,}",
             "alert" if n_hidden_flagged else ""),
        ])

        # --- transfers summary ---
        st.markdown("###### Transfers and memo lines in this filing")
        st.caption("Attributed total can exceed the counted parent amount: a "
                   "joint-fundraising committee often itemizes every donor behind it "
                   "even when only part of that money was transferred here. The parent "
                   "amount is zero for standalone memo adjustments.")
        st.dataframe(
            transfers, use_container_width=True, hide_index=True,
            column_config={
                "Counted parent amount":
                    st.column_config.NumberColumn("Counted parent amount", format="$%.2f"),
                "Attributed total":
                    st.column_config.NumberColumn("Attributed total", format="$%.2f"),
                "Underlying donors": st.column_config.NumberColumn("Underlying donors", format="%d"),
                "Flagged underlying": st.column_config.NumberColumn("Flagged underlying", format="%d"),
            },
        )

        st.divider()

        # --- underlying donor detail ---
        st.markdown("###### Underlying donors")
        fc1, fc2 = st.columns([3, 1])
        tsearch = fc1.text_input("Search underlying donor / employer",
                                 placeholder="e.g. DeJoy, Kalikow, Citadel",
                                 label_visibility="collapsed")
        only_flagged_attr = fc2.toggle("Flagged only", key="attr_flagged_only")

        av = attributed.copy()
        if tsearch:
            s = tsearch.strip().lower()
            av = av[av["display_name"].str.lower().str.contains(s, na=False)
                    | av["employer"].str.lower().str.contains(s, na=False)]
        if only_flagged_attr:
            av = av[av["flagged"]]

        av = av.sort_values(["flagged", "attributed_amount"], ascending=[False, False])
        detail = pd.DataFrame({
            "Donor": av["display_name"],
            "State": av["state"],
            "Employer": av["employer"],
            "Attributed amount": av["attributed_amount"],
            "Attributed to / memo": av["transfer_via"],
            "Flagged via": av["flagged_via"].replace("", "-"),
        })
        st.caption(f"Showing {len(detail):,} of {len(attributed):,} underlying donors.")
        st.dataframe(
            detail, use_container_width=True, hide_index=True, height=420,
            column_config={
                "Attributed amount":
                    st.column_config.NumberColumn("Attributed amount", format="$%.2f"),
            },
        )

        attr_csv = av.assign(
            match_summary=av["flag_details"].map(
                lambda ds: " || ".join(f"{d['display']}: {d['detail']}" for d in ds))
        )[["display_name", "donor_type", "state", "employer", "attributed_amount",
           "transfer_via", "flagged_via", "match_summary"]].to_csv(index=False)
        st.download_button("Download underlying donors (CSV)", attr_csv,
                           file_name="transfer_attributed_donors.csv", mime="text/csv",
                           icon=":material/download:")


# ======================= TAB 4: INDUSTRY & PAC FILTERS ==================== #
with tab_filters:
    st.markdown('<div class="eyebrow">Feature 3</div>', unsafe_allow_html=True)
    st.markdown("#### Filter the donor master by special-interest category")

    sub_lobby, sub_corp, sub_lead = st.tabs([
        "Lobbyist sectors", "Corporate PAC sectors", "Leadership PACs"])

    # ---- 3.1 Lobbyist sectors (federal lobbyist masterlist, LDA descriptions) ----
    with sub_lobby:
        st.caption("Donors matched to the federal lobbyist masterlist, filterable by "
                   "the LDA disclosure area (natural-language description, not the "
                   "three-letter code).")
        lob = work[work["lobbyist_sectors"].map(len) > 0].copy()
        if lob.empty:
            st.info("No donors matched the federal lobbyist masterlist for these cycles.",
                    icon=":material/info:")
        else:
            all_sectors = sorted({s for lst in lob["lobbyist_sectors"] for s in lst})
            picked = st.multiselect("LDA disclosure sectors", all_sectors,
                                    placeholder="All sectors")
            if picked:
                lob = lob[lob["lobbyist_sectors"].map(
                    lambda lst: any(s in lst for s in picked))]
            lob = lob.sort_values("sel_total", ascending=False)
            tbl = pd.DataFrame({
                "Donor": lob["display_name"], "State": lob["state"],
                "Lobbyist sectors": lob["lobbyist_sectors"].map(lambda l: ", ".join(l)),
                "Total (selected)": lob["sel_total"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True,
                         column_config={"Total (selected)": TOTAL_COL})

    # ---- 3.2 Corporate PAC sectors (industry masterlist, org_type == C) ----
    with sub_corp:
        st.caption("Corporate PACs (industry masterlist org type C), broken down by "
                   "macro category.")
        corp = work[work["is_corporate_pac"]].copy()
        if corp.empty:
            st.info("No corporate PACs matched for these cycles.", icon=":material/info:")
        else:
            macros = sorted(c for c in corp["industry_category"].unique() if c)
            picked = st.multiselect("Macro categories", macros, placeholder="All categories")
            if picked:
                corp = corp[corp["industry_category"].isin(picked)]

            by_cat = (corp.groupby("industry_category")["sel_total"].sum()
                      .sort_values(ascending=False))
            if not by_cat.empty:
                chips_html = '<div class="legend">'
                for cat, amt in by_cat.items():
                    chips_html += chip(f"{cat}: {money(amt)}", category_color(cat))
                st.markdown(chips_html + "</div>", unsafe_allow_html=True)
                st.bar_chart(by_cat, color=None, height=240)

            corp = corp.sort_values("sel_total", ascending=False)
            tbl = pd.DataFrame({
                "PAC": corp["display_name"],
                "Committee ID": corp["committee_id"],
                "Macro category": corp["industry_category"],
                "Total (selected)": corp["sel_total"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True,
                         column_config={"Total (selected)": TOTAL_COL})

    # ---- 3.3 Leadership PACs (owner pulled from masterlist relationship) ----
    with sub_lead:
        st.caption("PACs contained in the leadership PAC masterlist, with the "
                   "owning candidate pulled from the local relationship data.")
        lead = work[work["is_leadership_pac"]].copy()
        if lead.empty:
            st.info("No leadership PACs matched for these cycles.", icon=":material/info:")
        else:
            lead["owner_display"] = lead["lpac_owner"].map(nz.format_person_display)
            lead = lead.sort_values("sel_total", ascending=False)
            tbl = pd.DataFrame({
                "Leadership PAC": lead["display_name"],
                "Committee ID": lead["committee_id"],
                "Owner / candidate": lead["owner_display"],
                "Total (selected)": lead["sel_total"],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True,
                         column_config={"Total (selected)": TOTAL_COL})


# ============================ TAB 5: SUMMARY ============================== #
with tab_summary:
    st.markdown('<div class="eyebrow">Feature 4</div>', unsafe_allow_html=True)
    st.markdown("#### Interactive summary")
    st.caption("Choose a row dimension and the metrics to compute. Everything "
               "reflects the currently selected election cycles.")

    c1, c2 = st.columns([1, 2])
    row_dim = c1.selectbox("Group by", list(aggregate.ROW_DIMENSIONS.keys()))
    metrics = c2.multiselect(
        "Metrics", list(aggregate.METRIC_FUNCS.keys()),
        default=["Total funds", "Unique donors", "Average donation"],
    )

    if not metrics:
        st.info("Select at least one metric to build the summary.", icon=":material/info:")
    else:
        table = aggregate.build_summary_table(work, sel_total, sel_count, row_dim, metrics)
        if table.empty:
            st.info("No rows to summarize for this dimension and cycle selection.",
                    icon=":material/info:")
        else:
            col_cfg = {}
            for m in metrics:
                if m in ("Total funds", "Average donation", "Median donor total",
                         "Largest donor total"):
                    col_cfg[m] = st.column_config.NumberColumn(m, format="$%.2f")
                else:
                    col_cfg[m] = st.column_config.NumberColumn(m, format="%d")
            st.dataframe(table, use_container_width=True, hide_index=True,
                         column_config=col_cfg)

            if "Total funds" in metrics:
                chart_df = table.set_index(row_dim)["Total funds"]
                st.bar_chart(chart_df, height=280)

            st.download_button("Download summary (CSV)", table.to_csv(index=False),
                               file_name="summary.csv", mime="text/csv",
                               icon=":material/download:")

st.divider()
st.caption("Donor Scope visualizes and cross-references disclosure data. Matches are "
           "investigative leads, not conclusions - validate any total or affiliation "
           "against the underlying FEC records before relying on it.")
