"""
Repository store.

Loads the nine cross-reference masterlists from ``repositories/`` and turns each
into fast lookup structures. Every list has its own real-world schema (see the
sample files), so each loader is written against that specific shape rather than
a one-size-fits-all importer.

The store exposes three kinds of lookups used by the matching engine:
    * individual name  -> bad donor / young republican / epstein / lobbyist hits
    * employer string  -> bad employer hit
    * committee id/name -> bad group / industry / leadership PAC hits
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import normalize as nz

try:                                     # fast fuzzy matching when available
    from rapidfuzz import fuzz, process as rf_process
    _HAVE_RAPIDFUZZ = True
except ImportError:                      # graceful fallback
    from difflib import SequenceMatcher
    _HAVE_RAPIDFUZZ = False


# Canonical registry. Order controls display order in the UI.
REPO_REGISTRY: List[dict] = [
    {"key": "bad_donor", "file": "bad_donor_masterlist.csv",
     "display": "bad donor masterlist", "scope": "individual"},
    {"key": "bad_employer", "file": "bad_employer_masterlist.csv",
     "display": "bad employer masterlist", "scope": "employer"},
    {"key": "bad_group", "file": "bad_group_masterlist.csv",
     "display": "bad group masterlist", "scope": "committee"},
    {"key": "epstein", "file": "epstein_persons_list.csv",
     "display": "epstein persons list", "scope": "individual"},
    {"key": "iowa_lobbyist", "file": "iowa_lobbyist_masterlist.csv",
     "display": "iowa lobbyist masterlist", "scope": "individual"},
    {"key": "industry_pac", "file": "industry_pac_masterlist.csv",
     "display": "industry pac masterlist", "scope": "committee"},
    {"key": "federal_lobbyist", "file": "federal_lobbyist_masterlist.csv",
     "display": "federal lobbyist masterlist", "scope": "individual"},
    {"key": "leadership_pac", "file": "leadership_pac_masterlist.csv",
     "display": "leadership pac masterlist", "scope": "committee"},
    {"key": "young_republicans", "file": "politico_young_republicans_list.csv",
     "display": "politico young republicans list", "scope": "individual"},
]
REPO_DISPLAY = {r["key"]: r["display"] for r in REPO_REGISTRY}


def _pick(df: pd.DataFrame, *names: str) -> pd.Series:
    """Return the first matching column (case/format insensitive), else empty."""
    lowered = {c.lower().replace(".", "").replace("_", "").replace(" ", ""): c
               for c in df.columns}
    for name in names:
        target = name.lower().replace(".", "").replace("_", "").replace(" ", "")
        if target in lowered:
            return df[lowered[target]].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _score(a: str, b: str) -> float:
    if _HAVE_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    return SequenceMatcher(None, a, b).ratio() * 100


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


@dataclass
class Hit:
    """A single repository match against one donor/entity."""
    repo_key: str
    detail: str
    confidence: str = "HIGH"
    extra: dict = field(default_factory=dict)


class RepositoryStore:
    def __init__(self, folder: str, fuzzy_threshold: int = 88):
        self.folder = folder
        self.fuzzy_threshold = fuzzy_threshold
        self.available: Dict[str, bool] = {}
        self.counts: Dict[str, int] = {}

        # individual-name indices
        self._bad_donor: Dict[str, List[str]] = {}
        self._yr_full: Dict[str, Tuple] = {}
        self._yr_name: Dict[str, Tuple] = {}
        self._iowa: Dict[str, dict] = {}
        self._fed_lobby: Dict[str, dict] = {}
        self._epstein: List[Tuple[str, dict]] = []

        # employer index
        self._bad_employer: Dict[str, str] = {}

        # committee indices
        self._bad_group_id: Dict[str, str] = {}
        self._bad_group_name: Dict[str, str] = {}
        self._industry_id: Dict[str, dict] = {}
        self._industry_name: List[Tuple[str, dict]] = []
        self._lpac_id: Dict[str, dict] = {}
        self._lpac_name: Dict[str, dict] = {}

        self._load_all()

    # ------------------------------------------------------------------ #
    def _path(self, filename: str) -> str:
        return os.path.join(self.folder, filename)

    def _read(self, filename: str) -> Optional[pd.DataFrame]:
        path = self._path(filename)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            return None

    def _load_all(self) -> None:
        loaders: Dict[str, Callable[[pd.DataFrame], int]] = {
            "bad_donor": self._load_bad_donor,
            "bad_employer": self._load_bad_employer,
            "bad_group": self._load_bad_group,
            "epstein": self._load_epstein,
            "iowa_lobbyist": self._load_iowa,
            "industry_pac": self._load_industry,
            "federal_lobbyist": self._load_federal_lobbyist,
            "leadership_pac": self._load_lpac,
            "young_republicans": self._load_young_republicans,
        }
        for entry in REPO_REGISTRY:
            key, fname = entry["key"], entry["file"]
            df = self._read(fname)
            if df is None or df.empty:
                self.available[key] = False
                self.counts[key] = 0
                continue
            self.counts[key] = loaders[key](df)
            self.available[key] = self.counts[key] > 0

    # ------------------------------------------------------------------ #
    # Individual-scope loaders
    # ------------------------------------------------------------------ #
    def _load_bad_donor(self, df: pd.DataFrame) -> int:
        names = _pick(df, "Name")
        blurbs = _pick(df, "Blurb", "Flag", "Affiliation")
        for name, blurb in zip(names, blurbs):
            first, last = nz.parse_person_name(name)
            key = f"{first} {last}".strip()
            if not key:
                continue
            self._bad_donor.setdefault(key, [])
            if blurb.strip():
                self._bad_donor[key].append(blurb.strip())
        return len(self._bad_donor)

    def _load_young_republicans(self, df: pd.DataFrame) -> int:
        names = _pick(df, "NAME", "Name")
        states = _pick(df, "State")
        quotes = _pick(df, "Politico.Quote", "Quote")
        types = _pick(df, "TYPE", "Type")
        n = 0
        for name, state, quote, yr_type in zip(names, states, quotes, types):
            first, last = nz.parse_person_name(name)
            name_key = f"{first} {last}".strip()
            if not name_key:
                continue
            payload = (name.strip(), state.strip().upper(), quote.strip(), yr_type.strip())
            full_key = f"{name_key} {state.strip().upper()}".strip()
            self._yr_full.setdefault(full_key, payload)
            self._yr_name.setdefault(name_key, payload)
            n += 1
        return n

    def _load_iowa(self, df: pd.DataFrame) -> int:
        names = _pick(df, "Name")
        clients = _pick(df, "clients")
        years = _pick(df, "year")
        addrs = _pick(df, "address")
        for name, client, year, addr in zip(names, clients, years, addrs):
            first, last = nz.parse_person_name(name)
            key = f"{first} {last}".strip()
            if not key:
                continue
            rec = self._iowa.setdefault(key, {"clients": set(), "years": set(), "address": ""})
            for c in str(client).split(","):
                if c.strip():
                    rec["clients"].add(c.strip())
            if str(year).strip():
                rec["years"].add(str(year).strip())
            if addr.strip() and not rec["address"]:
                rec["address"] = addr.strip()
        return len(self._iowa)

    def _load_federal_lobbyist(self, df: pd.DataFrame) -> int:
        names = _pick(df, "lobbyist_full_name")
        code_desc = _pick(df, "code_desc")
        issue_name = _pick(df, "issue_name")
        registrant = _pick(df, "registrant_name")
        client = _pick(df, "client_name")
        years = _pick(df, "filing_year")
        for name, desc, issue, reg, cli, year in zip(
            names, code_desc, issue_name, registrant, client, years
        ):
            first, last = nz.parse_person_name(name)
            key = f"{first} {last}".strip()
            if not key:
                continue
            rec = self._fed_lobby.setdefault(
                key, {"sectors": set(), "registrants": set(),
                      "clients": set(), "years": set()}
            )
            sector = desc.strip() or issue.strip()      # natural-language, not the 3-letter code
            if sector:
                rec["sectors"].add(sector)
            if reg.strip():
                rec["registrants"].add(reg.strip())
            if cli.strip():
                rec["clients"].add(cli.strip())
            if str(year).strip():
                rec["years"].add(str(year).strip())
        return len(self._fed_lobby)

    def _load_epstein(self, df: pd.DataFrame) -> int:
        names = _pick(df, "Name")
        aliases = _pick(df, "Aliases")
        category = _pick(df, "Category")
        bio = _pick(df, "Bio")
        flights = _pick(df, "Flights")
        docs = _pick(df, "Documents")
        conns = _pick(df, "Connections")
        black = _pick(df, "In.Black.Book")
        nat = _pick(df, "Nationality")
        for i in range(len(df)):
            meta = {
                "name": names.iloc[i].strip(), "category": category.iloc[i].strip(),
                "bio": bio.iloc[i].strip(), "flights": flights.iloc[i].strip(),
                "documents": docs.iloc[i].strip(), "connections": conns.iloc[i].strip(),
                "in_black_book": black.iloc[i].strip(), "nationality": nat.iloc[i].strip(),
            }
            candidates = [nz.clean_person_name(names.iloc[i])]
            for alias in str(aliases.iloc[i]).split(";"):
                cleaned = nz.clean_person_name(alias)
                if cleaned:
                    candidates.append(cleaned)
            for cand in candidates:
                if cand:
                    self._epstein.append((cand, meta))
        return len(self._epstein)

    # ------------------------------------------------------------------ #
    # Employer-scope loader
    # ------------------------------------------------------------------ #
    def _load_bad_employer(self, df: pd.DataFrame) -> int:
        names = _pick(df, "NAME", "Name")
        flags = _pick(df, "Flag")
        for name, flag in zip(names, flags):
            key = nz.clean_text(name)
            if key:
                self._bad_employer.setdefault(key, flag.strip() or "Flagged employer")
        return len(self._bad_employer)

    # ------------------------------------------------------------------ #
    # Committee-scope loaders
    # ------------------------------------------------------------------ #
    def _load_bad_group(self, df: pd.DataFrame) -> int:
        ids = _pick(df, "Committee.ID", "Committee_Id", "committee_id")
        names = _pick(df, "Committee.Name", "Committee_Name", "committee_name")
        flags = _pick(df, "Flag")
        for cid, name, flag in zip(ids, names, flags):
            label = flag.strip() or "Flagged group"
            if cid.strip():
                self._bad_group_id.setdefault(cid.strip().upper(), label)
            nkey = nz.normalize_committee_name(name)
            if nkey:
                self._bad_group_name.setdefault(nkey, label)
        return max(len(self._bad_group_id), len(self._bad_group_name))

    def _load_industry(self, df: pd.DataFrame) -> int:
        ids = _pick(df, "Committee.ID", "committee_id")
        names = _pick(df, "Committee.Name", "committee_name")
        org = _pick(df, "Org.type", "org_type")
        smaller = _pick(df, "Smaller.Categories", "smaller_categories")
        larger = _pick(df, "Larger.Categories", "larger_categories")
        connected = _pick(df, "Connected.Organization", "connected_organization")
        for i in range(len(df)):
            data = {
                "name": names.iloc[i].strip(),
                "org_type": org.iloc[i].strip().upper(),
                "smaller": smaller.iloc[i].strip(),
                "larger": (larger.iloc[i].strip() or "Uncategorized").strip(),
                "connected": connected.iloc[i].strip(),
            }
            cid = ids.iloc[i].strip().upper()
            if cid:
                self._industry_id.setdefault(cid, data)
            nkey = nz.normalize_committee_name(names.iloc[i])
            if nkey:
                self._industry_name.append((nkey, data))
        return len(self._industry_id) + len(self._industry_name)

    def _load_lpac(self, df: pd.DataFrame) -> int:
        ids = _pick(df, "Committee_Id", "Committee.ID", "committee_id")
        names = _pick(df, "Committee_Name", "Committee.Name", "committee_name")
        sponsors = _pick(df, "Sponsor_Name", "sponsor_name")
        for cid, name, sponsor in zip(ids, names, sponsors):
            data = {"name": name.strip(), "owner": sponsor.strip() or "Unknown sponsor"}
            if cid.strip():
                self._lpac_id.setdefault(cid.strip().upper(), data)
            nkey = nz.normalize_committee_name(name)
            if nkey:
                self._lpac_name.setdefault(nkey, data)
        return max(len(self._lpac_id), len(self._lpac_name))

    # ------------------------------------------------------------------ #
    # Lookups (used by matching engine). `enabled` gates each repository.
    # ------------------------------------------------------------------ #
    def match_individual(self, first: str, last: str, state: str,
                         enabled: set) -> List[Hit]:
        hits: List[Hit] = []
        name_key = f"{first} {last}".strip()
        if not name_key:
            return hits

        if "bad_donor" in enabled and name_key in self._bad_donor:
            blurbs = self._bad_donor[name_key]
            detail = "; ".join(sorted(set(blurbs))) if blurbs else "Listed bad donor"
            hits.append(Hit("bad_donor", _truncate(detail), "HIGH"))

        if "young_republicans" in enabled:
            full_key = f"{name_key} {state}".strip()
            rec = self._yr_full.get(full_key)
            conf = "HIGH"
            if rec is None:
                rec = self._yr_name.get(name_key)
                conf = "MEDIUM"
            if rec:
                _, yr_state, quote, yr_type = rec
                detail = f"{yr_type or 'Young Republican'} ({yr_state}) - {_truncate(quote, 120)}"
                hits.append(Hit("young_republicans", detail, conf,
                                {"state": yr_state, "quote": quote}))

        if "iowa_lobbyist" in enabled and name_key in self._iowa:
            rec = self._iowa[name_key]
            clients = ", ".join(sorted(rec["clients"])) if rec["clients"] else "Registered Iowa lobbyist"
            years = "/".join(sorted(rec["years"]))
            detail = f"Iowa lobbyist{f' ({years})' if years else ''}: {_truncate(clients)}"
            hits.append(Hit("iowa_lobbyist", detail, "HIGH", {"clients": clients}))

        if "federal_lobbyist" in enabled and name_key in self._fed_lobby:
            rec = self._fed_lobby[name_key]
            sectors = sorted(rec["sectors"])
            detail = "Federal lobbyist: " + (", ".join(sectors) if sectors else "registered")
            hits.append(Hit("federal_lobbyist", _truncate(detail), "HIGH",
                            {"sectors": sectors,
                             "clients": sorted(rec["clients"])}))

        if "epstein" in enabled and self._epstein:
            ep = self._best_epstein(name_key)
            if ep:
                meta, score = ep
                detail = (f"{meta['name']} ({meta['category'] or 'listed'}) - "
                          f"fuzzy match {score:.0f}%")
                hits.append(Hit("epstein", detail, "HIGH",
                                {"meta": meta, "score": score}))
        return hits

    def _best_epstein(self, name_key: str) -> Optional[Tuple[dict, float]]:
        if _HAVE_RAPIDFUZZ:
            choices = [c for c, _ in self._epstein]
            match = rf_process.extractOne(
                name_key, choices, scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if match:
                _, score, idx = match
                return self._epstein[idx][1], score
            return None
        best = None
        for cand, meta in self._epstein:
            s = _score(name_key, cand)
            if s >= self.fuzzy_threshold and (best is None or s > best[1]):
                best = (meta, s)
        return best

    def match_employer(self, employer: str, enabled: set) -> Optional[Hit]:
        if "bad_employer" not in enabled:
            return None
        key = nz.clean_text(employer)
        if key and key in self._bad_employer:
            return Hit("bad_employer", self._bad_employer[key], "HIGH",
                       {"employer": employer})
        return None

    def match_committee(self, committee_id: str, committee_name: str,
                        enabled: set) -> List[Hit]:
        hits: List[Hit] = []
        cid = (committee_id or "").strip().upper()
        nkey = nz.normalize_committee_name(committee_name)

        # bad group -----------------------------------------------------
        if "bad_group" in enabled:
            flag = self._bad_group_id.get(cid) or self._bad_group_name.get(nkey)
            if flag:
                hits.append(Hit("bad_group", flag, "HIGH"))

        # industry ------------------------------------------------------
        if "industry_pac" in enabled:
            data = self._industry_id.get(cid)
            conf = "HIGH"
            if data is None and nkey:
                data, score = self._best_industry(nkey)
                conf = "MEDIUM" if data else conf
            if data:
                is_corp = data["org_type"] == "C"
                detail = f"{data['larger']}" + (" (Corporate PAC)" if is_corp else "")
                hits.append(Hit("industry_pac", detail, conf, {
                    "larger": data["larger"], "smaller": data["smaller"],
                    "org_type": data["org_type"], "is_corporate": is_corp,
                    "connected": data["connected"],
                }))

        # leadership PAC ------------------------------------------------
        if "leadership_pac" in enabled:
            data = self._lpac_id.get(cid) or self._lpac_name.get(nkey)
            if data:
                hits.append(Hit("leadership_pac", f"Leadership PAC - {data['owner']}",
                                "HIGH", {"owner": data["owner"]}))
        return hits

    def _best_industry(self, nkey: str) -> Tuple[Optional[dict], float]:
        if not self._industry_name:
            return None, 0.0
        if _HAVE_RAPIDFUZZ:
            choices = [c for c, _ in self._industry_name]
            match = rf_process.extractOne(
                nkey, choices, scorer=fuzz.token_sort_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if match:
                _, score, idx = match
                return self._industry_name[idx][1], score
            return None, 0.0
        best, best_s = None, 0.0
        for cand, data in self._industry_name:
            s = _score(nkey, cand)
            if s >= self.fuzzy_threshold and s > best_s:
                best, best_s = data, s
        return best, best_s
