"""
Build all_playoff_series.csv for logistic regression training.

1. Reads series results from data/model/series_history.csv (exported from BBRef)
2. Scrapes playoff advanced stats (VORP) and standings (SRS) per year
3. Joins everything into one analysis-ready dataset

Run locally:
    pip install beautifulsoup4 lxml requests pandas
    python scrape_all_playoffs.py

~92 requests total (~5.5 minutes).
"""

import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/model")
RAW_DIR = DATA_DIR / "raw"
SERIES_CSV = DATA_DIR / "series_history.csv"
OUTPUT_CSV = DATA_DIR / "all_playoff_series.csv"

START_YEAR = 1980
END_YEAR = 2025

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 3.5

TEAM_NAME_TO_ABBR = {
    "Chicago Bulls": "CHI",
    "Los Angeles Lakers": "LAL",
    "Portland Trail Blazers": "POR",
    "Phoenix Suns": "PHO",
    "Seattle SuperSonics": "SEA",
    "Utah Jazz": "UTA",
    "Cleveland Cavaliers": "CLE",
    "San Antonio Spurs": "SAS",
    "Dallas Mavericks": "DAL",
    "Miami Heat": "MIA",
    "Oklahoma City Thunder": "OKC",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Boston Celtics": "BOS",
    "Detroit Pistons": "DET",
    "Indiana Pacers": "IND",
    "Milwaukee Bucks": "MIL",
    "New York Knicks": "NYK",
    "Philadelphia 76ers": "PHI",
    "Toronto Raptors": "TOR",
    "Washington Wizards": "WAS",
    "Orlando Magic": "ORL",
    "Charlotte Hornets": "CHO",
    "Atlanta Hawks": "ATL",
    "Brooklyn Nets": "BRK",
    "New Jersey Nets": "NJN",
    "Denver Nuggets": "DEN",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "Sacramento Kings": "SAC",
    "Los Angeles Clippers": "LAC",
    "Memphis Grizzlies": "MEM",
    "Charlotte Bobcats": "CHA",
    "New Orleans Hornets": "NOH",
    "New Orleans/Oklahoma City Hornets": "NOK",
    "Vancouver Grizzlies": "VAN",
    "Washington Bullets": "WAS",
    "Kansas City Kings": "KCK",
    "San Diego Clippers": "SDC",
    "Buffalo Braves": "BUF",
    "New York Nets": "NYN",
    "Seattle Supersonics": "SEA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    """GET with rate limiting."""
    time.sleep(REQUEST_DELAY)
    print(f"    Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_all_tables(html: str) -> list:
    """Extract all tables including those in HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    tables = list(soup.find_all("table"))

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            comment_soup = BeautifulSoup(comment, "lxml")
            tables.extend(comment_soup.find_all("table"))

    return tables


def _table_to_df(table) -> pd.DataFrame:
    """Convert BS4 table to DataFrame, flattening multi-level headers."""
    try:
        dfs = pd.read_html(StringIO(str(table)))
        if dfs:
            df = dfs[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _strip_seed(team_name: str) -> str:
    """Remove seed from team name: 'Detroit Pistons (1)' -> 'Detroit Pistons'"""
    return re.sub(r"\s*\(\d+\)\s*$", "", str(team_name).strip())


def _team_to_abbr(team_name: str) -> str | None:
    """Convert team name to abbreviation."""
    clean = _strip_seed(team_name)
    return TEAM_NAME_TO_ABBR.get(clean)


# ---------------------------------------------------------------------------
# Load series history
# ---------------------------------------------------------------------------


def load_series_history() -> pd.DataFrame:
    """Load and clean the exported series_history.csv from BBRef."""
    # the CSV has a messy multi-row header; skip it and assign columns manually
    df = pd.read_csv(SERIES_CSV, skiprows=2, header=None)

    # based on the export format:
    # 0=Yr, 1=Lg, 2=Series, 3=Dates, 4=empty, 5=Winner, 6=Winner_W,
    # 7=empty, 8=Loser, 9=Loser_W, 10=empty, 11=Favorite, 12=Underdog
    if len(df.columns) < 10:
        print(f"  WARNING: Expected 10+ columns, got {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].tolist()}")
        return pd.DataFrame()

    series_df = pd.DataFrame(
        {
            "Year": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
            "League": df.iloc[:, 1],
            "Round": df.iloc[:, 2],
            "Winner": df.iloc[:, 5],
            "Winner_W": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
            "Loser": df.iloc[:, 8],
            "Loser_W": pd.to_numeric(df.iloc[:, 9], errors="coerce"),
        }
    )

    # filter: NBA only, valid years, completed series
    series_df = series_df[
        (series_df["League"] == "NBA")
        & (series_df["Year"] >= START_YEAR)
        & (series_df["Year"] <= END_YEAR)
        & (series_df["Year"].notna())
        & (series_df["Winner_W"].notna())
        & (series_df["Loser_W"].notna())
    ].copy()

    series_df["Year"] = series_df["Year"].astype(int)
    series_df["Winner_W"] = series_df["Winner_W"].astype(int)
    series_df["Loser_W"] = series_df["Loser_W"].astype(int)

    # resolve team abbreviations
    series_df["Winner_Abbr"] = series_df["Winner"].apply(_team_to_abbr)
    series_df["Loser_Abbr"] = series_df["Loser"].apply(_team_to_abbr)

    # report unresolved teams
    unresolved = series_df[
        series_df["Winner_Abbr"].isna() | series_df["Loser_Abbr"].isna()
    ]
    if len(unresolved) > 0:
        print(f"\n  WARNING: {len(unresolved)} series with unresolved team names:")
        for _, row in unresolved.iterrows():
            w = row["Winner"] if pd.isna(row["Winner_Abbr"]) else ""
            l = row["Loser"] if pd.isna(row["Loser_Abbr"]) else ""
            print(f"    {row['Year']}: {w} {l}")

    # drop unresolved
    series_df = series_df.dropna(subset=["Winner_Abbr", "Loser_Abbr"])

    print(f"\n  Loaded {len(series_df)} completed series ({START_YEAR}-{END_YEAR})")
    return series_df


# ---------------------------------------------------------------------------
# Scrapers (same as before — advanced stats + SRS only)
# ---------------------------------------------------------------------------


def scrape_playoff_advanced(year: int) -> pd.DataFrame:
    """Scrape full playoff advanced stats for all players in a given year."""
    url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_advanced.html"
    html = _fetch(url)
    tables = _parse_all_tables(html)

    for table in tables:
        df = _table_to_df(table)
        if df.empty:
            continue
        if "BPM" in df.columns and "Player" in df.columns:
            for col in ["BPM", "VORP", "PER", "WS", "WS/48", "MP", "G"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["VORP"])
            df = df[df["Player"] != "Player"]
            df["Year"] = year
            return df

    print(f"    WARNING: No advanced stats table found for {year}")
    return pd.DataFrame()


def scrape_team_srs(year: int) -> dict[str, float]:
    """Scrape SRS for all teams from the standings page."""
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    html = _fetch(url)
    tables = _parse_all_tables(html)

    srs_map = {}

    for table in tables:
        df = _table_to_df(table)
        if df.empty or "SRS" not in df.columns:
            continue

        df["SRS"] = pd.to_numeric(df["SRS"], errors="coerce")
        team_col = df.columns[0]

        for _, row in df.iterrows():
            team_name = (
                str(row[team_col])
                .strip()
                .replace("*", "")
                .replace("(", "")
                .replace(")", "")
                .strip()
            )
            srs_val = row["SRS"]
            if team_name and pd.notna(srs_val) and team_name != team_col:
                abbr = TEAM_NAME_TO_ABBR.get(team_name)
                if abbr:
                    srs_map[abbr] = srs_val

    return srs_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_all_playoff_series():
    """Build the complete all_playoff_series.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Load series history ---
    print("Loading series history...")
    series_df = load_series_history()
    if series_df.empty:
        print("ERROR: No series data loaded. Check series_history.csv.")
        return

    # get unique years we need to scrape
    years = sorted(series_df["Year"].unique())
    print(f"  Years to scrape: {years[0]}-{years[-1]} ({len(years)} seasons)")
    print(f"  Estimated time: ~{len(years) * 2 * REQUEST_DELAY / 60:.1f} minutes")

    # --- 2. Scrape advanced stats + SRS per year ---
    adv_cache: dict[int, pd.DataFrame] = {}
    srs_cache: dict[int, dict] = {}

    for i, year in enumerate(years):
        print(f"\n[{i + 1}/{len(years)}] Scraping {year}...")

        # advanced stats
        try:
            adv_df = scrape_playoff_advanced(year)
            if not adv_df.empty:
                adv_cache[year] = adv_df
                adv_df.to_csv(RAW_DIR / f"{year}_playoff_adv.csv", index=False)
            else:
                print(f"    No advanced stats for {year}")
        except Exception as e:
            print(f"    ERROR (advanced) {year}: {e}")

        # SRS
        try:
            srs_map = scrape_team_srs(year)
            if srs_map:
                srs_cache[year] = srs_map
            else:
                print(f"    No SRS data for {year}")
        except Exception as e:
            print(f"    ERROR (SRS) {year}: {e}")

    # --- 3. Join series with VORP + SRS ---
    print("\nJoining data...")
    all_rows = []
    skipped = 0

    for _, series in series_df.iterrows():
        year = series["Year"]
        w_abbr = series["Winner_Abbr"]
        l_abbr = series["Loser_Abbr"]

        if year not in adv_cache or year not in srs_cache:
            skipped += 1
            continue

        adv_df = adv_cache[year]
        srs_map = srs_cache[year]

        # VORP for winner's roster
        w_team = adv_df[adv_df["Tm"] == w_abbr].sort_values("VORP", ascending=False)
        l_team = adv_df[adv_df["Tm"] == l_abbr].sort_values("VORP", ascending=False)

        if w_team.empty or l_team.empty:
            skipped += 1
            continue

        # top 2 VORP per team
        w_v1 = w_team.iloc[0]["VORP"]
        w_p1 = w_team.iloc[0]["Player"]
        w_v2 = w_team.iloc[1]["VORP"] if len(w_team) > 1 else None
        w_p2 = w_team.iloc[1]["Player"] if len(w_team) > 1 else "N/A"

        l_v1 = l_team.iloc[0]["VORP"]
        l_p1 = l_team.iloc[0]["Player"]
        l_v2 = l_team.iloc[1]["VORP"] if len(l_team) > 1 else None
        l_p2 = l_team.iloc[1]["Player"] if len(l_team) > 1 else "N/A"

        w_carry = round(w_v1 - w_v2, 2) if w_v2 is not None else None
        l_carry = round(l_v1 - l_v2, 2) if l_v2 is not None else None

        w_srs = srs_map.get(w_abbr)
        l_srs = srs_map.get(l_abbr)
        srs_delta = (
            round(w_srs - l_srs, 2) if w_srs is not None and l_srs is not None else None
        )

        # winner row
        all_rows.append(
            {
                "Season": year,
                "Round": series["Round"],
                "Team": w_abbr,
                "Opp": l_abbr,
                "Result": "W",
                "Series": f"{series['Winner_W']}-{series['Loser_W']}",
                "Best_Player": w_p1,
                "Best_VORP": w_v1,
                "Second_Player": w_p2,
                "Second_VORP": w_v2,
                "Carry_Delta": w_carry,
                "Opp_Best_Player": l_p1,
                "Opp_Best_VORP": l_v1,
                "Team_SRS": w_srs,
                "Opp_SRS": l_srs,
                "SRS_Delta": srs_delta,
            }
        )

        # loser row
        all_rows.append(
            {
                "Season": year,
                "Round": series["Round"],
                "Team": l_abbr,
                "Opp": w_abbr,
                "Result": "L",
                "Series": f"{series['Loser_W']}-{series['Winner_W']}",
                "Best_Player": l_p1,
                "Best_VORP": l_v1,
                "Second_Player": l_p2,
                "Second_VORP": l_v2,
                "Carry_Delta": l_carry,
                "Opp_Best_Player": w_p1,
                "Opp_Best_VORP": w_v1,
                "Team_SRS": l_srs,
                "Opp_SRS": w_srs,
                "SRS_Delta": -srs_delta if srs_delta is not None else None,
            }
        )

    # --- 4. Save ---
    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"  Rows: {len(all_rows)} ({len(all_rows) // 2} series)")
    print(f"  Skipped: {skipped} series (missing data)")
    print(f"  Seasons: {final_df['Season'].nunique()}")
    print(f"  Saved to: {OUTPUT_CSV}")
    print(f"{'=' * 60}")

    return final_df


if __name__ == "__main__":
    build_all_playoff_series()
