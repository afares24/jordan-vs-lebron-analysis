import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/finals")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_CSV = DATA_DIR / "finals_context.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

REQUEST_DELAY = 3.5  # seconds between requests

# (year, player_team_abbr, opp_team_abbr, player_name, result)
FINALS = [
    # Jordan
    (1991, "CHI", "LAL", "Michael Jordan", "W"),
    (1992, "CHI", "POR", "Michael Jordan", "W"),
    (1993, "CHI", "PHO", "Michael Jordan", "W"),
    (1996, "CHI", "SEA", "Michael Jordan", "W"),
    (1997, "CHI", "UTA", "Michael Jordan", "W"),
    (1998, "CHI", "UTA", "Michael Jordan", "W"),
    # LeBron
    (2007, "CLE", "SAS", "LeBron James", "L"),
    (2011, "MIA", "DAL", "LeBron James", "L"),
    (2012, "MIA", "OKC", "LeBron James", "W"),
    (2013, "MIA", "SAS", "LeBron James", "W"),
    (2014, "MIA", "SAS", "LeBron James", "L"),
    (2015, "CLE", "GSW", "LeBron James", "L"),
    (2016, "CLE", "GSW", "LeBron James", "W"),
    (2017, "CLE", "GSW", "LeBron James", "L"),
    (2018, "CLE", "GSW", "LeBron James", "L"),
    (2020, "LAL", "MIA", "LeBron James", "W"),
]

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
    "Vancouver Grizzlies": "VAN",
    "Washington Bullets": "WSB",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str:
    """GET a URL with rate limiting."""
    time.sleep(REQUEST_DELAY)
    print(f"  Fetching: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_all_tables(html: str) -> list:
    """Extract all tables including those hidden in HTML comments."""
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


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------


def scrape_playoff_advanced(year: int) -> pd.DataFrame:
    """Scrape full playoff advanced stats for all players in a given year.
    Source: basketball-reference.com/playoffs/NBA_{year}_advanced.html
    """
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

            df = df.dropna(subset=["BPM"])
            df = df[df["Player"] != "Player"]  # drop repeated header rows
            df["Year"] = year
            return df

    print(f"  WARNING: No table with BPM found for {year} playoffs")
    for table in tables:
        tid = table.get("id", "")
        df = _table_to_df(table)
        if not df.empty and tid:
            print(f"    id={tid}: {list(df.columns[:10])}")

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


def build_finals_context():
    """Build the complete finals_context.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    playoff_cache: dict[int, pd.DataFrame] = {}
    srs_cache: dict[int, dict] = {}

    for year, player_team, opp_team, player_name, result in FINALS:
        print(f"\n{'=' * 60}")
        print(f"{year} Finals: {player_team} vs {opp_team} ({player_name})")
        print(f"{'=' * 60}")

        # --- 1. Get playoff advanced stats ---
        if year not in playoff_cache:
            playoff_cache[year] = scrape_playoff_advanced(year)
            if not playoff_cache[year].empty:
                playoff_cache[year].to_csv(
                    RAW_DIR / f"{year}_playoff_adv.csv", index=False
                )

        adv_df = playoff_cache[year]
        if adv_df.empty:
            print(f"  Skipping {year}")
            continue

        # --- 2. Filter to Finals teams ---
        player_team_df = adv_df[adv_df["Tm"] == player_team].copy()
        opp_team_df = adv_df[adv_df["Tm"] == opp_team].copy()

        if player_team_df.empty:
            print(f"  WARNING: No players found for {player_team}")
            print(f"  Available teams: {sorted(adv_df['Tm'].unique())}")
            continue

        # --- 3. Carry delta: top 2 VORP on player's team ---
        pt_sorted = player_team_df.sort_values("VORP", ascending=False)
        player_vorp = pt_sorted["VORP"].iloc[0]
        second_player = pt_sorted["Player"].iloc[1] if len(pt_sorted) > 1 else "N/A"
        second_vorp = pt_sorted["VORP"].iloc[1] if len(pt_sorted) > 1 else None

        # best opponent by VORP
        opp_sorted = opp_team_df.sort_values("VORP", ascending=False)
        opp_best_player = opp_sorted["Player"].iloc[0] if len(opp_sorted) > 0 else "N/A"
        opp_best_vorp = opp_sorted["VORP"].iloc[0] if len(opp_sorted) > 0 else None

        # --- 4. Team SRS ---
        if year not in srs_cache:
            srs_cache[year] = scrape_team_srs(year)

        team_srs = srs_cache[year].get(player_team)
        opp_srs = srs_cache[year].get(opp_team)

        carry_delta = (
            round(player_vorp - second_vorp, 2) if second_vorp is not None else None
        )
        srs_delta = round(team_srs - opp_srs, 2) if team_srs and opp_srs else None

        results.append(
            {
                "Season": year,
                "Player": player_name,
                "Team": player_team,
                "Opp": opp_team,
                "Result": result,
                "Player_VORP": player_vorp,
                "Second_Best_Player": second_player,
                "Second_Best_VORP": second_vorp,
                "Carry_Delta": carry_delta,
                "Opp_Best_Player": opp_best_player,
                "Opp_Best_VORP": opp_best_vorp,
                "Team_SRS": team_srs,
                "Opp_SRS": opp_srs,
                "SRS_Delta": srs_delta,
            }
        )

        print(
            f"  {player_name}: VORP={player_vorp}, #2={second_player} ({second_vorp})"
        )
        print(f"  Carry Delta: {carry_delta}")
        print(f"  Opp best: {opp_best_player} ({opp_best_vorp})")
        print(
            f"  SRS: {player_team}={team_srs}, {opp_team}={opp_srs}, delta={srs_delta}"
        )

    # --- 5. Save ---
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"Saved {len(results)} rows to {OUTPUT_CSV}")
    print(f"{'=' * 60}")
    print(final_df.to_string(index=False))

    return final_df


if __name__ == "__main__":
    build_finals_context()
