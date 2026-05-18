from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_PATH = Path("data/player_playoffs")
FINALS_PATH = Path("data/finals/finals_context.csv")
PO_ADV_STATS_PATH = "_po_adv_stats.csv"

PLAYERS = ("lbj", "mj", "kobe")


def _to_numeric(col):
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove BBRef mid-table headers, summary rows, and cast types."""
    return (
        df.dropna(how="all")
        .dropna(subset=["Season"])
        .loc[lambda d: d["Age"] != "Age"]
        .loc[lambda d: ~d["Season"].astype(str).str.contains("Yrs|Avg")]
        .reset_index(drop=True)
        .apply(_to_numeric)
    )


def _load_player(player: str) -> pd.DataFrame:
    """Load and clean a player's playoff advanced stats."""
    df = pd.read_csv(DATA_PATH / f"{player}{PO_ADV_STATS_PATH}")
    return _clean_df(df).sort_values("Season").reset_index(drop=True)


def _parse_season_year(season_str: str) -> int:
    """'2008-09' -> 2008"""
    return int(str(season_str).split("-")[0])


def _best_window(values: list[float], window: int) -> tuple[float, int]:
    """Return (best_avg, start_index) for the best consecutive window."""
    best_avg = float("-inf")
    best_start = 0
    for i in range(len(values) - window + 1):
        w_avg = sum(values[i : i + window]) / window
        if w_avg > best_avg:
            best_avg = w_avg
            best_start = i
    return best_avg, best_start


def compare_player_adv_stats(
    players: Iterable[str] = PLAYERS,
    metrics: list[str] = ["WS/48", "VORP", "PER"],
) -> pd.DataFrame:
    """Career playoff advanced stats: peak, average, floor per metric."""
    report_data = []

    for player in players:
        clean_df = _load_player(player)
        player_stats = {
            "Player": player.upper(),
            "Playoff Seasons": len(clean_df),
        }
        for metric in metrics:
            s = clean_df[metric]
            player_stats[f"{metric} Peak"] = s.max()
            player_stats[f"{metric} Avg"] = s.mean()
            player_stats[f"{metric} Lowest"] = s.min()

        report_data.append(player_stats)

    report = pd.DataFrame(report_data).set_index("Player").T

    print("\n--- Playoff Advanced Stats Comparison ---")
    print(
        report.to_string(
            float_format=lambda x: str(int(x)) if x == int(x) else f"{x:.3f}"
        )
    )
    return report


def prime_metrics(
    players: Iterable[str] = PLAYERS,
    metrics: list[str] = ["WS/48", "VORP", "PER"],
    window: int = 5,
) -> pd.DataFrame:
    """Best consecutive playoff-season stretch per metric."""
    report_data = []

    for player in players:
        clean_df = _load_player(player)
        seasons = clean_df["Season"]

        if len(clean_df) < window:
            continue

        player_stats = {"Player": player.upper()}
        for metric in metrics:
            values = clean_df[metric].tolist()
            best_avg, best_start = _best_window(values, window)
            w_values = values[best_start : best_start + window]

            start_year = _parse_season_year(seasons.iloc[best_start])
            end_year = _parse_season_year(seasons.iloc[best_start + window - 1]) + 1

            player_stats[f"{metric} Prime Avg"] = round(best_avg, 3)
            player_stats[f"{metric} Prime Peak"] = round(max(w_values), 3)
            player_stats[f"{metric} Prime Floor"] = round(min(w_values), 3)
            player_stats[f"{metric} Seasons"] = f"{start_year}-{end_year}"

        report_data.append(player_stats)

    report = pd.DataFrame(report_data).set_index("Player").T

    print(f"\n--- Best {window}-Season Playoff Prime ---")
    print(report)
    return report


def longevity(
    players: Iterable[str] = PLAYERS,
    metrics: list[str] = ["WS/48", "VORP", "PER"],
) -> pd.DataFrame:
    """Cumulative career value and prime-to-career dropoff."""
    cumulative_data = []
    dropoff_data = []

    for player in players:
        clean_df = _load_player(player)

        # cumulative totals
        cum = {"Player": player.upper()}
        cum["Playoff Seasons"] = len(clean_df)
        cum["Total Games"] = (
            int(clean_df["G"].sum()) if "G" in clean_df.columns else None
        )
        cum["Cumulative VORP"] = round(clean_df["VORP"].sum(), 1)
        cum["Cumulative WS"] = (
            round(clean_df["WS"].sum(), 1) if "WS" in clean_df.columns else None
        )
        for metric in metrics:
            cum[f"{metric} Career Avg"] = round(clean_df[metric].mean(), 3)
        cumulative_data.append(cum)

        # prime-to-career dropoff
        drop = {"Player": player.upper()}
        for metric in metrics:
            values = clean_df[metric].tolist()
            career_avg = sum(values) / len(values)
            best_avg, _ = _best_window(values, 5)

            dropoff = round(career_avg - best_avg, 3)
            dropoff_pct = round((dropoff / best_avg) * 100, 1) if best_avg else 0

            drop[f"{metric} Prime Avg"] = round(best_avg, 3)
            drop[f"{metric} Career Avg"] = round(career_avg, 3)
            drop[f"{metric} Dropoff"] = dropoff
            drop[f"{metric} Dropoff %"] = f"{dropoff_pct}%"
        dropoff_data.append(drop)

    cum_df = pd.DataFrame(cumulative_data).set_index("Player").T
    drop_df = pd.DataFrame(dropoff_data).set_index("Player").T

    print("\n--- Sustained Excellence: Cumulative Career Value ---")
    print(
        cum_df.to_string(
            float_format=lambda x: str(int(x)) if x == int(x) else f"{x:.3f}"
        )
    )
    print("\n--- Prime-to-Career Dropoff ---")
    print(drop_df)

    return cum_df


def finals_impact() -> tuple[dict, pd.DataFrame]:
    """Context-adjusted Finals record using logistic regression."""
    from model import (
        compute_weighted_record,
        fit_model,
        print_model_report,
        print_weighted_record,
    )

    results = fit_model()
    print_model_report(results)

    weighted_df = compute_weighted_record(FINALS_PATH, results)
    print_weighted_record(weighted_df)

    return results, weighted_df


if __name__ == "__main__":
    compare_player_adv_stats()
    prime_metrics()
    longevity()
    finals_impact()
