from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DATA = Path("data/model/all_playoff_series.csv")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_training_data(path: Path = MODEL_DATA) -> pd.DataFrame:
    """Load, deduplicate, and prepare the training dataset.

    Deduplication: each series has two mirrored rows (winner/loser perspective).
    We keep one row per series using a consistent perspective (alphabetically
    first team) to ensure independent observations.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["SRS_Delta", "Carry_Delta"])

    # create a unique series key from the sorted team pair
    df["_series_key"] = df.apply(
        lambda r: (r["Season"], r["Round"], *sorted([r["Team"], r["Opp"]])),
        axis=1,
    )

    # keep the row where Team comes first alphabetically for consistency
    df["_alpha_first"] = df["Team"] < df["Opp"]
    deduped = df.sort_values("_alpha_first", ascending=False).drop_duplicates(
        subset=["_series_key"], keep="first"
    )
    deduped = deduped.drop(columns=["_series_key", "_alpha_first"])

    # binary target
    deduped["Win"] = (deduped["Result"] == "W").astype(int)

    win_count = deduped["Win"].sum()
    loss_count = len(deduped) - win_count

    print(f"Training data: {len(deduped)} series (deduplicated)")
    print(f"  Outcomes: {win_count} W, {loss_count} L from selected perspective")
    print(f"  Seasons: {deduped['Season'].min()}-{deduped['Season'].max()}")
    print(
        f"  SRS_Delta: [{deduped['SRS_Delta'].min():.2f}, {deduped['SRS_Delta'].max():.2f}] "
        f"(SD={deduped['SRS_Delta'].std():.2f})"
    )
    print(
        f"  Carry_Delta: [{deduped['Carry_Delta'].min():.2f}, {deduped['Carry_Delta'].max():.2f}] "
        f"(SD={deduped['Carry_Delta'].std():.2f})"
    )

    return deduped.reset_index(drop=True)


def fit_model(df: pd.DataFrame | None = None) -> dict:
    """Fit logistic regression and return model details.

    Reports both raw coefficients (for prediction) and standardized
    coefficients (for relative importance comparison).
    """
    if df is None:
        df = load_training_data()

    features = ["SRS_Delta", "Carry_Delta"]
    X = df[features].values
    y = df["Win"].values

    # store predictor statistics for standardized coefficients
    sds = {f: df[f].std() for f in features}
    means = {f: df[f].mean() for f in features}

    # fit logistic regression (C=1e10 ≈ no regularization)
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    model.fit(X, y)

    # predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # metrics
    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    logloss = log_loss(y, y_prob)

    # 5-fold cross-validation
    cv_scores = cross_val_score(
        LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000),
        X,
        y,
        cv=5,
        scoring="accuracy",
    )
    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()

    # raw coefficients (for prediction in original units)
    raw_coefficients = {f: model.coef_[0][i] for i, f in enumerate(features)}

    # standardized coefficients (for relative importance)
    # standardized = raw_coef * sd(predictor)
    std_coefficients = {f: model.coef_[0][i] * sds[f] for i, f in enumerate(features)}

    results = {
        "model": model,
        "raw_coefficients": raw_coefficients,
        "std_coefficients": std_coefficients,
        "predictor_sds": sds,
        "predictor_means": means,
        "intercept": model.intercept_[0],
        "accuracy": accuracy,
        "auc": auc,
        "log_loss": logloss,
        "cv_accuracy": cv_accuracy,
        "cv_std": cv_std,
        "n_observations": len(df),
    }

    return results


def predict_win_prob(
    srs_delta: float,
    carry_delta: float,
    model_results: dict,
) -> float:
    """Predict win probability for a single series given context."""
    model = model_results["model"]
    X = np.array([[srs_delta, carry_delta]])
    return model.predict_proba(X)[0][1]


def compute_weighted_record(
    finals_csv: str | Path,
    model_results: dict,
) -> pd.DataFrame:
    """Computes weighted wins/losses using a weight formula.

    Weight formula:
        P(win) = logistic(b0 + b1*SRS_Delta + b2*Carry_Delta)
        For a WIN:  weight = 2 - P(win)   -> underdog win worth more
        For a LOSS: weight = P(win)        -> underdog loss penalized less
    """
    df = pd.read_csv(finals_csv)
    df = df.dropna(subset=["SRS_Delta", "Carry_Delta"])

    df["P_Win"] = df.apply(
        lambda row: predict_win_prob(
            row["SRS_Delta"], row["Carry_Delta"], model_results
        ),
        axis=1,
    )

    df["Weighted"] = df.apply(
        lambda row: (2 - row["P_Win"]) if row["Result"] == "W" else row["P_Win"],
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_model_report(results: dict):
    """Print model diagnostics."""
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: P(Win) ~ SRS_Delta + Carry_Delta")
    print(f"Observations: {results['n_observations']} series (1 per series)")
    print("=" * 60)

    # raw coefficients
    print("\nRaw Coefficients (for prediction):")
    for feat, coef in results["raw_coefficients"].items():
        print(f"  {feat}: {coef:.4f}")
    print(f"  Intercept: {results['intercept']:.4f}")

    # intercept interpretation
    p_at_zero = 1 / (1 + np.exp(-results["intercept"]))
    print(f"  -> P(Win) at SRS_Delta=0, Carry_Delta=0: {p_at_zero:.3f}")

    # standardized coefficients
    print("\nStandardized Coefficients (raw_coef x SD, for relative importance):")
    for feat in results["std_coefficients"]:
        raw = results["raw_coefficients"][feat]
        sd = results["predictor_sds"][feat]
        std = results["std_coefficients"][feat]
        print(f"  {feat}: {std:.4f}  (raw={raw:.4f} x SD={sd:.2f})")

    # relative importance from STANDARDIZED coefficients
    std_total = sum(abs(v) for v in results["std_coefficients"].values())
    print("\nRelative Importance (standardized):")
    for feat, std_coef in results["std_coefficients"].items():
        pct = abs(std_coef) / std_total * 100
        print(f"  {feat}: {pct:.1f}%")

    # odds ratios (per 1-unit change in original scale)
    print("\nOdds Ratios (per +1 unit in original scale):")
    for feat, coef in results["raw_coefficients"].items():
        odds_ratio = np.exp(coef)
        print(f"  {feat}: {odds_ratio:.3f}x")

    # odds ratios per 1-SD change
    print("\nOdds Ratios (per +1 SD change):")
    for feat, std_coef in results["std_coefficients"].items():
        odds_ratio = np.exp(std_coef)
        sd = results["predictor_sds"][feat]
        print(f"  {feat}: {odds_ratio:.3f}x  (1 SD = {sd:.2f} units)")

    # model performance
    print("\nModel Performance:")
    print(f"  Accuracy:             {results['accuracy']:.3f}")
    print(f"  AUC-ROC:              {results['auc']:.3f}")
    print(f"  Log Loss:             {results['log_loss']:.3f}")
    print(
        f"  CV Accuracy (5-fold): {results['cv_accuracy']:.3f} (+/- {results['cv_std']:.3f})"
    )


def print_weighted_record(df: pd.DataFrame):
    """Print the weighted record analysis for Finals appearances."""
    print("\n" + "=" * 60)
    print("WEIGHTED FINALS RECORD")
    print("=" * 60)

    print(
        f"\n{'Season':<8} {'Player':<16} {'Result':<8} {'SRS_Δ':>7} {'Carry_Δ':>9} "
        f"{'P(Win)':>8} {'Weight':>8}"
    )
    print("-" * 72)

    for _, row in df.iterrows():
        print(
            f"{row['Season']:<8} {row['Player']:<16} {row['Result']:<8} "
            f"{row['SRS_Delta']:>7.2f} {row['Carry_Delta']:>9.2f} "
            f"{row['P_Win']:>8.3f} {row['Weighted']:>8.3f}"
        )

    # summary by player
    print("\n" + "-" * 72)
    print(
        f"\n{'Player':<20} {'Raw':>10} {'Weighted W':>12} {'Weighted L':>12} {'Net':>8}"
    )
    print("-" * 64)

    for player in df["Player"].unique():
        p_df = df[df["Player"] == player]
        raw_w = (p_df["Result"] == "W").sum()
        raw_l = (p_df["Result"] == "L").sum()
        raw = f"{raw_w}-{raw_l}"

        w_wins = p_df[p_df["Result"] == "W"]["Weighted"].sum()
        w_losses = p_df[p_df["Result"] == "L"]["Weighted"].sum()
        net = w_wins - w_losses

        print(f"{player:<20} {raw:>10} {w_wins:>12.2f} {w_losses:>12.2f} {net:>8.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 1. Load and fit
    df = load_training_data()
    results = fit_model(df)
    print_model_report(results)

    # 2. Apply to Finals
    finals_path = Path("data/finals/finals_context.csv")
    if finals_path.exists():
        weighted_df = compute_weighted_record(finals_path, results)
        print_weighted_record(weighted_df)
    else:
        print(f"\nFinals data not found at {finals_path}")
        print("Run scrape_finals.py first, then re-run this module.")
