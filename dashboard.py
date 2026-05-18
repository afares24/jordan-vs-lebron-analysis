from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from main import (
    DATA_PATH,
    PO_ADV_STATS_PATH,
    _clean_df,
)
from model import compute_weighted_record, fit_model, load_training_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FINALS_PATH = Path("data/finals/finals_context.csv")

PLAYER_COLORS = {
    "LBJ": "#FFD700",
    "MJ": "#CE1141",
    "KOBE": "#552583",
}

PLAYER_LABELS = {
    "LBJ": "LeBron James",
    "MJ": "Michael Jordan",
    "KOBE": "Kobe Bryant",
}

st.set_page_config(
    page_title="NBA Player Analysis",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


def _parse_year(season_str: str) -> int:
    """Extract start year from season string, e.g. '2008-09' -> 2008."""
    return int(str(season_str).split("-")[0])


@st.cache_data
def load_player_data():
    """Load and clean player playoff data."""
    players = {}
    for tag in ["lbj", "mj", "kobe"]:
        df = pd.read_csv(DATA_PATH / f"{tag}{PO_ADV_STATS_PATH}")
        df = _clean_df(df).sort_values("Season").reset_index(drop=True)
        df["Year"] = df["Season"].apply(_parse_year)
        players[tag.upper()] = df
    return players


@st.cache_data
def load_model():
    """Fit model and compute weighted record."""
    training_df = load_training_data()
    results = fit_model(training_df)
    weighted_df = compute_weighted_record(FINALS_PATH, results)
    return results, weighted_df


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("NBA Player Analysis: Jordan vs. James")
st.caption(
    "Evaluating player legacy through playoff advanced statistics, contextual modeling, "
    "and sustained production. Kobe Bryant is included as a benchmark — a consensus top-10 "
    "player used to validate that the metrics separate tiers of greatness."
)

players = load_player_data()
model_results, weighted_df = load_model()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Peak Performance",
        "Finals Impact",
        "Sustained Excellence",
        "Methodology",
    ]
)


# ---------------------------------------------------------------------------
# Tab 1: Peak Performance
# ---------------------------------------------------------------------------

with tab1:
    st.header("Criterion 1: Peak Performance")
    st.markdown("At their absolute best, who produced more in the playoffs?")

    # --- Metric selector ---
    metric = st.selectbox("Metric", ["VORP", "WS/48", "PER"], index=0)

    col1, col2 = st.columns(2)

    # --- Season-by-season comparison ---
    with col1:
        st.subheader(f"Season-by-Season Playoff {metric}")

        fig = go.Figure()
        for tag in ["LBJ", "MJ", "KOBE"]:
            df = players[tag]
            fig.add_trace(
                go.Scatter(
                    x=df["Year"],
                    y=df[metric],
                    mode="lines+markers",
                    name=PLAYER_LABELS[tag],
                    line=dict(color=PLAYER_COLORS[tag], width=2),
                    marker=dict(size=6),
                    text=df["Season"],
                    hovertemplate="%{text}<br>" + metric + ": %{y:.3f}",
                )
            )

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title=metric,
            hovermode="x unified",
            template="plotly_dark",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Peak / Avg / Floor comparison ---
    with col2:
        st.subheader(f"{metric}: Peak, Average, Floor")

        comparison_data = []
        for tag in ["LBJ", "MJ", "KOBE"]:
            df = players[tag]
            series = df[metric]
            comparison_data.append(
                {
                    "Player": PLAYER_LABELS[tag],
                    "Peak": series.max(),
                    "Average": series.mean(),
                    "Floor": series.min(),
                }
            )

        comp_df = pd.DataFrame(comparison_data)

        fig = go.Figure()
        for measure, symbol in [
            ("Peak", "triangle-up"),
            ("Average", "circle"),
            ("Floor", "triangle-down"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=comp_df["Player"],
                    y=comp_df[measure],
                    mode="markers",
                    name=measure,
                    marker=dict(size=14, symbol=symbol),
                )
            )

        fig.update_layout(
            yaxis_title=metric,
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 5-Year Prime Window ---
    st.subheader("Best 5-Playoff-Season Prime")

    prime_data = []
    for tag in ["LBJ", "MJ", "KOBE"]:
        df = players[tag]
        values = df[metric].tolist()
        seasons = df["Season"].tolist()

        best_avg = float("-inf")
        best_start = 0

        for i in range(len(values) - 5 + 1):
            w_avg = sum(values[i : i + 5]) / 5
            if w_avg > best_avg:
                best_avg = w_avg
                best_start = i

        w_values = values[best_start : best_start + 5]
        w_seasons = seasons[best_start : best_start + 5]

        for idx, (s, v) in enumerate(zip(w_seasons, w_values)):
            prime_data.append(
                {
                    "Player": PLAYER_LABELS[tag],
                    "Prime Season": f"Year {idx + 1}",
                    "Season": s,
                    metric: v,
                }
            )

    prime_df = pd.DataFrame(prime_data)

    fig = px.bar(
        prime_df,
        x="Prime Season",
        y=metric,
        color="Player",
        barmode="group",
        template="plotly_dark",
        height=400,
        hover_data={"Season": True, "Prime Season": False},
        color_discrete_map={PLAYER_LABELS[k]: v for k, v in PLAYER_COLORS.items()},
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Finals Impact
# ---------------------------------------------------------------------------

with tab2:
    st.header("Criterion 2: Playoff/Finals Impact")
    st.markdown(
        "Does the 6-0 vs 4-6 record reflect individual performance or team context? "
        "A logistic regression trained on 654 historical playoff series provides "
        "context-adjusted weights for each Finals appearance."
    )

    col1, col2 = st.columns(2)

    with col1:
        # --- Model diagnostics ---
        st.subheader("Model Diagnostics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observations", model_results["n_observations"])
        m2.metric("Accuracy", f"{model_results['accuracy']:.1%}")
        m3.metric("AUC-ROC", f"{model_results['auc']:.3f}")
        m4.metric("CV Accuracy", f"{model_results['cv_accuracy']:.1%}")

        # relative importance
        st.subheader("Standardized Relative Importance")
        std_total = sum(abs(v) for v in model_results["std_coefficients"].values())
        for feat, std_coef in model_results["std_coefficients"].items():
            pct = abs(std_coef) / std_total * 100
            label = (
                "Team Quality (SRS Δ)" if "SRS" in feat else "Star Dominance (Carry Δ)"
            )
            st.progress(pct / 100, text=f"{label}: {pct:.1f}%")

    with col2:
        # --- SRS Delta vs Outcome scatter ---
        st.subheader("Finals Context: SRS Δ vs Win Probability")

        fig = go.Figure()

        for player in ["Michael Jordan", "LeBron James"]:
            pdf = weighted_df[weighted_df["Player"] == player]
            color = PLAYER_COLORS["MJ"] if "Jordan" in player else PLAYER_COLORS["LBJ"]

            wins = pdf[pdf["Result"] == "W"]
            losses = pdf[pdf["Result"] == "L"]

            fig.add_trace(
                go.Scatter(
                    x=wins["SRS_Delta"],
                    y=wins["P_Win"],
                    mode="markers",
                    name=f"{player} W",
                    marker=dict(color=color, size=12, symbol="circle"),
                    text=wins["Season"].astype(str),
                    hovertemplate="%{text}<br>SRS Δ: %{x:.2f}<br>P(Win): %{y:.3f}",
                )
            )

            if len(losses) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=losses["SRS_Delta"],
                        y=losses["P_Win"],
                        mode="markers",
                        name=f"{player} L",
                        marker=dict(
                            color=color,
                            size=12,
                            symbol="x",
                            line=dict(width=2),
                        ),
                        text=losses["Season"].astype(str),
                        hovertemplate="%{text}<br>SRS Δ: %{x:.2f}<br>P(Win): %{y:.3f}",
                    )
                )

        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            xaxis_title="SRS Delta (+ = team advantage)",
            yaxis_title="Model P(Win)",
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Weighted record table ---
    st.subheader("Weighted Finals Record")

    display_cols = [
        "Season",
        "Player",
        "Team",
        "Opp",
        "Result",
        "SRS_Delta",
        "Carry_Delta",
        "P_Win",
        "Weighted",
    ]
    st.dataframe(
        weighted_df[display_cols]
        .style.format(
            {
                "SRS_Delta": "{:.2f}",
                "Carry_Delta": "{:.2f}",
                "P_Win": "{:.3f}",
                "Weighted": "{:.3f}",
            }
        )
        .apply(
            lambda row: (
                [
                    "background-color: rgba(0,100,0,0.3)"
                    if row["Result"] == "W"
                    else "background-color: rgba(100,0,0,0.3)"
                ]
                * len(row)
            ),
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # --- Finals Opponent Strength ---
    st.subheader("Finals Team Context (SRS)")

    col1, col2 = st.columns(2)

    finals_df = pd.read_csv(FINALS_PATH)

    with col1:
        st.markdown("**Michael Jordan**")
        mj_df = finals_df[finals_df["Player"] == "Michael Jordan"][
            ["Season", "Team_SRS", "Opp", "Opp_SRS", "SRS_Delta", "Result"]
        ].copy()
        mj_df = mj_df.rename(
            columns={"Opp_SRS": "Opp SRS", "Team_SRS": "Team SRS", "SRS_Delta": "SRS Δ"}
        )
        st.dataframe(
            mj_df.style.format(
                {"Opp SRS": "{:.2f}", "Team SRS": "{:.2f}", "SRS Δ": "{:+.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        avg_team = mj_df["Team SRS"].mean()
        avg_opp = mj_df["Opp SRS"].mean()
        st.caption(f"Avg team SRS: {avg_team:.2f} | Avg opponent SRS: {avg_opp:.2f}")

    with col2:
        st.markdown("**LeBron James**")
        lbj_df = finals_df[finals_df["Player"] == "LeBron James"][
            ["Season", "Team_SRS", "Opp", "Opp_SRS", "SRS_Delta", "Result"]
        ].copy()
        lbj_df = lbj_df.rename(
            columns={"Opp_SRS": "Opp SRS", "Team_SRS": "Team SRS", "SRS_Delta": "SRS Δ"}
        )
        st.dataframe(
            lbj_df.style.format(
                {"Opp SRS": "{:.2f}", "Team SRS": "{:.2f}", "SRS Δ": "{:+.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        avg_team = lbj_df["Team SRS"].mean()
        avg_opp = lbj_df["Opp SRS"].mean()
        st.caption(f"Avg team SRS: {avg_team:.2f} | Avg opponent SRS: {avg_opp:.2f}")

    # --- Summary ---
    st.subheader("Summary")
    col1, col2 = st.columns(2)

    for player, col in [("Michael Jordan", col1), ("LeBron James", col2)]:
        pdf = weighted_df[weighted_df["Player"] == player]
        raw_w = (pdf["Result"] == "W").sum()
        raw_l = (pdf["Result"] == "L").sum()
        w_wins = pdf[pdf["Result"] == "W"]["Weighted"].sum()
        w_losses = pdf[pdf["Result"] == "L"]["Weighted"].sum()
        net = w_wins - w_losses

        with col:
            st.markdown(f"**{player}**")
            st.metric("Raw Record", f"{raw_w}-{raw_l}")
            st.metric("Weighted Wins", f"{w_wins:.2f}")
            st.metric("Weighted Losses", f"{w_losses:.2f}")
            st.metric("Net", f"{net:.2f}")


# ---------------------------------------------------------------------------
# Tab 3: Sustained Excellence
# ---------------------------------------------------------------------------

with tab3:
    st.header("Criterion 3: Sustained Excellence")
    st.markdown("Who maintained elite production longer?")

    col1, col2 = st.columns(2)

    with col1:
        # --- Cumulative VORP over career ---
        st.subheader("Cumulative Playoff VORP Over Career")

        fig = go.Figure()
        for tag in ["LBJ", "MJ", "KOBE"]:
            df = players[tag]
            cum_vorp = df["VORP"].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cum_vorp) + 1)),
                    y=cum_vorp,
                    mode="lines+markers",
                    name=PLAYER_LABELS[tag],
                    line=dict(color=PLAYER_COLORS[tag], width=2),
                    marker=dict(size=5),
                )
            )

        fig.update_layout(
            xaxis_title="Playoff Season Number",
            yaxis_title="Cumulative VORP",
            template="plotly_dark",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # --- Cumulative totals ---
        st.subheader("Career Totals")

        totals = []
        for tag in ["LBJ", "MJ", "KOBE"]:
            df = players[tag]
            totals.append(
                {
                    "Player": PLAYER_LABELS[tag],
                    "Playoff Seasons": len(df),
                    "Total Games": int(df["G"].sum()),
                    "Cumulative VORP": round(df["VORP"].sum(), 1),
                    "Cumulative WS": round(df["WS"].sum(), 1)
                    if "WS" in df.columns
                    else None,
                    "VORP per Season": round(df["VORP"].mean(), 3),
                }
            )

        totals_df = pd.DataFrame(totals).set_index("Player")
        st.dataframe(totals_df, use_container_width=True)

    # --- Season-by-season VORP heatmap style ---
    st.subheader("Season-by-Season Playoff VORP")

    fig = go.Figure()
    for tag in ["LBJ", "MJ", "KOBE"]:
        df = players[tag]
        fig.add_trace(
            go.Bar(
                x=df["Year"],
                y=df["VORP"],
                name=PLAYER_LABELS[tag],
                marker_color=PLAYER_COLORS[tag],
                opacity=0.8,
                text=df["Season"],
                hovertemplate="%{text}<br>VORP: %{y:.1f}",
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Season",
        yaxis_title="VORP",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Methodology
# ---------------------------------------------------------------------------

with tab4:
    st.header("Methodology")

    methodology_path = Path("analysis.md")
    if methodology_path.exists():
        st.markdown(methodology_path.read_text())
    else:
        st.markdown("""
### Framework

This analysis evaluates Michael Jordan and LeBron James across three measurable criteria:

1. **Peak Performance** — Best 5-playoff-season stretches using WS/48, VORP, and PER.
2. **Playoff/Finals Impact** — Logistic regression model trained on 654 historical playoff series (1980-2025) to context-adjust the 6-0 vs 4-6 Finals record using SRS Delta (team quality) and Carry Delta (star dominance).
3. **Sustained Excellence** — Cumulative career value and production decay curves.

### Why These Metrics

- **VORP** — Cumulative measure capturing both quality and playing time. Used for carry delta because it naturally filters small-sample noise.
- **WS/48** — Per-minute rate stat normalizing for pace and playing time.
- **PER** — Per-minute statistical production rating, used as cross-validation.

### Why These Criteria

Criteria were selected for measurability and non-overlap. Two-way ability, elevating teammates, and accolades were excluded because they are either captured within the advanced stats already used or lack objective measurement frameworks.

### Model Details

The logistic regression uses no regularization (C=1e10) to produce unbiased coefficients. Standardized coefficients are used for relative importance to account for the different scales of SRS Delta (SD=4.21) and Carry Delta (SD=0.50).

See `analysis.md` for the full write-up with limitations and detailed rationale.
        """)
