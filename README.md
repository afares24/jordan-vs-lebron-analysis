# NBA Playoff Impact Analysis: Jordan vs. James

Evaluating player legacy through playoff advanced statistics, logistic regression modeling, and sustained production. Kobe Bryant included as a benchmark.

## Criteria

1. **Peak Performance** — Best 5-playoff-season stretches using WS/48, VORP, PER
2. **Playoff/Finals Impact** — Context-adjusted Finals record using SRS delta and carry delta
3. **Sustained Excellence** — Cumulative career value and production decay

## Model

Logistic regression trained on 654 NBA playoff series (1980-2025) predicting series outcomes from team quality differential (SRS delta) and individual star dominance (carry delta). Applied to 16 Jordan/LeBron Finals to produce weighted win/loss records.

## Run

```bash
pip install -r requirements.txt
python main.py          # full analysis output
streamlit run dashboard.py  # interactive dashboard
```

## Structure

```
├── analysis.md          # full write-up with methodology and findings
├── main.py              # analysis pipeline
├── model.py             # logistic regression model
├── dashboard.py         # streamlit dashboard
├── data/                # player stats, finals context, model training data
└── scripts/             # one-time data scrapers
```

## Data

All data sourced from [Basketball Reference](https://www.basketball-reference.com). Training set: 654 playoff series across 46 seasons.