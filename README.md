# Inflation Regime Forecasting with AR-HMM and NLP Sentiment Analysis

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQL%20Server-red.svg)](https://www.microsoft.com/sql-server/)

## Abstract

This thesis develops a step-by-step framework for forecasting inflation and identifying inflation regimes with an Autoregressive Hidden Markov Model. The current implementation covers data preparation, AR(1), ARX with lagged FinBERT sentiment, a two-state HMM, and a two-state AR-HMM with regime-specific autoregressive equations. US macroeconomic data, two alternative US news sources, and a normalized SQL Server database form the supporting data pipeline.

**Author:** Zahra Younes Pour Langaroudi

**Institution:** University of Trieste — M.Sc. in Data Science and Artificial Intelligence

---

## Project Overview

This thesis builds an end-to-end pipeline that connects three complementary areas:

1. **Data Infrastructure** — A normalized SQL Server database that ingests macroeconomic indicators from FRED and financial news from public datasets, scores articles with FinBERT, and exposes analytics-ready views through a Streamlit dashboard.
2. **NLP Sentiment Extraction** — FinBERT (a BERT model fine-tuned on financial text) classifies each news article as positive, negative, or neutral. Article-level scores are aggregated into a monthly sentiment signal.
3. **Forecasting and Regime Modelling** — The current learning workflow contains AR(1), ARX, HMM, and AR-HMM. The HMM notebook separates full-sample historical interpretation from no-lookahead forecasting, while AR-HMM adds a separate autoregressive equation to every hidden state. Adding sentiment to AR-HMM is the next thesis stage; the older US regime notebook is retained only as reference.


### Pipeline Architecture

```
FRED macro data ───────────────────────────────┐
                                               │
FinSen CSV ──┐                                 ▼
             ├─ FinBERT ─ monthly sentiment ─ SQL Server
NYT API ─────┘                                 │
                                               ├─ Streamlit dashboard
                                               │
                                               ▼
                           01 Data and Inflation
                                      │
                                      ▼
                              02 AR(1) baseline
                                      │
                                      ▼
                         03 ARX with lagged news
                                      │
                                      ▼
                         Forecast-error comparison
                                      │
                                      ▼
                        04 Historical + forecasting HMM
                                      │
                                      ▼
                                  05 AR-HMM
                                      │
                                      ▼
                         Next: AR-HMM + sentiment
```

---

## Repository Structure

```
Thesis/
│
├── README.md                       ← This file
├── .gitignore
├── .env.example                    ← Safe API and database configuration template
├── requirements.txt                ← Merged Python dependencies
│
├── database/                       ← Data infrastructure & schema definitions
│   ├── scripts/                    ← SQL schema, views, security & news-source upgrade (01–07)
│   ├── docs/
│   │   └── ER.png                  ← Entity-Relationship diagram
│   └── metadata/
│       ├── DMP.md                  ← Data Management Plan
│       ├── data_dictionary.json    ← Field-level metadata
│       ├── data_lineage.jsonld     ← Source & transformation lineage
│       └── dataset_metadata.jsonld ← Descriptive dataset metadata
│
├── modelling/                      ← Data pipeline, NLP, and probabilistic models
│   ├── config.py                   ← Central `.env` loader and SQL connection builder
│   ├── news_pipeline.py            ← Reusable NYT filtering, FinBERT & database helpers
│   ├── streamlit_app.py            ← Interactive analytics dashboard
│   └── notebooks/
│       ├── learning/
│       │   ├── 00_NYT_News_FinBERT_and_DB.ipynb
│       │   │                                ← NYT acquisition and scoring pipeline
│       │   ├── 01_Data_and_Inflation.ipynb ←  Data and inflation
│       │   ├── 02_AR1_Baseline.ipynb        ← Simple AR(1) forecast
│       │   ├── 03_ARX_with_News.ipynb       ← Add lagged sentiment
│       │   ├── 04_HMM.ipynb                 ← Historical regimes and HMM forecasts
│       │   └── 05_AR_HMM.ipynb              ← Regime-specific AR forecasts
│       └── US/
│           ├── 01_Data_Prep.ipynb              ← Data cleaning, ETL & FinBERT scoring
│           └── 02_AR_HMM_and_Evaluation.ipynb  ← HMM & AR-HMM regime detection
│
├── data/                           ← Shared data layer
│   ├── raw/
│   │   └── news/                   ← FinSen CSV and generated NYT monthly cache
│   └── processed/
│       ├── US/                     ← US macro data and generated NYT monthly sentiment
│       ├── finbert_article_scores.csv  ← FinSen FinBERT scores
│       └── nyt_finbert_article_scores.csv ← Generated NYT FinBERT scores
│
└── plots/                          ← Exploratory notebook visualisations
    └── US/                         ← Historical regimes and BIC comparisons

```

---

## Data Sources

| Source | Provider | Description | Period |
|---|---|---|---|
| CPI (`CPIAUCSL`) | FRED | Consumer Price Index for All Urban Consumers | 2007–2023 |
| Unemployment (`UNRATE`) | FRED | Civilian Unemployment Rate | 2007–2023 |
| Industrial Production (`INDPRO`) | FRED | Industrial Production Index | 2007–2023 |
| US News — original source | Kaggle ([FinSen](https://www.kaggle.com/datasets/eaglewhl/finsen-financial-sentiment-dataset)) | 15,534 categorized articles; retained as the historical baseline | 2007–2023 |
| US News — replacement candidate | [New York Times Archive API](https://developer.nytimes.com/) | Publication-dated metadata filtered for US macroeconomic topics and scored locally with FinBERT | 2007–2023 |

FinSen and NYT are not averaged together. The database source registry identifies one active news source for analytical views and the dashboard. FinSen remains available for comparison while NYT is downloaded, audited, scored, and validated.

---

## Methodology

### 1. NLP Sentiment Extraction — FinBERT

Articles from either US news source are scored using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT-based model fine-tuned on financial text. FinBERT outputs class probabilities for *positive*, *negative*, and *neutral* labels. A net sentiment score is computed as:

```
score = P(positive) − P(negative)     ∈ [−1, +1]
```

Article-level scores are aggregated within each calendar month to produce `News_Sentiment`. The NYT workflow also retains the monthly article count, median sentiment, sentiment dispersion, and negative-article share for data-quality and robustness checks.

To prevent future information from entering a forecast, the ARX model uses the previous month's sentiment (`Sentiment_Lag1`) rather than contemporaneous sentiment.

### 2. Current Forecasting Models — AR(1) and ARX

The implemented learning workflow currently contains:

- **AR(1):** forecasts month-over-month inflation from its value one month earlier.
- **ARX:** extends AR(1) with news sentiment lagged by one month.

Both models use the same chronological train/test split and the same test months. This makes their forecasting errors directly comparable.

### 3. Evaluation Metrics — RMSE, MAE, and Mean Error

Every forecasting notebook reports three complementary measures on identical test months:

```text
RMSE = sqrt(mean((actual inflation - predicted inflation)²))
MAE  = mean(abs(actual inflation - predicted inflation))
ME   = mean(actual inflation - predicted inflation)
```

Lower RMSE and MAE indicate better accuracy. RMSE penalizes large errors more heavily, while MAE describes the typical absolute mistake. Mean error measures bias rather than accuracy: a positive value means the model underpredicts inflation on average, and a negative value means it overpredicts.

Verified results from the frozen US CSV snapshot on the common 2017–2023 test period are:

### 4. HMM and AR-HMM

The first section of the implemented two-state HMM is fitted to year-over-year inflation (`Inflation_YoY`) because it represents persistent inflation environments more clearly than noisy monthly changes. It estimates a state-specific average inflation rate, smoothed historical state probabilities, and transition probabilities.

The second section is a separate forecasting experiment fitted only to pre-2017 month-over-month inflation. It creates each test forecast from predicted state probabilities, records the forecast before observing the actual month, and then updates filtered probabilities for the following month. Its RMSE, MAE, and mean error are compared with AR(1) using identical test months. This keeps smoothed historical interpretation separate from genuine no-lookahead forecasting.

The implemented AR-HMM then assigns a different constant and lag coefficient to each hidden state. Its one-month-ahead forecast combines the state-specific AR forecasts using predicted state probabilities, and actual test inflation updates those probabilities only after each forecast is recorded.

### 5. Step-by-Step Modelling

The simplified learning sequence keeps five models in scope:

1. AR(1) as the understandable inflation forecast baseline.
2. ARX to test whether lagged sentiment helps without hidden states.
3. HMM for historical low/high inflation regimes.
4. AR-HMM to combine regimes with past inflation.
5. AR-HMM with news sentiment to test the NLP contribution.


### 6. Database Architecture

The data infrastructure uses a normalized SQL Server schema:

- **`core` schema:** Regions, agencies, indicators, monthly observations, daily news, and selectable news sources
- **`analytics` schema:** Aggregated monthly views based on the active news source, with month-over-month macroeconomic changes
- **`auth` schema:** Users, roles, companies, and scoped permissions

See the [ER diagram](database/docs/ER.png) and [Data Management Plan](database/metadata/DMP.md) for full documentation.

---

## Quick Start

### Prerequisites

- Python 3.8 or later
- Microsoft SQL Server (with ODBC Driver 17)
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)
- A free [NYT Developer API key](https://developer.nytimes.com/) when using the NYT replacement source

### Installation

```bash
pip install -r requirements.txt
```

### Step 1 — Set Up the Database

Create a SQL Server database named `MacroSentimentDB`, then run the SQL scripts in order:

```text
database/scripts/01_create_schemas.sql
database/scripts/02a_create_source_regions.sql
database/scripts/02b_build_core_tables.sql
database/scripts/03_create_analytics_view.sql
database/scripts/03b_create_monthly_region_wide_view.sql
database/scripts/04_create_auth_tables.sql
database/scripts/05_create_access_views.sql
database/scripts/06_create_assets_table.sql
database/scripts/07_upgrade_news_sources.sql
```

### Step 2 — Configure the Environment

Copy `.env.example` to a project-root `.env` file and replace the placeholders with local values:

```text
FRED_API_KEY=your-fred-key
NYT_API_KEY=your-nyt-key
DB_SERVER=your-server
DB_DATABASE=MacroSentimentDB
DB_DRIVER=ODBC Driver 17 for SQL Server
```

### Step 3 — Prepare the News Source

- **FinSen:** use the included raw CSV and precomputed scores for the historical baseline.
- **NYT:** run **`modelling/notebooks/learning/00_NYT_News_FinBERT_and_DB.ipynb`**. It downloads and filters Archive API metadata, audits coverage, runs FinBERT, creates monthly indicators, and optionally loads the database.

### Step 4 — Run the Five Learning Notebooks

Begin with:

- **`modelling/notebooks/learning/01_Data_and_Inflation.ipynb`** — Explains the database/CSV roles, CPI, month-over-month inflation, year-over-year inflation, the target, and possible predictors.
- **`modelling/notebooks/learning/02_AR1_Baseline.ipynb`** — After Part 1, explains a one-month lag, a fixed training/test split, the AR(1) equation, forecasts, RMSE, MAE, and mean error.
- **`modelling/notebooks/learning/03_ARX_with_News.ipynb`** — Adds lagged news sentiment to AR(1) and compares both models fairly using the same test months and three error measures.
- **`modelling/notebooks/learning/04_HMM.ipynb`** — First explains two historical US inflation regimes using year-over-year inflation, then compares the forecasting HMM with AR(1) using three error measures.
- **`modelling/notebooks/learning/05_AR_HMM.ipynb`** — Combines hidden states with state-specific AR(1) equations and compares AR(1), HMM, and AR-HMM using identical test months, RMSE, MAE, and mean error.

### Step 5 — Launch the Dashboard

The dashboard reads its database connection from the same project-root `.env` file.

```bash
streamlit run modelling/streamlit_app.py
```

---

## Documentation

| Document | Description |
|---|---|
| [`DMP.md`](database/metadata/DMP.md) | Data Management Plan — scope, architecture, governance, security |
| [`data_dictionary.json`](database/metadata/data_dictionary.json) | Field-level metadata for all database tables |
| [`data_lineage.jsonld`](database/metadata/data_lineage.jsonld) | Source and transformation lineage |
| [`dataset_metadata.jsonld`](database/metadata/dataset_metadata.jsonld) | Descriptive dataset metadata |
| [`ER.png`](database/docs/ER.png) | Entity-Relationship diagram |

---

## Future Work

- Validate NYT coverage and relevance against the historical FinSen baseline.
- Re-run the ARX experiment with NYT sentiment and compare all three error measures on the same test months.
- Add the AR-HMM-with-news learning notebook.
- Understand and document each model before adding the next one.
- Extend the understood US workflow to Eurostat/ECB data for the euro area.
- Consider richer news indicators only after the generic FinBERT baseline is understood.

---

## License

Project code, metadata, and eligible derived outputs are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Upstream data remains subject to the terms of FRED, Kaggle, The New York Times, Eurostat, and the original data providers.
