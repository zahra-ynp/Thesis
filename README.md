# Inflation Regime Forecasting with AR-HMM and NLP Sentiment Analysis

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQL%20Server-red.svg)](https://www.microsoft.com/sql-server/)

## Abstract

Development and implementation of an advanced probabilistic model (Autoregressive Hidden Markov Model) for forecasting macroeconomic scenarios, with a specific focus on inflation regimes. The project integrates the analysis of historical macroeconomic data with Natural Language Processing (NLP) techniques applied to financial news in order to extract real-time sentiment indicators.

**Author:** Zahra Younes Pour Langaroudi

**Institution:** University of Trieste — M.Sc. in Data Science and Artificial Intelligence

---

## Project Overview

This thesis builds an end-to-end pipeline that connects three complementary areas:

1. **Data Infrastructure** — A normalized SQL Server database that ingests macroeconomic indicators from FRED and financial news from public datasets, scores articles with FinBERT, and exposes analytics-ready views through a Streamlit dashboard.
2. **NLP Sentiment Extraction** — FinBERT (a BERT model fine-tuned on financial text) classifies each news article as positive, negative, or neutral. Article-level scores are aggregated into a monthly sentiment signal.
3. **Regime Detection** — Hidden Markov Models (HMM) and Autoregressive HMMs (AR-HMM) identify latent inflation regimes from the combined macroeconomic and sentiment features, using incremental BIC-guided feature selection.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                    │
│  ┌──────────┐  ┌─────────────────┐  ┌───────────┐  ┌──────────────┐   │
│  │ FRED API │  │ Kaggle (FinSen) │  │ HuggingFace│  │   Eurostat   │   │
│  └────┬─────┘  └───────┬─────────┘  └─────┬──────┘  └──────┬───────┘   │
└───────┼────────────────┼──────────────────┼──────────────────┼──────────┘
        │                │                  │                  │
        ▼                ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATABASE LAYER  (database/)                          │
│  ┌──────────────┐  ┌────────────────────────────────────────────────┐  │
│  │  SQL Scripts  │  │          SQL Server Database                   │  │
│  │  (schemas,    │  │  ┌──────┐  ┌──────────┐  ┌──────┐            │  │
│  │   tables,     │  │  │ core │  │ analytics│  │ auth │            │  │
│  │   views)      │  │  └──────┘  └──────────┘  └──────┘            │  │
│  └──────────────┘  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              NLP & FEATURE ENGINEERING  (modelling/)                    │
│  ┌─────────────────────────┐     ┌────────────────────────────────┐    │
│  │  01_Data_Prep.ipynb     │     │   FinBERT Sentiment Scoring    │    │
│  │  • Data cleaning & ETL  │────▶│   score = P(pos) - P(neg)      │    │
│  │  • DB ingestion         │     │   Monthly aggregation          │    │
│  └─────────────────────────┘     └────────────────────────────────┘    │
│                                     ┌────────────────────────────────┐ │
│                                     │   Streamlit Dashboard          │ │
│                                     │   streamlit_app.py             │ │
│                                     └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REGIME DETECTION  (modelling/)                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  02_AR_HMM_and_Evaluation.ipynb                                 │   │
│  │  • HMM baseline on Inflation_MoM                               │   │
│  │  • AR-HMM with autoregressive dynamics                         │   │
│  │  • Incremental feature selection (BIC-guided)                   │   │
│  │  • Regime probability estimation & visualization                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
Thesis/
│
├── README.md                       ← This file
├── .gitignore
├── requirements.txt                ← Merged Python dependencies
│
├── database/                       ← Data infrastructure & schema definitions
│   ├── scripts/                    ← SQL schema & setup (01–06_*.sql)
│   ├── docs/
│   │   └── ER.png                  ← Entity-Relationship diagram
│   └── metadata/
│       ├── DMP.md                  ← Data Management Plan
│       ├── data_dictionary.json    ← Field-level metadata
│       ├── data_lineage.jsonld     ← Source & transformation lineage
│       └── dataset_metadata.jsonld ← Descriptive dataset metadata
│
├── modelling/                      ← Data pipeline, NLP, and probabilistic models
│   ├── streamlit_app.py            ← Interactive analytics dashboard
│   └── notebooks/
│       └── US/
│           ├── 01_Data_Prep.ipynb              ← Data cleaning, ETL & FinBERT scoring
│           └── 02_AR_HMM_and_Evaluation.ipynb  ← HMM & AR-HMM regime detection
│
├── data/                           ← Shared data layer
│   ├── raw/
│   │   └── news/                   ← FinSen_US_Categorized_Timestamp.csv
│   └── processed/
│       ├── US/                     ← us_macro_2007_2023.csv
│       └── finbert_article_scores.csv  ← Precomputed FinBERT checkpoint
│
└── plots/                          ← Model output visualisations
    └── US/                         ← Regime probabilities, BIC comparisons, etc.
```

---

## Data Sources

| Source | Provider | Description | Period |
|---|---|---|---|
| CPI (`CPIAUCSL`) | FRED | Consumer Price Index for All Urban Consumers | 2007–2023 |
| Unemployment (`UNRATE`) | FRED | Civilian Unemployment Rate | 2007–2023 |
| Interest Rate (`FEDFUNDS`) | FRED | Effective Federal Funds Rate | 2007–2023 |
| Industrial Production (`INDPRO`) | FRED | Industrial Production Index | 2007–2023 |
| Financial News (US) | Kaggle ([FinSen](https://www.kaggle.com/datasets/eaglewhl/finsen-financial-sentiment-dataset)) | ~15,000 categorized financial news articles | 2007–2023 |
| Financial News (Multi) | HuggingFace ([Financial News Multisource](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource)) | Additional financial news corpus | 2007–2023 |
| European Macro | Eurostat | HICP and European macroeconomic indicators | 2007–2023 |

---

## Methodology

### 1. NLP Sentiment Extraction — FinBERT

Each financial news article is scored using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT-based model fine-tuned on financial text. FinBERT outputs class probabilities for *positive*, *negative*, and *neutral* labels. A net sentiment score is computed as:

```
score = P(positive) − P(negative)     ∈ [−1, +1]
```

Article-level scores are averaged within each calendar month to produce a single `News_Sentiment` signal, which is merged with the macroeconomic indicators.

### 2. Regime Detection — HMM & AR-HMM

Latent inflation regimes are identified using two classes of Markov-Switching models fitted to the month-over-month inflation series (`Inflation_MoM`):

- **HMM (Hidden Markov Model):** Assumes observations are conditionally independent given the hidden state.
- **AR-HMM (Autoregressive HMM):** Extends HMM by modelling autoregressive dynamics within each regime, capturing serial dependence in inflation.

### 3. Incremental Feature Selection (BIC-Guided)

Starting from a baseline model (inflation only), macroeconomic indicators are added sequentially. The Bayesian Information Criterion (BIC) is recorded at each step to quantify each variable's marginal contribution:

| Step | Model Specification |
|---|---|
| M1 | Baseline — no exogenous variables |
| M2 | + News Sentiment (FinBERT monthly average) |
| M3 | + Unemployment Rate |
| M4 | + Industrial Production MoM |

The specification with the lowest BIC is selected as the best model within each family (HMM and AR-HMM).

### 4. Database Architecture

The data infrastructure uses a normalized SQL Server schema:

- **`core` schema:** Regions, agencies, indicators, monthly observations, and daily news
- **`analytics` schema:** Aggregated monthly views with month-over-month changes
- **`auth` schema:** Users, roles, companies, and scoped permissions

See the [ER diagram](database/docs/ER.png) and [Data Management Plan](database/metadata/DMP.md) for full documentation.

---

## Quick Start

### Prerequisites

- Python 3.8 or later
- Microsoft SQL Server (with ODBC Driver 17)
- A free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

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
```

A precomputed FinBERT checkpoint is included at `data/processed/finbert_article_scores.csv`.

### Step 2 — Run the Modelling Pipeline

Notebooks must be run in order:

1. **`modelling/notebooks/US/01_Data_Prep.ipynb`** — Cleans and ingests raw data into SQL Server, downloads macroeconomic indicators from FRED, and scores news articles with FinBERT. Set `FRED_API_KEY` and configure the database connection at the top of the notebook.
2. **`modelling/notebooks/US/02_AR_HMM_and_Evaluation.ipynb`** — Fits HMM and AR-HMM models with incremental feature selection. Produces regime probability plots and BIC comparisons.

### Step 3 — Launch the Dashboard

Configure the database connection if the defaults do not match your environment:

```text
DB_SERVER=your-server
DB_DATABASE=MacroSentimentDB
DB_DRIVER=ODBC Driver 17 for SQL Server
```

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

- **European analysis:** Extend the AR-HMM framework to European inflation using Eurostat HICP data and multilingual financial news.
- **Real-time pipeline:** Connect the sentiment scoring to a live news feed for near-real-time regime probability updates.
- **Model extensions:** Explore higher-order AR-HMM, Bayesian HMM, and regime-switching VAR models.

---

## License

Project code, metadata, and eligible derived outputs are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Upstream data remains subject to the terms of FRED, Eurostat, Kaggle, Hugging Face, and the original data providers.
