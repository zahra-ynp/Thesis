# European Inflation Regime Forecasting with AR-HMM and News Sentiment

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQL%20Server-red.svg)](https://www.microsoft.com/sql-server/)

M.Sc. thesis by **Zahra Younes Pour Langaroudi**, University of Trieste.

This project studies monthly inflation in the **European Union (EU-27)** using autoregressive models, hidden inflation regimes, macroeconomic indicators, and financial-news sentiment. It combines Eurostat data with FinBERT scores from **29,080 Guardian articles covering 2007–2025**, and evaluates whether regime switching and news information improve inflation predictions.

The European results show that a two-state AR-HMM is competitive with linear autoregression. The value of additional predictors depends on the information set: energy-risk sentiment alone does not improve accuracy over the full final period, while a broader macroeconomic and sentiment specification achieves the lowest final-period RMSE in the matched machine-learning comparison. Earlier model and data checks were carried out as exploratory analysis to inform the European study.

## Data and Sentiment

| Input | Source | Role |
|---|---|---|
| All-items HICP | Eurostat | EU-27 inflation target |
| Energy HICP and energy producer price index (PPI) | Eurostat | Energy-price predictors |
| Unemployment rate and industrial production | Eurostat | Labour-market and activity predictors |
| News headlines, introductory text, and keyword tags | Guardian Open Platform | Generic sentiment and subject-risk indicators |

Monthly inflation is calculated from the HICP index:

```text
Inflation_MoM = 100 × (HICP_t / HICP_(t−1) − 1)
```

FinBERT scores article headlines and introductory text. Monthly net sentiment aggregates `P(positive) − P(negative)`. Subject-risk indicators measure news coverage of **energy, geopolitics, and supply chains**, weighted by each article's negative-sentiment probability. They are text-based indicators rather than direct measures of inflation expectations.

The European dataset, source metadata, article scores, and saved results are stored under [`data/processed/europe/`](data/processed/europe/).

## Evaluation Design

- **Model selection:** validation over 2017–2021, with rolling folds for the seasonal, macroeconomic, sentiment, and machine-learning comparisons.
- **Final evaluation:** January 2022–December 2025, with separate reporting for 2022 and 2023–2025.
- **Final estimation sample:** data through December 2021; model inputs and settings are selected before the final evaluation period.
- **Seasonality:** calendar-month means are estimated from the relevant training sample, removed before fitting, and added back to predictions. This is a project-specific seasonal adjustment.
- **Information timing:** inflation and macroeconomic predictors are lagged. Specifications using target-month news are **month-end nowcasts**, because that news is unavailable at the start of the month.
- **Metrics:** RMSE, MAE, and mean error on common evaluation dates. RMSE and MAE are expressed in percentage points of monthly inflation; lower values indicate better accuracy.

The macroeconomic experiments use the latest revised Eurostat data, so the evaluation is retrospective rather than a real-time release-vintage exercise. Results from different information sets and experiment configurations should be interpreted separately.

## European Results

### Regime structure and selected inputs

Pre-2022 validation selects a **two-state AR-HMM with one inflation lag** from the tested regime-switching structures. Macroeconomic lag selection identifies **energy PPI at lag 1**, and the sentiment nowcast comparison selects **energy risk**. The fixed comparison below evaluates those inputs individually and together with the seasonal AR-HMM baseline.

| Specification | RMSE 2022–2025 | MAE 2022–2025 | RMSE 2022 | RMSE 2023–2025 |
|---|---:|---:|---:|---:|
| Seasonal AR-HMM(1) baseline | **0.4331** | **0.3263** | 0.6820 | **0.3083** |
| + Energy PPI, lag 1 | 0.4594 | 0.3570 | 0.7010 | 0.3430 |
| + Target-month energy-risk sentiment | 0.4337 | 0.3288 | **0.6749** | 0.3145 |
| + Energy PPI and energy-risk sentiment | 0.4743 | 0.3586 | 0.7428 | 0.3406 |

Energy PPI is preferred in rolling validation, but the baseline has the lowest RMSE in this final-period comparison. Energy-risk sentiment slightly reduces RMSE in 2022 alone; it does not improve the full 2022–2025 period or add predictive value when combined with energy PPI. Errors are substantially larger during the 2022 inflation surge than during 2023–2025.

Source: [`final_europe_comparison` in the saved European results](data/processed/europe/eu27_guardian_results_2022_2025.json). Earlier structure and seasonal checks are retained in separate sections of the same file.

### Matched machine-learning benchmarks

The benchmark compares linear AR, two-state AR-HMM, Random Forest, and Gradient Boosting using the same inputs within each information set. Every model includes seasonally adjusted inflation at lag 1. The broader information set adds unemployment, industrial production, energy HICP, and energy PPI at validation-selected lags, plus target-month generic sentiment and all three subject-risk indicators.

| Model | Validation RMSE: lag 1 only | Final RMSE: lag 1 only | Validation RMSE: macro + sentiment | Final RMSE: macro + sentiment |
|---|---:|---:|---:|---:|
| Linear AR | **0.2414** | **0.4331** | **0.2356** | 0.4556 |
| Two-state AR-HMM | 0.2416 | **0.4331** | 0.2432 | **0.3959** |
| Random Forest | 0.2515 | 0.4466 | 0.2476 | 0.4615 |
| Gradient Boosting | 0.2441 | 0.4523 | 0.2446 | 0.4665 |

Validation covers 2017–2021; final RMSE covers 2022–2025. With inflation lag 1 alone, AR-HMM and linear AR are effectively tied at four decimal places and outperform both tree models. With the broader macroeconomic and sentiment inputs, AR-HMM achieves the lowest final-period RMSE, but linear AR has the lowest validation RMSE. This richer specification is a month-end nowcast and differs from the energy-only comparison above. Its result does not establish that sentiment alone caused the improvement or that AR-HMM is consistently superior across periods.

## Repository Structure

```text
database/
  scripts/                 SQL Server schemas, ingestion, and analytics views
  docs/                    Entity-Relationship diagram
  metadata/                Data dictionary, management plan, and lineage
modelling/
  config.py                Environment and database connection
  data_pipeline.py         Shared data preparation
  news_pipeline.py         News acquisition and FinBERT helpers
  sentiment.py             Monthly sentiment and subject-risk features
  models.py                Shared forecasting implementations
  evaluation.py            Forecast metrics and comparison checks
  run_experiments.py       Earlier exploratory experiment runner
  europe_pipeline.py       Eurostat/Guardian preparation and European experiments
  europe_figures.py        European result visualizations
  notebooks/               Analysis and presentation notebooks
  notebooks/archive/       Historical exploratory notebooks
  streamlit_app.py         Database-backed dashboard
data/
  raw/europe/              Cached Eurostat source responses
  processed/europe/        European dataset, sentiment scores, and results
  processed/modelling_datasets/ Earlier prepared experiment snapshots
  processed/model_results/ Earlier exploratory result bundles
```

## Setup and Reproduction

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The European notebooks read the saved CSV and JSON files directly. Start with notebook 03 to inspect the European analysis and notebook 04 for the matched machine-learning benchmarks.

| Notebook | Purpose |
|---|---|
| [`01_Data_and_Sentiment_Analysis.ipynb`](modelling/notebooks/01_Data_and_Sentiment_Analysis.ipynb) | Earlier exploratory data and sentiment checks |
| [`02_Forecasting_and_Regime_Results.ipynb`](modelling/notebooks/02_Forecasting_and_Regime_Results.ipynb) | Earlier exploratory forecasting and regime checks |
| [`03_European_Union_Replication.ipynb`](modelling/notebooks/03_European_Union_Replication.ipynb) | Main EU-27 analysis, validation, and robustness comparisons |
| [`04_Machine_Learning_Benchmarks.ipynb`](modelling/notebooks/04_Machine_Learning_Benchmarks.ipynb) | Matched European model comparison and broader information sets |

To rebuild the European monthly dataset from saved article scores and cached Eurostat inputs:

```bash
python -m modelling.europe_pipeline --stage build
```

To re-estimate the European experiments in dependency order:

```bash
python -m modelling.europe_pipeline --stage models
python -m modelling.europe_pipeline --stage structure
python -m modelling.europe_pipeline --stage seasonal
python -m modelling.europe_pipeline --stage macro
python -m modelling.europe_pipeline --stage macro-lags
python -m modelling.europe_pipeline --stage sentiment-nowcast
python -m modelling.europe_pipeline --stage final-comparison
```

These commands update the saved European result bundle. Run notebook 04 separately to reproduce its machine-learning tables. Numerical optimization can produce variation between AR-HMM fits.

To collect the news and regenerate article scores, copy `.env.example` to `.env`, add `GUARDIAN_API_KEY`, then run the `download` and `score` stages before `build`. The scoring stage defaults to offline mode and requires the `ProsusAI/finbert` model to be cached locally. Downloads can resume from saved progress when API quotas interrupt collection.

### Database and dashboard

For the database workflow, create a SQL Server database named `MacroSentimentDB`, run the files under `database/scripts/` in numerical order, and configure the database settings in `.env`:

```text
DB_SERVER=your-server
DB_DATABASE=MacroSentimentDB
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_AUTH_MODE=trusted
```

Launch the database-backed dashboard with:

```bash
streamlit run modelling/streamlit_app.py
```

## Documentation

- [Data Management Plan](database/metadata/DMP.md)
- [Data dictionary](database/metadata/data_dictionary.json)
- [Data lineage](database/metadata/data_lineage.jsonld)
- [Dataset metadata](database/metadata/dataset_metadata.jsonld)
- [Entity-Relationship diagram](database/docs/ER.png)

## License

Project code, metadata, and eligible derived outputs are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Source datasets remain subject to their providers' terms.
