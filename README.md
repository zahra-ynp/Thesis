# Inflation Regime Forecasting with AR-HMM and News Sentiment

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQL%20Server-red.svg)](https://www.microsoft.com/sql-server/)

M.Sc. thesis by **Zahra Younes Pour Langaroudi**, University of Trieste.

This project forecasts inflation and identifies inflation regimes using AR, HMM, AR-HMM, and related benchmark models. It combines macroeconomic data with FinBERT sentiment extracted from financial news. The current experiments cover the United States and the European Union.

## Project Components

- **Database:** normalized SQL Server schemas, ingestion scripts, analytics views, and metadata.
- **NLP:** FinBERT article scoring and monthly sentiment aggregation.
- **Forecasting:** chronological one-month-ahead experiments with separate model-selection and final-test periods.
- **Dashboard:** interactive access to the processed data and model results.

## Data

| Input | Source | Use |
|---|---|---|
| CPI (`CPIAUCSL`) | FRED | Inflation target |
| Unemployment (`UNRATE`) | FRED | Future macro extension |
| Industrial production (`INDPRO`) | FRED | Future macro extension |
| WTI oil (`MCOILWTICO`) | FRED | Future macro extension |
| FinSen news | Kaggle | Historical sentiment baseline |
| NYT news | Archive API | Alternative US news source |
| EU-27 HICP, energy HICP, energy PPI, unemployment, industrial production | Eurostat | European target and macro indicators |
| Guardian news | Guardian Open Platform | European sentiment and subject-risk indicators |

FinBERT calculates monthly net sentiment as:

```text
sentiment = P(positive) - P(negative)
```

Each experiment selects a named news source explicitly; FinSen and NYT are not averaged together. Predictor timing is explicit: lagged inputs support forecasts, while target-month sentiment supports month-end nowcasts. SQL Server is the primary store, while exported experiment datasets provide auditable, reproducible snapshots.

## Models

| Model | Predictors | Purpose |
|---|---|---|
| AR(p) | Inflation lags 1 through p | Forecast baseline; p = 1–6 in the lag experiment |
| ARX | Inflation lag 1, sentiment lag 1 | Simple sentiment test |
| HMM | Hidden inflation state | Regime identification and forecast |
| AR-HMM (2 states) | State-specific AR coefficients | Regime-dependent forecast; p = 1–6 tested |
| AR-HMM (4 states) | Inflation lag 1 | Low, Moderate, High, and Crisis regime experiment |
| AR-HMM + sentiment | AR-HMM inputs, sentiment lag 1 | Regime-dependent sentiment test |
| LASSO | Inflation lags 1–6 | Regularized forecast |
| LASSO + sentiment | Inflation and sentiment lags 1–6 | Regularized sentiment test |

Forecasts are compared on common out-of-sample dates using RMSE, MAE, and mean error. Lower RMSE and MAE are better; mean error indicates systematic underprediction or overprediction.

## Current Status

- The US experiments compare standard AR and regime-switching models with generic and subject-specific news sentiment.
- The European replication covers 2007–2025 using EU-27 Eurostat data and 29,080 Guardian articles.
- European model selection uses rolling validation over 2017–2021; the final evaluation remains 2022–2025.
- The selected European structure is a two-state AR-HMM(1). Seasonal adjustment reduces its RMSE from 0.4658 to 0.4331; adding all subject risks gives 0.4396.
- The fixed European comparison selects energy PPI lag 1 in pre-2022 validation, but the seasonal AR-HMM baseline remains best over 2022–2025. Energy-risk sentiment slightly improves 2022 alone but does not improve the full test period or add value when combined with energy PPI.
- In the matched European machine-learning benchmark, every model uses inflation lag 1. The two-state AR-HMM is essentially tied with linear AR and outperforms Random Forest and gradient boosting; adding macro and sentiment gives AR-HMM the lowest final-period error but not the lowest validation error.

## Historical experiments (2017–2023)

The following archived results used January 2017 through December 2023. Notebook 02 now runs the 2022–2025 comparison described above.

| Experiment | Result |
|---|---|
| State count | AR(1): 0.2641 RMSE; 2-state AR-HMM: 0.2766; 4-state AR-HMM: 0.2771 |
| Lag order | The best 2-state AR-HMM was p=1 with 0.2779 RMSE, versus 0.2645 for AR(1) |
| YoY robustness | 2-state AR-HMM(p=2): 0.3962 RMSE; AR(2): 0.4047; naïve: 0.4376 |
| YoY macro | Best AR: p=2 + oil + unemployment, 0.3811 RMSE; best AR-HMM: p=2 + oil, 0.3849 |
| YoY sentiment | Sentiment increased RMSE in all four tested baseline and best-macro specifications |

The four-state specification did not improve on the two-state model. In the lag-order comparison, none of the 2-state AR-HMM models with p = 1–6 beat the matching AR(p) model. For p=1, the absolute RMSE difference is 0.0133 and the AR-HMM improvement over AR is -5.04%. The negative sign means the AR-HMM RMSE is 5.04% higher than the AR RMSE.

For YoY inflation, AR-HMM(p=2) improves on AR(2) by 2.12% and also beats the naïve previous-month YoY forecast. YoY is calculated directly from CPI as `100 * (CPI_t / CPI_t-12 - 1)`.

The YoY macro experiment tests all 16 combinations of oil, unemployment, industrial production, and the Fed rate for both AR and two-state AR-HMM. AR-HMM beats AR for some matching inputs, with the largest gain of 1.84% for industry alone, but the best overall model is AR(2) with oil and unemployment. Adding mean FinBERT sentiment does not improve the YoY baseline or best-macro models.

The experiments use their own common training samples, so results should be compared within each experiment. Outputs are stored under `data/processed/model_results/` in `arhmm_four_states/`, `arhmm_lag_orders/`, `arhmm_yoy_lag_orders/`, `yoy_macro/`, and `yoy_sentiment/`, and are retained as historical outputs.

## Repository Structure

```text
database/
  scripts/                 SQL Server setup and views
  docs/                    ER diagram
  metadata/                Data management and lineage
modelling/
  config.py                Environment and database connection
  data_pipeline.py         Shared experiment data preparation
  news_pipeline.py         NYT and FinBERT helpers
  sentiment.py             Shared sentiment features
  models.py                Shared forecasting implementations
  evaluation.py            Metrics and comparison checks
  run_experiments.py       Repeatable experiment runner
  europe_pipeline.py       Guardian and Eurostat replication pipeline
  notebooks/               Three active thesis notebooks
  notebooks/archive/       Preserved historical notebooks
  streamlit_app.py         Dashboard
data/
  raw/                     Source data
  processed/modelling_datasets/ Prepared datasets and metadata used by models
  processed/model_results/ US model result bundles
  processed/europe/        European results
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create the database

Create a SQL Server database named `MacroSentimentDB`, then run the files under `database/scripts/` in numerical order.

### 3. Configure the environment

Copy `.env.example` to `.env` in the project root and set:

```text
FRED_API_KEY=your-fred-key
NYT_API_KEY=your-nyt-key
GUARDIAN_API_KEY=your-guardian-key
DB_SERVER=your-server
DB_DATABASE=MacroSentimentDB
DB_DRIVER=ODBC Driver 17 for SQL Server
```

## Notebook Order

Use the three active notebooks under `modelling/notebooks/` in this order:

| Part | Notebook | Main output |
|---:|---|---|
| 01 | `01_Data_and_Sentiment_Analysis.ipynb` | Data provenance, coverage, and current FinBERT sentiment analysis |
| 02 | `02_Forecasting_and_Regime_Results.ipynb` | Existing forecast comparison and regime interpretation |
| 03 | `03_European_Union_Replication.ipynb` | EU-27 Guardian/Eurostat replication and robustness results |
| 04 | `04_Machine_Learning_Benchmarks.ipynb` | European Random Forest and gradient-boosting benchmarks |

The old learning notebooks are preserved under `modelling/notebooks/archive/`. Reusable logic is no longer defined inside notebooks. To check inputs without fitting anything, run:

```bash
python -m modelling.run_experiments --model all --check-only
```

To re-estimate all existing models, omit `--check-only`. Results are written as a coherent bundle under `data/processed/model_results/baseline/`.

## Dashboard

```bash
streamlit run modelling/streamlit_app.py
```

The dashboard uses the database connection defined in `.env`.

## Documentation

- [Data Management Plan](database/metadata/DMP.md)
- [Data dictionary](database/metadata/data_dictionary.json)
- [Data lineage](database/metadata/data_lineage.jsonld)
- [Dataset metadata](database/metadata/dataset_metadata.jsonld)
- [Entity-Relationship diagram](database/docs/ER.png)

## Selected References

- Chahad, M., Hofmann-Drahonsky, A.-C., Meunier, B., Page, A., and Tirpák, M. (2022). [What explains recent errors in the inflation projections of Eurosystem and ECB staff?](https://www.ecb.europa.eu/press/economic-bulletin/focus/2022/html/ecb.ebbox202203_05~6d1fb8f5b0.en.html)
- Kuik, F., Adolfsen, J. F., Lis, E. M., and Meyler, A. (2022). [Energy price developments in and out of the COVID-19 pandemic](https://www.ecb.europa.eu/press/economic-bulletin/articles/2022/html/ecb.ebart202204_01~7b32d31b29.en.html)
- Bańbura, M., Bobeica, E., Giammaria, A., Porqueddu, M., and van Spronsen, J. (2025). [A new model to forecast energy inflation in the euro area](https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp3062~abe560353f.en.pdf). ECB Working Paper No. 3062.

## License

Project code, metadata, and eligible derived outputs are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Source datasets remain subject to their providers' terms.
