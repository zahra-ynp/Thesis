# Inflation Regime Forecasting with AR-HMM and News Sentiment

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-SQL%20Server-red.svg)](https://www.microsoft.com/sql-server/)

M.Sc. thesis by **Zahra Younes Pour Langaroudi**, University of Trieste.

This project forecasts US inflation and identifies inflation regimes using AR(1), ARX, HMM, AR-HMM, and LASSO. It combines monthly macroeconomic data from FRED with FinBERT sentiment extracted from financial news. Data are managed in SQL Server and exposed through a Streamlit dashboard.

## Project Components

- **Database:** normalized SQL Server schemas, ingestion scripts, analytics views, and metadata.
- **NLP:** FinBERT article scoring and monthly sentiment aggregation.
- **Forecasting:** chronological one-month-ahead experiments with a common 2017–2023 test period.
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

FinBERT calculates monthly net sentiment as:

```text
sentiment = P(positive) - P(negative)
```

The database source registry selects the active news source; FinSen and NYT are not averaged together. Forecasts use only lagged information to prevent look-ahead bias. The notebooks read from SQL Server by default and support the processed CSV as a fallback.

## Models

| Model | Predictors | Purpose |
|---|---|---|
| AR(1) | Inflation lag 1 | Forecast baseline |
| ARX | Inflation lag 1, sentiment lag 1 | Simple sentiment test |
| HMM | Hidden inflation state | Regime identification and forecast |
| AR-HMM | State-specific inflation dynamics | Regime-dependent forecast |
| AR-HMM + sentiment | AR-HMM inputs, sentiment lag 1 | Regime-dependent sentiment test |
| LASSO | Inflation lags 1–6 | Regularized forecast |
| LASSO + sentiment | Inflation and sentiment lags 1–6 | Regularized sentiment test |

Forecasts are compared on common out-of-sample dates using RMSE, MAE, and mean error. Lower RMSE and MAE are better; mean error indicates systematic underprediction or overprediction.

## Repository Structure

```text
database/
  scripts/                 SQL Server setup and views
  docs/                    ER diagram
  metadata/                Data management and lineage
modelling/
  notebooks/learning/      Step-by-step learning workflow
  config.py                Environment and database connection
  news_pipeline.py         NYT and FinBERT helpers
  streamlit_app.py         Dashboard
data/
  raw/                     Source data
  processed/US/            Processed US dataset
  processed/model_results/ One CSV and one JSON per model notebook
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
DB_SERVER=your-server
DB_DATABASE=MacroSentimentDB
DB_DRIVER=ODBC Driver 17 for SQL Server
```

## Notebook Order

Run the notebooks under `modelling/notebooks/learning/` in this order:

| Part | Notebook | Main output |
|---:|---|---|
| 00 | `00_NYT_News_FinBERT_and_DB.ipynb` | Optional NYT acquisition, FinBERT scoring, and database load |
| 01 | `01_Data_and_Inflation.ipynb` | Data and inflation explanation |
| 02 | `02_AR1_Baseline.ipynb` | `ar1_forecasts.csv`, `ar1_metrics.json` |
| 03 | `03_ARX_with_News.ipynb` | `arx_forecasts.csv`, `arx_metrics.json` |
| 04 | `04_HMM.ipynb` | `hmm_forecasts.csv`, `hmm_metrics.json` |
| 05 | `05_AR_HMM.ipynb` | `ar_hmm_forecasts.csv`, `ar_hmm_metrics.json` |
| 06 | `06_LASSO.ipynb` | `lasso_forecasts.csv`, `lasso_metrics.json` |

Parts 05 and 06 store their inflation-only and sentiment variants together. All result files are written to `data/processed/model_results/`.

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

## Next Steps

- Add oil-price growth, unemployment changes, and industrial-production growth to the forecasting experiments.
- Validate NYT sentiment against the FinSen baseline.
- Consider Ridge and Random Forest after the macroeconomic extension.
- Extend the validated workflow to the euro area using Eurostat and ECB data.
- Connect the final forecasts and regime probabilities to the dashboard.

## License

Project code, metadata, and eligible derived outputs are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Source datasets remain subject to their providers' terms.
