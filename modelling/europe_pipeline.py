from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
from pathlib import Path
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
import requests

from modelling.evaluation import forecast_metrics
from modelling.models import ForecastResult, fit_ar_hmm, fit_ols_forecast
from modelling.news_pipeline import score_with_finbert
from modelling.sentiment import aggregate_monthly_sentiment, aggregate_monthly_subject_risk


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/europe"
PROCESSED = ROOT / "data/processed/europe"
ARTICLES = PROCESSED / "guardian_eu_articles_2007_2025.csv"
PROGRESS = PROCESSED / "guardian_eu_progress.json"
SCORES = PROCESSED / "guardian_eu_finbert_scores_2007_2025.csv"
DATASET = PROCESSED / "eu27_guardian_monthly_experiment.csv"
RESULTS = PROCESSED / "eu27_guardian_results_2022_2025.json"

GUARDIAN_QUERY = (
    "(eurozone OR Europe) AND "
    "(inflation OR oil OR energy OR unemployment OR war OR sanctions OR supply-chain OR shipping)"
)

EUROSTAT_SERIES = {
    "HICP": ("prc_hicp_midx", {"freq": "M", "unit": "I15", "coicop": "CP00", "geo": "EU27_2020"}),
    "HICP_Energy": ("prc_hicp_midx", {"freq": "M", "unit": "I15", "coicop": "NRG", "geo": "EU27_2020"}),
    "Unemployment_Rate": ("une_rt_m", {"freq": "M", "unit": "PC_ACT", "sex": "T", "age": "TOTAL", "s_adj": "SA", "geo": "EU27_2020"}),
    "Industrial_Production": ("sts_inpr_m", {"freq": "M", "unit": "I21", "s_adj": "SCA", "nace_r2": "B-D", "geo": "EU27_2020"}),
    "PPI_Energy": ("sts_inppd_m", {"freq": "M", "indic_bt": "PRC_PRR_DOM", "unit": "I21", "s_adj": "NSA", "nace_r2": "MIG_NRG", "geo": "EU27_2020"}),
}


def _clean_html(value: str | None) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def download_guardian(api_key: str) -> pd.DataFrame:
    if not api_key:
        raise ValueError("GUARDIAN_API_KEY is missing from .env")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    completed = set()
    if PROGRESS.exists():
        completed = set(json.loads(PROGRESS.read_text(encoding="utf-8")).get("completed_months", []))
    existing = pd.read_csv(ARTICLES, dtype=str).fillna("") if ARTICLES.exists() else pd.DataFrame()
    rows = []
    calls = 0
    for month in pd.period_range("2007-01", "2025-12", freq="M"):
        month_code = str(month)
        if month_code in completed:
            continue
        start = month.start_time.strftime("%Y-%m-%d")
        end = month.end_time.strftime("%Y-%m-%d")
        page = 1
        while True:
            response = requests.get(
                "https://content.guardianapis.com/search",
                params={
                    "api-key": api_key,
                    "q": GUARDIAN_QUERY,
                    "section": "business|world",
                    "from-date": start,
                    "to-date": end,
                    "page-size": 200,
                    "page": page,
                    "order-by": "oldest",
                    "show-fields": "headline,trailText",
                    "show-tags": "keyword",
                },
                timeout=90,
            )
            if response.status_code == 429:
                raise RuntimeError("Guardian daily API quota reached; rerun tomorrow to resume.")
            response.raise_for_status()
            payload = response.json()["response"]
            for item in payload["results"]:
                fields = item.get("fields", {})
                title = _clean_html(fields.get("headline") or item.get("webTitle"))
                trail = _clean_html(fields.get("trailText"))
                rows.append({
                    "external_article_id": item["id"],
                    "published_at": item["webPublicationDate"],
                    "news_date": item["webPublicationDate"][:10],
                    "title": title,
                    "abstract": trail,
                    "keyword_values": " | ".join(tag.get("webTitle", "") for tag in item.get("tags", [])),
                    "article_url": item.get("webUrl", ""),
                    "source": "The Guardian",
                    "section_name": item.get("sectionName", ""),
                    "text_for_finbert": "\n\n".join(part for part in (title, trail) if part),
                })
            calls += 1
            if page >= payload["pages"]:
                break
            page += 1
            if calls >= 480:
                raise RuntimeError("Stopped before the daily Guardian quota; rerun to resume.")
            time.sleep(1.05)
        completed.add(month_code)
        print(f"Guardian {month_code}: {payload['total']} articles", flush=True)
        if month.month in (6, 12) or month == pd.Period("2025-12", freq="M"):
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            combined = combined.drop_duplicates("external_article_id").sort_values("published_at")
            combined.to_csv(ARTICLES, index=False)
            PROGRESS.write_text(json.dumps({"completed_months": sorted(completed)}, indent=2), encoding="utf-8")
            existing, rows = combined, []
        time.sleep(1.05)
    return pd.read_csv(ARTICLES, parse_dates=["published_at", "news_date"])


def _jsonstat_series(payload: dict, name: str) -> pd.DataFrame:
    time_index = payload["dimension"]["time"]["category"]["index"]
    values = payload.get("value", {})
    rows = [(pd.Period(code, freq="M").to_timestamp(), values.get(str(position)))
            for code, position in time_index.items()]
    frame = pd.DataFrame(rows, columns=["Date", name]).sort_values("Date")
    frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def download_eurostat() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    macro = pd.DataFrame({"Date": pd.date_range("2007-01-01", "2025-12-01", freq="MS")})
    for name, (dataset, filters) in EUROSTAT_SERIES.items():
        path = RAW / f"{dataset}_{name}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            response = requests.get(
                f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}",
                params={"lang": "en", "sinceTimePeriod": "2007-01", "untilTimePeriod": "2025-12", **filters},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            path.write_text(json.dumps(payload), encoding="utf-8")
        part = _jsonstat_series(payload, name)
        macro = macro.merge(part, on="Date", how="left", validate="one_to_one")
    if macro[list(EUROSTAT_SERIES)].isna().any().any():
        raise ValueError(f"Incomplete Eurostat coverage: {macro.isna().sum().to_dict()}")
    return macro


def score_guardian(model_name: str = "ProsusAI/finbert") -> pd.DataFrame:
    articles = pd.read_csv(ARTICLES, keep_default_na=False)
    if articles.empty:
        raise ValueError("Download Guardian articles before scoring.")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return score_with_finbert(articles, SCORES, model_name=model_name, batch_size=16,
                              checkpoint_every=200)


def build_dataset() -> pd.DataFrame:
    macro = download_eurostat()
    scored = pd.read_csv(SCORES, parse_dates=["news_date"])
    generic = aggregate_monthly_sentiment(scored, source_code="GUARDIAN")
    subjects = aggregate_monthly_subject_risk(scored, source_code="GUARDIAN")
    news = generic.merge(subjects.drop(columns="News_Source"), on="Date", validate="one_to_one")
    frame = macro.merge(news, on="Date", how="left", validate="one_to_one")
    if frame.News_Sentiment.isna().any():
        raise ValueError("At least one month has no Guardian news coverage.")
    frame["Inflation_MoM"] = frame.HICP.pct_change(fill_method=None) * 100
    frame["Inflation_YoY"] = frame.HICP.pct_change(12, fill_method=None) * 100
    frame["Energy_Inflation_MoM"] = frame.HICP_Energy.pct_change(fill_method=None) * 100
    frame["PPI_Energy_Growth_MoM"] = frame.PPI_Energy.pct_change(fill_method=None) * 100
    frame["Unemployment_Change"] = frame.Unemployment_Rate.diff()
    frame["IP_Growth_MoM"] = frame.Industrial_Production.pct_change(fill_method=None) * 100
    lagged = ["Inflation_MoM", "News_Sentiment", "Sentiment_Median", "Sentiment_Std", "Negative_Share",
              "Geopolitical_Risk", "Energy_Risk", "Supply_Chain_Risk",
              "Unemployment_Change", "IP_Growth_MoM", "Energy_Inflation_MoM",
              "PPI_Energy_Growth_MoM"]
    for column in lagged:
        frame[f"{column}_Lag1"] = frame[column].shift(1)
    for column in ["Unemployment_Change", "IP_Growth_MoM", "Energy_Inflation_MoM",
                   "PPI_Energy_Growth_MoM"]:
        for lag in range(2, 7):
            frame[f"{column}_Lag{lag}"] = frame[column].shift(lag)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATASET, index=False)
    metadata = {
        "region": "EU27_2020",
        "period": ["2007-01", "2025-12"],
        "news_source": "The Guardian Open Platform",
        "guardian_query": GUARDIAN_QUERY,
        "article_count": int(frame.Article_Count.sum()),
        "macro_source": "Eurostat API",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_design": "Train through 2021-12; one-month-lagged predictors; test 2022-01 to 2025-12",
    }
    DATASET.with_name(DATASET.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return frame


def run_models() -> dict:
    from scipy.stats import t as student_t

    frame = pd.read_csv(DATASET, parse_dates=["Date"])
    features = {
        "AR(1)": ["Inflation_MoM_Lag1"],
        "AR(1) + lagged sentiment": ["Inflation_MoM_Lag1", "News_Sentiment_Lag1"],
        "AR(1) + sentiment dispersion": ["Inflation_MoM_Lag1", "Sentiment_Std_Lag1"],
        "AR(1) + negative share": ["Inflation_MoM_Lag1", "Negative_Share_Lag1"],
        "AR(1) + geopolitical risk": ["Inflation_MoM_Lag1", "Geopolitical_Risk_Lag1"],
        "AR(1) + energy risk": ["Inflation_MoM_Lag1", "Energy_Risk_Lag1"],
        "AR(1) + supply-chain risk": ["Inflation_MoM_Lag1", "Supply_Chain_Risk_Lag1"],
        "AR(1) + all subject risks": ["Inflation_MoM_Lag1", "Geopolitical_Risk_Lag1", "Energy_Risk_Lag1", "Supply_Chain_Risk_Lag1"],
        "AR(1) + macro": ["Inflation_MoM_Lag1", "Unemployment_Change_Lag1", "IP_Growth_MoM_Lag1"],
    }
    required = sorted({"Inflation_MoM", *(item for values in features.values() for item in values)})
    sample = frame.loc[frame[required].notna().all(axis=1)].copy()
    results = {"sample": {"training_end": "2021-12-01", "test_start": "2022-01-01",
                           "test_end": "2025-12-01"}, "models": []}

    def save(name: str, family: str, model_features: list[str], result) -> None:
        results["models"].append({
            "model": name,
            "family": family,
            "features": model_features,
            "metrics": forecast_metrics(result.forecasts.Actual, result.forecasts.Forecast),
            "forecasts": result.forecasts.assign(
                Date=pd.to_datetime(result.forecasts.Date).dt.strftime("%Y-%m-%d")
            ).to_dict("records"),
            "parameters": result.parameters,
        })

    test = sample.loc[sample.Date.ge("2022-01-01")]
    naive = ForecastResult("Naive previous month", pd.DataFrame({
        "Date": test.Date, "Actual": test.Inflation_MoM,
        "Forecast": test.Inflation_MoM_Lag1,
    }), {"features": ["Inflation_MoM_Lag1"]})
    save("Naive previous month", "Naive", ["Inflation_MoM_Lag1"], naive)

    for label, model_features in features.items():
        save(label, "AR", model_features, fit_ols_forecast(
            sample, features=model_features, name=label, target="Inflation_MoM", test_start="2022-01-01"
        ))
        ms_name = label.replace("AR(1)", "MS-Variance AR(1)")
        save(ms_name, "MS-Variance", model_features, fit_ar_hmm(
            sample, features=model_features, name=ms_name, target="Inflation_MoM",
            test_start="2022-01-01", n_states=2, switching_intercept=False,
            switching_ar=False, switching_variance=True,
        ))

    for label, model_features in {
        "AR-HMM(1)": features["AR(1)"],
        "AR-HMM(1) + lagged sentiment": features["AR(1) + lagged sentiment"],
        "AR-HMM(1) + sentiment dispersion": features["AR(1) + sentiment dispersion"],
        "AR-HMM(1) + geopolitical risk": features["AR(1) + geopolitical risk"],
        "AR-HMM(1) + energy risk": features["AR(1) + energy risk"],
        "AR-HMM(1) + supply-chain risk": features["AR(1) + supply-chain risk"],
        "AR-HMM(1) + all subject risks": features["AR(1) + all subject risks"],
    }.items():
        save(label, "AR-HMM", model_features, fit_ar_hmm(
            sample, features=model_features, name=label, target="Inflation_MoM",
            test_start="2022-01-01", n_states=2,
        ))

    by_name = {item["model"]: item for item in results["models"]}
    winner = by_name["AR-HMM(1) + all subject risks"]
    winner_errors = np.array([row["Actual"] - row["Forecast"] for row in winner["forecasts"]])
    loss_tests = {}
    for baseline in ["Naive previous month", "AR(1)", "AR-HMM(1)"]:
        baseline_errors = np.array([
            row["Actual"] - row["Forecast"] for row in by_name[baseline]["forecasts"]
        ])
        differential = baseline_errors ** 2 - winner_errors ** 2
        centered = differential - differential.mean()
        n = len(differential)
        variance = centered @ centered / n + centered[1:] @ centered[:-1] / n
        statistic = differential.mean() / np.sqrt(variance / n)
        loss_tests[baseline] = float(2 * student_t.sf(abs(statistic), n - 1))

    seed_results = []
    for seed in [7, 42, 99, 123, 2026]:
        fitted = fit_ar_hmm(
            sample, features=winner["features"], name="robustness", target="Inflation_MoM",
            test_start="2022-01-01", n_states=2, random_seed=seed,
        )
        errors = fitted.forecasts.Actual - fitted.forecasts.Forecast
        seed_results.append({"seed": seed, "rmse": float(np.sqrt(np.mean(errors ** 2))),
                             "log_likelihood": fitted.parameters["log_likelihood"],
                             "converged": fitted.parameters["converged"]})
    results["diagnostics"] = {
        "winner": winner["model"],
        "newey_west_lag1_loss_test_pvalues": loss_tests,
        "winner_initialization_robustness": seed_results,
    }

    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return results


def run_structure_validation() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    for lag in (2, 3):
        frame[f"Inflation_MoM_Lag{lag}"] = frame.Inflation_MoM.shift(lag)

    validation = frame.loc[frame.Date.le("2021-12-01")].copy()
    candidates = []

    def assess(name: str, family: str, features: list[str], **settings) -> None:
        fitter = fit_ols_forecast if family == "AR" else fit_ar_hmm
        result = fitter(
            validation, features=features, name=name, target="Inflation_MoM",
            test_start="2017-01-01", **settings,
        )
        candidates.append({
            "model": name,
            "family": family,
            "features": features,
            "settings": settings,
            "metrics": forecast_metrics(result.forecasts.Actual, result.forecasts.Forecast),
            "converged": result.parameters.get("converged", True),
        })

    for lag_order in (1, 2, 3):
        lag_features = [f"Inflation_MoM_Lag{lag}" for lag in range(1, lag_order + 1)]
        assess(f"AR({lag_order})", "AR", lag_features)
        for states in (2, 3):
            assess(
                f"AR-HMM({lag_order}), {states} states", "AR-HMM", lag_features,
                n_states=states,
            )

    hmm_candidates = [item for item in candidates if item["family"] == "AR-HMM"]
    preliminary = min(hmm_candidates, key=lambda item: item["metrics"]["rmse"])
    selected_features = preliminary["features"]
    selected_states = preliminary["settings"]["n_states"]
    assess(
        f"AR-HMM({len(selected_features)}), {selected_states} states, common AR",
        "AR-HMM", selected_features, n_states=selected_states, switching_ar=False,
    )
    assess(
        f"MS-Variance AR({len(selected_features)}), {selected_states} states",
        "AR-HMM", selected_features, n_states=selected_states,
        switching_intercept=False, switching_ar=False, switching_variance=True,
    )

    chosen = min(
        (item for item in candidates if item["family"] == "AR-HMM"),
        key=lambda item: item["metrics"]["rmse"],
    )
    final_models = []
    for suffix, extra_features in (
        ("baseline", []),
        ("+ all subject risks", [
            "Geopolitical_Risk_Lag1", "Energy_Risk_Lag1", "Supply_Chain_Risk_Lag1"
        ]),
    ):
        features = [*chosen["features"], *extra_features]
        name = f"Selected {chosen['model']} {suffix}"
        fitted = fit_ar_hmm(
            frame, features=features, name=name, target="Inflation_MoM",
            test_start="2022-01-01", **chosen["settings"],
        )
        final_models.append({
            "model": name,
            "features": features,
            "metrics": forecast_metrics(fitted.forecasts.Actual, fitted.forecasts.Forecast),
            "parameters": fitted.parameters,
        })

    output = {
        "design": {
            "selection_period": "2017-01 to 2021-12",
            "final_test_period": "2022-01 to 2025-12",
            "selection_rule": "lowest validation RMSE; test period not used for selection",
        },
        "validation_candidates": candidates,
        "selected_structure": chosen,
        "final_test": final_models,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["structure_validation"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def run_seasonal_experiment() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    specifications = {
        "Seasonally adjusted AR(1)": ("AR", ["Inflation_MoM_Adjusted_Lag1"]),
        "Seasonally adjusted AR-HMM(1)": ("AR-HMM", ["Inflation_MoM_Adjusted_Lag1"]),
        "Seasonally adjusted AR-HMM(1) + all subject risks": (
            "AR-HMM", ["Inflation_MoM_Adjusted_Lag1", "Geopolitical_Risk_Lag1",
                       "Energy_Risk_Lag1", "Supply_Chain_Risk_Lag1"]
        ),
    }

    def prepare(training_end: str) -> tuple[pd.DataFrame, dict[int, float]]:
        training = frame.loc[frame.Date.le(training_end)].dropna(subset=["Inflation_MoM"])
        factors = training.groupby(training.Date.dt.month).Inflation_MoM.mean().to_dict()
        adjusted = frame.copy()
        adjusted["Seasonal_Factor"] = adjusted.Date.dt.month.map(factors)
        adjusted["Inflation_MoM_Adjusted"] = adjusted.Inflation_MoM - adjusted.Seasonal_Factor
        adjusted["Inflation_MoM_Adjusted_Lag1"] = adjusted.Inflation_MoM_Adjusted.shift(1)
        return adjusted, factors

    def run_period(training_end: str, test_start: str, test_end: str) -> tuple[list[dict], dict]:
        prepared, factors = prepare(training_end)
        prepared = prepared.loc[prepared.Date.le(test_end)].copy()
        period_results = []
        for name, (family, features) in specifications.items():
            fitter = fit_ols_forecast if family == "AR" else fit_ar_hmm
            fitted = fitter(
                prepared, features=features, name=name, target="Inflation_MoM_Adjusted",
                test_start=test_start, **({"n_states": 2} if family == "AR-HMM" else {}),
            )
            forecasts = fitted.forecasts.copy()
            seasonal = forecasts.Date.dt.month.map(factors).to_numpy()
            forecasts["Actual"] += seasonal
            forecasts["Forecast"] += seasonal
            forecasts["Error"] = forecasts.Actual - forecasts.Forecast
            period_results.append({
                "model": name,
                "family": family,
                "features": features,
                "metrics": forecast_metrics(forecasts.Actual, forecasts.Forecast),
                "forecasts": forecasts.assign(
                    Date=pd.to_datetime(forecasts.Date).dt.strftime("%Y-%m-%d")
                ).to_dict("records"),
                "parameters": fitted.parameters,
                "converged": fitted.parameters.get("converged", True),
            })
        return period_results, {str(month): value for month, value in factors.items()}

    validation, validation_factors = run_period("2016-12-01", "2017-01-01", "2021-12-01")
    fold_definitions = [
        ("2007-2016 → 2017-2018", "2016-12-01", "2017-01-01", "2018-12-01"),
        ("2007-2018 → 2019-2020", "2018-12-01", "2019-01-01", "2020-12-01"),
        ("2007-2020 → 2021", "2020-12-01", "2021-01-01", "2021-12-01"),
    ]
    rolling_folds = []
    combined_forecasts = {name: [] for name in specifications}
    for label, training_end, test_start, test_end in fold_definitions:
        fold_models, _ = run_period(training_end, test_start, test_end)
        rolling_folds.append({
            "fold": label,
            "training_end": training_end,
            "validation_start": test_start,
            "validation_end": test_end,
            "models": [{"model": item["model"], "metrics": item["metrics"],
                        "converged": item["converged"]} for item in fold_models],
        })
        for item in fold_models:
            combined_forecasts[item["model"]].extend(item["forecasts"])
    rolling_models = []
    for name, forecasts in combined_forecasts.items():
        combined = pd.DataFrame(forecasts).sort_values("Date")
        rolling_models.append({
            "model": name,
            "metrics": forecast_metrics(combined.Actual, combined.Forecast),
            "forecast_months": len(combined),
        })

    final_test, final_factors = run_period("2021-12-01", "2022-01-01", "2025-12-01")
    preferred = min(validation, key=lambda item: item["metrics"]["rmse"])["model"]
    rolling_preferred = min(rolling_models, key=lambda item: item["metrics"]["rmse"])["model"]
    output = {
        "design": {
            "method": "calendar-month mean removed before fitting and added back to forecasts",
            "validation_factor_training_end": "2016-12-01",
            "final_factor_training_end": "2021-12-01",
            "validation_period": "2017-01 to 2021-12",
            "final_test_period": "2022-01 to 2025-12",
        },
        "validation_seasonal_factors": validation_factors,
        "final_seasonal_factors": final_factors,
        "validation_models": validation,
        "preferred_on_validation": preferred,
        "rolling_validation": {
            "folds": rolling_folds,
            "combined_models": rolling_models,
            "preferred_model": rolling_preferred,
        },
        "final_test": final_test,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["seasonal_experiment"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def run_macro_experiment() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    base = ["Inflation_MoM_Adjusted_Lag1"]
    macro_sets = {
        "Seasonal AR-HMM(1)": [],
        "Seasonal AR-HMM(1) + unemployment": ["Unemployment_Change_Lag1"],
        "Seasonal AR-HMM(1) + industrial production": ["IP_Growth_MoM_Lag1"],
        "Seasonal AR-HMM(1) + energy HICP": ["Energy_Inflation_MoM_Lag1"],
        "Seasonal AR-HMM(1) + energy PPI": ["PPI_Energy_Growth_MoM_Lag1"],
        "Seasonal AR-HMM(1) + energy block": [
            "Energy_Inflation_MoM_Lag1", "PPI_Energy_Growth_MoM_Lag1"
        ],
        "Seasonal AR-HMM(1) + slow macro": [
            "Unemployment_Change_Lag1", "IP_Growth_MoM_Lag1"
        ],
        "Seasonal AR-HMM(1) + all macro": [
            "Unemployment_Change_Lag1", "IP_Growth_MoM_Lag1",
            "Energy_Inflation_MoM_Lag1", "PPI_Energy_Growth_MoM_Lag1"
        ],
    }

    def run_period(training_end: str, test_start: str, test_end: str) -> list[dict]:
        training = frame.loc[frame.Date.le(training_end)].dropna(subset=["Inflation_MoM"])
        factors = training.groupby(training.Date.dt.month).Inflation_MoM.mean().to_dict()
        prepared = frame.loc[frame.Date.le(test_end)].copy()
        prepared["Seasonal_Factor"] = prepared.Date.dt.month.map(factors)
        prepared["Inflation_MoM_Adjusted"] = prepared.Inflation_MoM - prepared.Seasonal_Factor
        prepared["Inflation_MoM_Adjusted_Lag1"] = prepared.Inflation_MoM_Adjusted.shift(1)
        period_results = []
        for name, macro_features in macro_sets.items():
            features = [*base, *macro_features]
            model_frame = prepared.copy()
            scale_sample = model_frame.loc[model_frame.Date.le(training_end), features]
            center = scale_sample.mean()
            scale = scale_sample.std().replace(0, 1)
            model_frame[features] = (model_frame[features] - center) / scale
            fitted = fit_ar_hmm(
                model_frame, features=features, name=name, target="Inflation_MoM_Adjusted",
                test_start=test_start, n_states=2,
            )
            forecasts = fitted.forecasts.copy()
            seasonal = forecasts.Date.dt.month.map(factors).to_numpy()
            forecasts["Actual"] += seasonal
            forecasts["Forecast"] += seasonal
            forecasts["Error"] = forecasts.Actual - forecasts.Forecast
            period_results.append({
                "model": name,
                "features": features,
                "metrics": forecast_metrics(forecasts.Actual, forecasts.Forecast),
                "forecasts": forecasts.assign(
                    Date=pd.to_datetime(forecasts.Date).dt.strftime("%Y-%m-%d")
                ).to_dict("records"),
                "converged": fitted.parameters["converged"],
                "predictor_scaling": {"center": center.to_dict(), "scale": scale.to_dict()},
            })
        return period_results

    fold_definitions = [
        ("2007-2016 → 2017-2018", "2016-12-01", "2017-01-01", "2018-12-01"),
        ("2007-2018 → 2019-2020", "2018-12-01", "2019-01-01", "2020-12-01"),
        ("2007-2020 → 2021", "2020-12-01", "2021-01-01", "2021-12-01"),
    ]
    folds = []
    combined = {name: [] for name in macro_sets}
    for label, training_end, test_start, test_end in fold_definitions:
        models = run_period(training_end, test_start, test_end)
        folds.append({
            "fold": label,
            "models": [{"model": item["model"], "metrics": item["metrics"],
                        "converged": item["converged"]} for item in models],
        })
        for item in models:
            combined[item["model"]].extend(item["forecasts"])
    rolling_models = []
    for name, forecasts in combined.items():
        values = pd.DataFrame(forecasts).sort_values("Date")
        rolling_models.append({
            "model": name,
            "metrics": forecast_metrics(values.Actual, values.Forecast),
            "forecast_months": len(values),
        })

    final_models = run_period("2021-12-01", "2022-01-01", "2025-12-01")
    for item in final_models:
        forecasts = pd.DataFrame(item["forecasts"])
        dates = pd.to_datetime(forecasts.Date)
        item["subperiod_metrics"] = {
            "2022": forecast_metrics(
                forecasts.loc[dates.dt.year.eq(2022), "Actual"],
                forecasts.loc[dates.dt.year.eq(2022), "Forecast"],
            ),
            "2023-2025": forecast_metrics(
                forecasts.loc[dates.dt.year.ge(2023), "Actual"],
                forecasts.loc[dates.dt.year.ge(2023), "Forecast"],
            ),
        }
    preferred = min(rolling_models, key=lambda item: item["metrics"]["rmse"])["model"]
    output = {
        "design": {
            "selection": "combined rolling-validation RMSE over 2017-2021",
            "final_test": "2022-2025, additionally reported as 2022 and 2023-2025",
            "predictor_timing": "all macroeconomic predictors lagged one month",
            "data_vintage": "latest revised Eurostat data, not real-time releases",
        },
        "rolling_validation": {"folds": folds, "combined_models": rolling_models},
        "preferred_model": preferred,
        "final_test": final_models,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["macro_experiment"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def run_macro_lag_experiment() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    indicators = {
        "unemployment": "Unemployment_Change",
        "industrial production": "IP_Growth_MoM",
        "energy HICP": "Energy_Inflation_MoM",
        "energy PPI": "PPI_Energy_Growth_MoM",
    }
    folds = [
        ("2007-2016 → 2017-2018", "2016-12-01", "2017-01-01", "2018-12-01"),
        ("2007-2018 → 2019-2020", "2018-12-01", "2019-01-01", "2020-12-01"),
        ("2007-2020 → 2021", "2020-12-01", "2021-01-01", "2021-12-01"),
    ]

    def prepare(training_end: str, test_end: str) -> pd.DataFrame:
        training = frame.loc[frame.Date.le(training_end)].dropna(subset=["Inflation_MoM"])
        factors = training.groupby(training.Date.dt.month).Inflation_MoM.mean().to_dict()
        prepared = frame.loc[frame.Date.le(test_end)].copy()
        prepared["Seasonal_Factor"] = prepared.Date.dt.month.map(factors)
        prepared["Inflation_MoM_Adjusted"] = prepared.Inflation_MoM - prepared.Seasonal_Factor
        prepared["Inflation_MoM_Adjusted_Lag1"] = prepared.Inflation_MoM_Adjusted.shift(1)
        return prepared

    def fit_period(name: str, features: list[str], training_end: str,
                   test_start: str, test_end: str) -> dict:
        prepared = prepare(training_end, test_end)
        scale_sample = prepared.loc[prepared.Date.le(training_end), features]
        center = scale_sample.mean()
        scale = scale_sample.std().replace(0, 1)
        prepared[features] = (prepared[features] - center) / scale
        fitted = fit_ar_hmm(
            prepared, features=features, name=name, target="Inflation_MoM_Adjusted",
            test_start=test_start, n_states=2,
        )
        forecasts = fitted.forecasts.copy()
        seasonal = forecasts.Date.map(
            frame.set_index("Date").loc[prepared.Date, "Inflation_MoM"]
            - prepared.set_index("Date").Inflation_MoM_Adjusted
        ).to_numpy()
        forecasts["Actual"] += seasonal
        forecasts["Forecast"] += seasonal
        forecasts["Error"] = forecasts.Actual - forecasts.Forecast
        return {
            "model": name,
            "features": features,
            "metrics": forecast_metrics(forecasts.Actual, forecasts.Forecast),
            "forecasts": forecasts.assign(
                Date=pd.to_datetime(forecasts.Date).dt.strftime("%Y-%m-%d")
            ).to_dict("records"),
            "converged": fitted.parameters["converged"],
        }

    candidates = []
    for indicator, column in indicators.items():
        for lag in range(1, 7):
            name = f"Seasonal AR-HMM(1) + {indicator} lag {lag}"
            features = ["Inflation_MoM_Adjusted_Lag1", f"{column}_Lag{lag}"]
            fold_results = [
                fit_period(name, features, training_end, test_start, test_end)
                for _, training_end, test_start, test_end in folds
            ]
            combined = pd.concat(
                [pd.DataFrame(item["forecasts"]) for item in fold_results], ignore_index=True
            )
            candidates.append({
                "model": name,
                "indicator": indicator,
                "lag": lag,
                "features": features,
                "metrics": forecast_metrics(combined.Actual, combined.Forecast),
                "fold_metrics": [
                    {"fold": fold[0], **item["metrics"]}
                    for fold, item in zip(folds, fold_results)
                ],
                "all_folds_converged": all(item["converged"] for item in fold_results),
            })

    selected = {
        indicator: min(
            (item for item in candidates if item["indicator"] == indicator),
            key=lambda item: item["metrics"]["rmse"],
        )
        for indicator in indicators
    }
    selected_features = ["Inflation_MoM_Adjusted_Lag1", *[
        item["features"][1] for item in selected.values()
    ]]
    compact_name = "Seasonal AR-HMM(1) + selected macro lags"
    compact_folds = [
        fit_period(compact_name, selected_features, training_end, test_start, test_end)
        for _, training_end, test_start, test_end in folds
    ]
    compact_combined = pd.concat(
        [pd.DataFrame(item["forecasts"]) for item in compact_folds], ignore_index=True
    )
    compact_validation = {
        "model": compact_name,
        "features": selected_features,
        "metrics": forecast_metrics(compact_combined.Actual, compact_combined.Forecast),
        "all_folds_converged": all(item["converged"] for item in compact_folds),
    }

    final_specs = {
        "Seasonal AR-HMM(1)": ["Inflation_MoM_Adjusted_Lag1"],
        **{item["model"]: item["features"] for item in selected.values()},
        compact_name: selected_features,
    }
    final_models = []
    for name, features in final_specs.items():
        item = fit_period(name, features, "2021-12-01", "2022-01-01", "2025-12-01")
        forecasts = pd.DataFrame(item["forecasts"])
        dates = pd.to_datetime(forecasts.Date)
        item["subperiod_metrics"] = {
            "2022": forecast_metrics(
                forecasts.loc[dates.dt.year.eq(2022), "Actual"],
                forecasts.loc[dates.dt.year.eq(2022), "Forecast"],
            ),
            "2023-2025": forecast_metrics(
                forecasts.loc[dates.dt.year.ge(2023), "Actual"],
                forecasts.loc[dates.dt.year.ge(2023), "Forecast"],
            ),
        }
        final_models.append(item)

    output = {
        "design": {
            "selection": "one lag per indicator selected by combined rolling-validation RMSE",
            "candidate_lags": [1, 2, 3, 4, 5, 6],
            "final_test": "2022-2025, additionally reported as 2022 and 2023-2025",
        },
        "candidates": candidates,
        "selected_lags": {indicator: item["lag"] for indicator, item in selected.items()},
        "selected_models": selected,
        "compact_validation": compact_validation,
        "final_test": final_models,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["macro_lag_experiment"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def run_sentiment_nowcast_experiment() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    base = ["Inflation_MoM_Adjusted_Lag1"]
    sentiment_sets = {
        "Seasonal AR-HMM(1) nowcast baseline": [],
        "Seasonal AR-HMM(1) + mean sentiment": ["News_Sentiment"],
        "Seasonal AR-HMM(1) + median sentiment": ["Sentiment_Median"],
        "Seasonal AR-HMM(1) + negative share": ["Negative_Share"],
        "Seasonal AR-HMM(1) + sentiment dispersion": ["Sentiment_Std"],
        "Seasonal AR-HMM(1) + geopolitical risk": ["Geopolitical_Risk"],
        "Seasonal AR-HMM(1) + energy risk": ["Energy_Risk"],
        "Seasonal AR-HMM(1) + supply-chain risk": ["Supply_Chain_Risk"],
        "Seasonal AR-HMM(1) + all subject risks": [
            "Geopolitical_Risk", "Energy_Risk", "Supply_Chain_Risk"
        ],
    }
    folds = [
        ("2007-2016 → 2017-2018", "2016-12-01", "2017-01-01", "2018-12-01"),
        ("2007-2018 → 2019-2020", "2018-12-01", "2019-01-01", "2020-12-01"),
        ("2007-2020 → 2021", "2020-12-01", "2021-01-01", "2021-12-01"),
    ]

    def run_period(training_end: str, test_start: str, test_end: str) -> list[dict]:
        training = frame.loc[frame.Date.le(training_end)].dropna(subset=["Inflation_MoM"])
        factors = training.groupby(training.Date.dt.month).Inflation_MoM.mean().to_dict()
        prepared = frame.loc[frame.Date.le(test_end)].copy()
        prepared["Seasonal_Factor"] = prepared.Date.dt.month.map(factors)
        prepared["Inflation_MoM_Adjusted"] = prepared.Inflation_MoM - prepared.Seasonal_Factor
        prepared["Inflation_MoM_Adjusted_Lag1"] = prepared.Inflation_MoM_Adjusted.shift(1)
        outputs = []
        for name, sentiment_features in sentiment_sets.items():
            features = [*base, *sentiment_features]
            model_frame = prepared.copy()
            scale_sample = model_frame.loc[model_frame.Date.le(training_end), features]
            center = scale_sample.mean()
            scale = scale_sample.std().replace(0, 1)
            model_frame[features] = (model_frame[features] - center) / scale
            fitted = fit_ar_hmm(
                model_frame, features=features, name=name, target="Inflation_MoM_Adjusted",
                test_start=test_start, n_states=2,
            )
            forecasts = fitted.forecasts.copy()
            seasonal = forecasts.Date.dt.month.map(factors).to_numpy()
            forecasts["Actual"] += seasonal
            forecasts["Forecast"] += seasonal
            forecasts["Error"] = forecasts.Actual - forecasts.Forecast
            outputs.append({
                "model": name,
                "features": features,
                "metrics": forecast_metrics(forecasts.Actual, forecasts.Forecast),
                "forecasts": forecasts.assign(
                    Date=pd.to_datetime(forecasts.Date).dt.strftime("%Y-%m-%d")
                ).to_dict("records"),
                "converged": fitted.parameters["converged"],
            })
        return outputs

    combined = {name: [] for name in sentiment_sets}
    fold_results = []
    for label, training_end, test_start, test_end in folds:
        models = run_period(training_end, test_start, test_end)
        fold_results.append({
            "fold": label,
            "models": [{"model": item["model"], "metrics": item["metrics"],
                        "converged": item["converged"]} for item in models],
        })
        for item in models:
            combined[item["model"]].extend(item["forecasts"])
    rolling_models = []
    for name, forecasts in combined.items():
        values = pd.DataFrame(forecasts).sort_values("Date")
        rolling_models.append({
            "model": name,
            "metrics": forecast_metrics(values.Actual, values.Forecast),
            "forecast_months": len(values),
        })
    preferred = min(rolling_models, key=lambda item: item["metrics"]["rmse"])["model"]

    final_models = run_period("2021-12-01", "2022-01-01", "2025-12-01")
    for item in final_models:
        forecasts = pd.DataFrame(item["forecasts"])
        dates = pd.to_datetime(forecasts.Date)
        item["subperiod_metrics"] = {
            "2022": forecast_metrics(
                forecasts.loc[dates.dt.year.eq(2022), "Actual"],
                forecasts.loc[dates.dt.year.eq(2022), "Forecast"],
            ),
            "2023-2025": forecast_metrics(
                forecasts.loc[dates.dt.year.ge(2023), "Actual"],
                forecasts.loc[dates.dt.year.ge(2023), "Forecast"],
            ),
        }
    output = {
        "design": {
            "timing": "month-end nowcast using Guardian information from the target month",
            "selection": "combined rolling-validation RMSE over 2017-2021",
            "final_test": "2022-2025, additionally reported as 2022 and 2023-2025",
        },
        "rolling_validation": {"folds": fold_results, "combined_models": rolling_models},
        "preferred_model": preferred,
        "final_test": final_models,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    results["sentiment_nowcast_experiment"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def run_final_europe_comparison() -> dict:
    frame = pd.read_csv(DATASET, parse_dates=["Date"]).sort_values("Date")
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    macro_results = results["macro_lag_experiment"]
    sentiment_results = results["sentiment_nowcast_experiment"]
    selected_macro = min(macro_results["candidates"], key=lambda item: item["metrics"]["rmse"])
    selected_sentiment_name = sentiment_results["preferred_model"]
    selected_sentiment = next(
        item for item in sentiment_results["final_test"]
        if item["model"] == selected_sentiment_name
    )
    macro_feature = selected_macro["features"][1]
    sentiment_feature = selected_sentiment["features"][1]
    specs = {
        "Seasonal AR-HMM(1) baseline": ["Inflation_MoM_Adjusted_Lag1"],
        f"Seasonal AR-HMM(1) + {selected_macro['indicator']} lag {selected_macro['lag']}": [
            "Inflation_MoM_Adjusted_Lag1", macro_feature
        ],
        "Seasonal AR-HMM(1) + energy-risk sentiment": [
            "Inflation_MoM_Adjusted_Lag1", sentiment_feature
        ],
        "Seasonal AR-HMM(1) + macro and sentiment": [
            "Inflation_MoM_Adjusted_Lag1", macro_feature, sentiment_feature
        ],
    }
    folds = [
        ("2007-2016 → 2017-2018", "2016-12-01", "2017-01-01", "2018-12-01"),
        ("2007-2018 → 2019-2020", "2018-12-01", "2019-01-01", "2020-12-01"),
        ("2007-2020 → 2021", "2020-12-01", "2021-01-01", "2021-12-01"),
    ]

    def run_period(training_end: str, test_start: str, test_end: str) -> list[dict]:
        training = frame.loc[frame.Date.le(training_end)].dropna(subset=["Inflation_MoM"])
        factors = training.groupby(training.Date.dt.month).Inflation_MoM.mean().to_dict()
        prepared = frame.loc[frame.Date.le(test_end)].copy()
        prepared["Seasonal_Factor"] = prepared.Date.dt.month.map(factors)
        prepared["Inflation_MoM_Adjusted"] = prepared.Inflation_MoM - prepared.Seasonal_Factor
        prepared["Inflation_MoM_Adjusted_Lag1"] = prepared.Inflation_MoM_Adjusted.shift(1)
        outputs = []
        for name, features in specs.items():
            model_frame = prepared.copy()
            scale_sample = model_frame.loc[model_frame.Date.le(training_end), features]
            center = scale_sample.mean()
            scale = scale_sample.std().replace(0, 1)
            model_frame[features] = (model_frame[features] - center) / scale
            fitted = fit_ar_hmm(
                model_frame, features=features, name=name, target="Inflation_MoM_Adjusted",
                test_start=test_start, n_states=2,
            )
            forecasts = fitted.forecasts.copy()
            seasonal = forecasts.Date.dt.month.map(factors).to_numpy()
            forecasts["Actual"] += seasonal
            forecasts["Forecast"] += seasonal
            forecasts["Error"] = forecasts.Actual - forecasts.Forecast
            outputs.append({
                "model": name,
                "features": features,
                "metrics": forecast_metrics(forecasts.Actual, forecasts.Forecast),
                "forecasts": forecasts.assign(
                    Date=pd.to_datetime(forecasts.Date).dt.strftime("%Y-%m-%d")
                ).to_dict("records"),
                "converged": fitted.parameters["converged"],
            })
        return outputs

    combined = {name: [] for name in specs}
    fold_results = []
    for label, training_end, test_start, test_end in folds:
        models = run_period(training_end, test_start, test_end)
        fold_results.append({
            "fold": label,
            "models": [{"model": item["model"], "metrics": item["metrics"],
                        "converged": item["converged"]} for item in models],
        })
        for item in models:
            combined[item["model"]].extend(item["forecasts"])
    rolling_models = []
    for name, forecasts in combined.items():
        values = pd.DataFrame(forecasts).sort_values("Date")
        rolling_models.append({
            "model": name,
            "metrics": forecast_metrics(values.Actual, values.Forecast),
            "forecast_months": len(values),
        })

    final_models = run_period("2021-12-01", "2022-01-01", "2025-12-01")
    for item in final_models:
        forecasts = pd.DataFrame(item["forecasts"])
        dates = pd.to_datetime(forecasts.Date)
        item["subperiod_metrics"] = {
            "2022": forecast_metrics(
                forecasts.loc[dates.dt.year.eq(2022), "Actual"],
                forecasts.loc[dates.dt.year.eq(2022), "Forecast"],
            ),
            "2023-2025": forecast_metrics(
                forecasts.loc[dates.dt.year.ge(2023), "Actual"],
                forecasts.loc[dates.dt.year.ge(2023), "Forecast"],
            ),
        }
    output = {
        "design": {
            "selection": "macro and sentiment inputs fixed using 2017-2021 rolling validation",
            "timing": "macro lagged one month; Guardian sentiment observed during the target month",
            "final_test": "2022-2025, additionally reported as 2022 and 2023-2025",
        },
        "selected_inputs": {
            "macro": {"indicator": selected_macro["indicator"], "lag": selected_macro["lag"],
                      "feature": macro_feature},
            "sentiment": {"model": selected_sentiment_name, "feature": sentiment_feature},
        },
        "rolling_validation": {"folds": fold_results, "combined_models": rolling_models},
        "preferred_model": min(
            rolling_models, key=lambda item: item["metrics"]["rmse"]
        )["model"],
        "final_test": final_models,
    }
    results["final_europe_comparison"] = output
    RESULTS.write_text(json.dumps(results, allow_nan=False), encoding="utf-8")
    return output


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "download", "score", "build", "models", "structure", "seasonal", "macro", "macro-lags", "sentiment-nowcast", "final-comparison"], default="all")
    args = parser.parse_args()
    if args.stage in ("all", "download"):
        download_guardian(os.environ.get("GUARDIAN_API_KEY", ""))
        download_eurostat()
    if args.stage in ("all", "score"):
        score_guardian()
    if args.stage in ("all", "build"):
        build_dataset()
    if args.stage in ("all", "models"):
        run_models()
    if args.stage in ("all", "structure"):
        run_structure_validation()
    if args.stage in ("all", "seasonal"):
        run_seasonal_experiment()
    if args.stage in ("all", "macro"):
        run_macro_experiment()
    if args.stage in ("all", "macro-lags"):
        run_macro_lag_experiment()
    if args.stage in ("all", "sentiment-nowcast"):
        run_sentiment_nowcast_experiment()
    if args.stage in ("all", "final-comparison"):
        run_final_europe_comparison()


if __name__ == "__main__":
    main()
