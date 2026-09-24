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
    "Unemployment_Rate": ("une_rt_m", {"freq": "M", "unit": "PC_ACT", "sex": "T", "age": "TOTAL", "s_adj": "SA", "geo": "EU27_2020"}),
    "Industrial_Production": ("sts_inpr_m", {"freq": "M", "unit": "I21", "s_adj": "SCA", "nace_r2": "B-D", "geo": "EU27_2020"}),
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
    frame["Unemployment_Change"] = frame.Unemployment_Rate.diff()
    frame["IP_Growth_MoM"] = frame.Industrial_Production.pct_change(fill_method=None) * 100
    lagged = ["Inflation_MoM", "News_Sentiment", "Sentiment_Std", "Negative_Share",
              "Geopolitical_Risk", "Energy_Risk", "Supply_Chain_Risk",
              "Unemployment_Change", "IP_Growth_MoM"]
    for column in lagged:
        frame[f"{column}_Lag1"] = frame[column].shift(1)
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


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "download", "score", "build", "models"], default="all")
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


if __name__ == "__main__":
    main()
