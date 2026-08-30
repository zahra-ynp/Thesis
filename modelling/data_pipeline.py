from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_NAME = "inflation_sentiment_forecasting"
REQUIRED_MACRO_COLUMNS = (
    "CPIAUCSL",
    "UNRATE",
    "FEDFUNDS",
    "INDPRO",
    "MCOILWTICO",
)


def load_source_registry(engine) -> pd.DataFrame:
    """Return available news sources and their database coverage."""
    from sqlalchemy import text

    query = text("""
        SELECT
            s.source_code,
            s.source_name,
            CAST(s.is_active AS INT) AS is_active,
            COUNT(n.article_id) AS article_count,
            MIN(n.news_date) AS first_news_date,
            MAX(n.news_date) AS last_news_date
        FROM core.news_sources s
        LEFT JOIN core.daily_news n ON n.source_id = s.source_id
        GROUP BY s.source_code, s.source_name, s.is_active
        ORDER BY s.source_code
    """)
    return pd.read_sql(query, engine, parse_dates=["first_news_date", "last_news_date"])


def load_monthly_macro_from_database(
    engine,
    *,
    region_code: str = "US",
) -> pd.DataFrame:
    """Load the current experiment macro series and reshape to one row per month."""
    from sqlalchemy import text

    query = text("""
        SELECT m.month_date, i.indicator_code, m.indicator_value
        FROM core.monthly_macro_indicators m
        INNER JOIN core.source_agencies a ON m.agency_id = a.agency_id
        INNER JOIN core.source_regions r ON a.region_id = r.region_id
        INNER JOIN core.indicators i ON m.indicator_id = i.indicator_id
        WHERE r.region_code = :region_code
          AND i.indicator_code IN (
              'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'INDPRO', 'MCOILWTICO'
          )
        ORDER BY m.month_date, i.indicator_code
    """)
    long_frame = pd.read_sql(
        query,
        engine,
        params={"region_code": region_code},
        parse_dates=["month_date"],
    )
    if long_frame.empty:
        raise ValueError(f"No macroeconomic rows found for region {region_code!r}.")

    wide = (
        long_frame.pivot(
            index="month_date",
            columns="indicator_code",
            values="indicator_value",
        )
        .rename_axis(columns=None)
        .reset_index()
        .rename(columns={"month_date": "Date"})
    )
    return wide


def load_article_sentiment_from_database(
    engine,
    *,
    source_code: str,
    region_code: str = "US",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load article scores for exactly one named source, regardless of active status."""
    from sqlalchemy import text

    source_query = text("""
        SELECT source_code, source_name, CAST(is_active AS INT) AS is_active
        FROM core.news_sources
        WHERE source_code = :source_code
    """)
    source = pd.read_sql(source_query, engine, params={"source_code": source_code})
    if len(source) != 1:
        raise ValueError(f"Expected one registered source for {source_code!r}; found {len(source)}.")

    news_query = text("""
        SELECT
            n.article_id,
            n.external_article_id,
            n.news_date,
            n.sentiment_score,
            n.sentiment_label
        FROM core.daily_news n
        INNER JOIN core.news_sources s ON n.source_id = s.source_id
        INNER JOIN core.source_regions r ON n.region_id = r.region_id
        WHERE s.source_code = :source_code
          AND r.region_code = :region_code
        ORDER BY n.news_date, n.article_id
    """)
    articles = pd.read_sql(
        news_query,
        engine,
        params={"source_code": source_code, "region_code": region_code},
        parse_dates=["news_date"],
    )
    if articles.empty:
        raise ValueError(
            f"No news rows found for source {source_code!r} and region {region_code!r}."
        )

    source_record = source.iloc[0].to_dict()
    source_record["is_active"] = bool(source_record["is_active"])
    return articles, source_record


def prepare_experiment_frame(
    macro: pd.DataFrame,
    monthly_news: pd.DataFrame,
    *,
    source_code: str,
) -> pd.DataFrame:
    """Create the common monthly frame and leakage-safe one-month lags."""
    if "Date" not in macro or "Date" not in monthly_news:
        raise ValueError("Both macro and monthly news data must contain a Date column.")

    missing_macro = set(REQUIRED_MACRO_COLUMNS) - set(macro.columns)
    if missing_macro:
        raise ValueError(f"Macro data are missing required series: {sorted(missing_macro)}")

    macro_frame = macro.copy()
    news_frame = monthly_news.copy()
    macro_frame["Date"] = pd.to_datetime(macro_frame["Date"]).dt.to_period("M").dt.to_timestamp()
    news_frame["Date"] = pd.to_datetime(news_frame["Date"]).dt.to_period("M").dt.to_timestamp()
    macro_frame = macro_frame.sort_values("Date").drop_duplicates("Date", keep="last")
    news_frame = news_frame.sort_values("Date").drop_duplicates("Date", keep="last")

    for column in REQUIRED_MACRO_COLUMNS:
        macro_frame[column] = pd.to_numeric(macro_frame[column], errors="coerce")

    data = macro_frame.merge(news_frame, on="Date", how="left", validate="one_to_one")
    data.insert(1, "Experiment_News_Source", source_code.upper())

    data["Inflation_MoM"] = data["CPIAUCSL"].pct_change(1, fill_method=None) * 100
    data["Inflation_YoY"] = data["CPIAUCSL"].pct_change(12, fill_method=None) * 100
    data["Unemployment_Change"] = data["UNRATE"].diff()
    data["Fed_Funds_Change"] = data["FEDFUNDS"].diff()
    data["IP_Growth_MoM"] = data["INDPRO"].pct_change(1, fill_method=None) * 100
    positive_oil = data["MCOILWTICO"].where(data["MCOILWTICO"] > 0)
    data["Oil_Price_Growth"] = np.log(positive_oil).diff() * 100

    lag_columns = {
        "Inflation_MoM": "Inflation_Lag1",
        "News_Sentiment": "News_Sentiment_Lag1",
        "Unemployment_Change": "Unemployment_Change_Lag1",
        "Fed_Funds_Change": "Fed_Funds_Change_Lag1",
        "IP_Growth_MoM": "IP_Growth_MoM_Lag1",
        "Oil_Price_Growth": "Oil_Price_Growth_Lag1",
    }
    for original, lagged in lag_columns.items():
        if original in data:
            data[lagged] = data[original].shift(1)

    return data.sort_values("Date").reset_index(drop=True)


def validate_experiment_frame(data: pd.DataFrame) -> dict[str, Any]:
    """Run structural checks and return a compact coverage report."""
    if data.empty:
        raise ValueError("The experiment frame is empty.")
    if data["Date"].duplicated().any():
        raise ValueError("The experiment frame contains duplicate months.")
    if not data["Date"].is_monotonic_increasing:
        raise ValueError("The experiment frame is not ordered chronologically.")

    expected_dates = pd.date_range(data["Date"].min(), data["Date"].max(), freq="MS")
    missing_months = expected_dates.difference(pd.DatetimeIndex(data["Date"]))
    missing_macro = {
        column: int(data[column].isna().sum()) for column in REQUIRED_MACRO_COLUMNS
    }
    if any(missing_macro.values()):
        raise ValueError(f"Macroeconomic coverage is incomplete: {missing_macro}")

    return {
        "row_count": int(len(data)),
        "first_month": str(data["Date"].min().date()),
        "last_month": str(data["Date"].max().date()),
        "missing_calendar_months": [str(value.date()) for value in missing_months],
        "months_with_sentiment": int(data["News_Sentiment"].notna().sum()),
        "months_without_sentiment": int(data["News_Sentiment"].isna().sum()),
        "article_count_total": int(data["Article_Count"].fillna(0).sum()),
        "macro_missing_values": missing_macro,
    }


def build_experiment_metadata(
    data: pd.DataFrame,
    *,
    source_record: dict[str, Any],
    coverage: dict[str, Any],
    test_start: str,
    region_code: str = "US",
) -> dict[str, Any]:
    """Build auditable metadata shared by all forecasting models."""
    test_timestamp = pd.Timestamp(test_start)
    train = data.loc[data["Date"] < test_timestamp]
    test = data.loc[data["Date"] >= test_timestamp]
    if train.empty or test.empty:
        raise ValueError("test_start must leave observations in both train and test periods.")

    return {
        "experiment": EXPERIMENT_NAME,
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "region_code": region_code,
        "news_source": {
            "source_code": str(source_record["source_code"]),
            "source_name": str(source_record["source_name"]),
            "was_active_at_export": bool(source_record["is_active"]),
            "selection_rule": "explicit source_code; experiment does not depend on active view",
        },
        "coverage": coverage,
        "target_definitions": {
            "primary": "Inflation_MoM = 100 * monthly percentage change in CPIAUCSL",
            "secondary": "Inflation_YoY = 100 * twelve-month percentage change in CPIAUCSL",
        },
        "predictor_timing": {
            "forecast_horizon_months": 1,
            "policy": "Only lag-1 predictors are used to forecast month t.",
        },
        "split": {
            "test_start": str(test_timestamp.date()),
            "train_first_month": str(train["Date"].min().date()),
            "train_last_month": str(train["Date"].max().date()),
            "test_first_month": str(test["Date"].min().date()),
            "test_last_month": str(test["Date"].max().date()),
        },
        "macro_transformations": {
            "UNRATE": "first difference",
            "FEDFUNDS": "first difference",
            "INDPRO": "monthly percentage growth",
            "MCOILWTICO": "monthly log growth",
        },
    }


def export_experiment_dataset(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Save the common experiment data and its metadata side by side."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_code = metadata["news_source"]["source_code"].lower()
    region_code = metadata["region_code"].lower()
    stem = f"{region_code}_{source_code}_monthly_experiment"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}_metadata.json"
    data.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return csv_path, json_path


def add_lags(
    data: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    lags: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Add consistently named lags for any existing monthly columns."""
    if lags < 1:
        raise ValueError("lags must be at least 1.")
    frame = data.copy()
    created: list[str] = []
    for column in columns:
        if column not in frame:
            raise ValueError(f"Cannot lag missing column {column!r}.")
        base = column.removesuffix("_MoM")
        for lag in range(1, lags + 1):
            name = f"{base}_Lag{lag}"
            frame[name] = frame[column].shift(lag)
            created.append(name)
    return frame, created
