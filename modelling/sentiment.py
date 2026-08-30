from __future__ import annotations

from typing import Any

import pandas as pd


def aggregate_monthly_sentiment(
    articles: pd.DataFrame,
    *,
    source_code: str,
) -> pd.DataFrame:
    """Aggregate article FinBERT scores and retain their monthly distribution."""
    required = {"news_date", "sentiment_score", "sentiment_label"}
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Article data are missing required columns: {sorted(missing)}")

    frame = articles.copy()
    frame["news_date"] = pd.to_datetime(frame["news_date"], errors="coerce")
    frame["sentiment_score"] = pd.to_numeric(frame["sentiment_score"], errors="coerce")
    frame["sentiment_label"] = frame["sentiment_label"].fillna("").str.lower()
    frame = frame.dropna(subset=["news_date", "sentiment_score"])
    if frame.empty:
        raise ValueError("No valid dated sentiment scores remain after cleaning.")

    frame["Date"] = frame["news_date"].dt.to_period("M").dt.to_timestamp()
    if "external_article_id" in frame and "article_id" in frame:
        external = frame["external_article_id"].astype("string")
        frame["_article_key"] = external.where(
            external.notna() & external.ne(""),
            "database:" + frame["article_id"].astype("string"),
        )
        id_column = "_article_key"
    elif "external_article_id" in frame:
        id_column = "external_article_id"
    else:
        id_column = "article_id"

    monthly = (
        frame.groupby("Date", as_index=False)
        .agg(
            News_Sentiment=("sentiment_score", "mean"),
            Sentiment_Median=("sentiment_score", "median"),
            Sentiment_Std=("sentiment_score", "std"),
            Negative_Share=("sentiment_label", lambda values: float((values == "negative").mean())),
            Neutral_Share=("sentiment_label", lambda values: float((values == "neutral").mean())),
            Positive_Share=("sentiment_label", lambda values: float((values == "positive").mean())),
            Article_Count=(id_column, "nunique"),
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    monthly.insert(1, "News_Source", source_code.upper())
    return monthly


def add_sentiment_lags(
    data: pd.DataFrame,
    *,
    column: str = "News_Sentiment",
    lags: int = 6,
) -> tuple[pd.DataFrame, list[str]]:
    """Create the lagged sentiment predictors used by the existing models."""
    if column not in data:
        raise ValueError(f"Cannot lag missing sentiment column {column!r}.")
    if lags < 1:
        raise ValueError("lags must be at least 1.")

    frame = data.copy()
    created = []
    for lag in range(1, lags + 1):
        name = f"Sentiment_Lag{lag}"
        frame[name] = frame[column].shift(lag)
        created.append(name)
    return frame, created


def current_method_catalog() -> list[dict[str, Any]]:
    return [
        {
            "method": "generic_finbert",
            "level": "article",
            "monthly_primary_feature": "News_Sentiment",
            "forecast_features": [f"Sentiment_Lag{lag}" for lag in range(1, 7)],
            "status": "implemented",
            "notes": "Generic financial sentiment; not yet manually validated for inflation direction.",
        }
    ]
