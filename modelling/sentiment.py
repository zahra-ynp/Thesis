from __future__ import annotations

import re
from typing import Any

import pandas as pd


ALTERNATIVE_FINBERT_FEATURES = (
    "News_Sentiment",
    "Sentiment_Median",
    "Negative_Share",
    "Sentiment_Std",
)

SUBJECT_RISK_TERMS = {
    "Geopolitical_Risk": (
        "war", "armed conflict", "invasion", "military", "missile", "airstrike",
        "attack", "sanction", "embargo", "blockade", "geopolitical", "terrorism",
    ),
    "Energy_Risk": (
        "oil", "crude", "petroleum", "gasoline", "diesel", "natural gas", "fuel",
        "refinery", "pipeline", "opec", "energy supply", "energy prices", "electricity",
    ),
    "Supply_Chain_Risk": (
        "supply chain", "shipping", "freight", "port", "logistics", "container",
        "cargo", "trade route", "semiconductor", "chip shortage", "suez", "red sea",
        "bottleneck",
    ),
}


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
    frame["sentiment_label"] = (
        frame["sentiment_label"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )
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
    elif "article_id" in frame:
        id_column = "article_id"
    else:
        frame["_article_key"] = range(len(frame))
        id_column = "_article_key"

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
    # Dispersion is zero, rather than undefined, when a month has one article.
    monthly["Sentiment_Std"] = monthly["Sentiment_Std"].fillna(0.0)
    monthly.insert(1, "News_Source", source_code.upper())
    return monthly


def aggregate_monthly_subject_risk(
    articles: pd.DataFrame,
    *,
    source_code: str,
    text_column: str = "text_for_finbert",
) -> pd.DataFrame:
    required = {"news_date", "prob_negative", text_column}
    missing = required - set(articles.columns)
    if missing:
        raise ValueError(f"Article data are missing required columns: {sorted(missing)}")

    frame = articles.copy()
    frame["news_date"] = pd.to_datetime(frame["news_date"], errors="coerce")
    frame["prob_negative"] = pd.to_numeric(frame["prob_negative"], errors="coerce")
    frame[text_column] = frame[text_column].astype("string").fillna("").str.lower()
    frame = frame.dropna(subset=["news_date", "prob_negative"])
    if frame.empty:
        raise ValueError("No valid dated article probabilities remain after cleaning.")
    if not frame["prob_negative"].between(0, 1).all():
        raise ValueError("Negative sentiment probabilities must lie between zero and one.")

    frame["Date"] = frame["news_date"].dt.to_period("M").dt.to_timestamp()
    monthly = frame[["Date"]].drop_duplicates().sort_values("Date").reset_index(drop=True)
    total = frame.groupby("Date").size()
    for feature, terms in SUBJECT_RISK_TERMS.items():
        pattern = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        match = frame[text_column].str.contains(pattern, regex=True, na=False)
        weighted = frame["prob_negative"].where(match, 0.0).groupby(frame["Date"]).sum()
        counts = match.groupby(frame["Date"]).sum()
        monthly[feature] = monthly["Date"].map(weighted.div(total)).fillna(0.0)
        monthly[f"{feature}_Articles"] = monthly["Date"].map(counts).fillna(0).astype(int)
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
        },
        {
            "method": "alternative_finbert_aggregations",
            "level": "month",
            "monthly_primary_feature": "News_Sentiment",
            "forecast_features": list(ALTERNATIVE_FINBERT_FEATURES),
            "status": "implemented",
            "notes": "Mean, median, negative share, and dispersion from the same FinBERT scores.",
        },
        {
            "method": "subject_negative_intensity",
            "level": "month",
            "monthly_primary_feature": list(SUBJECT_RISK_TERMS),
            "forecast_features": [f"{name}_Lag1" for name in SUBJECT_RISK_TERMS],
            "status": "implemented",
            "notes": "Subject coverage weighted by article-level FinBERT negative probability.",
        },
    ]
