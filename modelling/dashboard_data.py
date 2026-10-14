"""Local, read-only data access for the research dashboard. No database imports."""

import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
US_WORKBOOK = DATA / "US" / "us_dashboard.xlsx"
STATE_LABELS = {
    "Europe": {"Level 1": "State 1 · Low inflation", "Level 2": "State 2 · High inflation"},
    "US": {"Low volatility": "State 1 · Low volatility", "High volatility": "State 2 · High volatility"},
}
REGIONS = {
    "Europe": {
        "area": "European Union · EU-27", "publisher": "The Guardian",
        "macro_source": "Eurostat", "price_index": "HICP", "state_kind": "inflation",
        "monthly": DATA / "europe" / "eu27_guardian_monthly_experiment.csv",
        "results": DATA / "europe" / "eu27_guardian_results_2022_2025.json",
        "news": [DATA / "europe" / "guardian_eu_finbert_scores_2007_2025.csv"],
        "raw_news": [DATA / "europe" / "guardian_eu_articles_2007_2025.csv"],
    },
    "US": {
        "area": "United States", "publisher": "The New York Times",
        "macro_source": "FRED", "price_index": "CPI", "state_kind": "volatility",
        "monthly": US_WORKBOOK, "results": US_WORKBOOK,
        "news": [DATA / "nyt_finbert_article_scores.csv", DATA / "nyt_finbert_article_scores_2024_2025.csv"],
        "raw_news": [DATA / "nyt_macro_articles_2007_2023.csv", DATA / "nyt_macro_articles_2024_2025.csv"],
    },
}


def normalize_dates(frame, column="Date"):
    frame = frame.copy()
    frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None)
    return frame.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def load_monthly(region):
    path = REGIONS[region]["monthly"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {region} monthly dataset: {path.relative_to(ROOT)}")
    frame = pd.read_excel(path, sheet_name="Monthly") if region == "US" else pd.read_csv(path)
    frame = normalize_dates(frame)
    if frame["Date"].duplicated().any():
        raise ValueError(f"The {region} monthly dataset contains duplicate months.")
    return frame.rename(columns={"UNRATE": "Unemployment_Rate", "INDPRO": "Industrial_Production"})


def load_states(region):
    path = REGIONS[region]["results"]
    if region == "US":
        frame = pd.read_excel(path, sheet_name="Estimated states")
        model_name = "Two-state AR-HMM(1) · Switching variance only"
    else:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        # Use the exact fit plotted by notebook 03 and the report's regime figure.
        model = next(item for item in bundle["seasonal_experiment"]["final_test"]
                     if item["model"] == "Seasonally adjusted AR-HMM(1)")
        frame = pd.DataFrame(model["forecasts"])
        model_name = model["model"]
    frame = normalize_dates(frame)
    probability_columns = [f"Probability_{label}" for label in STATE_LABELS[region]]
    required = ["Actual", "Forecast", *probability_columns]
    if frame["Date"].duplicated().any():
        raise ValueError(f"The {region} state estimates contain duplicate months.")
    values = frame[required].apply(pd.to_numeric, errors="coerce")
    probabilities = values[probability_columns]
    if (values.isna().any().any() or not probabilities.ge(0).all().all()
            or not probabilities.le(1).all().all()
            or not probabilities.sum(axis=1).sub(1).abs().lt(1e-6).all()):
        raise ValueError(f"Invalid saved state probabilities for {region}.")
    frame[required] = values
    assigned = probabilities.idxmax(axis=1).str.removeprefix("Probability_")
    if not assigned.eq(frame["PredictedRegime"]).all():
        raise ValueError(f"The {region} state labels disagree with their probabilities.")
    return frame, model_name


def load_news(region):
    """Prefer scored articles; use local source articles where scores are absent."""
    config = REGIONS[region]
    frames, missing = [], []
    for scored, raw in zip(config["news"], config["raw_news"]):
        path = scored if scored.is_file() else raw
        if path.is_file():
            frames.append(pd.read_csv(path, low_memory=False))
        else:
            missing.append(str(raw.relative_to(ROOT)))
    columns = ["external_article_id", "news_date", "title", "source", "sentiment_label", "sentiment_score"]
    if not frames:
        return pd.DataFrame({column: pd.Series(dtype="datetime64[ns]" if column == "news_date" else "object")
                             for column in columns}), missing
    news = normalize_dates(pd.concat(frames, ignore_index=True), "news_date")
    news = news.dropna(subset=["title"])
    news = news.loc[news["title"].astype(str).str.strip().ne("")].copy()
    fallback = news["news_date"].astype(str) + "|" + news["title"]
    news["external_article_id"] = news["external_article_id"].fillna(fallback) if "external_article_id" in news else fallback
    news = news.drop_duplicates("external_article_id", keep="last")
    news["source"] = config["publisher"]
    if "sentiment_label" not in news:
        news["sentiment_label"] = "Unscored"
    news["sentiment_label"] = news["sentiment_label"].fillna("Unscored").astype(str).str.capitalize()
    if "sentiment_score" not in news:
        news["sentiment_score"] = float("nan")
    news["sentiment_score"] = pd.to_numeric(news["sentiment_score"], errors="coerce")
    return news.sort_values(["news_date", "external_article_id"], ascending=[False, True]).reset_index(drop=True), missing


def filter_period(frame, start, end, column="Date"):
    """Inclusive calendar days, including articles published on the final day."""
    return frame.loc[frame[column].ge(pd.Timestamp(start))
                     & frame[column].lt(pd.Timestamp(end) + pd.Timedelta(days=1))].copy()


def filter_news(news, query="", sentiment="All sentiments"):
    frame = news
    if query.strip():
        searchable = frame["title"].fillna("").astype(str)
        for column in ("abstract", "lead_paragraph", "text_for_finbert", "keyword_values"):
            if column in frame:
                searchable = searchable + " " + frame[column].fillna("").astype(str)
        frame = frame.loc[searchable.str.contains(query.strip(), case=False, regex=False)]
    if sentiment != "All sentiments":
        frame = frame.loc[frame["sentiment_label"].eq(sentiment)]
    return frame.reset_index(drop=True)


def plain_text(value):
    if value is None or pd.isna(value):
        return ""
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", str(value), flags=re.I | re.S)
    text = re.sub(r"</?(?:p|div|br|h[1-6])\b[^>]*>", "\n", text, flags=re.I)
    return unescape(re.sub(r"<[^>]+>", "", text)).strip()


def article_text(article):
    """Display existing article text without repeating the headline."""
    full = next((plain_text(article.get(key)) for key in ("bodyText", "body", "Content")
                 if plain_text(article.get(key))), "")
    if full:
        return full, True
    paragraphs = []
    for key in ("abstract", "lead_paragraph", "snippet"):
        text = plain_text(article.get(key))
        if text and text not in paragraphs:
            paragraphs.append(text)
    if not paragraphs:
        text = plain_text(article.get("text_for_finbert"))
        title = plain_text(article.get("title"))
        if title and text.startswith(title):
            text = text[len(title):].strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs), False


def original_url(article):
    value = article.get("article_url")
    if value is None or pd.isna(value):
        return None
    url = str(value).strip()
    parsed = urlparse(url)
    expected = "nytimes.com" if article.get("source") == "The New York Times" else "theguardian.com"
    if parsed.scheme in {"https", "http"} and (
        parsed.hostname == expected or (parsed.hostname or "").endswith("." + expected)
    ):
        return url
    return None
