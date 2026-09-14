from __future__ import annotations

import calendar
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


NYT_ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

# Terms are deliberately explicit so the thesis can report the selection rule.
MACRO_TERMS = (
    "inflation", "consumer price", "producer price", "cost of living",
    "federal reserve", "interest rate", "monetary policy", "fed funds",
    "unemployment", "employment", "labor market", "labour market", "jobless",
    "nonfarm payroll", "wage growth", "gross domestic product", "gdp",
    "economic growth", "recession", "consumer spending", "retail sales",
    "industrial production", "supply chain", "energy prices", "oil prices",
)

US_TERMS = (
    "united states", "u.s.", " u.s ", "american economy", "federal reserve",
    "fed funds", "washington", "wall street", "bureau of labor statistics",
    "commerce department", "treasury department",
)

OUTPUT_COLUMNS = [
    "external_article_id", "published_at", "news_date", "title", "abstract",
    "lead_paragraph", "snippet", "article_url", "source", "news_desk",
    "section_name", "type_of_material", "keyword_values", "matched_macro_terms",
    "matched_us_terms", "text_for_finbert",
]


def find_project_root(start: str | Path = ".") -> Path:
    """Find the repository root from either the repo or a notebook directory."""
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the thesis project root.")


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _term_matches(text_value: str, terms: tuple[str, ...]) -> list[str]:
    lowered = f" {_clean(text_value).lower()} "
    return [term for term in terms if term in lowered]


def normalize_and_filter_docs(docs: list[dict]) -> pd.DataFrame:
    """Normalize NYT metadata and retain reproducibly defined US macro articles."""
    records: list[dict] = []
    for doc in docs:
        headline = doc.get("headline") or {}
        keywords = doc.get("keywords") or []
        keyword_values = " | ".join(
            _clean(item.get("value")) for item in keywords if isinstance(item, dict)
        )
        title = _clean(headline.get("main"))
        abstract = _clean(doc.get("abstract"))
        lead = _clean(doc.get("lead_paragraph"))
        snippet = _clean(doc.get("snippet"))
        metadata_text = " ".join(
            [title, abstract, lead, snippet, keyword_values,
             _clean(doc.get("news_desk")), _clean(doc.get("section_name"))]
        )
        macro_hits = _term_matches(metadata_text, MACRO_TERMS)
        us_hits = _term_matches(metadata_text, US_TERMS)
        if not macro_hits or not us_hits:
            continue

        published = pd.to_datetime(doc.get("pub_date"), utc=True, errors="coerce")
        if pd.isna(published) or not title:
            continue

        text_parts = list(dict.fromkeys(part for part in [title, abstract, lead, snippet] if part))
        records.append({
            "external_article_id": _clean(doc.get("_id") or doc.get("uri") or doc.get("web_url")),
            "published_at": published.isoformat(),
            "news_date": published.date().isoformat(),
            "title": title,
            "abstract": abstract,
            "lead_paragraph": lead,
            "snippet": snippet,
            "article_url": _clean(doc.get("web_url")),
            "source": _clean(doc.get("source")) or "The New York Times",
            "news_desk": _clean(doc.get("news_desk")),
            "section_name": _clean(doc.get("section_name")),
            "type_of_material": _clean(doc.get("type_of_material")),
            "keyword_values": keyword_values,
            "matched_macro_terms": " | ".join(macro_hits),
            "matched_us_terms": " | ".join(us_hits),
            "text_for_finbert": "\n\n".join(text_parts),
        })

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _request_month(session, api_key: str, year: int, month: int, timeout: int = 120) -> list[dict]:
    import requests
    url = NYT_ARCHIVE_URL.format(year=year, month=month)
    for attempt in range(6):
        try:
            response = session.get(url, params={"api-key": api_key}, timeout=timeout)
        except requests.RequestException:
            raise RuntimeError(f"NYT request failed for {year}-{month:02d}; check connectivity.") from None
        if response.status_code == 429:
            time.sleep(min(60, 5 * (2 ** attempt)))
            continue
        if not response.ok:
            # Requests' default exception includes the URL and its API key.
            raise RuntimeError(f"NYT returned HTTP {response.status_code} for {year}-{month:02d}.")
        payload = response.json()
        docs = payload.get("response", {}).get("docs")
        if not isinstance(docs, list) or not docs:
            raise RuntimeError(f"NYT archive is empty or malformed for {year}-{month:02d}.")
        return docs
    raise RuntimeError(f"NYT rate limit persisted for {year}-{month:02d}.")


def download_nyt_archive(
    api_key: str,
    start_year: int,
    end_year: int,
    cache_dir: str | Path,
    pause_seconds: float = 6.1,
) -> pd.DataFrame:
    """Download monthly archives, cache only filtered metadata, and combine them."""
    if not api_key:
        raise ValueError("NYT_API_KEY is missing. Add it to the project .env file.")
    import requests

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "inflation-regime-thesis/1.0"})
    monthly_frames = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            cache_file = cache_dir / f"nyt_macro_{year}-{month:02d}.csv"
            if cache_file.exists():
                frame = pd.read_csv(cache_file, dtype=str).fillna("")
            else:
                docs = _request_month(session, api_key, year, month)
                frame = normalize_and_filter_docs(docs)
                frame.to_csv(cache_file, index=False)
                time.sleep(pause_seconds)
            monthly_frames.append(frame)
            print(f"{year}-{month:02d}: {len(frame):,} retained")

    combined = pd.concat(monthly_frames, ignore_index=True)
    combined = combined.drop_duplicates("external_article_id", keep="first")
    combined = combined.drop_duplicates(["news_date", "title"], keep="first")
    return combined.sort_values(["published_at", "external_article_id"]).reset_index(drop=True)


def score_with_finbert(
    articles: pd.DataFrame,
    output_path: str | Path,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32,
    checkpoint_every: int = 500,
) -> pd.DataFrame:
    """Score articles with resumable checkpoints and save a database-ready CSV."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    raw_output_path = str(output_path)
    control_characters = sorted({repr(char) for char in raw_output_path if ord(char) < 32})
    if control_characters:
        raise ValueError(
            "The FinBERT output path contains a control character "
            f"({', '.join(control_characters)}): {raw_output_path!r}. "
            "On Windows, build paths with pathlib.Path / 'filename.csv' or use a raw "
            "string (r'D:\\\\folder\\\\file.csv'); do not use an ordinary backslash string."
        )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fail before loading/scoring the model if Windows cannot write in this directory.
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".finbert_write_test_", suffix=".tmp", dir=output_path.parent, delete=True
        ):
            pass
    except OSError as exc:
        raise OSError(
            f"Cannot write FinBERT checkpoints under {output_path.parent!s}. "
            "Check the path, permissions, free disk space, and whether the folder is synced or locked."
        ) from exc

    score_columns = ["prob_positive", "prob_negative", "prob_neutral", "sentiment_score", "sentiment_label"]

    def save_checkpoint(frame: pd.DataFrame) -> None:
        """Write a complete checkpoint and replace the old file only after success."""
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            frame.to_csv(temporary_path, index=False)
            os.replace(temporary_path, output_path)
        except OSError as exc:
            raise OSError(
                f"Could not save the FinBERT checkpoint to {output_path!s}. "
                "Close the CSV if it is open in Excel and verify that the path contains no "
                "hidden newline or tab characters."
            ) from exc

    if output_path.exists():
        existing = pd.read_csv(output_path, dtype={"external_article_id": str})
        existing_ids = set(existing["external_article_id"].astype(str))
    else:
        existing = pd.DataFrame(columns=[*articles.columns, *score_columns])
        existing_ids = set()

    pending = articles[~articles["external_article_id"].astype(str).isin(existing_ids)].copy()
    if pending.empty:
        return existing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    id_to_label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    new_chunks = []

    for start in range(0, len(pending), batch_size):
        batch = pending.iloc[start:start + batch_size].copy()
        encoded = tokenizer(
            batch["text_for_finbert"].fillna("").tolist(), padding=True,
            truncation=True, max_length=512, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits, dim=1).cpu().numpy()

        label_to_index = {label: idx for idx, label in id_to_label.items()}
        batch["prob_positive"] = probabilities[:, label_to_index["positive"]]
        batch["prob_negative"] = probabilities[:, label_to_index["negative"]]
        batch["prob_neutral"] = probabilities[:, label_to_index["neutral"]]
        batch["sentiment_score"] = batch["prob_positive"] - batch["prob_negative"]
        batch["sentiment_label"] = [id_to_label[int(i)] for i in probabilities.argmax(axis=1)]
        new_chunks.append(batch)

        processed = min(start + batch_size, len(pending))
        if processed % checkpoint_every < batch_size or processed == len(pending):
            result = pd.concat([existing, *new_chunks], ignore_index=True)
            save_checkpoint(result)
            print(f"Scored {processed:,}/{len(pending):,} new articles on {device}.")

    return pd.read_csv(output_path, parse_dates=["published_at", "news_date"])


def monthly_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Create richer monthly indicators than the mean alone."""
    frame = scored.copy()
    frame["month_date"] = pd.to_datetime(frame["news_date"]).dt.to_period("M").dt.to_timestamp()
    return (
        frame.groupby("month_date", as_index=False)
        .agg(
            News_Sentiment=("sentiment_score", "mean"),
            Sentiment_Median=("sentiment_score", "median"),
            Sentiment_Std=("sentiment_score", "std"),
            Negative_Share=("sentiment_label", lambda values: (values == "negative").mean()),
            Article_Count=("external_article_id", "nunique"),
        )
    )


def load_scored_news_to_database(
    scored: pd.DataFrame,
    engine,
    scored_csv_path: str | Path,
    activate_nyt: bool = False,
) -> int:
    """Idempotently append NYT rows after migration 07 has been applied."""
    from sqlalchemy import inspect, text

    required = {"source_id", "external_article_id", "published_at", "article_url"}
    columns = {column["name"] for column in inspect(engine).get_columns("daily_news", schema="core")}
    missing = required - columns
    if missing:
        raise RuntimeError(
            "Apply database/scripts/07_upgrade_news_sources.sql first. "
            f"Missing columns: {sorted(missing)}"
        )

    with engine.begin() as connection:
        source_id = connection.execute(
            text("SELECT source_id FROM core.news_sources WHERE source_code = 'NYT'")
        ).scalar_one()
        region_id = connection.execute(
            text("SELECT region_id FROM core.source_regions WHERE region_code = 'US'")
        ).scalar_one()
        existing = set(pd.read_sql(
            text("SELECT external_article_id FROM core.daily_news WHERE source_id = :source_id"),
            connection, params={"source_id": source_id},
        )["external_article_id"].dropna().astype(str))

    pending = scored[~scored["external_article_id"].astype(str).isin(existing)].copy()
    if pending.empty:
        inserted = 0
    else:
        published = pd.to_datetime(pending["published_at"], utc=True).dt.tz_localize(None)
        insert_frame = pd.DataFrame({
            "news_date": pd.to_datetime(pending["news_date"]).dt.date,
            "title": pending["title"].fillna("").str.slice(0, 500),
            "source_name": "The New York Times",
            "source_id": source_id,
            "region_id": region_id,
            "sentiment_score": pending["sentiment_score"].astype(float),
            "sentiment_label": pending["sentiment_label"].astype(str).str.slice(0, 50),
            "lake_pointer_url": str(Path(scored_csv_path).resolve())[:500],
            "external_article_id": pending["external_article_id"].astype(str).str.slice(0, 300),
            "published_at": published,
            "article_url": pending["article_url"].fillna("").str.slice(0, 1000),
        })
        insert_frame.to_sql(
            "daily_news", engine, schema="core", if_exists="append", index=False, chunksize=250,
        )
        inserted = len(insert_frame)

    if activate_nyt:
        with engine.begin() as connection:
            connection.execute(text("UPDATE core.news_sources SET is_active = CASE WHEN source_code = 'NYT' THEN 1 ELSE 0 END"))
    return inserted
