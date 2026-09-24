"""Extend FRED/NYT through 2025 without rescoring historical NYT articles.

Run from the repository root: python -m modelling.extend_dataset --stage all
The old 2007–2023 experiment remains unchanged. No forecasts are fitted here.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from modelling.data_pipeline import (
    REQUIRED_MACRO_COLUMNS, prepare_experiment_frame,
    build_experiment_metadata, export_experiment_dataset,
)
from modelling.news_pipeline import download_nyt_archive, score_with_finbert, MACRO_TERMS, US_TERMS
from modelling.sentiment import aggregate_monthly_sentiment, aggregate_monthly_subject_risk

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'data/processed/modelling_datasets/us_nyt_monthly_experiment.csv'
OUT = ROOT / 'data/processed/modelling_datasets/extended_2025'
NEW_ARTICLES = ROOT / 'data/processed/nyt_macro_articles_2024_2025.csv'
NEW_SCORES = ROOT / 'data/processed/nyt_finbert_article_scores_2024_2025.csv'
OLD_SCORES = ROOT / 'data/processed/nyt_finbert_article_scores.csv'
FRED = ROOT / 'data/raw/fred/extension_2025'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_fred():
    import requests
    key = os.environ.get('FRED_API_KEY', '')
    if not key:
        raise ValueError('FRED_API_KEY is missing in .env')
    FRED.mkdir(parents=True, exist_ok=True)
    for series in REQUIRED_MACRO_COLUMNS:
        path = FRED / f'{series}.json'
        if path.exists():
            continue
        try:
            response = requests.get('https://api.stlouisfed.org/fred/series/observations', params={
                'api_key': key, 'series_id': series, 'file_type': 'json',
                'observation_start': '2007-01-01', 'observation_end': '2025-12-31',
            }, timeout=120)
        except requests.RequestException:
            raise RuntimeError(f'FRED connection failed for {series}.') from None
        if not response.ok:
            raise RuntimeError(f'FRED returned HTTP {response.status_code} for {series}.')
        payload = response.json()
        if not payload.get('observations'):
            raise RuntimeError(f'FRED returned no observations for {series}.')
        path.write_text(json.dumps({
            'series_id': series, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'vintage_policy': 'latest available at retrieval, not a real-time vintage backtest',
            'response': payload,
        }, indent=2), encoding='utf-8')
        print(f'FRED {series}: {len(payload["observations"])} observations cached', flush=True)


def download_news():
    articles = download_nyt_archive(
        os.environ.get('NYT_API_KEY', ''), 2024, 2025,
        ROOT / 'data/raw/news/nyt_archive', pause_seconds=12.1,
    )
    dates = pd.to_datetime(articles.news_date)
    articles = articles.loc[dates.between('2024-01-01', '2025-12-31')].copy()
    old_ids = set(pd.read_csv(OLD_SCORES, usecols=['external_article_id'], dtype=str).external_article_id)
    articles = articles.loc[~articles.external_article_id.isin(old_ids)]
    articles.to_csv(NEW_ARTICLES, index=False)
    print(f'New articles only: {len(articles)}; historical IDs excluded', flush=True)


def score_news(model_name):
    articles = pd.read_csv(NEW_ARTICLES, keep_default_na=False)
    if articles.empty:
        raise ValueError('No new articles to score.')
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    score_with_finbert(articles, NEW_SCORES, model_name=model_name, batch_size=16,
                       checkpoint_every=100)


def build_dataset():
    old = pd.read_csv(OLD, parse_dates=['Date'])
    expected = pd.date_range('2007-01-01', '2025-12-01', freq='MS')
    macro = pd.DataFrame({'Date': expected})
    fred_sources = {}
    for series in REQUIRED_MACRO_COLUMNS:
        path = FRED / f'{series}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        values = pd.DataFrame(payload['response']['observations'])
        part = pd.DataFrame({'Date': pd.to_datetime(values.date),
                             series: pd.to_numeric(values.value, errors='coerce')})
        macro = macro.merge(part, on='Date', how='left', validate='one_to_one')
        fred_sources[series] = {'sha256': sha256(path), 'retrieved_at_utc': payload['retrieved_at_utc']}
    missing = {s: macro.loc[macro[s].isna(), 'Date'].dt.strftime('%Y-%m').tolist()
               for s in REQUIRED_MACRO_COLUMNS if macro[s].isna().any()}
    expected_missing = {'CPIAUCSL': ['2025-10'], 'UNRATE': ['2025-10']}
    if any(month not in expected_missing.get(s, []) for s, months in missing.items() for month in months):
        raise ValueError(f'Unexpected FRED coverage gaps: {missing}')

    scored = pd.read_csv(NEW_SCORES, parse_dates=['news_date'])
    articles = pd.read_csv(NEW_ARTICLES, usecols=['external_article_id'], dtype=str)
    if scored.external_article_id.duplicated().any() or set(scored.external_article_id) != set(articles.external_article_id):
        raise ValueError('Scoring is incomplete or has duplicate/unexpected article IDs.')
    if not scored.news_date.between('2024-01-01', '2025-12-31').all():
        raise ValueError('New scoring file contains historical or out-of-period articles.')
    probability_cols = ['prob_positive', 'prob_negative', 'prob_neutral']
    probs = scored[probability_cols].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError('Invalid sentiment probabilities.')
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(scored.sentiment_score, scored.prob_positive-scored.prob_negative, atol=1e-6)
    new_monthly = aggregate_monthly_sentiment(scored, source_code='NYT')
    if not pd.DatetimeIndex(new_monthly.Date).equals(pd.date_range('2024-01-01', '2025-12-01', freq='MS')):
        raise ValueError('New sentiment does not cover all 24 test months.')
    # Preserve the original historical monthly sentiment values verbatim.
    monthly = pd.concat([old[list(new_monthly.columns)], new_monthly], ignore_index=True)
    historical_scored = pd.read_csv(OLD_SCORES, parse_dates=['news_date'])
    all_scored = pd.concat([historical_scored, scored], ignore_index=True)
    if all_scored.external_article_id.duplicated().any():
        raise ValueError('Historical and new article scores overlap.')
    subject_monthly = aggregate_monthly_subject_risk(all_scored, source_code='NYT')
    monthly = monthly.merge(subject_monthly.drop(columns='News_Source'), on='Date',
                            how='left', validate='one_to_one')
    frame = prepare_experiment_frame(macro, monthly, source_code='NYT')
    # Keep calendar rows intact: dropping October before shifting would create false lags.
    coverage = {
        'row_count': len(frame), 'first_month': str(frame.Date.min().date()),
        'last_month': str(frame.Date.max().date()),
        'missing_calendar_months': expected.difference(frame.Date).strftime('%Y-%m-%d').tolist(),
        'months_with_sentiment': int(frame.News_Sentiment.notna().sum()),
        'months_without_sentiment': int(frame.News_Sentiment.isna().sum()),
        'article_count_total': int(frame.Article_Count.sum()),
        'macro_missing_values': {s: int(frame[s].isna().sum()) for s in REQUIRED_MACRO_COLUMNS},
    }
    if coverage['missing_calendar_months'] or coverage['months_without_sentiment']:
        raise ValueError('Extended monthly dataset is incomplete.')
    metadata = build_experiment_metadata(frame, source_record={
        'source_code': 'NYT', 'source_name': 'The New York Times Archive API', 'is_active': True,
    }, coverage=coverage, test_start='2022-01-01')
    metadata['split']['test_last_month'] = '2025-11-01'
    metadata['extension'] = {
        'historical_dataset_sha256': sha256(OLD), 'historical_scores_sha256': sha256(OLD_SCORES),
        'new_articles': len(scored), 'new_scores_sha256': sha256(NEW_SCORES),
        'sentiment_model': 'ProsusAI/finbert', 'historical_articles_rescored': 0,
        'sentiment_methods': ['generic_finbert', 'subject_negative_intensity'],
        'macro_terms': list(MACRO_TERMS), 'us_terms': list(US_TERMS),
        'fred_sources': fred_sources,
        'fred_policy': 'Refresh the full macro history consistently; preserve the original experiment separately.',
        'evaluation_caveat': 'Latest revised FRED data, not vintage data available at each historical forecast origin.',
        'missing_observations': missing, 'missing_data_policy': 'Preserve missing values; never impute test actuals.',
        'missing_data_source': 'https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm',
    }
    paths = export_experiment_dataset(frame, metadata, output_dir=OUT)
    development_path = OUT / 'development_2007_2023.csv'
    test_path = OUT / 'test_2024_2025.csv'
    frame.loc[frame.Date.le('2023-12-01')].to_csv(development_path, index=False)
    frame.loc[frame.Date.ge('2024-01-01')].to_csv(test_path, index=False)
    print(json.dumps({'coverage': coverage, 'new_articles_scored': len(scored),
                       'files': [str(p) for p in (*paths, development_path, test_path)]},
                     indent=2), flush=True)
    return frame, metadata


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'download', 'score', 'build'], default='all')
    parser.add_argument('--model', default='ProsusAI/finbert', help='Existing FinBERT model or local snapshot')
    args = parser.parse_args()
    if args.stage in ('all', 'download'):
        download_fred()
        download_news()
    if args.stage in ('all', 'score'):
        score_news(args.model)
    if args.stage in ('all', 'build'):
        build_dataset()


if __name__ == '__main__':
    main()
